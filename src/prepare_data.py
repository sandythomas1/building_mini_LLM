"""Build tokenized training datasets: <out>/train.bin, <out>/val.bin, <out>/meta.json.

Two datasets, matching a two-stage recipe (pretrain on general text, then fine-tune on support chats):

    # Stage 1 corpus: high-quality educational web text (streamed, bounded memory)
    python src/prepare_data.py fineweb --out ~/mini_llm_data/fineweb_edu --max-tokens 1000000000

    # Stage 2 corpus: Bitext support Q/A + the local CSV with its random-noise suffixes removed
    python src/prepare_data.py support --out ~/mini_llm_data/support --local-csv data/customer_support_data.csv

Tokens are GPT-2 BPE (tiktoken) stored as uint16; every document ends with <|endoftext|>.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType

import numpy as np
import tiktoken

from chat_format import Turn, format_conversation

FINEWEB_DATASET = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
BITEXT_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# The local CSV is synthetic: each message is a template sentence followed by a run of 20+
# random lowercase pseudo-words ("... as expected. tyugwmtolj qripqglpz ..."). That noise was
# ~91% of all characters. Require 8+ such words so a real lowercase tail is never stripped.
_NOISE_SUFFIX = re.compile(r"^(.*?[.!?])(?:\s+[a-z]{1,12}){8,}\s*$", re.DOTALL)


def strip_noise(text: str) -> str:
    """Remove a trailing run of random pseudo-words that follows sentence-ending punctuation."""
    match = _NOISE_SUFFIX.match(text)
    return (match.group(1) if match else text).strip()


def iter_local_support_conversations(csv_path: str | Path) -> Iterator[list[Turn]]:
    """Yield cleaned conversations from the local CSV, turns ordered by turn_index.

    Streams rows (the file is ~600 MB) and keeps only the short cleaned text in memory.
    """
    conversations: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = strip_noise(row["text"])
            if text:
                conversations[row["conv_id"]].append((int(row["turn_index"]), row["role"], text))
    for turns in conversations.values():
        yield [(role, text) for _, role, text in sorted(turns)]


def iter_bitext_conversations() -> Iterator[list[Turn]]:
    from datasets import load_dataset

    for row in load_dataset(BITEXT_DATASET, split="train"):
        yield [("customer", row["instruction"]), ("agent", row["response"])]


def dedupe(documents: Iterable[str]) -> list[str]:
    """Drop exact duplicates while preserving first-seen order."""
    return list(dict.fromkeys(documents))


class SplitTokenWriter:
    """Tokenizes documents and appends them to train/val .bin files.

    Every `val_every`-th document goes to validation, so the split is at document granularity
    (no window straddles train and val). Files are written as *.tmp and only renamed into place
    on a clean exit, so an interrupted run never leaves a dataset that looks complete.
    """

    def __init__(self, out_dir: str | Path, val_every: int, sources: list[str]):
        if val_every < 2:
            raise ValueError("val_every must be >= 2")
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.val_every = val_every
        self.sources = sources
        self.encoder = tiktoken.get_encoding("gpt2")
        self.eot = self.encoder.eot_token
        self.counts = {"train": 0, "val": 0}
        self.documents = 0
        self._files = {split: open(self._tmp_path(split), "wb") for split in ("train", "val")}

    def _tmp_path(self, split: str) -> Path:
        return self.out_dir / f"{split}.bin.tmp"

    @property
    def total_tokens(self) -> int:
        return self.counts["train"] + self.counts["val"]

    def add_documents(self, texts: list[str]) -> None:
        encoded = self.encoder.encode_ordinary_batch(texts, num_threads=os.cpu_count() or 1)
        buffers: dict[str, list[int]] = {"train": [], "val": []}
        for tokens in encoded:
            split = "val" if self.documents % self.val_every == 0 else "train"
            buffers[split].extend(tokens)
            buffers[split].append(self.eot)
            self.documents += 1
        for split, tokens in buffers.items():
            if tokens:
                np.asarray(tokens, dtype=np.uint16).tofile(self._files[split])
                self.counts[split] += len(tokens)

    def __enter__(self) -> SplitTokenWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for f in self._files.values():
            f.close()
        if exc_type is not None:
            for split in self._files:
                self._tmp_path(split).unlink(missing_ok=True)
            return
        for split in self._files:
            os.replace(self._tmp_path(split), self.out_dir / f"{split}.bin")
        meta = {
            "tokenizer": "gpt2",
            "vocab_size": self.encoder.n_vocab,
            "eot_token": self.eot,
            "dtype": "uint16",
            "documents": self.documents,
            "train_tokens": self.counts["train"],
            "val_tokens": self.counts["val"],
            "sources": self.sources,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def prepare_fineweb(out_dir: str, max_tokens: int, batch_docs: int = 1024) -> None:
    from datasets import load_dataset

    name, config = FINEWEB_DATASET
    stream = load_dataset(name, name=config, split="train", streaming=True)
    started = time.time()
    next_report = 50_000_000
    with SplitTokenWriter(out_dir, val_every=200, sources=[f"{name}/{config}"]) as writer:
        batch: list[str] = []
        for doc in stream:
            batch.append(doc["text"])
            if len(batch) < batch_docs:
                continue
            writer.add_documents(batch)
            batch = []
            if writer.total_tokens >= next_report:
                rate = writer.total_tokens / (time.time() - started)
                print(f"{writer.total_tokens / 1e6:,.0f}M tokens ({rate / 1e6:.2f}M tok/s)", flush=True)
                next_report += 50_000_000
            if writer.total_tokens >= max_tokens:
                break
        if batch and writer.total_tokens < max_tokens:
            writer.add_documents(batch)
    print(f"Done: {writer.counts} tokens -> {writer.out_dir}")


def prepare_support(out_dir: str, local_csv: str | None, max_local_conversations: int, seed: int) -> None:
    sources: list[str] = []
    documents: list[str] = []

    bitext = dedupe(format_conversation(turns) for turns in iter_bitext_conversations())
    documents += bitext
    sources.append(BITEXT_DATASET)
    print(f"Bitext: {len(bitext):,} unique conversations")

    if local_csv:
        local = dedupe(format_conversation(t) for t in iter_local_support_conversations(local_csv))
        # The local data is built from only a few hundred template sentences, so it is capped
        # to keep those templates from dominating what the model learns.
        random.Random(seed).shuffle(local)
        local = local[:max_local_conversations]
        documents += local
        sources.append(f"local:{Path(local_csv).name}")
        print(f"Local CSV: using {len(local):,} unique cleaned conversations")

    random.Random(seed).shuffle(documents)  # mix sources before the deterministic val split
    with SplitTokenWriter(out_dir, val_every=20, sources=sources) as writer:
        for start in range(0, len(documents), 1024):
            writer.add_documents(documents[start : start + 1024])
    print(f"Done: {writer.counts} tokens -> {writer.out_dir}")


def positive_int(value: str) -> int:
    number = int(float(value))  # accepts 1e9-style input
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)

    fineweb = sub.add_parser("fineweb", help="General-text pretraining corpus (FineWeb-Edu)")
    fineweb.add_argument("--out", required=True)
    fineweb.add_argument("--max-tokens", type=positive_int, default=1_000_000_000)

    support = sub.add_parser("support", help="Customer-support fine-tuning corpus")
    support.add_argument("--out", required=True)
    support.add_argument("--local-csv", default=None, help="Optional path to customer_support_data.csv")
    support.add_argument("--max-local-conversations", type=positive_int, default=5000)
    support.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dataset == "fineweb":
        prepare_fineweb(args.out, args.max_tokens)
    else:
        prepare_support(args.out, args.local_csv, args.max_local_conversations, args.seed)


if __name__ == "__main__":
    main()
