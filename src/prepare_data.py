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
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import tempfile
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path
from types import TracebackType

import numpy as np
import tiktoken

from chat_format import Turn, format_conversation

FINEWEB_DATASET = ("HuggingFaceFW/fineweb-edu", "sample-100BT")
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

    Splits use document index or a normalized-text hash, always at document granularity
    (no window straddles train and val). Files are written as *.tmp and only renamed into place
    on a clean exit, so an interrupted run never leaves a dataset that looks complete.
    """

    def __init__(
        self,
        out_dir: str | Path,
        val_every: int,
        sources: list[str],
        *,
        max_tokens: int | None = None,
        hash_split: bool = False,
        seed: int = 1337,
        num_threads: int = 8,
    ):
        if val_every < 2:
            raise ValueError("val_every must be >= 2")
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if any((self.out_dir / name).exists() for name in ("train.bin", "val.bin", "meta.json")):
            raise FileExistsError(f"{self.out_dir} already contains a dataset; choose a new --out directory")
        self.val_every = val_every
        self.max_tokens = max_tokens
        self.hash_split = hash_split
        self.seed = seed
        self.num_threads = num_threads
        self.extra_meta: dict = {}
        self._hashes = {split: hashlib.sha256() for split in ("train", "val")}
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
        encoded = self.encoder.encode_ordinary_batch(texts, num_threads=self.num_threads)
        buffers: dict[str, list[int]] = {"train": [], "val": []}
        pending = 0
        for text, tokens in zip(texts, encoded):
            if self.max_tokens is not None and self.total_tokens + pending + len(tokens) + 1 > self.max_tokens:
                continue  # Keep whole documents, never overflow the requested budget.
            if self.hash_split:
                canonical = " ".join(unicodedata.normalize("NFKC", text).split())
                key = hashlib.sha256(f"{self.seed}:{canonical}".encode("utf-8")).digest()
                is_val = int.from_bytes(key[:8], "big") % self.val_every == 0
            else:
                is_val = self.documents % self.val_every == 0
            split = "val" if is_val else "train"
            buffers[split].extend(tokens)
            buffers[split].append(self.eot)
            pending += len(tokens) + 1
            self.documents += 1
        for split, tokens in buffers.items():
            if tokens:
                raw = np.asarray(tokens, dtype="<u2").tobytes()
                self._files[split].write(raw)
                self._hashes[split].update(raw)
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
            "sha256": {split: h.hexdigest() for split, h in self._hashes.items()},
            "split_method": "normalized-text-sha256" if self.hash_split else "document-index",
            **self.extra_meta,
        }
        meta_tmp = self.out_dir / "meta.json.tmp"
        meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        os.replace(meta_tmp, self.out_dir / "meta.json")


def prepare_fineweb(
    out_dir: str,
    max_tokens: int,
    batch_docs: int = 256,
    *,
    config: str = "sample-100BT",
    revision: str | None = None,
    seed: int = 1337,
    shuffle_buffer: int = 1000,
    min_chars: int = 200,
    max_chars: int = 100_000,
    num_threads: int = 8,
    val_every: int = 200,
) -> None:
    """Stream, filter, disk-dedupe and tokenize a pinned FineWeb snapshot with bounded RAM.

    The budget includes validation. Stops below the limit at a document boundary.
    SQLite fingerprints avoid keeping millions of Python strings in memory.
    """
    from datasets import load_dataset

    out = Path(out_dir).expanduser()
    if any((out / name).exists() for name in ("train.bin", "val.bin", "meta.json")):
        raise FileExistsError(f"{out} already contains a dataset; choose a new --out directory")
    if max_tokens <= 0 or batch_docs <= 0 or num_threads <= 0 or shuffle_buffer <= 0:
        raise ValueError("token budget, batch size, threads and shuffle buffer must be positive")
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("require 0 <= min_chars <= max_chars")
    name = FINEWEB_DATASET[0]
    if revision is None:
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(name).sha
    print(f"Streaming {name}/{config} at revision {revision}; budget {max_tokens:,} tokens", flush=True)
    stream = load_dataset(name, name=config, split="train", streaming=True, revision=revision, columns=["text"])
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    started = time.time()
    quality = {"seen": 0, "duplicates": 0, "filtered": 0}
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dedupe-", dir=out) as temp_dir:
        with (
            closing(sqlite3.connect(str(Path(temp_dir) / "seen.sqlite"))) as db,
            SplitTokenWriter(
                out,
                val_every=val_every,
                sources=[f"{name}/{config}@{revision}"],
                max_tokens=max_tokens,
                hash_split=True,
                seed=seed,
                num_threads=num_threads,
            ) as writer,
        ):
            db.execute("CREATE TABLE seen (hash BLOB PRIMARY KEY) WITHOUT ROWID")
            batch: list[str] = []
            next_report = 10_000_000
            for doc in stream:
                quality["seen"] += 1
                text = unicodedata.normalize("NFKC", doc["text"]).replace("\r\n", "\n").strip()
                if not min_chars <= len(text) <= max_chars or "\x00" in text:
                    quality["filtered"] += 1
                    continue
                digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()
                cursor = db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,))
                if not cursor.rowcount:
                    quality["duplicates"] += 1
                    continue
                batch.append(text)
                if len(batch) < batch_docs:
                    continue
                before = writer.total_tokens
                writer.add_documents(batch)
                batch = []
                db.commit()
                if writer.total_tokens >= next_report:
                    rate = writer.total_tokens / max(time.time() - started, 0.001)
                    print(f"{writer.total_tokens / 1e6:,.1f}M tokens ({rate:,.0f} tok/s)", flush=True)
                    next_report = writer.total_tokens + 10_000_000
                # Stop once a full batch cannot fit, rather than scanning the entire web
                # trying to find tiny documents to fill the last few tokens.
                if writer.total_tokens == before or writer.total_tokens >= max_tokens:
                    break
            if batch:
                writer.add_documents(batch)
            writer.extra_meta = {
                "preparation": {
                    "revision": revision,
                    "config": config,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "shuffle_buffer": shuffle_buffer,
                    "min_chars": min_chars,
                    "max_chars": max_chars,
                    "batch_docs": batch_docs,
                    "num_threads": num_threads,
                    "val_every": val_every,
                    "normalization": "NFKC; CRLF to LF; trim; whitespace-folded exact dedupe",
                    "deduplication": "sha256-sqlite",
                    "license": "ODC-BY-1.0 (upstream content rights apply)",
                },
                "quality": quality,
            }
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

    documents = dedupe(documents)  # Also dedupe across sources before splitting.
    random.Random(seed).shuffle(documents)  # mix sources before the deterministic val split
    with SplitTokenWriter(out_dir, val_every=20, sources=sources) as writer:
        for start in range(0, len(documents), 1024):
            writer.add_documents(documents[start : start + 1024])
    print(f"Done: {writer.counts} tokens -> {writer.out_dir}")


def positive_int(value: str) -> int:
    raw = float(value)  # accepts 1e9-style input
    if not math.isfinite(raw) or raw <= 0 or not raw.is_integer():
        raise argparse.ArgumentTypeError("must be a positive integer")
    return int(raw)


def verify_dataset(data_dir: str | Path) -> dict:
    """Verify saved corpus counts, GPT-2 token range and SHA-256 with bounded RAM."""
    directory = Path(data_dir).expanduser()
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta.get("tokenizer") != "gpt2" or meta.get("dtype") != "uint16" or "sha256" not in meta:
        raise ValueError("Expected a versioned GPT-2 uint16 corpus with SHA-256 hashes")
    for split in ("train", "val"):
        path = directory / f"{split}.bin"
        if path.stat().st_size != 2 * meta[f"{split}_tokens"] or not meta[f"{split}_tokens"]:
            raise ValueError(f"{split}: token count mismatch or empty split")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while raw := handle.read(8 * 1024 * 1024):
                digest.update(raw)
                if np.frombuffer(raw, dtype="<u2").max() >= 50257:
                    raise ValueError(f"{split}: token outside GPT-2 vocabulary")
        if digest.hexdigest() != meta["sha256"][split]:
            raise ValueError(f"{split}: SHA-256 mismatch; rebuild or recopy the corpus")
    print(f"Verified {meta['train_tokens']:,} train + {meta['val_tokens']:,} validation tokens in {directory}")
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)

    fineweb = sub.add_parser("fineweb", help="General-text pretraining corpus (FineWeb-Edu)")
    fineweb.add_argument("--out", required=True)
    fineweb.add_argument(
        "--max-tokens",
        type=positive_int,
        default=5_000_000_000,
        help="Train + validation token budget (default: 5B, about 10 GB of token files)",
    )
    fineweb.add_argument("--config", choices=["sample-10BT", "sample-100BT", "sample-350BT"], default="sample-100BT")
    fineweb.add_argument("--revision", help="HF dataset commit SHA (default: resolve and record current SHA)")
    fineweb.add_argument("--seed", type=int, default=1337)
    fineweb.add_argument("--batch-docs", type=positive_int, default=256)
    fineweb.add_argument("--num-threads", type=positive_int, default=min(8, os.cpu_count() or 1))
    fineweb.add_argument("--shuffle-buffer", type=positive_int, default=1000)
    fineweb.add_argument("--min-chars", type=int, default=200)
    fineweb.add_argument("--max-chars", type=positive_int, default=100_000)
    fineweb.add_argument("--val-every", type=positive_int, default=200)

    support = sub.add_parser("support", help="Customer-support fine-tuning corpus")
    support.add_argument("--out", required=True)
    support.add_argument("--local-csv", default=None, help="Optional path to customer_support_data.csv")
    support.add_argument("--max-local-conversations", type=positive_int, default=5000)
    support.add_argument("--seed", type=int, default=1337)
    verify = sub.add_parser("verify", help="Check corpus token counts, vocabulary and SHA-256 after preparation/copy")
    verify.add_argument("--data-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dataset == "fineweb":
        prepare_fineweb(
            args.out,
            args.max_tokens,
            args.batch_docs,
            config=args.config,
            revision=args.revision,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            num_threads=args.num_threads,
            val_every=args.val_every,
        )
    elif args.dataset == "support":
        prepare_support(args.out, args.local_csv, args.max_local_conversations, args.seed)
    else:
        verify_dataset(args.data_dir)


if __name__ == "__main__":
    main()
