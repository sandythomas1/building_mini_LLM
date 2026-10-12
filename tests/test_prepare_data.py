import csv
import json

import numpy as np
import pytest
import tiktoken

from chat_format import build_prompt, format_conversation, format_turn
from prepare_data import SplitTokenWriter, dedupe, iter_local_support_conversations, strip_noise

NOISE = "tyugwmtolj qripqglpz rlefnglpkd ooxsdlqmal fms qekwxufwjs xsgmw ppj efuoxelp zvq dik osynwai"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (f"Hello, my SSO is not working as expected. {NOISE}", "Hello, my SSO is not working as expected."),
        (f"Got it. I’ve retried, and it works now. {NOISE}", "Got it. I’ve retried, and it works now."),
        (f"Still not fixed. Can you escalate? {NOISE}  ", "Still not fixed. Can you escalate?"),
        ("Thanks, that helps.", "Thanks, that helps."),
        # A short real lowercase tail is not noise and must survive.
        ("Okay. see you then", "Okay. see you then"),
        ("No punctuation then many words a b c d e f g h", "No punctuation then many words a b c d e f g h"),
    ],
)
def test_strip_noise(raw, expected):
    assert strip_noise(raw) == expected


def test_format_conversation_normalizes_whitespace_and_skips_empty_turns():
    text = format_conversation([("customer", "  my  order\nis late "), ("agent", ""), ("Agent", "On it.")])
    assert text == "Customer: my order is late\nAgent: On it.\n"


def test_format_turn_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown role"):
        format_turn("system", "hi")


def test_build_prompt_ends_with_agent_cue():
    prompt = build_prompt([("customer", "Hi"), ("agent", "Hello!")], "Where is my refund?")
    assert prompt == "Customer: Hi\nAgent: Hello!\nCustomer: Where is my refund?\nAgent:"


def test_dedupe_keeps_first_occurrence_order():
    assert dedupe(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_local_conversations_are_cleaned_grouped_and_ordered(tmp_path):
    path = tmp_path / "support.csv"
    rows = [
        ("C2", 0, "customer", f"Need help with Wallet. {NOISE}"),
        ("C1", 1, "agent", f"Let me check your account. {NOISE}"),
        ("C1", 0, "customer", f"My SSO is broken. {NOISE}"),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conv_id", "turn_index", "role", "text"])
        writer.writerows(rows)

    conversations = sorted(iter_local_support_conversations(path))
    assert conversations == [
        [("customer", "My SSO is broken."), ("agent", "Let me check your account.")],
        [("customer", "Need help with Wallet.")],
    ]


def test_split_token_writer_routes_documents_and_writes_meta(tmp_path):
    docs = [f"document number {i}" for i in range(10)]
    with SplitTokenWriter(tmp_path, val_every=5, sources=["unit-test"]) as writer:
        writer.add_documents(docs[:6])
        writer.add_documents(docs[6:])

    enc = tiktoken.get_encoding("gpt2")
    train = np.fromfile(tmp_path / "train.bin", dtype=np.uint16).tolist()
    val = np.fromfile(tmp_path / "val.bin", dtype=np.uint16).tolist()
    expected_val = enc.encode_ordinary(docs[0]) + [enc.eot_token] + enc.encode_ordinary(docs[5]) + [enc.eot_token]
    assert val == expected_val
    assert train.count(enc.eot_token) == 8
    assert enc.decode(train[: train.index(enc.eot_token)]) == docs[1]

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["train_tokens"] == len(train)
    assert meta["val_tokens"] == len(val)
    assert meta["documents"] == 10
    assert meta["sources"] == ["unit-test"]
    assert not list(tmp_path.glob("*.tmp"))


def test_split_token_writer_discards_partial_output_on_error(tmp_path):
    with pytest.raises(RuntimeError):
        with SplitTokenWriter(tmp_path, val_every=5, sources=[]) as writer:
            writer.add_documents(["some text"])
            raise RuntimeError("download failed")
    assert list(tmp_path.iterdir()) == []


def test_split_token_writer_validates_val_every(tmp_path):
    with pytest.raises(ValueError):
        SplitTokenWriter(tmp_path, val_every=1, sources=[])
