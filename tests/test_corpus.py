import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from prepare_data import SplitTokenWriter, prepare_fineweb, verify_dataset


class Stream:
    def shuffle(self, **kwargs):
        return self

    def __iter__(self):
        for i in range(100):
            text = f"This is educational document {i}. " * 20
            yield {"text": text}
            yield {"text": "  " + text + "\n"}
        yield {"text": ""}


def test_corpus_is_bounded_deduplicated_and_versioned(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: Stream()))
    prepare_fineweb(str(tmp_path), 1000, batch_docs=7, revision="fixture", min_chars=50)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert 0 < meta["train_tokens"] + meta["val_tokens"] <= 1000
    assert meta["preparation"]["revision"] == "fixture"
    assert meta["quality"]["duplicates"] > 0
    assert all(len(meta["sha256"][split]) == 64 for split in ("train", "val"))
    assert np.fromfile(tmp_path / "train.bin", dtype=np.uint16).size > 0
    with pytest.raises(FileExistsError):
        prepare_fineweb(str(tmp_path), 1000, revision="fixture")


def test_verification_detects_same_size_corruption(tmp_path):
    with SplitTokenWriter(tmp_path, val_every=2, sources=["fixture"]) as writer:
        writer.add_documents(["a useful educational text", "another useful educational text"])
    verify_dataset(tmp_path)
    path = tmp_path / "train.bin"
    contents = bytearray(path.read_bytes())
    contents[0] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_dataset(tmp_path)
