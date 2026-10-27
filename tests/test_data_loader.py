import numpy as np
import pytest

from data_loader import TokenBatchLoader


def write_split(directory, split, tokens):
    np.asarray(tokens, dtype=np.uint16).tofile(directory / f"{split}.bin")


@pytest.fixture
def data_dir(tmp_path):
    write_split(tmp_path, "train", np.arange(1000) % 50000)
    write_split(tmp_path, "val", np.arange(200) + 60000)  # > int16 range: checks unsigned handling
    return tmp_path


def test_batches_are_shifted_by_one_token(data_dir):
    loader = TokenBatchLoader(data_dir, batch_size=4, block_size=8, device="cpu")
    x, y = loader.get_batch("train")
    assert x.shape == y.shape == (4, 8)
    assert (y[:, :-1] == x[:, 1:]).all()
    # The train file is a ramp, so every target is its input + 1.
    assert (y == x + 1).all()


def test_val_split_reads_val_file_with_uint16_values(data_dir):
    loader = TokenBatchLoader(data_dir, batch_size=2, block_size=8, device="cpu")
    x, _ = loader.get_batch("val")
    assert x.min() >= 60000
    assert x.dtype.is_floating_point is False


def test_same_seed_gives_same_batches(data_dir):
    a = TokenBatchLoader(data_dir, 4, 8, "cpu", seed=7).get_batch("train")[0]
    b = TokenBatchLoader(data_dir, 4, 8, "cpu", seed=7).get_batch("train")[0]
    c = TokenBatchLoader(data_dir, 4, 8, "cpu", seed=8).get_batch("train")[0]
    assert (a == b).all()
    assert not (a == c).all()


def test_rejects_files_shorter_than_a_window(tmp_path):
    write_split(tmp_path, "train", range(100))
    write_split(tmp_path, "val", range(5))
    with pytest.raises(ValueError, match="val.bin"):
        TokenBatchLoader(tmp_path, 2, block_size=8, device="cpu")


def test_rejects_unknown_split(data_dir):
    loader = TokenBatchLoader(data_dir, 2, 8, "cpu")
    with pytest.raises(ValueError, match="split"):
        loader.get_batch("test")


def test_loader_state_restores_next_batch(data_dir):
    loader = TokenBatchLoader(data_dir, 4, 8, "cpu")
    loader.get_batch("train")
    state = loader.state_dict()
    expected = loader.get_batch("train")[0]
    loader.load_state_dict(state)
    assert (loader.get_batch("train")[0] == expected).all()


def test_exactly_one_window_is_valid(tmp_path):
    for split in ("train", "val"):
        write_split(tmp_path, split, range(9))
    x, y = TokenBatchLoader(tmp_path, 2, 8, "cpu").get_batch("train")
    assert x.tolist() == [list(range(8))] * 2
    assert y.tolist() == [list(range(1, 9))] * 2
