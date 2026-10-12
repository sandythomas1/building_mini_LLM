"""Random-window batch sampler over a flat uint16 token file (nanoGPT-style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint16


class TokenBatchLoader:
    """Samples (x, y) batches of `block_size` windows where y is x shifted by one token.

    Expects `<data_dir>/train.bin` and `<data_dir>/val.bin` written by prepare_data.py.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int,
        block_size: int,
        device: str | torch.device,
        seed: int = 1337,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self._offsets = np.arange(block_size + 1)
        for split in ("train", "val"):
            n = self.num_tokens(split)
            if n <= block_size + 1:
                raise ValueError(
                    f"{self._path(split)} has {n} tokens; need more than block_size + 1 ({block_size + 1})"
                )

    def _path(self, split: str) -> Path:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        return self.data_dir / f"{split}.bin"

    def _open(self, split: str) -> np.memmap:
        # Re-opened per batch on purpose: a long-lived np.memmap keeps every page it has
        # touched referenced, which slowly grows RSS on multi-GB files.
        return np.memmap(self._path(split), dtype=TOKEN_DTYPE, mode="r")

    def num_tokens(self, split: str) -> int:
        return self._path(split).stat().st_size // np.dtype(TOKEN_DTYPE).itemsize

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._open(split)
        starts = self._rng.integers(0, len(data) - self.block_size - 1, size=self.batch_size)
        windows = data[starts[:, None] + self._offsets].astype(np.int64)  # (B, block_size + 1)
        chunk = torch.from_numpy(windows)
        x, y = chunk[:, :-1], chunk[:, 1:]
        if self.device.type == "cuda":
            # Pinned host memory lets the copy overlap with GPU compute.
            return (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True),
            )
        return x.contiguous().to(self.device), y.contiguous().to(self.device)
