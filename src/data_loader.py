"""Random-window batch sampler over a flat uint16 token file (nanoGPT-style)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint16


def mask_tokens(
    tokens: torch.Tensor, generator: torch.Generator, probability: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic BERT-style 80% MASK / 10% random / 10% unchanged corruption.

    50257 is an unused padded-vocabulary slot reserved for MASK; EOT is never masked.
    """
    if not 0 < probability <= 1:
        raise ValueError("mask probability must be in (0, 1]")
    eligible = tokens != 50256
    selected = (torch.rand(tokens.shape, generator=generator) < probability) & eligible
    if not selected.any():
        candidates = eligible.flatten().nonzero().flatten()
        if not candidates.numel():
            raise ValueError("MLM batch contains only end-of-text tokens")
        index = torch.randint(candidates.numel(), (1,), generator=generator)
        selected.view(-1)[candidates[index]] = True
    targets = tokens.clone().masked_fill(~selected, -100)
    inputs = tokens.clone()
    draw = torch.rand(tokens.shape, generator=generator)
    inputs[selected & (draw < 0.8)] = 50257
    random_positions = selected & (draw >= 0.8) & (draw < 0.9)
    random_tokens = torch.randint(50256, tokens.shape, generator=generator)
    inputs[random_positions] = random_tokens[random_positions]
    return inputs, targets


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
        if batch_size < 1 or block_size < 1:
            raise ValueError("batch_size and block_size must be positive")
        self._rng = np.random.default_rng(seed)
        self._mask_rng = torch.Generator().manual_seed(seed)
        self._offsets = np.arange(block_size + 1)
        for split in ("train", "val"):
            n = self.num_tokens(split)
            if n < block_size + 1:
                raise ValueError(f"{self._path(split)} has {n} tokens; need at least block_size + 1 ({block_size + 1})")

    def _path(self, split: str) -> Path:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        return self.data_dir / f"{split}.bin"

    def _open(self, split: str) -> np.memmap:
        # Re-opened per batch on purpose: a long-lived np.memmap keeps every page it has
        # touched referenced, which slowly grows RSS on multi-GB files.
        return np.memmap(self._path(split), dtype=TOKEN_DTYPE, mode="r")

    def num_tokens(self, split: str) -> int:
        size = self._path(split).stat().st_size
        if size % 2:
            raise ValueError(f"{self._path(split)} is not a complete uint16 token file")
        return size // np.dtype(TOKEN_DTYPE).itemsize

    def state_dict(self) -> dict:
        """Save sampler and dynamic masking state without serializing NumPy objects."""
        return {"rng": deepcopy(self._rng.bit_generator.state), "mask_rng": self._mask_rng.get_state()}

    def load_state_dict(self, state: dict) -> None:
        """Restore the next sampled window and MLM corruption exactly."""
        self._rng.bit_generator.state = deepcopy(state["rng"])
        self._mask_rng.set_state(state["mask_rng"].cpu())

    def get_batch(
        self, split: str, *, objective: str = "causal", mask_probability: float = 0.15
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._open(split)
        starts = self._rng.integers(0, len(data) - self.block_size, size=self.batch_size)
        windows = data[starts[:, None] + self._offsets].astype(np.int64)  # (B, block_size + 1)
        chunk = torch.from_numpy(windows)
        x, y = chunk[:, :-1], chunk[:, 1:]
        if objective == "mlm":
            x, y = mask_tokens(x.contiguous(), self._mask_rng, mask_probability)
        elif objective != "causal":
            raise ValueError("objective must be causal or mlm")
        if self.device.type == "cuda":
            # Pinned host memory lets the copy overlap with GPU compute.
            return (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True),
            )
        return x.contiguous().to(self.device), y.contiguous().to(self.device)
