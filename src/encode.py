"""Extract contextual encoder vectors: python src/encode.py --ckpt runs/encoder/best.pt --text 'Hello'."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tiktoken
import torch
from torch.nn import functional as F

from model import Encoder, GPTConfig
from train import load_checkpoint


def load_encoder(path: str, device: torch.device) -> Encoder:
    """Load an MLM checkpoint, refusing causal GPT weights."""
    checkpoint = load_checkpoint(Path(path).expanduser())
    config = GPTConfig(**checkpoint["model_config"])
    if config.causal:
        raise ValueError("Expected an encoder checkpoint trained with --objective mlm")
    model = Encoder(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def embed_text(model: Encoder, text: str) -> torch.Tensor:
    """Mean-pool and normalize token representations, without silently truncating text."""
    tokens = tiktoken.get_encoding("gpt2").encode_ordinary(text)
    if not tokens:
        raise ValueError("Text must contain at least one token")
    if len(tokens) > model.config.block_size:
        raise ValueError(f"Text has {len(tokens)} tokens; encoder context is {model.config.block_size}")
    x = torch.tensor([tokens], device=next(model.parameters()).device)
    return F.normalize(model.encode(x).float().mean(dim=1), dim=-1).squeeze(0).cpu()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", default="embedding.npy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    vector = embed_text(load_encoder(args.ckpt, torch.device(args.device)), args.text)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vector.numpy())
    print(f"Saved {vector.numel()}-dimensional contextual embedding to {out}")


if __name__ == "__main__":
    main()
