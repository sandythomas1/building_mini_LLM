"""Measure a real training step on the lab GPU without datasets or saved weights.

python src/benchmark.py --profile rtx4090 --objective mlm --steps 20 --out benchmark.json
Accepts train.py model, batch, compilation and memory flags. Uses synthetic resident
tokens: this measures compute throughput, not disk/network or model quality.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from data_loader import mask_tokens
from model import GPT, Encoder
from train import build_model_config, parse_args


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="benchmark.json")
    bench, remaining = parser.parse_known_args(argv)
    args = parse_args(["--data-dir", "unused", "--out-dir", "unused", *remaining])
    if bench.steps < 1 or bench.warmup < 1:
        parser.error("--steps and --warmup must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Benchmark requires a CUDA GPU. Run this command on a lab computer.")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = build_model_config(args, None)
    model = (GPT(config) if config.causal else Encoder(config)).to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    model.loss_chunk_size = args.loss_chunk_size
    optimizer = model.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.95), "cuda")
    if args.compile:
        if os.name == "nt":
            print("Native Windows: eager benchmark; use Linux/WSL for torch.compile.")
        else:
            model = torch.compile(model)
    tokens = torch.randint(0, 50256, (args.micro_batch, config.block_size + 1))
    if config.causal:
        x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    else:
        x, y = mask_tokens(tokens[:, :-1].contiguous(), torch.Generator().manual_seed(args.seed), args.mask_probability)
    x, y = x.to(device), y.to(device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else contextlib.nullcontext()
    )

    def step() -> None:
        for _ in range(args.grad_accum):
            with autocast:
                _, loss = model(x, y, return_logits=False)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    try:
        for _ in range(bench.warmup):
            step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        for _ in range(bench.steps):
            step()
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "Out of VRAM. Halve --micro-batch and double --grad-accum; "
            "then try --gradient-checkpointing or a smaller --loss-chunk-size."
        ) from None
    seconds = start.elapsed_time(end) / 1000
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "compiled": bool(args.compile and os.name != "nt"),
        "precision": "bf16-autocast" if torch.cuda.is_bf16_supported() else "fp32",
        "model_config": asdict(config),
        "train_args": vars(args),
        "warmup": bench.warmup,
        "steps": bench.steps,
        "input_tokens_per_second": bench.steps * args.micro_batch * args.grad_accum * config.block_size / seconds,
        "seconds_per_step": seconds / bench.steps,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "total_vram_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
        "scope": "synthetic resident batches; forward + backward + optimizer; excludes compile warmup and data I/O",
    }
    out = Path(bench.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
