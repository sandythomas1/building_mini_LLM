"""Train or fine-tune the GPT model on a dataset built by prepare_data.py.

Stage 1, pretrain from scratch:
    python src/train.py --data-dir ~/mini_llm_data/fineweb_edu --out-dir ~/mini_llm_runs/pretrain

Stage 2, fine-tune on support chats from the best pretrained weights:
    python src/train.py --data-dir ~/mini_llm_data/support --out-dir ~/mini_llm_runs/support \
        --init-from ~/mini_llm_runs/pretrain/best.pt --lr 1e-4 --epochs 4 --dropout 0.1

Ctrl-C saves last.pt; re-run the same command with --resume to continue.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from data_loader import TokenBatchLoader
from model import GPT, MODEL_PRESETS, GPTConfig


def get_lr(step: int, *, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup to max_lr, then cosine decay to min_lr at max_steps (held after)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Directory containing train.bin / val.bin")
    p.add_argument("--out-dir", required=True, help="Where checkpoints and log.jsonl are written")
    start = p.add_mutually_exclusive_group()
    start.add_argument("--resume", action="store_true", help="Continue from <out-dir>/last.pt")
    start.add_argument("--init-from", default=None, help="Start from a checkpoint's weights (fine-tuning)")
    p.add_argument("--overwrite", action="store_true", help="Allow a new run to replace checkpoints in --out-dir")

    p.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="small")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)

    # Measured on an 8 GB RTX 5060 (small preset, block 512): micro-batch 8 peaks at 2.7 GiB and ~79k tok/s.
    # Micro-batch 32 overflows VRAM via the (B, T, 50304) logits tensor.
    p.add_argument("--micro-batch", type=int, default=8, help="Sequences per forward pass")
    p.add_argument("--grad-accum", type=int, default=16, help="Micro-batches per optimizer step")
    p.add_argument("--epochs", type=float, default=1.0, help="Passes over train.bin (sets max steps)")
    p.add_argument("--max-steps", type=int, default=None, help="Overrides --epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=None, help="Default: 5%% of steps, max 500")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--no-compile", dest="compile", action="store_false", help="Disable torch.compile")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args(argv)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    # weights_only=True refuses arbitrary pickled objects, so a tampered checkpoint can't run code.
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Write to a temp file then rename, so a crash mid-save never corrupts the previous checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def ensure_out_dir_is_safe(out_dir: Path, resume: bool, overwrite: bool) -> None:
    """Refuse to start a fresh run on top of an existing one: its first eval would replace best.pt."""
    existing = [name for name in ("best.pt", "last.pt") if (out_dir / name).exists()]
    if existing and not (resume or overwrite):
        raise FileExistsError(
            f"{out_dir} already contains {', '.join(existing)}. Use --resume to continue that run, "
            "--overwrite to replace it, or choose a different --out-dir."
        )


def resolve_max_steps(args: argparse.Namespace, train_tokens: int, tokens_per_step: int) -> int:
    if args.max_steps is not None:
        return args.max_steps
    return max(1, math.ceil(args.epochs * train_tokens / tokens_per_step))


def build_model_config(args: argparse.Namespace, checkpoint: dict[str, Any] | None) -> GPTConfig:
    if checkpoint is None:
        return GPTConfig(block_size=args.block_size, dropout=args.dropout, **MODEL_PRESETS[args.model_size])
    config = GPTConfig(**checkpoint["model_config"])
    config.dropout = args.dropout  # dropout is a training choice, not part of the learned weights
    return config


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, loader: TokenBatchLoader, eval_iters: int, autocast: contextlib.AbstractContextManager
) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.get_batch(split)
            with autocast:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_out_dir_is_safe(out_dir, args.resume, args.overwrite)
    data_dir = str(Path(args.data_dir).expanduser())
    if data_dir.startswith("/mnt/"):
        print("Warning: data is on the Windows filesystem (/mnt/...). Random reads over WSL's 9P bridge "
              "are slow; copy it to the Linux filesystem (e.g. ~/mini_llm_data) for faster training.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # bf16 autocast: ~2x faster tensor-core math and half the activation memory, with fp32's
    # exponent range so no loss scaling is needed (unlike fp16). Weights stay in fp32.
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda and torch.cuda.is_bf16_supported()
        else contextlib.nullcontext()
    )

    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(out_dir / "last.pt")
    elif args.init_from:
        checkpoint = load_checkpoint(Path(args.init_from).expanduser())

    config = build_model_config(args, checkpoint)
    model = GPT(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    print(f"Model: {model.num_params() / 1e6:.1f}M non-embedding params, "
          f"{model.num_params(non_embedding=False) / 1e6:.1f}M total, config={asdict(config)}")

    tokens_per_step = args.micro_batch * args.grad_accum * config.block_size
    start_step = checkpoint["step"] if args.resume else 0
    loader = TokenBatchLoader(data_dir, args.micro_batch, config.block_size, device, seed=args.seed + start_step)
    max_steps = resolve_max_steps(args, loader.num_tokens("train"), tokens_per_step)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else min(500, max(1, max_steps // 20))
    min_lr = args.lr * args.min_lr_ratio
    print(f"Training {max_steps:,} steps x {tokens_per_step:,} tokens/step "
          f"({max_steps * tokens_per_step / 1e9:.2f}B tokens; train set {loader.num_tokens('train') / 1e6:,.1f}M)")

    optimizer = model.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.95), device.type)
    best_val = math.inf
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_val = checkpoint["best_val_loss"]
    checkpoint = None  # free host memory held by the loaded state dicts

    raw_model = model
    if args.compile and use_cuda:
        print("Compiling model (first steps will be slow)...")
        model = torch.compile(model)

    def snapshot(step: int) -> dict[str, Any]:
        return {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(config),
            "step": step,
            "best_val_loss": best_val,
            "train_args": vars(args),
        }

    # A resumed run extends its log; a new run starts a clean one.
    log_file = open(out_dir / "log.jsonl", "a" if args.resume else "w", encoding="utf-8")

    def log(record: dict[str, Any]) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    step = start_step
    last_eval: dict[str, float] = {}
    x, y = loader.get_batch("train")
    t0 = time.time()
    try:
        while step < max_steps:
            lr = get_lr(step, max_lr=args.lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if step % args.eval_interval == 0 or step == max_steps - 1:
                last_eval = estimate_loss(model, loader, args.eval_iters, autocast)
                # Nothing is saved before the first optimizer step of this run (no new weights yet).
                is_new_best = last_eval["val"] < best_val and step > start_step
                print(f"step {step:>6}: train {last_eval['train']:.4f}  val {last_eval['val']:.4f}"
                      f"{'  (new best, saved)' if is_new_best else ''}", flush=True)
                log({"step": step, "event": "eval", **last_eval, "lr": lr})
                if is_new_best:
                    best_val = last_eval["val"]
                    save_checkpoint(out_dir / "best.pt", snapshot(step))
                if step > start_step:
                    save_checkpoint(out_dir / "last.pt", snapshot(step))

            for _ in range(args.grad_accum):
                with autocast:
                    _, loss = model(x, y)
                # Prefetch the next batch while the GPU is still busy with this one.
                x, y = loader.get_batch("train")
                (loss / args.grad_accum).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_interval == 0:
                loss_value = loss.item()  # .item() syncs the GPU, so only do it when logging
                elapsed = time.time() - t0
                t0 = time.time()
                tok_per_s = args.log_interval * tokens_per_step / elapsed
                peak_mem = f" | {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB" if use_cuda else ""
                print(f"step {step:>6} | loss {loss_value:.4f} | lr {lr:.2e} | grad {grad_norm.item():.2f} "
                      f"| {1000 * elapsed / args.log_interval:.0f} ms/step | {tok_per_s:,.0f} tok/s{peak_mem}",
                      flush=True)
                log({"step": step, "event": "train", "loss": loss_value, "lr": lr, "tok_per_s": tok_per_s})
    except KeyboardInterrupt:
        print(f"\nInterrupted at step {step}; saving last.pt (resume with --resume)")
    finally:
        if step > start_step:
            save_checkpoint(out_dir / "last.pt", snapshot(step))
        log_file.close()

    return {"step": step, "best_val_loss": best_val, **{f"last_{k}": v for k, v in last_eval.items()}}


if __name__ == "__main__":
    try:
        result = train(parse_args())
    except FileExistsError as error:
        raise SystemExit(f"Error: {error}") from None
    print(f"Finished: {result}")
