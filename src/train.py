"""Single-GPU causal GPT or masked-language-model encoder training.

Prepare train.bin / val.bin first with prepare_data.py. Use --profile rtx4090
or rtx5090 on the lab GPU, and --objective mlm to train the encoder.
Ctrl-C / SIGTERM saves last.pt; repeat the command with --resume to continue.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from data_loader import TokenBatchLoader
from model import GPT, MODEL_PRESETS, Encoder, GPTConfig

# Conservative starting points, not measured throughput/fit guarantees. All have
# an effective batch of 65,536 input tokens. CLI values override profile values.
GPU_PROFILES = {
    "default": dict(model_size="small", block_size=512, micro_batch=8, grad_accum=16, lr=1e-3, loss_chunk_size=0),
    "rtx4090": dict(model_size="medium", block_size=1024, micro_batch=8, grad_accum=8, lr=6e-4, loss_chunk_size=1024),
    "rtx5090": dict(model_size="medium", block_size=1024, micro_batch=16, grad_accum=4, lr=6e-4, loss_chunk_size=1024),
}


def get_lr(step: int, *, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup, then cosine decay to the minimum learning rate."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    start = p.add_mutually_exclusive_group()
    start.add_argument("--resume", action="store_true")
    start.add_argument("--init-from", help="Load weights for a new run with the same architecture/objective")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--profile", choices=GPU_PROFILES, default="default")
    p.add_argument(
        "--objective",
        choices=["causal", "mlm"],
        default=None,
        help="Default: causal, or inherit the checkpoint's objective",
    )
    p.add_argument("--mask-probability", type=float, default=0.15)
    p.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--loss-chunk-size", type=int, default=None, help="Bound head logits memory; 0 disables chunking")
    p.add_argument("--gradient-checkpointing", action="store_true", help="Recompute blocks to save activation memory")
    p.add_argument("--epochs", type=float, default=1.0, help="Token exposure budget; windows are randomly sampled")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--save-interval", type=int, default=250, help="Save last.pt independently of evaluation")
    p.add_argument("--patience", type=int, default=0, help="Stop after this many unimproved evaluations; 0 disables")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)
    for key, value in GPU_PROFILES[args.profile].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    positive = (
        "block_size",
        "micro_batch",
        "grad_accum",
        "epochs",
        "lr",
        "grad_clip",
        "eval_interval",
        "eval_iters",
        "save_interval",
        "log_interval",
    )
    for key in positive:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive and finite")
    if args.max_steps is not None and args.max_steps <= 0:
        p.error("--max-steps must be positive")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        p.error("--warmup-steps must be nonnegative")
    if args.loss_chunk_size < 0 or args.patience < 0 or args.weight_decay < 0:
        p.error("loss chunk size, patience and weight decay must be nonnegative")
    if not 0 < args.mask_probability <= 1 or not 0 <= args.dropout < 1 or not 0 <= args.min_lr_ratio <= 1:
        p.error("require mask-probability in (0,1], dropout in [0,1), min-lr-ratio in [0,1]")
    return args


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load tensors and primitives only; never execute arbitrary checkpoint pickle code."""
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Publish atomically so a failed save preserves the previous checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def ensure_out_dir_is_safe(out_dir: Path, resume: bool, overwrite: bool) -> None:
    """Refuse an accidental fresh run over existing trained weights."""
    existing = [name for name in ("best.pt", "last.pt") if (out_dir / name).exists()]
    if existing and not (resume or overwrite):
        raise FileExistsError(
            f"{out_dir} contains {', '.join(existing)}; use --resume, --overwrite, or a new directory"
        )


def resolve_max_steps(args: argparse.Namespace, train_tokens: int, tokens_per_step: int) -> int:
    """Convert a token exposure budget to optimizer steps."""
    return (
        args.max_steps
        if args.max_steps is not None
        else max(1, math.ceil(args.epochs * train_tokens / tokens_per_step))
    )


def build_model_config(args: argparse.Namespace, checkpoint: dict[str, Any] | None) -> GPTConfig:
    """Keep old GPT checkpoints compatible; never silently convert encoder/decoder weights."""
    if checkpoint is None:
        return GPTConfig(
            block_size=args.block_size,
            dropout=args.dropout,
            causal=args.objective != "mlm",
            **MODEL_PRESETS[args.model_size],
        )
    config = GPTConfig(**checkpoint["model_config"])
    objective = "causal" if config.causal else "mlm"
    if args.objective is not None and args.objective != objective:
        raise ValueError(f"Checkpoint objective is {objective}; start a separate {args.objective} run from scratch")
    if not args.resume:
        config.dropout = args.dropout
    return config


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    loader: TokenBatchLoader,
    eval_iters: int,
    autocast: contextlib.AbstractContextManager,
    *,
    objective: str = "causal",
    mask_probability: float = 0.15,
) -> dict[str, float]:
    """Score fixed windows and corruptions without consuming the training sampler or RNG."""
    was_training = model.training
    model.eval()
    out = {}
    try:
        # A dedicated fixed evaluation stream makes best-checkpoint comparisons meaningful.
        eval_loader = TokenBatchLoader(loader.data_dir, loader.batch_size, loader.block_size, loader.device, seed=1729)
        for split in ("train", "val"):
            loss_sum, count = 0.0, 0
            for _ in range(eval_iters):
                x, y = eval_loader.get_batch(split, objective=objective, mask_probability=mask_probability)
                with autocast:
                    _, loss = model(x, y, return_logits=False)
                n = int((y != -100).sum().item())
                loss_sum += loss.item() * n
                count += n
            out[split] = loss_sum / max(1, count)
    finally:
        model.train(was_training)
    return out


def dataset_identity(data_dir: Path) -> dict[str, Any]:
    """Record provenance and sizes; full hashes are produced once by data preparation."""
    meta_path = data_dir / "meta.json"
    identity: dict[str, Any] = {"sizes": {s: (data_dir / f"{s}.bin").stat().st_size for s in ("train", "val")}}
    if meta_path.exists():
        raw = meta_path.read_bytes()
        meta = json.loads(raw)
        if meta.get("tokenizer") != "gpt2" or meta.get("dtype") != "uint16":
            raise ValueError("Expected a GPT-2 uint16 corpus from prepare_data.py")
        for split in ("train", "val"):
            if meta.get(f"{split}_tokens") != identity["sizes"][split] // 2:
                raise ValueError(f"{split}.bin size disagrees with meta.json; rebuild the dataset")
        identity.update(meta_sha256=hashlib.sha256(raw).hexdigest(), metadata=meta)
    else:
        print("Warning: no meta.json; legacy token files have no tokenizer or content-hash provenance.")
    return identity


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train one model on one device, with reproducible sampling and resumable state."""
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_out_dir_is_safe(out_dir, args.resume, args.overwrite)
    data_dir = Path(args.data_dir).expanduser()
    if str(data_dir).startswith("/mnt/"):
        print("Warning: WSL data under /mnt/ has slow random I/O; copy to the Linux filesystem.")
    identity = dataset_identity(data_dir)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; install a compatible PyTorch CUDA build or use --device cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
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
    args.objective = "causal" if config.causal else "mlm"
    if args.resume and checkpoint.get("data_identity", identity) != identity:
        raise ValueError("Resume dataset differs from the checkpoint; use --init-from for a new corpus")
    if args.resume and "loader_state" in checkpoint:
        for key in ("micro_batch", "grad_accum", "seed", "mask_probability"):
            if checkpoint["train_args"].get(key, getattr(args, key)) != getattr(args, key):
                raise ValueError(f"--resume requires the original {key}; use --init-from to change it")
    model = (GPT(config) if config.causal else Encoder(config)).to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    model.loss_chunk_size = args.loss_chunk_size
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    print(f"{args.objective}: {model.num_params(non_embedding=False) / 1e6:.1f}M parameters, {asdict(config)}")
    tokens_per_step = args.micro_batch * args.grad_accum * config.block_size
    start_step = checkpoint["step"] if args.resume else 0
    loader = TokenBatchLoader(data_dir, args.micro_batch, config.block_size, device, seed=args.seed)
    max_steps = resolve_max_steps(args, loader.num_tokens("train"), tokens_per_step)
    if max_steps < start_step:
        raise ValueError("--max-steps is below the checkpoint's completed step")
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else min(500, max(1, max_steps // 20))
    min_lr = args.lr * args.min_lr_ratio
    print(
        f"Budget: {max_steps:,} steps x {tokens_per_step:,} input tokens; "
        f"{loader.num_tokens('train'):,} train tokens (random-window sampling)"
    )
    if args.objective == "mlm":
        print(f"MLM scores approximately {args.mask_probability:.0%} of input tokens per pass.")
    optimizer = model.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.95), device.type)
    best_val, bad_evals = math.inf, 0
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_val = checkpoint["best_val_loss"]
        bad_evals = checkpoint.get("bad_evals", 0)
        if "loader_state" in checkpoint:
            loader.load_state_dict(checkpoint["loader_state"])
            torch.set_rng_state(checkpoint["rng_state"].cpu())
            if use_cuda and checkpoint.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        else:
            print("Warning: legacy checkpoint lacks RNG state; continuation is not an exact replay.")
    checkpoint = None
    raw_model = model
    if args.compile and use_cuda:
        if os.name == "nt":
            print("Native Windows: running eager; use Linux/WSL for torch.compile and Triton.")
        else:
            print("Compiling model; first steps include compilation overhead...")
            model = torch.compile(model)

    runtime = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if use_cuda else None,
        "compiled": bool(args.compile and use_cuda and os.name != "nt"),
        "precision": "bf16-autocast" if use_cuda and torch.cuda.is_bf16_supported() else "fp32",
        "train_args": vars(args),
        "model_config": asdict(config),
        "data_identity": identity,
        "tokens_per_step": tokens_per_step,
        "max_steps": max_steps,
    }
    (out_dir / (f"resume-{start_step}.json" if args.resume else "run.json")).write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    step, last_eval_step = start_step, -1
    last_eval: dict[str, float] = {}

    def snapshot() -> dict[str, Any]:
        return {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(config),
            "step": step,
            "best_val_loss": best_val,
            "train_args": vars(args),
            "loader_state": loader.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if use_cuda else None,
            "data_identity": identity,
            "last_eval_step": last_eval_step,
            "bad_evals": bad_evals,
        }

    stop_requested = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    old_sigint = signal.signal(signal.SIGINT, request_stop)
    log_file = open(out_dir / "log.jsonl", "a" if args.resume else "w", encoding="utf-8")

    def log(record: dict[str, Any]) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    def evaluate() -> None:
        nonlocal best_val, bad_evals, last_eval, last_eval_step
        last_eval = estimate_loss(
            model, loader, args.eval_iters, autocast, objective=args.objective, mask_probability=args.mask_probability
        )
        last_eval_step = step
        if not all(math.isfinite(value) for value in last_eval.values()):
            raise FloatingPointError("Non-finite evaluation loss; check data and learning rate")
        improved = last_eval["val"] < best_val and step > start_step
        if improved:
            best_val, bad_evals = last_eval["val"], 0
            save_checkpoint(out_dir / "best.pt", snapshot())
        elif step > start_step:
            bad_evals += 1
        print(
            f"step {step:>6}: train {last_eval['train']:.4f} val {last_eval['val']:.4f}"
            f"{' (best saved)' if improved else ''}",
            flush=True,
        )
        log({"step": step, "event": "eval", **last_eval, "objective": args.objective})

    elapsed_train, logged_steps = 0.0, 0
    failed = False
    try:
        evaluate()
        while step < max_steps and not stop_requested:
            t0 = time.perf_counter()
            lr = get_lr(step, max_lr=args.lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum = torch.zeros((), device=device)
            for _ in range(args.grad_accum):
                x, y = loader.get_batch("train", objective=args.objective, mask_probability=args.mask_probability)
                with autocast:
                    _, loss = model(x, y, return_logits=False)
                loss_sum += loss.detach()
                (loss / args.grad_accum).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if use_cuda and (step % args.log_interval == 0 or step % args.eval_interval == 0 or step == max_steps):
                torch.cuda.synchronize(device)
            elapsed_train += time.perf_counter() - t0
            logged_steps += 1
            if step % args.log_interval == 0 or step == max_steps:
                loss_value = (loss_sum / args.grad_accum).item()
                tok_per_s = logged_steps * tokens_per_step / max(elapsed_train, 0.001)
                peak = torch.cuda.max_memory_allocated(device) / 2**30 if use_cuda else 0.0
                print(
                    f"step {step:>6} | loss {loss_value:.4f} | lr {lr:.2e} | "
                    f"{tok_per_s:,.0f} input tok/s | peak {peak:.2f} GiB",
                    flush=True,
                )
                log(
                    {
                        "step": step,
                        "event": "train",
                        "loss": loss_value,
                        "lr": lr,
                        "tok_per_s": tok_per_s,
                        "grad_norm": grad_norm.item(),
                        "peak_memory_gib": peak,
                    }
                )
                elapsed_train, logged_steps = 0.0, 0
            if step % args.eval_interval == 0 or step == max_steps:
                evaluate()
            if step % args.save_interval == 0:
                save_checkpoint(out_dir / "last.pt", snapshot())
            if args.patience and bad_evals >= args.patience:
                print(f"Early stopping after {bad_evals} evaluations without improvement.")
                break
    except BaseException:
        # A failure mid-update must not replace a valid checkpoint with mismatched
        # weights, optimizer, sampler and RNG state. Resume the last periodic save.
        failed = True
        raise
    finally:
        # SIGINT/SIGTERM request a stop at the completed optimizer-step boundary.
        if not failed and step > start_step:
            save_checkpoint(out_dir / "last.pt", snapshot())
        log_file.close()
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)
    return {"step": step, "best_val_loss": best_val, **{f"last_{k}": v for k, v in last_eval.items()}}


if __name__ == "__main__":
    try:
        print(f"Finished: {train(parse_args())}")
    except (FileExistsError, ValueError, FloatingPointError) as error:
        raise SystemExit(f"Error: {error}") from None
