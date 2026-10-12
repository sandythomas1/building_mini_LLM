import json

import numpy as np
import pytest
import tiktoken
import torch

import sample
from train import get_lr, load_checkpoint, parse_args, train

SCHEDULE = dict(max_lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)


def test_lr_warms_up_linearly():
    assert get_lr(0, **SCHEDULE) == pytest.approx(0.1)
    assert get_lr(4, **SCHEDULE) == pytest.approx(0.5)
    assert get_lr(9, **SCHEDULE) == pytest.approx(1.0)


def test_lr_cosine_decays_to_min_and_holds():
    assert get_lr(10, **SCHEDULE) == pytest.approx(1.0)
    assert get_lr(60, **SCHEDULE) == pytest.approx(0.55)  # halfway through decay
    assert get_lr(110, **SCHEDULE) == pytest.approx(0.1)
    assert get_lr(5000, **SCHEDULE) == pytest.approx(0.1)


def test_lr_is_monotonic_after_warmup():
    lrs = [get_lr(s, **SCHEDULE) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


@pytest.fixture
def tiny_dataset(tmp_path):
    rng = np.random.default_rng(0)
    # A repeating pattern is learnable, so loss should fall quickly.
    pattern = rng.integers(0, 1000, size=37)
    for split, reps in (("train", 200), ("val", 20)):
        np.tile(pattern, reps).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
    return tmp_path


def tiny_args(data_dir, out_dir, *extra):
    return parse_args([
        "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--model-size", "tiny",
        "--block-size", "32", "--micro-batch", "4", "--grad-accum", "2", "--eval-interval", "10",
        "--eval-iters", "2", "--log-interval", "5", "--device", "cpu", "--lr", "3e-3", "--no-compile", *extra,
    ])


def test_training_learns_and_writes_checkpoints(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    result = train(tiny_args(tiny_dataset, out, "--max-steps", "30"))

    assert result["step"] == 30
    assert result["last_val"] < 4.0  # uniform over 50304 tokens would be ~10.8
    for name in ("best.pt", "last.pt", "log.jsonl"):
        assert (out / name).exists()
    ckpt = load_checkpoint(out / "last.pt")
    assert ckpt["step"] == 30
    assert ckpt["model_config"]["n_layer"] == 6


def test_resume_continues_from_saved_step(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    train(tiny_args(tiny_dataset, out, "--max-steps", "10"))
    result = train(tiny_args(tiny_dataset, out, "--max-steps", "15", "--resume"))
    assert result["step"] == 15
    assert load_checkpoint(out / "last.pt")["step"] == 15


def test_fresh_run_refuses_to_clobber_existing_checkpoints(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    train(tiny_args(tiny_dataset, out, "--max-steps", "3"))
    before = (out / "last.pt").read_bytes()

    with pytest.raises(FileExistsError, match="--resume"):
        train(tiny_args(tiny_dataset, out, "--max-steps", "3"))
    assert (out / "last.pt").read_bytes() == before

    result = train(tiny_args(tiny_dataset, out, "--max-steps", "3", "--overwrite"))
    assert result["step"] == 3


def test_init_from_reuses_architecture_and_weights(tiny_dataset, tmp_path):
    pretrain = tmp_path / "pretrain"
    train(tiny_args(tiny_dataset, pretrain, "--max-steps", "20"))
    # --model-size is ignored when starting from a checkpoint; its stored config wins.
    finetune_args = tiny_args(tiny_dataset, tmp_path / "ft", "--max-steps", "2", "--init-from",
                              str(pretrain / "best.pt"))
    finetune_args.model_size = "medium"
    train(finetune_args)
    # The step-0 eval runs before any update, so it measures the loaded weights directly.
    first_eval = json.loads((tmp_path / "ft" / "log.jsonl").read_text().splitlines()[0])
    pretrain_best = load_checkpoint(pretrain / "best.pt")["best_val_loss"]
    assert first_eval["step"] == 0
    assert first_eval["val"] == pytest.approx(pretrain_best, rel=0.05)
    assert load_checkpoint(tmp_path / "ft" / "last.pt")["model_config"]["n_embd"] == 384


def test_generate_text_returns_only_new_text_and_stops_at_eot(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    train(tiny_args(tiny_dataset, out, "--max-steps", "2"))
    model = sample.load_model(str(out / "last.pt"), torch.device("cpu"))
    encoder = tiktoken.get_encoding("gpt2")
    text = sample.generate_text(model, encoder, "Hello there", max_new_tokens=8, temperature=1.0, top_k=10)
    assert isinstance(text, str)
    assert not text.startswith("Hello there")
    assert "<|endoftext|>" not in text


def test_extract_agent_reply_drops_hallucinated_customer_turn():
    generated = " Sure, I can help.\nCustomer: thanks\nAgent: welcome"
    assert sample.extract_agent_reply(generated) == "Sure, I can help."


def test_extract_agent_reply_drops_second_agent_turn():
    generated = " Thanks, that helps.\nAgent: Let me check your account.\nCustomer: ok"
    assert sample.extract_agent_reply(generated) == "Thanks, that helps."


def test_extract_agent_reply_keeps_multiline_reply_and_inline_role_words():
    generated = " Steps:\n1. Open Settings.\n2. Ask the Agent: it will help.\nCustomer: thanks"
    assert sample.extract_agent_reply(generated) == "Steps:\n1. Open Settings.\n2. Ask the Agent: it will help."
