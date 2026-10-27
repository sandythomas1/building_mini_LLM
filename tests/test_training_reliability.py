import contextlib
import signal

import numpy as np
import pytest
import torch

import train as training
from data_loader import TokenBatchLoader
from model import GPT, MODEL_PRESETS, GPTConfig
from train import estimate_loss, load_checkpoint, parse_args, train


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setitem(MODEL_PRESETS, "tiny", dict(n_layer=1, n_head=1, n_embd=16))
    for split in ("train", "val"):
        np.tile(np.arange(20), 100).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
    return tmp_path


def args(corpus, out, *extra):
    return parse_args(
        [
            "--data-dir",
            str(corpus),
            "--out-dir",
            str(out),
            "--model-size",
            "tiny",
            "--block-size",
            "8",
            "--micro-batch",
            "2",
            "--grad-accum",
            "1",
            "--max-steps",
            "2",
            "--eval-iters",
            "1",
            "--eval-interval",
            "10",
            "--device",
            "cpu",
            "--no-compile",
            *extra,
        ]
    )


def test_eval_preserves_training_sampler_and_repeats(corpus):
    model = GPT(GPTConfig(vocab_size=100, block_size=8, n_layer=1, n_head=1, n_embd=16))
    loader = TokenBatchLoader(corpus, 2, 8, "cpu")
    expected = TokenBatchLoader(corpus, 2, 8, "cpu").get_batch("train")[0]
    a = estimate_loss(model, loader, 2, contextlib.nullcontext())
    b = estimate_loss(model, loader, 2, contextlib.nullcontext())
    assert a == b
    assert torch.equal(expected, loader.get_batch("train")[0])


@pytest.mark.parametrize("objective", ["causal", "mlm"])
def test_final_update_is_evaluated_and_rng_is_saved(corpus, tmp_path, objective):
    out = tmp_path / objective
    result = train(args(corpus, out, "--objective", objective, "--loss-chunk-size", "4"))
    ckpt = load_checkpoint(out / "last.pt")
    assert ckpt["step"] == 2
    assert ckpt["last_eval_step"] == 2
    assert ckpt["rng_state"].dtype == torch.uint8
    assert "loader_state" in ckpt
    assert (out / "best.pt").exists()
    assert result["last_val"] > 0
    assert ckpt["model_config"]["causal"] == (objective == "causal")


def test_lab_presets_and_explicit_overrides(corpus, tmp_path):
    cfg = args(corpus, tmp_path / "run", "--profile", "rtx4090", "--micro-batch", "3")
    assert cfg.micro_batch == 3
    assert cfg.loss_chunk_size > 0


@pytest.mark.parametrize(
    "flag,value", [("--grad-accum", "0"), ("--eval-interval", "0"), ("--mask-probability", "0"), ("--lr", "-1")]
)
def test_invalid_training_settings_fail_early(corpus, tmp_path, flag, value):
    with pytest.raises(SystemExit):
        args(corpus, tmp_path / "bad", flag, value)


@pytest.mark.parametrize("objective", ["causal", "mlm"])
def test_resume_replays_uninterrupted_training(corpus, tmp_path, monkeypatch, objective):
    common = ["--max-steps", "4", "--save-interval", "2", "--dropout", "0.1", "--objective", objective]
    train(args(corpus, tmp_path / "whole", *common))
    original_save = training.save_checkpoint
    interrupted = False

    def save_and_interrupt(path, state):
        nonlocal interrupted
        original_save(path, state)
        if path.name == "last.pt" and state["step"] == 2 and not interrupted:
            interrupted = True
            signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(training, "save_checkpoint", save_and_interrupt)
    result = train(args(corpus, tmp_path / "split", *common))
    assert result["step"] == 2
    train(args(corpus, tmp_path / "split", *common, "--resume"))
    whole = load_checkpoint(tmp_path / "whole" / "last.pt")
    split = load_checkpoint(tmp_path / "split" / "last.pt")
    for name, tensor in whole["model"].items():
        torch.testing.assert_close(tensor, split["model"][name], rtol=0, atol=0)


def test_encoder_can_export_embeddings_and_rejects_generation(corpus, tmp_path):
    import sample
    from encode import embed_text, load_encoder

    out = tmp_path / "encoder"
    train(args(corpus, out, "--objective", "mlm"))
    encoder = load_encoder(str(out / "best.pt"), torch.device("cpu"))
    vector = embed_text(encoder, "Hello there")
    assert vector.shape == (16,)
    assert vector.norm().item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="encoder"):
        sample.load_model(str(out / "best.pt"), torch.device("cpu"))


def test_failed_update_does_not_replace_completed_checkpoint(corpus, tmp_path, monkeypatch):
    out = tmp_path / "failure"
    original_step = torch.optim.AdamW.step
    calls = 0

    def failing_step(optimizer, *a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated optimizer failure")
        return original_step(optimizer, *a, **kw)

    monkeypatch.setattr(torch.optim.AdamW, "step", failing_step)
    with pytest.raises(RuntimeError, match="simulated"):
        train(args(corpus, out, "--save-interval", "1"))
    assert load_checkpoint(out / "last.pt")["step"] == 1


def test_resume_rejects_changed_corpus(corpus, tmp_path):
    out = tmp_path / "identity"
    train(args(corpus, out))
    with (corpus / "train.bin").open("ab") as file:
        file.write(b"\x01\x00")
    with pytest.raises(ValueError, match="dataset differs"):
        train(args(corpus, out, "--resume"))
