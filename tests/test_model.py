import math

import pytest
import torch

from model import GPT, MODEL_PRESETS, GPTConfig

VOCAB = 97


@pytest.fixture
def config() -> GPTConfig:
    return GPTConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, n_embd=32)


@pytest.fixture
def model(config: GPTConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(config)


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(n_head=3, n_embd=32)


def test_forward_with_targets_returns_full_logits_and_scalar_loss(model):
    idx = torch.randint(0, VOCAB, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, VOCAB)
    assert loss.ndim == 0


def test_forward_without_targets_returns_last_position_only(model):
    logits, loss = model(torch.randint(0, VOCAB, (2, 7)))
    assert logits.shape == (2, 1, VOCAB)
    assert loss is None


def test_initial_loss_is_close_to_uniform(model):
    idx = torch.randint(0, VOCAB, (8, 16))
    _, loss = model(idx, idx)
    assert loss.item() == pytest.approx(math.log(VOCAB), rel=0.1)


def test_rejects_sequences_longer_than_block_size(model):
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros((1, 17), dtype=torch.long))


def test_attention_is_causal(model):
    model.eval()
    idx = torch.randint(0, VOCAB, (1, 12))
    changed = idx.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % VOCAB
    logits_a, _ = model(idx, idx)
    logits_b, _ = model(changed, changed)
    # Positions before the edit must not see future tokens.
    torch.testing.assert_close(logits_a[:, :8], logits_b[:, :8])
    assert not torch.allclose(logits_a[:, 8:], logits_b[:, 8:])


def test_embeddings_are_tied(model):
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_presets_use_64_dim_heads():
    for preset in MODEL_PRESETS.values():
        assert preset["n_embd"] // preset["n_head"] == 64


def test_optimizer_decays_only_matrices(model):
    optimizer = model.configure_optimizer(0.1, 1e-3, (0.9, 0.95), "cpu")
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.dim() < 2 for p in no_decay_group["params"])
    # Tied weights must appear once, not twice.
    total = sum(len(g["params"]) for g in optimizer.param_groups)
    assert total == len(list(model.parameters()))


def test_model_can_overfit_a_single_batch(model):
    idx = torch.randint(0, VOCAB, (4, 16))
    optimizer = model.configure_optimizer(0.0, 3e-3, (0.9, 0.95), "cpu")
    _, initial = model(idx[:, :-1], idx[:, 1:])
    for _ in range(60):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial.item() * 0.3


def test_generate_appends_tokens_and_slides_past_block_size(model):
    out = model.generate(torch.zeros((2, 3), dtype=torch.long), max_new_tokens=20, top_k=5)
    assert out.shape == (2, 23)
    assert out.max() < VOCAB


def fixed_next_token(token: int):
    """A forward() stub whose next-token distribution is a spike on `token`."""

    def forward(idx, targets=None):
        logits = torch.full((idx.size(0), 1, VOCAB), -1e4)
        logits[..., token] = 1e4
        return logits, None

    return forward


def test_generate_stops_when_all_rows_hit_stop_token(model, monkeypatch):
    monkeypatch.setattr(model, "forward", fixed_next_token(5))
    out = model.generate(torch.ones((2, 1), dtype=torch.long), max_new_tokens=10, stop_token=5)
    assert out.tolist() == [[1, 5], [1, 5]]


def test_generate_runs_to_max_tokens_without_stop_token(model, monkeypatch):
    monkeypatch.setattr(model, "forward", fixed_next_token(5))
    out = model.generate(torch.ones((1, 1), dtype=torch.long), max_new_tokens=4, stop_token=6)
    assert out.tolist() == [[1, 5, 5, 5, 5]]


def test_generate_rejects_non_positive_temperature(model):
    with pytest.raises(ValueError, match="temperature"):
        model.generate(torch.zeros((1, 1), dtype=torch.long), 1, temperature=0)
