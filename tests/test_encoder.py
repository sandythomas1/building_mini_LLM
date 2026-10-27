import pytest
import torch

from data_loader import mask_tokens
from model import GPT, Encoder, GPTConfig


def test_encoder_sees_future_tokens_and_returns_embeddings():
    torch.manual_seed(7)
    model = Encoder(GPTConfig(vocab_size=97, block_size=16, n_layer=2, n_head=2, n_embd=32, causal=False))
    model.eval()
    tokens = torch.randint(0, 97, (2, 10))
    changed = tokens.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 97
    a = model.encode(tokens)
    b = model.encode(changed)
    assert a.shape == (2, 10, 32)
    assert not torch.allclose(a[:, :8], b[:, :8])
    with pytest.raises(ValueError, match="encoder"):
        model.generate(tokens, 2)


def test_masking_preserves_eot_and_only_scores_selected_tokens():
    tokens = torch.tensor([[1, 2, 3, 50256], [4, 5, 6, 50256]])
    x, y = mask_tokens(tokens, torch.Generator().manual_seed(3), probability=1.0)
    assert (x[:, -1] == 50256).all()
    assert (y[:, -1] == -100).all()
    torch.testing.assert_close(y[:, :-1], tokens[:, :-1])
    assert (x[:, :-1] == 50257).any()


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("chunk_size", [0, 5])
def test_chunked_loss_and_checkpointing_match_full_gradients(causal, chunk_size):
    torch.manual_seed(1)
    cls = GPT if causal else Encoder
    config = GPTConfig(vocab_size=97, block_size=16, n_layer=2, n_head=2, n_embd=32, causal=causal)
    full, chunked = cls(config), cls(config)
    chunked.load_state_dict(full.state_dict())
    chunked.gradient_checkpointing = True
    chunked.loss_chunk_size = chunk_size
    x = torch.randint(0, 97, (2, 10))
    y = x.clone()
    y[:, ::3] = -100
    _, loss_a = full(x, y)
    logits, loss_b = chunked(x, y, return_logits=False)
    assert logits is None
    torch.testing.assert_close(loss_a, loss_b)
    loss_a.backward()
    loss_b.backward()
    for a, b in zip(full.parameters(), chunked.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


def test_compiled_encoder_does_not_recompile_for_each_mask_count():
    model = Encoder(GPTConfig(vocab_size=97, block_size=16, n_layer=1, n_head=1, n_embd=16, causal=False))
    model.loss_chunk_size = 5
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(model, backend=backend)
    tokens = torch.randint(0, 97, (2, 10))
    count_after_first = None
    for count in (2, 7, 11, 17):
        labels = torch.full_like(tokens, -100)
        labels.view(-1)[:count] = tokens.view(-1)[:count]
        _, loss = compiled(tokens, labels, return_logits=False)
        loss.backward()
        model.zero_grad(set_to_none=True)
        if count_after_first is None:
            count_after_first = len(graphs)
        assert len(graphs) == count_after_first
