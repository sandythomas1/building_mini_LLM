"""GPT-style decoder-only transformer.

Architecture follows GPT-2 (pre-norm blocks, learned positional embeddings,
GELU MLP) with the efficiency fixes that matter on a single consumer GPU:
fused QKV projection, PyTorch's flash/memory-efficient attention kernel,
tied input/output embeddings, and a vocab padded to a multiple of 64.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

GPT2_VOCAB_SIZE = 50257
PADDED_VOCAB_SIZE = 50304  # next multiple of 64 above 50257 -> faster tensor-core matmuls


@dataclass
class GPTConfig:
    vocab_size: int = PADDED_VOCAB_SIZE
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    causal: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")


# Head size is 64 in every preset, which is the sweet spot for flash attention kernels.
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "tiny": dict(n_layer=6, n_head=6, n_embd=384),  # ~30M params: fast experiments
    "small": dict(n_layer=8, n_head=8, n_embd=512),  # ~51M params: default for an 8 GB card
    "medium": dict(n_layer=12, n_head=12, n_embd=768),  # ~124M params: GPT-2 small
    "large": dict(n_layer=24, n_head=16, n_embd=1024),  # ~354M: longer lab runs
}


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.causal = config.causal
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.proj.is_residual_projection = True
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.proj.is_residual_projection = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: attention then MLP, each with a residual connection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.loss_chunk_size = 0
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output projection reuses the token embedding matrix.
        # Saves vocab_size * n_embd params (~26M for the small preset) and regularizes.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "is_residual_projection", False):
                # GPT-2 scaling: keeps the residual stream's variance from growing with depth.
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.position_embedding.weight.numel()
            n -= self.token_embedding.weight.numel()
        return n

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Return contextual token representations using the configured attention direction."""
        _, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(pos))
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.gradient_checkpointing and self.training else block(x)
        return self.ln_f(x)

    def _loss_sum(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # FP32 reduction avoids bf16 rounding accumulating across chunks.
        logits = self.lm_head(hidden)
        return F.cross_entropy(logits.float(), targets, ignore_index=-100, reduction="sum")

    def _chunked_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        total = hidden.new_zeros((), dtype=torch.float32)
        chunk_size = self.loss_chunk_size or max(1, labels.numel())
        for start in range(0, labels.numel(), chunk_size):
            h, y = hidden[start : start + chunk_size], labels[start : start + chunk_size]
            # Recomputing head chunks in backward bounds retained vocabulary activations.
            total = total + (
                checkpoint(self._loss_sum, h, y, use_reentrant=False) if self.loss_chunk_size else self._loss_sum(h, y)
            )
        return total / (labels != -100).sum().clamp_min(1)

    @torch.compiler.disable
    def _masked_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Dynamic masking changes the selected-token count on each batch. Keep the
        # sparse head eager to prevent per-count compiler specialization; the expensive
        # transformer still runs in compiled graphs.
        labels = targets.reshape(-1)
        selected = labels != -100
        return self._chunked_loss(hidden.reshape(-1, hidden.size(-1))[selected], labels[selected])

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, *, return_logits: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = self.hidden_states(idx)

        if targets is None:
            # Inference only needs the next-token distribution; skip the full (B, T, V) matmul.
            return self.lm_head(x[:, [-1], :] if self.config.causal else x), None

        if not return_logits and not self.config.causal:
            return None, self._masked_loss(x, targets)
        if not return_logits and self.loss_chunk_size:
            return None, self._chunked_loss(x.reshape(-1, x.size(-1)), targets.reshape(-1))

        logits = self.lm_head(x)
        labels = targets.reshape(-1)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels, reduction="sum")
        loss = loss / (labels != -100).sum().clamp_min(1)
        return logits if return_logits else None, loss

    def configure_optimizer(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        """AdamW with decay on matrices only; norms, biases and embeddings' 1-D params are exempt."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend `idx` (B, T). Stops early when every row emits `stop_token`."""
        if not self.config.causal:
            raise ValueError("A bidirectional encoder cannot generate autoregressively; use encode().")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            # Never sample the padding slots added past the real GPT-2 vocab.
            logits[:, GPT2_VOCAB_SIZE:] = -math.inf
            if top_k is not None:
                kth_largest = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits[logits < kth_largest] = -math.inf
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if stop_token is not None:
                finished |= next_token.squeeze(1) == stop_token
                if finished.all():
                    break
        return idx


class Encoder(GPT):
    """Bidirectional transformer trained with masked language modeling, sharing GPT's blocks.

    This produces contextual embeddings. Sentence retrieval quality requires downstream
    supervision; MLM alone is not a contrastive sentence-embedding training objective.
    """

    def __init__(self, config: GPTConfig):
        if config.causal:
            raise ValueError("Encoder requires causal=False")
        super().__init__(config)

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        """Return one contextual vector per input token (no vocabulary projection)."""
        return self.hidden_states(idx)
