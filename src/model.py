#building model architecture

class TransformerConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

#imports
import torch
import torch.nn as nn
from torch.nn import functional as F #matrix operations

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()

        #define Query, Key, Value
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        #buffer for masking
        self.register_buffer('tril', torch.tril(torch.ones(head_size, head_size)))

    def forward(self, x):
        B, T, C = x.shape

        #generate keys, queries, values
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        #compute attention scores
        head_size = k.size(-1)
        weights = q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, T)

        #scaling factor
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1) # (B, T, T)

        #aggregate the values
        out = weights @ v # (B, T, head_size)

        return out

