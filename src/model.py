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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()

        #calculate head size
        head_size = n_embed // n_head

        #create multiple heads

        self.heads = nn.ModuleList([
            Head(n_embed, head_size) for _ in range(n_head)
        ])

        #final projection layer
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        #concatenate outputs from all heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
#feed forward network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        #pass #will implement later
        return self.net(x)
    
class Block(nn.Module):
    pass
    