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
        pass #just for now 
        pass #to be implemented later
