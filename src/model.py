#building model architecture

#imports
import torch
import torch.nn as nn
from torch.nn import functional as F #matrix operations

# Enable Tensor Core math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable automatic fastest cuDNN kernel selection
torch.backends.cudnn.benchmark = True

class TransformerConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

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
    #transformer block
    def __init__(self, n_embd, n_head):
        super().__init__()
        #multihead attention sublayer
        self.sa = MultiHeadAttention(n_head, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        #token embedding table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        #position embedding table
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        #transformer blocks
        self.blocks = nn.Sequential(*[
            Block(config.n_embd, config.n_head) for _ in range(config.n_layer)
        ])

        #layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        #final linear layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
    
    def forward(self, idx, targets=None):
        pass