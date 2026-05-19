from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfigs:
    blocksize: int = 1024 # Maximum sequence lenght (context window)
    vocabsize: int = 50257 # (Number of possible token)
    n_layer: int = 12 # Number of layers
    n_head: int = 12 # Number of heads
    n_embd:int = 768 # Embedding dimension

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.c_attn = nn.Linear(self.config.n_embd, 3 * self.config.n_embd)
        self.c_proj = nn.Linear(self.config.n_embd, self.config.nb_embd)
        self.nhead = self.config.nhead

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        attn = q @ k.view.transpose(-2, -1) * (1 / torch.sqrt(self.config.n_embd))
        attn = attn.masked_fill_(torch.tril(attn, diagonal=-1).bool(), -torch.inf)
        #Softmax
        y = attn @ v
        #Re-assemble
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(self.config.n_embd, 4 * self.config.n_embd)
        self.gelu = nn.GELU("tanh")
        self.c_proj = nn.Linear(4 * self.config.n_embd, self.config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(self.config.n_embd)
        self.attn = MultiHeadedAttention(self.config)
        self.ln2 = nn.LayerNorm(self.config.n_embd)
        self.mlp = MLP(self.config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
        
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            ("wte", nn.Embedding(self.config.vocabsize, self.config.n_embd)),
            ("wpe", nn.Embedding(self.config.blocksize, self.config.n_embd)),
            ("h", [Block(self.config) for _ in self.config.n_layer]),
            ("ln_f", nn.LayerNorm(self.config.n_embd, self.config.n_embd))
        )

        self.ln_head = nn.Linear()

    def forward(self, x):
        y = x @ self.transformer["wte"]
        #x (1024, 50k)
        #wte (50k, 768)
        #out (1024, 768)

        y = y.T @ self.transformer["wpe"]
        #y (1024, 768)
        #wtp (1024, 768)
        #out (1024, 768)

        y = self.tran


