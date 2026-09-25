import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    """Single-head causal self-attention."""

    def __init__(self, embed_size, head_size, dropout=0.0):
        super().__init__()
        self.embed_size = embed_size
        self.head_size = head_size

        self.q_proj = nn.Linear(embed_size, head_size, bias=False)
        self.k_proj = nn.Linear(embed_size, head_size, bias=False)
        self.v_proj = nn.Linear(embed_size, head_size, bias=False)
        self.out_proj = nn.Linear(head_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        # x: (batch, seq_len, embed_size)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.matmul(weights, v)
        return self.out_proj(output)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, embed_size, num_heads, dropout=0.0):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList(
            [SelfAttentionHead(embed_size, head_size, dropout) for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        output = torch.cat([head(x, causal_mask) for head in self.heads], dim=-1)
        return self.dropout(self.out_proj(output))


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embed_size, hidden_size=None, dropout=0.0):
        super().__init__()
        hidden_size = hidden_size or 4 * embed_size
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-layer-normalized transformer block with residual connections."""

    def __init__(self, embed_size, num_heads, hidden_size=None, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, hidden_size, dropout)

    def forward(self, x, causal_mask=None):
        x = x + self.attention(self.ln1(x), causal_mask)
        return x + self.feed_forward(self.ln2(x))


