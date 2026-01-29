"""Multi-head attention implementation."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Allows the model to jointly attend to information from different
    representation subspaces.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize multi-head attention.

        Args:
            embed_dim: Dimension of input/output embeddings
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (optional, defaults to x for self-attention)
            value: Value tensor (optional, defaults to x for self-attention)
            mask: Attention mask [batch, seq_len, seq_len] or [batch, 1, seq_len]

        Returns:
            Attended output [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Default to self-attention
        if key is None:
            key = x
        if value is None:
            value = x

        # Linear projections and reshape for multi-head
        # Shape: [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # [batch, n_heads, seq_len, head_dim] @ [batch, n_heads, head_dim, key_len]
        # -> [batch, n_heads, seq_len, key_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch, n_heads, seq_len, key_len] @ [batch, n_heads, key_len, head_dim]
        # -> [batch, n_heads, seq_len, head_dim]
        out = torch.matmul(attn_weights, v)

        # Reshape back
        # [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Attention mask

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask=mask)
        x = x + attn_out

        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x
