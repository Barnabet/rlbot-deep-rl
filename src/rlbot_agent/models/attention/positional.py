"""Positional encoding for transformer."""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding using sinusoidal functions.

    Adds position information to embeddings so the attention mechanism
    can differentiate between different positions in the sequence.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 10,
        dropout: float = 0.0,
    ):
        """Initialize positional encoding.

        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length (max players)
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Input with positional encoding added [batch, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding.

    Uses learned embeddings instead of sinusoidal functions.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 10,
        dropout: float = 0.0,
    ):
        """Initialize learned positional encoding.

        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Input with positional encoding added [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class EntityPositionalEncoding(nn.Module):
    """Entity-type based positional encoding.

    Adds different encodings based on entity type (self, teammate, opponent).
    """

    def __init__(
        self,
        embed_dim: int,
        max_players: int = 6,
        dropout: float = 0.0,
    ):
        """Initialize entity positional encoding.

        Args:
            embed_dim: Embedding dimension
            max_players: Maximum number of players
            dropout: Dropout probability
        """
        super().__init__()

        # Entity type embeddings: self (0), teammate (1), opponent (2)
        self.entity_type_embedding = nn.Embedding(3, embed_dim)

        # Position within type embedding
        self.position_embedding = nn.Embedding(max_players, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        entity_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add entity-aware positional encoding.

        Args:
            x: Input tensor [batch, n_players, embed_dim]
            entity_types: Entity type for each position [batch, n_players]
                         0=self, 1=teammate, 2=opponent

        Returns:
            Input with positional encoding [batch, n_players, embed_dim]
        """
        batch_size, n_players, _ = x.shape

        if entity_types is None:
            # Default: first is self, rest alternate
            entity_types = torch.zeros(batch_size, n_players, dtype=torch.long, device=x.device)
            entity_types[:, 0] = 0  # Self
            # Assume sorted: teammates then opponents

        # Add entity type encoding
        x = x + self.entity_type_embedding(entity_types)

        # Add position encoding
        positions = torch.arange(n_players, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)

        return self.dropout(x)
