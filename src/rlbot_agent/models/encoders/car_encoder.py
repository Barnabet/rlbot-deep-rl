"""Car state encoder."""

from typing import Tuple

import torch
import torch.nn as nn

from ...core.config import EncoderConfig


class CarEncoder(nn.Module):
    """MLP encoder for car state features.

    Encodes car position, velocity, rotation, boost, etc. into a fixed-size embedding.
    """

    def __init__(
        self,
        input_dim: int,
        config: EncoderConfig = None,
    ):
        """Initialize car encoder.

        Args:
            input_dim: Dimension of input features
            config: Encoder configuration
        """
        super().__init__()
        config = config or EncoderConfig()

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(config.activation),
            ])
            prev_dim = hidden_dim

        # Output projection
        layers.append(nn.Linear(prev_dim, config.output_dim))

        self.mlp = nn.Sequential(*layers)
        self.output_dim = config.output_dim

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode car state.

        Args:
            x: Car state features [batch, input_dim] or [batch, n_cars, input_dim]

        Returns:
            Car embedding [batch, output_dim] or [batch, n_cars, output_dim]
        """
        return self.mlp(x)
