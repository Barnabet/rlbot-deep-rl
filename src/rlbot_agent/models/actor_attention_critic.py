"""Actor-Attention-Critic neural network architecture."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..core.config import NetworkConfig, ObservationConfig
from .attention import PositionalEncoding, TransformerBlock
from .distributions import CategoricalDistribution
from .encoders import BallEncoder, CarEncoder


class ActorAttentionCritic(nn.Module):
    """Actor-Attention-Critic network for Rocket League.

    Architecture:
    1. Parse flat observation into self_car, other_cars, ball
    2. Encode each entity with dedicated MLPs
    3. Apply multi-head self-attention across all players
    4. Extract self player's attended embedding
    5. Concatenate with ball embedding
    6. Separate policy and value heads

    Input: Flat observation [batch, obs_dim]
    Output: Action logits [batch, n_actions], Value [batch, 1]
    """

    def __init__(
        self,
        obs_config: ObservationConfig,
        network_config: NetworkConfig,
        n_actions: int = 1944,
        max_players: int = 6,
    ):
        """Initialize Actor-Attention-Critic network.

        Args:
            obs_config: Observation configuration
            network_config: Network architecture configuration
            n_actions: Number of discrete actions
            max_players: Maximum number of players
        """
        super().__init__()
        self.obs_config = obs_config
        self.network_config = network_config
        self.n_actions = n_actions
        self.max_players = max_players

        # Observation parsing dimensions
        self.self_car_dim = obs_config.self_car_dim  # 19
        self.other_car_dim = obs_config.other_car_dim  # 14
        self.ball_dim = obs_config.ball_dim  # 15

        # Calculate total observation size
        self.obs_dim = (
            self.self_car_dim
            + self.ball_dim
            + (max_players - 1) * self.other_car_dim
        )

        # Entity encoders
        self.self_car_encoder = CarEncoder(
            input_dim=self.self_car_dim,
            config=network_config.car_encoder,
        )
        self.other_car_encoder = CarEncoder(
            input_dim=self.other_car_dim,
            config=network_config.car_encoder,
        )
        self.ball_encoder = BallEncoder(
            input_dim=self.ball_dim,
            config=network_config.ball_encoder,
        )

        # Attention mechanism
        embed_dim = network_config.car_encoder.output_dim
        attn_config = network_config.attention

        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_players,
            dropout=attn_config.dropout,
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=attn_config.n_heads,
                ff_dim=attn_config.ff_dim,
                dropout=attn_config.dropout,
            )
            for _ in range(attn_config.n_layers)
        ])

        # Combined dimension after attention + ball
        combined_dim = embed_dim + network_config.ball_encoder.output_dim

        # Policy head
        self.policy_head = self._build_mlp(
            combined_dim,
            network_config.policy_hidden_dims,
            n_actions,
            network_config.policy_activation,
        )

        # Value head
        self.value_head = self._build_mlp(
            combined_dim,
            network_config.value_hidden_dims,
            1,
            network_config.value_activation,
        )

        # Action distribution
        self.distribution = CategoricalDistribution(n_actions)

        # Initialize weights
        self._init_weights()

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        activation: str,
    ) -> nn.Sequential:
        """Build MLP with given architecture.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function name

        Returns:
            MLP as Sequential module
        """
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

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

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Special initialization for value head (smaller scale)
        if hasattr(self.value_head[-1], 'weight'):
            nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def _parse_observation(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat observation into components.

        Args:
            obs: Flat observation [batch, obs_dim]

        Returns:
            Tuple of (self_car, ball, other_cars)
            - self_car: [batch, self_car_dim]
            - ball: [batch, ball_dim]
            - other_cars: [batch, max_players-1, other_car_dim]
        """
        batch_size = obs.shape[0]

        # Split observation
        idx = 0
        self_car = obs[:, idx:idx + self.self_car_dim]
        idx += self.self_car_dim

        ball = obs[:, idx:idx + self.ball_dim]
        idx += self.ball_dim

        # Reshape other cars
        other_cars_flat = obs[:, idx:]
        n_other_cars = self.max_players - 1
        other_cars = other_cars_flat.view(batch_size, n_other_cars, self.other_car_dim)

        return self_car, ball, other_cars

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            obs: Observations [batch, obs_dim]
            action_mask: Boolean mask of valid actions [batch, n_actions]

        Returns:
            Tuple of (logits, value)
            - logits: Action logits [batch, n_actions]
            - value: State value [batch, 1]
        """
        batch_size = obs.shape[0]

        # Parse observation
        self_car, ball, other_cars = self._parse_observation(obs)

        # Encode entities
        self_embed = self.self_car_encoder(self_car)  # [batch, embed_dim]
        ball_embed = self.ball_encoder(ball)  # [batch, ball_embed_dim]
        other_embeds = self.other_car_encoder(other_cars)  # [batch, n_others, embed_dim]

        # Stack all player embeddings for attention
        # Self is first, then others
        player_embeds = torch.cat([
            self_embed.unsqueeze(1),  # [batch, 1, embed_dim]
            other_embeds,  # [batch, n_others, embed_dim]
        ], dim=1)  # [batch, max_players, embed_dim]

        # Add positional encoding
        player_embeds = self.positional_encoding(player_embeds)

        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            player_embeds = transformer(player_embeds)

        # Extract self player's attended embedding (first position)
        self_attended = player_embeds[:, 0]  # [batch, embed_dim]

        # Concatenate with ball embedding
        combined = torch.cat([self_attended, ball_embed], dim=-1)

        # Policy and value heads
        logits = self.policy_head(combined)
        value = self.value_head(combined)

        # Apply action mask if provided
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy.

        Args:
            obs: Observations [batch, obs_dim]
            action_mask: Boolean mask of valid actions
            deterministic: If True, return mode instead of sampling

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self.forward(obs, action_mask)
        action, log_prob, entropy = self.distribution.sample(
            logits, action_mask, deterministic
        )
        return action, log_prob, entropy, value.squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Args:
            obs: Observations [batch, obs_dim]
            actions: Actions to evaluate [batch]
            action_mask: Boolean mask of valid actions

        Returns:
            Tuple of (log_prob, entropy, value)
        """
        logits, value = self.forward(obs, action_mask)

        log_prob = self.distribution.log_prob(logits, actions, action_mask)
        entropy = self.distribution.entropy(logits, action_mask)

        return log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only.

        Args:
            obs: Observations [batch, obs_dim]

        Returns:
            Value estimates [batch]
        """
        _, value = self.forward(obs)
        return value.squeeze(-1)
