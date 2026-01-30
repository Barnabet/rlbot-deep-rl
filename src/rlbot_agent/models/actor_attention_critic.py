"""Actor-Attention-Critic neural network architecture."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..core.config import NetworkConfig, ObservationConfig, LSTMConfig
from .attention import PositionalEncoding, TransformerBlock
from .distributions import CategoricalDistribution, MultiDiscreteDistribution
from .encoders import BallEncoder, CarEncoder


class ActorAttentionCritic(nn.Module):
    """Actor-Attention-Critic network for Rocket League with optional LSTM.

    Architecture:
    1. Parse flat observation into self_car, other_cars, ball
    2. Encode each entity with dedicated MLPs
    3. Apply multi-head self-attention across all players
    4. Extract self player's attended embedding
    5. Concatenate with ball embedding
    6. (Optional) Apply LSTM for temporal memory with residual connection
    7. Separate policy and value heads

    Input: Flat observation [batch, obs_dim] or [batch, seq_len, obs_dim]
    Output: Action logits, Value, (optional) hidden state
    """

    def __init__(
        self,
        obs_config: ObservationConfig,
        network_config: NetworkConfig,
        n_actions: int = 1944,
        max_players: int = 6,
        use_multi_discrete: bool = True,
    ):
        """Initialize Actor-Attention-Critic network.

        Args:
            obs_config: Observation configuration
            network_config: Network architecture configuration
            n_actions: Number of discrete actions (only used if use_multi_discrete=False)
            max_players: Maximum number of players
            use_multi_discrete: If True, use 8 independent action heads (21 logits)
                               If False, use single flat distribution (1944 logits)
        """
        super().__init__()
        self.obs_config = obs_config
        self.network_config = network_config
        self.n_actions = n_actions
        self.max_players = max_players
        self.use_multi_discrete = use_multi_discrete

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

        # LSTM for temporal memory (optional)
        self.lstm_config = network_config.lstm
        self.use_lstm = network_config.lstm.use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=combined_dim,
                hidden_size=network_config.lstm.hidden_size,
                num_layers=network_config.lstm.num_layers,
                batch_first=True,
            )
            # Residual connection: concat original + LSTM output
            policy_value_input_dim = combined_dim + network_config.lstm.hidden_size
        else:
            self.lstm = None
            policy_value_input_dim = combined_dim

        # Policy head - output size depends on action space type
        if use_multi_discrete:
            policy_output_dim = MultiDiscreteDistribution.TOTAL_LOGITS  # 21
            self.distribution = MultiDiscreteDistribution()
        else:
            policy_output_dim = n_actions  # 1944
            self.distribution = CategoricalDistribution(n_actions)

        self.policy_head = self._build_mlp(
            policy_value_input_dim,
            network_config.policy_hidden_dims,
            policy_output_dim,
            network_config.policy_activation,
        )

        # Value head
        self.value_head = self._build_mlp(
            policy_value_input_dim,
            network_config.value_hidden_dims,
            1,
            network_config.value_activation,
        )

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
        """Initialize network weights with proper gains.

        Uses orthogonal initialization with:
        - gain=sqrt(2) for hidden layers (ReLU/GELU activations)
        - gain=1.0 for output layers (policy and value)

        Note: For large discrete action spaces (1944 actions), using gain=0.01
        for policy output produces near-zero logits, resulting in uniform
        softmax output and maximum entropy. We use gain=1.0 to ensure
        meaningful logit spread while maintaining high initial entropy.
        """
        import math

        # Get the output layers for special handling
        policy_output = self.policy_head[-1]
        value_output = self.value_head[-1]

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is policy_output or module is value_output:
                    # Standard gain for output layers
                    nn.init.orthogonal_(module.weight, gain=1.0)
                else:
                    # sqrt(2) gain for hidden layers with ReLU/GELU
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network.

        Args:
            obs: Observations [batch, obs_dim] or [batch, seq_len, obs_dim] for sequences
            hidden: Optional LSTM hidden state tuple (h, c), each [num_layers, batch, hidden_size]
            action_mask: Boolean mask of valid actions [batch, n_actions]

        Returns:
            Tuple of (logits, value, new_hidden)
            - logits: Action logits [batch, n_actions] or [batch, seq_len, n_actions]
            - value: State value [batch, 1] or [batch, seq_len, 1]
            - new_hidden: Updated LSTM hidden state (or None if LSTM disabled)
        """
        # Handle sequence dimension
        has_sequence = obs.dim() == 3
        if has_sequence:
            batch_size, seq_len, _ = obs.shape
            # Flatten for batch processing through encoders
            obs_flat = obs.view(batch_size * seq_len, -1)
        else:
            batch_size = obs.shape[0]
            seq_len = 1
            obs_flat = obs

        # Parse observation
        self_car, ball, other_cars = self._parse_observation(obs_flat)

        # Encode entities
        self_embed = self.self_car_encoder(self_car)  # [batch*seq, embed_dim]
        ball_embed = self.ball_encoder(ball)  # [batch*seq, ball_embed_dim]
        other_embeds = self.other_car_encoder(other_cars)  # [batch*seq, n_others, embed_dim]

        # Stack all player embeddings for attention
        # Self is first, then others
        player_embeds = torch.cat([
            self_embed.unsqueeze(1),  # [batch*seq, 1, embed_dim]
            other_embeds,  # [batch*seq, n_others, embed_dim]
        ], dim=1)  # [batch*seq, max_players, embed_dim]

        # Add positional encoding
        player_embeds = self.positional_encoding(player_embeds)

        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            player_embeds = transformer(player_embeds)

        # Extract self player's attended embedding (first position)
        self_attended = player_embeds[:, 0]  # [batch*seq, embed_dim]

        # Concatenate with ball embedding
        combined = torch.cat([self_attended, ball_embed], dim=-1)  # [batch*seq, combined_dim]

        # LSTM processing
        new_hidden = None
        if self.use_lstm:
            if has_sequence:
                # Reshape for LSTM: [batch, seq_len, combined_dim]
                combined_seq = combined.view(batch_size, seq_len, -1)
                lstm_out, new_hidden = self.lstm(combined_seq, hidden)
                # Residual connection: concat original + LSTM output
                combined = torch.cat([combined_seq, lstm_out], dim=-1)
                # Reshape back for policy/value heads
                combined = combined.view(batch_size * seq_len, -1)
            else:
                # Single step: [batch, 1, combined_dim]
                combined_seq = combined.unsqueeze(1)
                lstm_out, new_hidden = self.lstm(combined_seq, hidden)
                lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]
                # Residual connection
                combined = torch.cat([combined, lstm_out], dim=-1)

        # Policy and value heads
        logits = self.policy_head(combined)
        value = self.value_head(combined)

        # Reshape outputs if sequence
        if has_sequence:
            logits = logits.view(batch_size, seq_len, -1)
            value = value.view(batch_size, seq_len, -1)

        # Apply action mask if provided
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits, value, new_hidden

    def get_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get action from policy.

        Args:
            obs: Observations [batch, obs_dim]
            hidden: Optional LSTM hidden state tuple (h, c)
            action_mask: Boolean mask of valid actions
            deterministic: If True, return mode instead of sampling

        Returns:
            Tuple of (action, log_prob, entropy, value, new_hidden)
        """
        logits, value, new_hidden = self.forward(obs, hidden, action_mask)
        action, log_prob, entropy = self.distribution.sample(
            logits, action_mask, deterministic
        )
        return action, log_prob, entropy, value.squeeze(-1), new_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Evaluate actions for PPO update.

        Args:
            obs: Observations [batch, obs_dim] or [batch, seq_len, obs_dim]
            actions: Actions to evaluate [batch] or [batch, seq_len]
            hidden: Optional LSTM hidden state tuple (h, c)
            action_mask: Boolean mask of valid actions

        Returns:
            Tuple of (log_prob, entropy, value, new_hidden)
        """
        logits, value, new_hidden = self.forward(obs, hidden, action_mask)

        # Handle sequence dimension for log_prob and entropy
        if logits.dim() == 3:
            # [batch, seq_len, n_logits] -> flatten for distribution
            batch_size, seq_len, n_logits = logits.shape
            logits_flat = logits.view(-1, n_logits)

            # Handle both flat and multi-discrete actions
            if self.use_multi_discrete:
                # actions: [batch, seq_len, 8] -> [batch*seq_len, 8]
                actions_flat = actions.view(-1, 8)
            else:
                # actions: [batch, seq_len] -> [batch*seq_len]
                actions_flat = actions.view(-1)

            mask_flat = action_mask.view(-1, n_logits) if action_mask is not None else None

            log_prob = self.distribution.log_prob(logits_flat, actions_flat, mask_flat)
            entropy = self.distribution.entropy(logits_flat, mask_flat)

            log_prob = log_prob.view(batch_size, seq_len)
            entropy = entropy.view(batch_size, seq_len)
            value = value.squeeze(-1)  # [batch, seq_len]
        else:
            log_prob = self.distribution.log_prob(logits, actions, action_mask)
            entropy = self.distribution.entropy(logits, action_mask)
            value = value.squeeze(-1)

        return log_prob, entropy, value, new_hidden

    def get_value(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get value estimate only.

        Args:
            obs: Observations [batch, obs_dim]
            hidden: Optional LSTM hidden state tuple (h, c)

        Returns:
            Value estimates [batch]
        """
        _, value, _ = self.forward(obs, hidden)
        return value.squeeze(-1)

    def get_initial_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get initial LSTM hidden state.

        Args:
            batch_size: Batch size for hidden state
            device: Device to create tensors on

        Returns:
            Tuple of (h, c) tensors, each [num_layers, batch, hidden_size],
            or None if LSTM is disabled.
        """
        if not self.use_lstm:
            return None

        h = torch.zeros(
            self.lstm_config.num_layers,
            batch_size,
            self.lstm_config.hidden_size,
            device=device,
        )
        c = torch.zeros(
            self.lstm_config.num_layers,
            batch_size,
            self.lstm_config.hidden_size,
            device=device,
        )
        return (h, c)
