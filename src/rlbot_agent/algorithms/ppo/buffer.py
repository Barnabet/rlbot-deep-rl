"""Rollout buffer for storing experience."""

from typing import Generator, NamedTuple, Optional, Tuple

import numpy as np
import torch

from ...core.config import LSTMConfig


class RolloutBufferSamples(NamedTuple):
    """Batch of samples from the rollout buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: Optional[torch.Tensor]


class SequenceRolloutSamples(NamedTuple):
    """Batch of sequences for LSTM training."""

    observations: torch.Tensor  # [batch, seq_len, obs_dim]
    actions: torch.Tensor  # [batch, seq_len]
    old_log_probs: torch.Tensor  # [batch, seq_len]
    old_values: torch.Tensor  # [batch, seq_len]
    advantages: torch.Tensor  # [batch, seq_len]
    returns: torch.Tensor  # [batch, seq_len]
    action_masks: Optional[torch.Tensor]  # [batch, seq_len, n_actions]
    hidden_states: Tuple[torch.Tensor, torch.Tensor]  # (h, c) each [num_layers, batch, hidden]
    masks: torch.Tensor  # [batch, seq_len] - 0 for episode boundaries


class RolloutBuffer:
    """Buffer for storing rollout experience.

    Stores transitions from multiple parallel environments and
    computes advantages using GAE when the buffer is full.
    """

    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        obs_dim: int,
        n_actions: int,
        device: torch.device,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        lstm_config: Optional[LSTMConfig] = None,
        action_dim: Optional[int] = None,
    ):
        """Initialize rollout buffer.

        Args:
            buffer_size: Number of steps to store per environment
            n_envs: Number of parallel environments
            obs_dim: Observation dimension
            n_actions: Number of actions (for action mask)
            device: PyTorch device
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            lstm_config: LSTM configuration (optional, for sequence-based training)
            action_dim: Action dimension (None for flat, 8 for multi-discrete)
        """
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lstm_config = lstm_config
        self.action_dim = action_dim

        # Storage arrays
        self.observations = np.zeros((buffer_size, n_envs, obs_dim), dtype=np.float32)
        if action_dim is not None:
            # Multi-discrete: [buffer_size, n_envs, action_dim]
            self.actions = np.zeros((buffer_size, n_envs, action_dim), dtype=np.int64)
        else:
            # Flat: [buffer_size, n_envs]
            self.actions = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.action_masks = np.ones((buffer_size, n_envs, n_actions), dtype=np.bool_)

        # LSTM hidden state storage
        if lstm_config is not None and lstm_config.use_lstm:
            self.hidden_h = np.zeros(
                (buffer_size, n_envs, lstm_config.num_layers, lstm_config.hidden_size),
                dtype=np.float32,
            )
            self.hidden_c = np.zeros(
                (buffer_size, n_envs, lstm_config.num_layers, lstm_config.hidden_size),
                dtype=np.float32,
            )
        else:
            self.hidden_h = None
            self.hidden_c = None

        # Computed after buffer is full
        self.advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_envs), dtype=np.float32)

        # Buffer state
        self.pos = 0
        self.full = False

        # For storing last observation for bootstrapping
        self._last_obs: Optional[np.ndarray] = None
        self._last_dones: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        hidden_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            obs: Observations [n_envs, obs_dim]
            action: Actions [n_envs]
            reward: Rewards [n_envs]
            done: Done flags [n_envs]
            log_prob: Log probabilities [n_envs]
            value: Value estimates [n_envs]
            action_mask: Action masks [n_envs, n_actions]
            hidden_state: LSTM hidden state tuple (h, c), each [num_layers, n_envs, hidden]
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value

        if action_mask is not None:
            self.action_masks[self.pos] = action_mask

        # Store hidden states if LSTM is enabled
        if hidden_state is not None and self.hidden_h is not None:
            h, c = hidden_state  # Each: [num_layers, n_envs, hidden_size]
            # Transpose to [n_envs, num_layers, hidden_size] for storage
            self.hidden_h[self.pos] = h.transpose(1, 0, 2)
            self.hidden_c[self.pos] = c.transpose(1, 0, 2)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def set_last_obs(
        self,
        last_obs: np.ndarray,
        last_dones: np.ndarray,
    ) -> None:
        """Set the last observation for bootstrapping.

        Args:
            last_obs: Last observations [n_envs, obs_dim]
            last_dones: Last done flags [n_envs]
        """
        self._last_obs = last_obs
        self._last_dones = last_dones

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
    ) -> None:
        """Compute returns and advantages using GAE.

        Args:
            last_values: Value estimates for last observation [n_envs]
        """
        # GAE computation
        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self._last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )

            last_gae_lam = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

            self.advantages[step] = last_gae_lam

        # Compute returns
        self.returns = self.advantages + self.values

    def get(
        self,
        batch_size: int,
    ) -> Generator[RolloutBufferSamples, None, None]:
        """Generate batches of experience.

        Args:
            batch_size: Size of each minibatch

        Yields:
            Batches of experience
        """
        # Flatten buffer
        total_size = self.buffer_size * self.n_envs

        observations = self.observations.reshape(total_size, self.obs_dim)
        if self.action_dim is not None:
            # Multi-discrete: [total_size, action_dim]
            actions = self.actions.reshape(total_size, self.action_dim)
        else:
            # Flat: [total_size]
            actions = self.actions.reshape(total_size)
        log_probs = self.log_probs.reshape(total_size)
        values = self.values.reshape(total_size)
        advantages = self.advantages.reshape(total_size)
        returns = self.returns.reshape(total_size)
        action_masks = self.action_masks.reshape(total_size, self.n_actions)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Generate random indices
        indices = np.random.permutation(total_size)

        # Generate batches
        start_idx = 0
        while start_idx < total_size:
            end_idx = min(start_idx + batch_size, total_size)
            batch_indices = indices[start_idx:end_idx]

            yield RolloutBufferSamples(
                observations=torch.tensor(observations[batch_indices], device=self.device),
                actions=torch.tensor(actions[batch_indices], device=self.device),
                old_log_probs=torch.tensor(log_probs[batch_indices], device=self.device),
                old_values=torch.tensor(values[batch_indices], device=self.device),
                advantages=torch.tensor(advantages[batch_indices], device=self.device),
                returns=torch.tensor(returns[batch_indices], device=self.device),
                action_masks=torch.tensor(action_masks[batch_indices], device=self.device),
            )

            start_idx = end_idx

    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return self.full

    @property
    def total_samples(self) -> int:
        """Get total number of samples in buffer."""
        return self.buffer_size * self.n_envs

    def get_sequences(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> Generator[SequenceRolloutSamples, None, None]:
        """Generate batches of sequences for LSTM training.

        Yields batches of shape [batch_size, sequence_length, ...].
        Sequences preserve temporal order within each sequence.

        Args:
            batch_size: Number of sequences per batch
            sequence_length: Length of each sequence

        Yields:
            SequenceRolloutSamples with properly shaped tensors
        """
        if self.lstm_config is None or not self.lstm_config.use_lstm:
            raise ValueError("get_sequences requires LSTM to be enabled")

        # Build list of valid sequence starts (env_idx, start_step)
        valid_starts = []
        stride = max(1, sequence_length // 2)  # Overlap sequences by 50%

        for env_idx in range(self.n_envs):
            for start in range(0, self.buffer_size - sequence_length + 1, stride):
                valid_starts.append((env_idx, start))

        # Shuffle sequence starts (but not within sequences!)
        np.random.shuffle(valid_starts)

        # Normalize advantages globally
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8

        # Generate batches
        for batch_start in range(0, len(valid_starts), batch_size):
            batch_indices = valid_starts[batch_start:batch_start + batch_size]
            actual_batch_size = len(batch_indices)

            # Allocate batch arrays
            obs_batch = np.zeros(
                (actual_batch_size, sequence_length, self.obs_dim),
                dtype=np.float32,
            )
            if self.action_dim is not None:
                # Multi-discrete: [batch, seq_len, action_dim]
                act_batch = np.zeros(
                    (actual_batch_size, sequence_length, self.action_dim),
                    dtype=np.int64,
                )
            else:
                # Flat: [batch, seq_len]
                act_batch = np.zeros(
                    (actual_batch_size, sequence_length),
                    dtype=np.int64,
                )
            logp_batch = np.zeros(
                (actual_batch_size, sequence_length),
                dtype=np.float32,
            )
            val_batch = np.zeros(
                (actual_batch_size, sequence_length),
                dtype=np.float32,
            )
            adv_batch = np.zeros(
                (actual_batch_size, sequence_length),
                dtype=np.float32,
            )
            ret_batch = np.zeros(
                (actual_batch_size, sequence_length),
                dtype=np.float32,
            )
            mask_batch = np.ones(
                (actual_batch_size, sequence_length),
                dtype=np.float32,
            )
            action_mask_batch = np.ones(
                (actual_batch_size, sequence_length, self.n_actions),
                dtype=np.bool_,
            )

            # Hidden states at sequence start
            h_batch = np.zeros(
                (self.lstm_config.num_layers, actual_batch_size, self.lstm_config.hidden_size),
                dtype=np.float32,
            )
            c_batch = np.zeros(
                (self.lstm_config.num_layers, actual_batch_size, self.lstm_config.hidden_size),
                dtype=np.float32,
            )

            for i, (env_idx, start) in enumerate(batch_indices):
                end = start + sequence_length

                obs_batch[i] = self.observations[start:end, env_idx]
                act_batch[i] = self.actions[start:end, env_idx]
                logp_batch[i] = self.log_probs[start:end, env_idx]
                val_batch[i] = self.values[start:end, env_idx]
                adv_batch[i] = (self.advantages[start:end, env_idx] - adv_mean) / adv_std
                ret_batch[i] = self.returns[start:end, env_idx]
                action_mask_batch[i] = self.action_masks[start:end, env_idx]

                # Mask: 1 for valid, 0 after episode boundaries
                # First step always valid
                mask_batch[i, 0] = 1.0
                # For subsequent steps, check if previous step was done
                mask_batch[i, 1:] = 1.0 - self.dones[start:end-1, env_idx]

                # Initial hidden state for this sequence
                if self.hidden_h is not None:
                    # hidden_h is [buffer, n_envs, num_layers, hidden]
                    # self.hidden_h[start, env_idx] gives [num_layers, hidden]
                    # h_batch is [num_layers, batch, hidden], so h_batch[:, i, :] is [num_layers, hidden]
                    h_batch[:, i, :] = self.hidden_h[start, env_idx]
                    c_batch[:, i, :] = self.hidden_c[start, env_idx]

            yield SequenceRolloutSamples(
                observations=torch.tensor(obs_batch, device=self.device),
                actions=torch.tensor(act_batch, device=self.device),
                old_log_probs=torch.tensor(logp_batch, device=self.device),
                old_values=torch.tensor(val_batch, device=self.device),
                advantages=torch.tensor(adv_batch, device=self.device),
                returns=torch.tensor(ret_batch, device=self.device),
                action_masks=torch.tensor(action_mask_batch, device=self.device),
                hidden_states=(
                    torch.tensor(h_batch, device=self.device),
                    torch.tensor(c_batch, device=self.device),
                ),
                masks=torch.tensor(mask_batch, device=self.device),
            )
