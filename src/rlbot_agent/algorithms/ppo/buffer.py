"""Rollout buffer for storing experience."""

from typing import Generator, NamedTuple, Optional

import numpy as np
import torch


class RolloutBufferSamples(NamedTuple):
    """Batch of samples from the rollout buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: Optional[torch.Tensor]


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
        """
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage arrays
        self.observations = np.zeros((buffer_size, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.action_masks = np.ones((buffer_size, n_envs, n_actions), dtype=np.bool_)

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
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value

        if action_mask is not None:
            self.action_masks[self.pos] = action_mask

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
