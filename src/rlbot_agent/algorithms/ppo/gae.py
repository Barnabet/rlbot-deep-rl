"""Generalized Advantage Estimation (GAE) computation."""

from typing import Tuple

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.

    GAE provides a trade-off between bias and variance in advantage estimation
    through the lambda parameter.

    Args:
        rewards: Rewards [T, N] where T=timesteps, N=envs
        values: Value estimates [T, N]
        dones: Done flags [T, N]
        last_values: Value estimates for final state [N]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (0=TD(0), 1=MC)

    Returns:
        Tuple of (advantages, returns) both [T, N]
    """
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae_lam = np.zeros(N)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = last_values
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]

        # GAE: A_t = δ_t + γλ * A_{t+1}
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    # Returns = advantages + values
    returns = advantages + values

    return advantages, returns


def compute_gae_torch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE using PyTorch tensors.

    Args:
        rewards: Rewards [T, N]
        values: Value estimates [T, N]
        dones: Done flags [T, N]
        last_values: Value estimates for final state [N]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns) both [T, N]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros_like(last_values)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = last_values
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values

    return advantages, returns


class GAE:
    """Generalized Advantage Estimation calculator.

    Provides a stateful interface for computing GAE, useful for
    integrating with training loops.
    """

    def __init__(
        self,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        normalize: bool = True,
    ):
        """Initialize GAE calculator.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize: Whether to normalize advantages
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize = normalize

    def __call__(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages and returns.

        Args:
            rewards: Rewards [T, N]
            values: Value estimates [T, N]
            dones: Done flags [T, N]
            last_values: Value estimates for final state [N]

        Returns:
            Tuple of (advantages, returns)
        """
        advantages, returns = compute_gae(
            rewards, values, dones, last_values,
            self.gamma, self.gae_lambda
        )

        if self.normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns
