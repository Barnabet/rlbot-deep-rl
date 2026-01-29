"""Action distributions for policy output."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CategoricalDistribution(nn.Module):
    """Categorical distribution for discrete action spaces.

    Supports action masking to prevent invalid actions.
    """

    def __init__(self, n_actions: int):
        """Initialize categorical distribution.

        Args:
            n_actions: Number of discrete actions
        """
        super().__init__()
        self.n_actions = n_actions

    def forward(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """Create distribution from logits.

        Args:
            logits: Action logits [batch, n_actions]
            action_mask: Boolean mask of valid actions [batch, n_actions]

        Returns:
            Categorical distribution
        """
        if action_mask is not None:
            # Set logits of invalid actions to very negative value
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return Categorical(logits=logits)

    def sample(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from distribution.

        Args:
            logits: Action logits [batch, n_actions]
            action_mask: Boolean mask of valid actions
            deterministic: If True, return mode instead of sampling

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        dist = self.forward(logits, action_mask)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def log_prob(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate log probability of actions.

        Args:
            logits: Action logits [batch, n_actions]
            actions: Taken actions [batch]
            action_mask: Boolean mask of valid actions

        Returns:
            Log probabilities [batch]
        """
        dist = self.forward(logits, action_mask)
        return dist.log_prob(actions)

    def entropy(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate entropy of distribution.

        Args:
            logits: Action logits [batch, n_actions]
            action_mask: Boolean mask of valid actions

        Returns:
            Entropy [batch]
        """
        dist = self.forward(logits, action_mask)
        return dist.entropy()

    def kl_divergence(
        self,
        logits_old: torch.Tensor,
        logits_new: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate KL divergence between two distributions.

        KL(old || new) = sum(p_old * (log p_old - log p_new))

        Args:
            logits_old: Old policy logits [batch, n_actions]
            logits_new: New policy logits [batch, n_actions]
            action_mask: Boolean mask of valid actions

        Returns:
            KL divergence [batch]
        """
        dist_old = self.forward(logits_old, action_mask)
        dist_new = self.forward(logits_new, action_mask)

        return torch.distributions.kl_divergence(dist_old, dist_new)


class MaskedCategorical(Categorical):
    """Categorical distribution with built-in masking."""

    def __init__(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Initialize masked categorical distribution.

        Args:
            logits: Raw logits [batch, n_actions]
            mask: Boolean mask of valid actions [batch, n_actions]
        """
        if mask is not None:
            # Apply mask by setting invalid actions to -inf
            logits = logits.masked_fill(~mask, float('-inf'))

        super().__init__(logits=logits)
        self.mask = mask

    def entropy(self) -> torch.Tensor:
        """Calculate entropy, only considering valid actions."""
        # Standard entropy calculation
        p_log_p = self.logits * self.probs

        if self.mask is not None:
            # Zero out invalid actions
            p_log_p = p_log_p.masked_fill(~self.mask, 0)

        return -p_log_p.sum(dim=-1)
