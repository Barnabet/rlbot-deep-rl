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


class MultiDiscreteDistribution(nn.Module):
    """Multi-head discrete distribution for independent action dimensions.

    Instead of 1944 actions, outputs 8 independent distributions:
    - 5 continuous controls (throttle, steer, pitch, yaw, roll): 3 options each
    - 3 binary controls (jump, boost, handbrake): 2 options each

    Total: 3+3+3+3+3+2+2+2 = 21 logits instead of 1944
    """

    # Action dimensions: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    ACTION_DIMS = (3, 3, 3, 3, 3, 2, 2, 2)
    TOTAL_LOGITS = sum(ACTION_DIMS)  # 21

    def __init__(self):
        super().__init__()
        self.action_dims = self.ACTION_DIMS
        self.n_heads = len(self.action_dims)

        # Precompute split indices
        self.split_sizes = list(self.action_dims)

    def _split_logits(self, logits: torch.Tensor) -> list:
        """Split flat logits into per-head logits."""
        return torch.split(logits, self.split_sizes, dim=-1)

    def sample(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from all heads.

        Args:
            logits: [batch, 21] flat logits
            action_mask: Not used (kept for API compatibility)
            deterministic: If True, return mode

        Returns:
            action: [batch, 8] action indices per head
            log_prob: [batch] sum of log probs
            entropy: [batch] sum of entropies
        """
        head_logits = self._split_logits(logits)

        actions = []
        log_probs = []
        entropies = []

        for head_idx, h_logits in enumerate(head_logits):
            dist = Categorical(logits=h_logits)

            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            actions.append(action)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

        # Stack actions, sum log_probs and entropies
        actions = torch.stack(actions, dim=-1)  # [batch, 8]
        total_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)  # [batch]
        total_entropy = torch.stack(entropies, dim=-1).sum(dim=-1)  # [batch]

        return actions, total_log_prob, total_entropy

    def log_prob(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate log probability of multi-discrete actions.

        Args:
            logits: [batch, 21] flat logits
            actions: [batch, 8] action indices per head

        Returns:
            [batch] sum of log probs
        """
        head_logits = self._split_logits(logits)

        log_probs = []
        for head_idx, h_logits in enumerate(head_logits):
            dist = Categorical(logits=h_logits)
            log_probs.append(dist.log_prob(actions[..., head_idx]))

        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def entropy(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate average entropy across all heads.

        We use MEAN instead of SUM because:
        - With sum, it's "easy" to maintain high entropy (each head just stays uniform)
        - With mean, entropy reflects per-decision uncertainty
        - This makes entropy_coef comparable to flat action spaces

        Args:
            logits: [batch, 21] flat logits

        Returns:
            [batch] mean entropy per head
        """
        head_logits = self._split_logits(logits)

        entropies = []
        for h_logits in head_logits:
            dist = Categorical(logits=h_logits)
            entropies.append(dist.entropy())

        return torch.stack(entropies, dim=-1).mean(dim=-1)

    def get_head_probs(self, logits: torch.Tensor) -> dict:
        """Get per-head probabilities for logging.

        Returns dict with jump_prob, boost_prob, etc.
        """
        head_logits = self._split_logits(logits)
        head_names = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']

        result = {}
        for name, h_logits in zip(head_names, head_logits):
            probs = F.softmax(h_logits, dim=-1)
            if probs.shape[-1] == 2:
                # Binary: return prob of action=1
                result[f'{name}_prob'] = probs[..., 1].mean().item()
            else:
                # 3-way: return prob of each
                result[f'{name}_neg_prob'] = probs[..., 0].mean().item()
                result[f'{name}_zero_prob'] = probs[..., 1].mean().item()
                result[f'{name}_pos_prob'] = probs[..., 2].mean().item()

        return result


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
