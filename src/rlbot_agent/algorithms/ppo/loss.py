"""PPO loss functions."""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F


class PPOLossOutput(NamedTuple):
    """Output from PPO loss computation."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    clip_fraction: float
    approx_kl: float


class PPOLoss:
    """PPO clipped surrogate loss with entropy bonus.

    Implements the standard PPO objective:
    L = L_clip - c1 * L_value + c2 * L_entropy
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        value_clip: bool = True,
    ):
        """Initialize PPO loss.

        Args:
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            value_clip: Whether to clip value loss
        """
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.value_clip = value_clip

    def __call__(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> PPOLossOutput:
        """Compute PPO loss.

        Args:
            log_probs: New log probabilities [batch]
            old_log_probs: Old log probabilities [batch]
            values: New value estimates [batch]
            old_values: Old value estimates [batch]
            advantages: Computed advantages [batch]
            returns: Computed returns [batch]
            entropy: Policy entropy [batch]

        Returns:
            PPOLossOutput with all loss components
        """
        # Policy loss (clipped surrogate)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (optionally clipped)
        if self.value_clip:
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            value_loss1 = F.mse_loss(values, returns)
            value_loss2 = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(values, returns)

        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # Compute metrics
        with torch.no_grad():
            clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()
            approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

        return PPOLossOutput(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )


def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute explained variance of predictions.

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        Explained variance (1.0 = perfect, 0.0 = baseline, negative = worse than baseline)
    """
    var_true = y_true.var()
    if var_true == 0:
        return 0.0

    return (1 - (y_true - y_pred).var() / var_true).item()
