"""Central learner for distributed training."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..algorithms.ppo import PPO, RolloutBuffer
from ..core.config import PPOConfig, NetworkConfig, ObservationConfig
from ..core.types import TrainingMetrics
from ..models import ActorAttentionCritic


class Learner:
    """Central learner that performs gradient updates.

    In distributed training, workers collect experience and send it
    to the learner, which performs the actual model updates.
    """

    def __init__(
        self,
        obs_config: ObservationConfig,
        network_config: NetworkConfig,
        ppo_config: PPOConfig,
        n_actions: int = 1944,
        max_players: int = 6,
        device: Optional[str] = None,
        use_multi_discrete: bool = True,
    ):
        """Initialize the learner.

        Args:
            obs_config: Observation configuration
            network_config: Network architecture configuration
            ppo_config: PPO configuration
            n_actions: Number of discrete actions
            max_players: Maximum number of players
            device: PyTorch device ('cuda', 'cpu', or None for auto)
            use_multi_discrete: Use 8 independent action heads (True) or flat 1944 (False)
        """
        # Set device
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Learner using device: {self.device}")
        print(f"Action space: {'multi-discrete (21 logits)' if use_multi_discrete else 'flat (1944 logits)'}")

        # Create model
        self.model = ActorAttentionCritic(
            obs_config=obs_config,
            network_config=network_config,
            n_actions=n_actions,
            max_players=max_players,
            use_multi_discrete=use_multi_discrete,
        ).to(self.device)

        # Create PPO trainer
        self.ppo = PPO(
            model=self.model,
            config=ppo_config,
            device=self.device,
        )

        # Calculate observation dimension
        self.obs_dim = self.model.obs_dim
        self.n_actions = n_actions

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights.

        Returns:
            Dictionary of model state
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set model weights.

        Args:
            weights: Dictionary of model state
        """
        self.model.load_state_dict({k: v.to(self.device) for k, v in weights.items()})

    def train_on_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        last_obs: Optional[np.ndarray] = None,
        last_dones: Optional[np.ndarray] = None,
    ) -> TrainingMetrics:
        """Train on a batch of experience.

        Args:
            observations: Observations [T, N, obs_dim]
            actions: Actions [T, N]
            rewards: Rewards [T, N]
            dones: Done flags [T, N]
            log_probs: Log probabilities [T, N]
            values: Value estimates [T, N]
            action_masks: Action masks [T, N, n_actions]
            last_obs: Last observations [N, obs_dim]
            last_dones: Last done flags [N]

        Returns:
            Training metrics
        """
        T, N = rewards.shape

        # Create buffer
        buffer = RolloutBuffer(
            buffer_size=T,
            n_envs=N,
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            device=self.device,
            gamma=self.ppo.config.gamma,
            gae_lambda=self.ppo.config.gae_lambda,
        )

        # Fill buffer
        for t in range(T):
            buffer.add(
                obs=observations[t],
                action=actions[t],
                reward=rewards[t],
                done=dones[t],
                log_prob=log_probs[t],
                value=values[t],
                action_mask=action_masks[t] if action_masks is not None else None,
            )

        # Set last observation
        if last_obs is not None:
            buffer.set_last_obs(last_obs, last_dones or np.zeros(N))

        # Compute last values for GAE
        with torch.no_grad():
            last_values = self.model.get_value(
                torch.tensor(last_obs, device=self.device)
            ).cpu().numpy()

        # Compute advantages
        buffer.compute_returns_and_advantages(last_values)

        # Train
        metrics = self.ppo.train_step(buffer)

        return metrics

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple:
        """Get action from policy.

        Args:
            obs: Observation [obs_dim] or [batch, obs_dim]
            action_mask: Action mask
            deterministic: If True, return mode

        Returns:
            Tuple of (action, log_prob, value)
        """
        self.model.eval()

        with torch.no_grad():
            if obs.ndim == 1:
                obs = obs[np.newaxis, :]

            obs_tensor = torch.tensor(obs, device=self.device)

            if action_mask is not None:
                if action_mask.ndim == 1:
                    action_mask = action_mask[np.newaxis, :]
                action_mask = torch.tensor(action_mask, device=self.device)

            action, log_prob, _, value = self.model.get_action(
                obs_tensor, action_mask, deterministic
            )

            return (
                action.cpu().numpy().squeeze(),
                log_prob.cpu().numpy().squeeze(),
                value.cpu().numpy().squeeze(),
            )

    def save(self, path: str) -> None:
        """Save learner state.

        Args:
            path: Path to save to
        """
        self.ppo.save(path)

    def load(self, path: str) -> None:
        """Load learner state.

        Args:
            path: Path to load from
        """
        self.ppo.load(path)
