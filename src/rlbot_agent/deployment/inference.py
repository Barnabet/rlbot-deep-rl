"""Optimized inference engine for deployment."""

from typing import Optional, Tuple

import numpy as np
import torch

from ..core.config import NetworkConfig, ObservationConfig
from ..models import ActorAttentionCritic


class InferenceEngine:
    """Optimized inference engine for fast action computation.

    Optimizations:
    - JIT compilation (optional)
    - Batched inference
    - GPU/CPU selection
    - Action caching
    """

    def __init__(
        self,
        model: ActorAttentionCritic,
        device: str = "cpu",
        use_jit: bool = False,
    ):
        """Initialize inference engine.

        Args:
            model: Trained model
            device: PyTorch device
            use_jit: Whether to use TorchScript JIT
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        # JIT compilation
        if use_jit:
            try:
                dummy_input = torch.zeros(1, model.obs_dim, device=self.device)
                self.model = torch.jit.trace(self.model, dummy_input)
                print("JIT compilation successful")
            except Exception as e:
                print(f"JIT compilation failed: {e}")

        # Pre-allocate tensors for inference
        self._obs_buffer = torch.zeros(1, model.obs_dim, device=self.device)

    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[int, float]:
        """Get action for single observation.

        Args:
            obs: Observation array
            action_mask: Optional action mask
            deterministic: If True, return mode

        Returns:
            Tuple of (action, value)
        """
        # Copy observation to buffer
        self._obs_buffer[0] = torch.from_numpy(obs)

        # Get action mask
        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)

        # Forward pass
        action, log_prob, _, value = self.model.get_action(
            self._obs_buffer,
            action_mask=mask_tensor,
            deterministic=deterministic,
        )

        return action.item(), value.item()

    @torch.no_grad()
    def get_actions_batch(
        self,
        obs_batch: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get actions for batch of observations.

        Args:
            obs_batch: Observations [batch, obs_dim]
            action_masks: Optional action masks [batch, n_actions]
            deterministic: If True, return modes

        Returns:
            Tuple of (actions, values)
        """
        obs_tensor = torch.from_numpy(obs_batch).to(self.device)

        mask_tensor = None
        if action_masks is not None:
            mask_tensor = torch.from_numpy(action_masks).to(self.device)

        actions, log_probs, _, values = self.model.get_action(
            obs_tensor,
            action_mask=mask_tensor,
            deterministic=deterministic,
        )

        return actions.cpu().numpy(), values.cpu().numpy()

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate for observation.

        Args:
            obs: Observation array

        Returns:
            Value estimate
        """
        self._obs_buffer[0] = torch.from_numpy(obs)
        value = self.model.get_value(self._obs_buffer)
        return value.item()

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str,
        device: str = "cpu",
        use_jit: bool = False,
    ) -> "InferenceEngine":
        """Create inference engine from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            device: PyTorch device
            use_jit: Whether to use JIT

        Returns:
            Inference engine
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        obs_config = ObservationConfig()
        network_config = NetworkConfig()

        model = ActorAttentionCritic(
            obs_config=obs_config,
            network_config=network_config,
            n_actions=1944,
            max_players=6,
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        return InferenceEngine(model, device, use_jit)


class CachedInferenceEngine(InferenceEngine):
    """Inference engine with action caching for tick skip."""

    def __init__(
        self,
        model: ActorAttentionCritic,
        device: str = "cpu",
        use_jit: bool = False,
        tick_skip: int = 8,
    ):
        """Initialize cached inference engine.

        Args:
            model: Trained model
            device: PyTorch device
            use_jit: Whether to use JIT
            tick_skip: Number of ticks to cache action
        """
        super().__init__(model, device, use_jit)
        self.tick_skip = tick_skip

        self._tick_count = 0
        self._cached_action: Optional[int] = None
        self._cached_value: Optional[float] = None

    def get_action_cached(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[int, float]:
        """Get action with caching.

        Only recomputes action every tick_skip ticks.

        Args:
            obs: Observation array
            action_mask: Optional action mask
            deterministic: If True, return mode

        Returns:
            Tuple of (action, value)
        """
        if self._tick_count < self.tick_skip and self._cached_action is not None:
            self._tick_count += 1
            return self._cached_action, self._cached_value

        # Recompute
        action, value = self.get_action(obs, action_mask, deterministic)

        self._cached_action = action
        self._cached_value = value
        self._tick_count = 1

        return action, value

    def reset_cache(self) -> None:
        """Reset action cache (e.g., on episode end)."""
        self._tick_count = 0
        self._cached_action = None
        self._cached_value = None
