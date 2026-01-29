"""Rollout worker for distributed training."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..core.config import ObservationConfig, NetworkConfig
from ..models import ActorAttentionCritic


class Worker:
    """Rollout worker that collects experience from the environment.

    Workers run environment simulations and send collected experience
    to the central learner for training.
    """

    def __init__(
        self,
        worker_id: int,
        env: Any,
        obs_config: ObservationConfig,
        network_config: NetworkConfig,
        n_actions: int = 1944,
        max_players: int = 6,
        device: str = "cpu",
    ):
        """Initialize worker.

        Args:
            worker_id: Unique worker identifier
            env: Environment instance
            obs_config: Observation configuration
            network_config: Network configuration
            n_actions: Number of discrete actions
            max_players: Maximum number of players
            device: PyTorch device (usually CPU for workers)
        """
        self.worker_id = worker_id
        self.env = env
        self.device = torch.device(device)

        # Create local model copy for inference
        self.model = ActorAttentionCritic(
            obs_config=obs_config,
            network_config=network_config,
            n_actions=n_actions,
            max_players=max_players,
        ).to(self.device)
        self.model.eval()

        # Environment state
        self._obs = None
        self._done = True

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Update local model weights from learner.

        Args:
            weights: Model state dict from learner
        """
        self.model.load_state_dict({k: v.to(self.device) for k, v in weights.items()})
        self.model.eval()

    def collect_rollout(
        self,
        n_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Collect a rollout of experience.

        Args:
            n_steps: Number of steps to collect

        Returns:
            Dictionary containing collected experience
        """
        # Storage
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        action_masks = []

        # Reset if needed
        if self._done:
            self._obs = self.env.reset()
            self._done = False

        with torch.no_grad():
            for _ in range(n_steps):
                obs_tensor = torch.tensor(
                    self._obs[np.newaxis, :] if self._obs.ndim == 1 else self._obs,
                    device=self.device,
                )

                # Get action from policy
                action, log_prob, _, value = self.model.get_action(obs_tensor)

                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

                # Store transition
                observations.append(self._obs.copy())
                actions.append(action.item() if action.ndim == 0 else action)
                log_probs.append(log_prob.item() if log_prob.ndim == 0 else log_prob)
                values.append(value.item() if value.ndim == 0 else value)

                # Environment step
                next_obs, reward, done, truncated, info = self.env.step(
                    action.item() if action.ndim == 0 else action
                )

                rewards.append(reward)
                dones.append(float(done or truncated))

                # Update state
                self._obs = next_obs
                self._done = done or truncated

                if self._done:
                    self._obs = self.env.reset()
                    self._done = False

        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'last_obs': self._obs.copy(),
            'last_done': float(self._done),
        }

    def get_stats(self) -> Dict[str, float]:
        """Get worker statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            'worker_id': self.worker_id,
        }


class VectorizedWorker:
    """Worker for vectorized environments (multiple envs per worker)."""

    def __init__(
        self,
        worker_id: int,
        envs: Any,  # VectorEnv
        obs_config: ObservationConfig,
        network_config: NetworkConfig,
        n_actions: int = 1944,
        max_players: int = 6,
        device: str = "cpu",
    ):
        """Initialize vectorized worker.

        Args:
            worker_id: Unique worker identifier
            envs: Vectorized environment
            obs_config: Observation configuration
            network_config: Network configuration
            n_actions: Number of discrete actions
            max_players: Maximum number of players
            device: PyTorch device
        """
        self.worker_id = worker_id
        self.envs = envs
        self.n_envs = envs.num_envs
        self.device = torch.device(device)

        # Create local model copy
        self.model = ActorAttentionCritic(
            obs_config=obs_config,
            network_config=network_config,
            n_actions=n_actions,
            max_players=max_players,
        ).to(self.device)
        self.model.eval()

        # Environment state
        self._obs = None

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Update local model weights."""
        self.model.load_state_dict({k: v.to(self.device) for k, v in weights.items()})
        self.model.eval()

    def collect_rollout(
        self,
        n_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Collect rollouts from all environments.

        Args:
            n_steps: Number of steps per environment

        Returns:
            Dictionary containing collected experience [T, N, ...]
        """
        # Storage [T, N, ...]
        observations = np.zeros((n_steps, self.n_envs, self.model.obs_dim), dtype=np.float32)
        actions = np.zeros((n_steps, self.n_envs), dtype=np.int64)
        rewards = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        dones = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        log_probs = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        values = np.zeros((n_steps, self.n_envs), dtype=np.float32)

        # Reset if needed
        if self._obs is None:
            self._obs = self.envs.reset()

        with torch.no_grad():
            for t in range(n_steps):
                obs_tensor = torch.tensor(self._obs, device=self.device)

                # Get actions for all envs
                action, log_prob, _, value = self.model.get_action(obs_tensor)

                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

                # Store
                observations[t] = self._obs
                actions[t] = action
                log_probs[t] = log_prob
                values[t] = value

                # Step all envs
                next_obs, reward, done, truncated, info = self.envs.step(action)

                rewards[t] = reward
                dones[t] = np.logical_or(done, truncated).astype(np.float32)

                self._obs = next_obs

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'log_probs': log_probs,
            'values': values,
            'last_obs': self._obs.copy(),
            'last_dones': dones[-1],
        }
