"""Distributed training coordinator."""

import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..algorithms.ppo import PPO, RolloutBuffer
from ..core.config import (
    EnvironmentConfig,
    NetworkConfig,
    ObservationConfig,
    PPOConfig,
    TrainingConfig,
)
from ..environment import create_environment
from ..models import ActorAttentionCritic
from .callbacks import CheckpointCallback, LoggingCallback, RewardAnnealingCallback
from .learner import Learner


class TrainingCoordinator:
    """Coordinates distributed PPO training.

    Manages multiple workers collecting experience in parallel
    and a central learner performing gradient updates.
    """

    def __init__(
        self,
        env_config: EnvironmentConfig,
        obs_config: ObservationConfig,
        network_config: NetworkConfig,
        ppo_config: PPOConfig,
        training_config: TrainingConfig,
        reward_fn: Optional[Any] = None,
    ):
        """Initialize training coordinator.

        Args:
            env_config: Environment configuration
            obs_config: Observation configuration
            network_config: Network configuration
            ppo_config: PPO configuration
            training_config: Training configuration
            reward_fn: Optional reward function (created if None)
        """
        self.env_config = env_config
        self.obs_config = obs_config
        self.network_config = network_config
        self.ppo_config = ppo_config
        self.training_config = training_config
        self.reward_fn = reward_fn

        # Set device
        if training_config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(training_config.device)

        # Set seed
        torch.manual_seed(training_config.seed)
        np.random.seed(training_config.seed)

        # Create learner
        self.learner = Learner(
            obs_config=obs_config,
            network_config=network_config,
            ppo_config=ppo_config,
            n_actions=1944,
            max_players=env_config.max_players,
            device=training_config.device,
        )

        # Callbacks
        self.checkpoint_callback = CheckpointCallback(
            checkpoint_dir=training_config.checkpoint_dir,
            save_interval=training_config.checkpoint_interval,
        )

        self.logging_callback = LoggingCallback(
            log_interval=training_config.log_interval,
            wandb_project=training_config.wandb_project,
            wandb_entity=training_config.wandb_entity,
            wandb_config={
                "env": env_config.__dict__,
                "obs": obs_config.__dict__,
                "network": str(network_config),
                "ppo": ppo_config.__dict__,
                "training": training_config.__dict__,
            },
        )

        if reward_fn is not None:
            self.reward_callback = RewardAnnealingCallback(reward_fn)
        else:
            self.reward_callback = None

        # Training state
        self.global_step = 0

    def train(
        self,
        env_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Run training loop.

        Args:
            env_factory: Factory function to create environments
        """
        # Create environments
        if env_factory is not None:
            envs = [env_factory() for _ in range(self.training_config.n_workers)]
        else:
            envs = [
                create_environment(
                    env_config=self.env_config,
                    obs_config=self.obs_config,
                    reward_config=None,
                )
                for _ in range(self.training_config.n_workers)
            ]

        # Calculate steps per update
        n_envs = len(envs) * getattr(envs[0], 'num_envs', 1)
        steps_per_rollout = self.ppo_config.batch_size // n_envs

        # Create rollout buffer
        buffer = RolloutBuffer(
            buffer_size=steps_per_rollout,
            n_envs=n_envs,
            obs_dim=self.learner.obs_dim,
            n_actions=self.learner.n_actions,
            device=self.device,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
        )

        # Training loop
        print(f"Starting training for {self.training_config.total_steps:,} steps")
        print(f"  Device: {self.device}")
        print(f"  Workers: {len(envs)}")
        print(f"  Steps per rollout: {steps_per_rollout}")
        print(f"  Batch size: {self.ppo_config.batch_size}")

        # Initialize observations
        obs_list = [env.reset() for env in envs]
        obs = np.stack(obs_list) if len(envs) > 1 else obs_list[0][np.newaxis, :]

        while self.global_step < self.training_config.total_steps:
            # Collect experience
            buffer.reset()

            self.learner.model.eval()
            with torch.no_grad():
                for _ in range(steps_per_rollout):
                    obs_tensor = torch.tensor(obs, device=self.device)
                    action, log_prob, _, value = self.learner.model.get_action(obs_tensor)

                    action_np = action.cpu().numpy()
                    log_prob_np = log_prob.cpu().numpy()
                    value_np = value.cpu().numpy()

                    # Step environments
                    next_obs_list = []
                    rewards = []
                    dones = []

                    for i, env in enumerate(envs):
                        next_obs, reward, done, truncated, info = env.step(action_np[i])
                        if done or truncated:
                            next_obs = env.reset()
                        next_obs_list.append(next_obs)
                        rewards.append(reward)
                        dones.append(float(done or truncated))

                    next_obs = np.stack(next_obs_list)
                    rewards = np.array(rewards)
                    dones = np.array(dones)

                    # Store transition
                    buffer.add(
                        obs=obs,
                        action=action_np,
                        reward=rewards,
                        done=dones,
                        log_prob=log_prob_np,
                        value=value_np,
                    )

                    obs = next_obs

            # Compute returns
            buffer.set_last_obs(obs, np.zeros(n_envs))

            with torch.no_grad():
                last_values = self.learner.model.get_value(
                    torch.tensor(obs, device=self.device)
                ).cpu().numpy()

            buffer.compute_returns_and_advantages(last_values)

            # Train
            metrics = self.learner.ppo.train_step(buffer)
            self.global_step += buffer.total_samples

            # Callbacks
            metrics_dict = {
                "train/policy_loss": metrics.policy_loss,
                "train/value_loss": metrics.value_loss,
                "train/entropy_loss": metrics.entropy_loss,
                "train/clip_fraction": metrics.clip_fraction,
                "train/approx_kl": metrics.approx_kl,
                "train/explained_variance": metrics.explained_variance,
                "train/learning_rate": metrics.learning_rate,
            }

            self.checkpoint_callback.on_step(self.global_step, self.learner.ppo, metrics_dict)
            self.logging_callback.on_step(self.global_step, metrics_dict)

            if self.reward_callback is not None:
                self.reward_callback.on_step(self.global_step)

        # Cleanup
        for env in envs:
            env.close()

        self.logging_callback.finish()
        print("Training complete!")

    def save(self, path: str) -> None:
        """Save training state.

        Args:
            path: Path to save to
        """
        self.learner.save(path)

    def load(self, path: str) -> None:
        """Load training state.

        Args:
            path: Path to load from
        """
        self.learner.load(path)
