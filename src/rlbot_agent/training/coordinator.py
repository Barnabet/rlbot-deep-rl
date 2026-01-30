"""Distributed training coordinator."""

import os
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..algorithms.ppo import PPO, RolloutBuffer
from ..core.config import (
    CurriculumConfig,
    EnvironmentConfig,
    NetworkConfig,
    ObservationConfig,
    PPOConfig,
    RewardConfig,
    TrainingConfig,
)
from ..environment import create_environment
from ..models import ActorAttentionCritic
from .callbacks import CheckpointCallback, LoggingCallback, RewardAnnealingCallback
from .callbacks.curriculum import CurriculumCallback
from .callbacks.evaluation import EvaluationCallback
from .learner import Learner
from .env_stats import EnvStatsTracker
from .policy_pool import PolicyPool
from .parallel_env import ParallelEnvManager


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
        reward_config: Optional[RewardConfig] = None,
        curriculum_config: Optional[CurriculumConfig] = None,
    ):
        """Initialize training coordinator.

        Args:
            env_config: Environment configuration
            obs_config: Observation configuration
            network_config: Network configuration
            ppo_config: PPO configuration
            training_config: Training configuration
            reward_fn: Optional reward function (created if None)
            reward_config: Reward configuration for curriculum
            curriculum_config: Curriculum configuration for phase-based training
        """
        self.env_config = env_config
        self.obs_config = obs_config
        self.network_config = network_config
        self.ppo_config = ppo_config
        self.training_config = training_config
        self.reward_fn = reward_fn
        self.reward_config = reward_config
        self.curriculum_config = curriculum_config

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
            use_multi_discrete=True,  # Use 8 independent action heads
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

        # Evaluation callback
        self.eval_callback = EvaluationCallback(
            eval_interval=training_config.eval_interval,
            eval_episodes=training_config.eval_episodes,
            env_config=env_config,
            obs_config=obs_config,
            device=str(self.device),
        )

        # Curriculum callback
        if curriculum_config is not None and curriculum_config.enabled:
            # Get initial reward weights for curriculum
            initial_weights = {}
            if reward_config is not None:
                initial_weights = {
                    'touch_velocity': reward_config.touch_velocity,
                    'velocity_ball_to_goal': reward_config.velocity_ball_to_goal,
                    'speed_toward_ball': reward_config.speed_toward_ball,
                    'goal': reward_config.goal,
                    'save_boost': reward_config.save_boost,
                    'demo': reward_config.demo,
                    'aerial_height': reward_config.aerial_height,
                    'team_spacing_penalty': reward_config.team_spacing_penalty,
                }

            self.curriculum_callback = CurriculumCallback(
                curriculum_config=curriculum_config,
                reward_fn=reward_fn,
                initial_weights=initial_weights,
                on_self_play_start=self._on_self_play_start,
            )
        else:
            self.curriculum_callback = None

        # Policy pool for self-play (initialized lazily when self-play starts)
        self.policy_pool: Optional[PolicyPool] = None

        # Training state
        self.global_step = 0

    def _on_self_play_start(self, step: int) -> None:
        """Called when self-play curriculum phase starts.

        Args:
            step: Current training step
        """
        print(f"\nInitializing self-play at step {step:,}")

        # Initialize policy pool
        self.policy_pool = PolicyPool(
            checkpoint_dir=self.training_config.checkpoint_dir,
            max_pool_size=10,
            device=str(self.device),
            obs_config=self.obs_config,
            network_config=self.network_config,
        )

        print(f"  Policy pool initialized with {len(self.policy_pool.policies)} checkpoints")

    def train(
        self,
        env_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Run training loop with parallel environment stepping.

        Args:
            env_factory: Factory function to create environments
        """
        n_workers = self.training_config.n_workers
        n_envs_per_worker = self.training_config.n_envs_per_worker
        n_games = n_workers * n_envs_per_worker

        # Create environment factory
        if env_factory is None:
            env_factory = partial(
                create_environment,
                env_config=self.env_config,
                obs_config=self.obs_config,
                reward_config=self.reward_config,
            )

        # Create parallel environment manager
        print(f"Creating parallel environment manager...")
        print(f"  Workers: {n_workers}")
        print(f"  Games per worker: {n_envs_per_worker}")
        print(f"  Total games: {n_games}")

        env_manager = ParallelEnvManager(
            env_factory=env_factory,
            n_workers=n_workers,
            n_envs_per_worker=n_envs_per_worker,
        )

        # Initialize observations - this also determines n_agents
        obs = env_manager.reset()
        n_envs = env_manager.n_envs  # Total agents across all games
        n_agents_per_game = env_manager.n_agents_per_game

        print(f"  Agents per game: {n_agents_per_game}")
        print(f"  Total agents (virtual envs): {n_envs}")

        # Calculate steps per update
        # batch_size is the fixed total samples per training update
        # steps_per_rollout is computed dynamically based on number of envs
        # This ensures scaling workers doesn't change the data-to-gradient ratio
        actual_batch_size = self.ppo_config.batch_size
        steps_per_rollout = max(1, actual_batch_size // n_envs)

        # Adjust batch size to be exact multiple of n_envs
        actual_batch_size = steps_per_rollout * n_envs

        # Warn if steps_per_rollout is very short (hurts LSTM learning)
        if steps_per_rollout < 32 and self.network_config.lstm.use_lstm:
            print(f"  WARNING: steps_per_rollout={steps_per_rollout} is very short for LSTM")
            print(f"           Consider increasing batch_size or reducing workers")

        # Create rollout buffer
        # Check if using multi-discrete action space
        use_multi_discrete = getattr(self.learner.model, 'use_multi_discrete', False)
        if use_multi_discrete:
            action_dim = 8
            n_actions_for_mask = 21  # Multi-discrete has 21 logits
        else:
            action_dim = None
            n_actions_for_mask = self.learner.n_actions  # Flat has 1944

        buffer = RolloutBuffer(
            buffer_size=steps_per_rollout,
            n_envs=n_envs,
            obs_dim=self.learner.obs_dim,
            n_actions=n_actions_for_mask,
            device=self.device,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            lstm_config=self.network_config.lstm if self.network_config.lstm.use_lstm else None,
            action_dim=action_dim,
        )

        # Create environment stats tracker
        env_stats = EnvStatsTracker(n_envs=n_envs)

        # Training loop
        updates_per_million = 1_000_000 // actual_batch_size
        minibatches = max(1, actual_batch_size // self.ppo_config.minibatch_size)
        grad_updates_per_batch = self.ppo_config.n_epochs * minibatches
        game_seconds = steps_per_rollout / 15  # 15 Hz decisions
        print(f"\nTraining: {n_envs} agents | {steps_per_rollout} steps/agent ({game_seconds:.1f}s game time) | {actual_batch_size:,} samples/update")
        print(f"  {updates_per_million} updates/1M steps | {grad_updates_per_batch} grad steps/update")

        # Initialize LSTM hidden states if enabled
        use_lstm = self.network_config.lstm.use_lstm
        if use_lstm:
            hidden = self.learner.model.get_initial_hidden(n_envs, self.device)
        else:
            hidden = None

        # Timing
        start_time = time.time()
        last_log_time = start_time

        while self.global_step < self.training_config.total_steps:
            rollout_start = time.time()

            # Collect experience
            buffer.reset()
            env_stats.reset()

            self.learner.model.eval()
            with torch.no_grad():
                for _ in range(steps_per_rollout):
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

                    # Get action (with hidden state if LSTM enabled)
                    action, log_prob, _, value, new_hidden = self.learner.model.get_action(
                        obs_tensor, hidden
                    )

                    action_np = action.cpu().numpy()
                    log_prob_np = log_prob.cpu().numpy()
                    value_np = value.cpu().numpy()

                    # Store hidden state before updating (for training)
                    if use_lstm and hidden is not None:
                        hidden_np = (
                            hidden[0].cpu().numpy(),
                            hidden[1].cpu().numpy(),
                        )
                    else:
                        hidden_np = None

                    # Step all environments in parallel
                    next_obs, rewards, dones, infos = env_manager.step(action_np)

                    # Track action distribution
                    env_stats.track_actions(action_np)

                    # Update environment stats
                    for i, info in enumerate(infos):
                        if 'state' in info and 'player' in info:
                            reward_breakdown = info.get('reward_breakdown', None)
                            env_stats.update(i, info['state'], info['player'], rewards[i], reward_breakdown, info)
                        if dones[i] > 0.5:
                            env_stats.reset_env(i)

                    # Store transition
                    buffer.add(
                        obs=obs,
                        action=action_np,
                        reward=rewards,
                        done=dones,
                        log_prob=log_prob_np,
                        value=value_np,
                        hidden_state=hidden_np,
                    )

                    # Update hidden state
                    if use_lstm:
                        hidden = new_hidden
                        # Reset hidden state for environments that are done
                        done_mask = dones > 0.5
                        if done_mask.any():
                            hidden[0][:, done_mask, :] = 0
                            hidden[1][:, done_mask, :] = 0

                    obs = next_obs

            rollout_time = time.time() - rollout_start

            # Compute returns
            buffer.set_last_obs(obs, np.zeros(n_envs))

            with torch.no_grad():
                last_values = self.learner.model.get_value(
                    torch.tensor(obs, dtype=torch.float32, device=self.device),
                    hidden,
                ).cpu().numpy()

            buffer.compute_returns_and_advantages(last_values)

            # Train
            train_start = time.time()
            metrics = self.learner.ppo.train_step(buffer)
            train_time = time.time() - train_start

            self.global_step += buffer.total_samples

            # Calculate throughput
            total_time = time.time() - start_time
            steps_per_sec = self.global_step / total_time
            hours = total_time / 3600

            # Callbacks
            metrics_dict = {
                "train/policy_loss": metrics.policy_loss,
                "train/value_loss": metrics.value_loss,
                "train/entropy_loss": metrics.entropy_loss,
                "train/clip_fraction": metrics.clip_fraction,
                "train/approx_kl": metrics.approx_kl,
                "train/explained_variance": metrics.explained_variance,
                "train/learning_rate": metrics.learning_rate,
                "time/steps_per_second": steps_per_sec,
                "time/total_hours": hours,
                "time/rollout_time": rollout_time,
                "time/train_time": train_time,
            }

            # Add environment statistics
            env_metrics = env_stats.get_rollout_stats()
            metrics_dict.update(env_metrics)

            # Checkpoint callback (may add checkpoint to policy pool)
            checkpoint_path = self.checkpoint_callback.on_step(
                self.global_step, self.learner.ppo, metrics_dict
            )

            # Add checkpoint to policy pool if self-play is active
            if checkpoint_path and self.policy_pool is not None:
                self.policy_pool.add_checkpoint(checkpoint_path, self.global_step)

            # Curriculum callback (updates reward weights and tracks phase)
            if self.curriculum_callback is not None:
                self.curriculum_callback.on_step(self.global_step, metrics_dict)

            # Add self-play statistics if active
            if self.policy_pool is not None:
                selfplay_stats = self.policy_pool.get_stats()
                metrics_dict.update(selfplay_stats)

            # Evaluation callback (runs periodic evaluation)
            self.eval_callback.on_step(self.global_step, self.learner, metrics_dict)

            # Logging callback (logs all metrics)
            self.logging_callback.on_step(self.global_step, metrics_dict)

            # Reward annealing callback
            if self.reward_callback is not None:
                self.reward_callback.on_step(self.global_step)

        # Cleanup
        env_manager.close()

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
