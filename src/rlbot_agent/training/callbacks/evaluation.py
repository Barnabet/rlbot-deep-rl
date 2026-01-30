"""Evaluation callback for periodic evaluation during training."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ...core.config import EnvironmentConfig, ObservationConfig
from ...environment import create_environment


class EvaluationCallback:
    """Callback for periodic evaluation during training.

    Runs deterministic evaluation episodes at specified intervals
    and logs metrics to the training metrics dict.
    """

    def __init__(
        self,
        eval_interval: int,
        eval_episodes: int,
        env_config: EnvironmentConfig,
        obs_config: ObservationConfig,
        device: str = "cpu",
        baseline_checkpoint_path: Optional[str] = None,
    ):
        """Initialize evaluation callback.

        Args:
            eval_interval: Steps between evaluations
            eval_episodes: Number of episodes per evaluation
            env_config: Environment configuration
            obs_config: Observation configuration
            device: PyTorch device for inference
            baseline_checkpoint_path: Path to fixed baseline checkpoint for consistent eval
        """
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.env_config = env_config
        self.obs_config = obs_config
        self.device = torch.device(device if device != "auto" else "cpu")
        self.baseline_checkpoint_path = baseline_checkpoint_path

        self._last_eval_step = 0
        self._eval_env = None

    def on_step(
        self,
        step: int,
        learner: Any,
        metrics_dict: Dict[str, float],
    ) -> None:
        """Called after each training step.

        Args:
            step: Current global step
            learner: Learner instance with model
            metrics_dict: Dictionary to add evaluation metrics to
        """
        if step - self._last_eval_step >= self.eval_interval:
            print(f"\nRunning evaluation at step {step:,}...")
            eval_metrics = self._run_evaluation(learner)

            # Add eval metrics to the metrics dict
            metrics_dict.update(eval_metrics)

            self._last_eval_step = step

            # Print summary
            print(f"  Eval mean reward: {eval_metrics['eval/mean_reward']:.2f}")
            print(f"  Eval mean length: {eval_metrics['eval/mean_length']:.1f}")
            if 'eval/win_rate' in eval_metrics:
                print(f"  Eval win rate: {eval_metrics['eval/win_rate']:.2%}")

    def _run_evaluation(self, learner: Any) -> Dict[str, float]:
        """Run evaluation episodes and return metrics.

        Args:
            learner: Learner instance with model

        Returns:
            Dictionary of evaluation metrics
        """
        # Create evaluation environment (fresh each time to avoid state leakage)
        env = create_environment(
            env_config=self.env_config,
            obs_config=self.obs_config,
            render=False,
        )

        # Put model in eval mode
        learner.model.eval()

        # Collect episode statistics
        episode_rewards = []
        episode_lengths = []
        goals_scored = []
        goals_conceded = []

        for ep in range(self.eval_episodes):
            obs = env.reset()
            n_agents = obs.shape[0] if obs.ndim > 1 else 1
            total_reward = 0.0
            steps = 0
            scored = 0
            conceded = 0
            done = False

            # Reset LSTM hidden state if model uses LSTM
            hidden = None
            if hasattr(learner.model, 'get_initial_hidden'):
                hidden = learner.model.get_initial_hidden(n_agents, self.device)

            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
                    # obs is [n_agents, obs_dim] from multi-agent wrapper

                    # Get deterministic action for all agents
                    action, _, _, _, new_hidden = learner.model.get_action(
                        obs_tensor, hidden, deterministic=True
                    )
                    hidden = new_hidden

                    action_np = action.cpu().numpy()  # [n_agents, action_dim] or [n_agents]

                # Multi-agent env returns (obs, rewards, dones, infos)
                obs, rewards, dones, infos = env.step(action_np)
                total_reward += rewards[0]  # Track first agent (blue)
                steps += 1
                done = dones[0] > 0.5  # Episode ends for all agents together

                # Track goals from info (FixedEpisodeLengthWrapper sets goal_scored_this_step)
                info = infos[0] if infos else {}
                if info.get('goal_scored_this_step'):
                    # Check reward to determine if scored or conceded
                    if rewards[0] > 5:  # Goal reward is typically large positive
                        scored += 1
                    elif rewards[0] < -5:
                        conceded += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            goals_scored.append(scored)
            goals_conceded.append(conceded)

        env.close()

        # Compute aggregate statistics
        metrics = {
            'eval/mean_reward': float(np.mean(episode_rewards)),
            'eval/std_reward': float(np.std(episode_rewards)),
            'eval/min_reward': float(np.min(episode_rewards)),
            'eval/max_reward': float(np.max(episode_rewards)),
            'eval/mean_length': float(np.mean(episode_lengths)),
        }

        # Add goal-based metrics if we tracked any goals
        if sum(goals_scored) > 0 or sum(goals_conceded) > 0:
            metrics['eval/mean_goals_scored'] = float(np.mean(goals_scored))
            metrics['eval/mean_goals_conceded'] = float(np.mean(goals_conceded))
            metrics['eval/goal_differential'] = float(
                np.mean(goals_scored) - np.mean(goals_conceded)
            )
            metrics['eval/win_rate'] = float(
                np.mean([s > c for s, c in zip(goals_scored, goals_conceded)])
            )

        return metrics

    def save_baseline(self, checkpoint_path: str) -> None:
        """Save current checkpoint as the baseline for evaluation.

        Args:
            checkpoint_path: Path to the checkpoint to use as baseline
        """
        self.baseline_checkpoint_path = checkpoint_path
        print(f"Evaluation baseline set to: {checkpoint_path}")

    def close(self) -> None:
        """Clean up resources."""
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None
