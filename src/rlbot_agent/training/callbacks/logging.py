"""Logging callback with WandB integration."""

import time
from typing import Any, Dict, Optional

import numpy as np


class LoggingCallback:
    """Callback for logging training metrics to WandB and console."""

    def __init__(
        self,
        log_interval: int = 10_000,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[dict] = None,
    ):
        """Initialize logging callback.

        Args:
            log_interval: Steps between logging
            wandb_project: WandB project name
            wandb_entity: WandB entity (team/user)
            wandb_config: Configuration to log to WandB
        """
        self.log_interval = log_interval
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        self._wandb_run = None
        self._last_log_step = 0
        self._start_time = time.time()
        self._step_times = []

        # Initialize WandB if configured
        if wandb_project:
            self._init_wandb(wandb_config)

    def _init_wandb(self, config: Optional[dict]) -> None:
        """Initialize WandB run."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=config,
            )
        except ImportError:
            print("WandB not installed. Logging to console only.")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")

    def on_step(
        self,
        step: int,
        metrics: Dict[str, float],
        trainer: Optional[Any] = None,
    ) -> None:
        """Called after each training step.

        Args:
            step: Current global step
            metrics: Training metrics
            trainer: PPO trainer instance
        """
        current_time = time.time()
        self._step_times.append(current_time)

        if step - self._last_log_step >= self.log_interval:
            self._log_metrics(step, metrics)
            self._last_log_step = step

    def _log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics to WandB and console."""
        # Calculate throughput
        elapsed = time.time() - self._start_time
        sps = step / elapsed if elapsed > 0 else 0

        # Add timing metrics
        metrics['time/steps_per_second'] = sps
        metrics['time/total_hours'] = elapsed / 3600

        # Minimal console output - just key metrics on one line
        reward = metrics.get('env/avg_episode_reward', 0)
        air_pct = metrics.get('env/air_pct', 0)
        ep_len = metrics.get('env/avg_episode_length', 0)
        ev = metrics.get('train/explained_variance', 0)
        print(f"[{step:,}] reward={reward:.1f} air={air_pct:.1f}% len={ep_len:.0f} ev={ev:.2f} sps={sps:.0f}")

        # WandB logging
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Failed to log to WandB: {e}")

    def on_episode_end(
        self,
        step: int,
        episode_metrics: Dict[str, float],
    ) -> None:
        """Called at the end of episodes for episode-level metrics.

        Args:
            step: Current global step
            episode_metrics: Episode-level metrics
        """
        if self._wandb_run is not None:
            try:
                import wandb
                prefixed = {f"episode/{k}": v for k, v in episode_metrics.items()}
                wandb.log(prefixed, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        """Finish logging and close WandB run."""
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
