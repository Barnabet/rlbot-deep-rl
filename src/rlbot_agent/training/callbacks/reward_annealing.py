"""Reward annealing callback."""

from typing import Any, Dict, Optional


class RewardAnnealingCallback:
    """Callback for annealing reward weights during training.

    Updates reward function weights based on training progress.
    """

    def __init__(
        self,
        reward_fn: Any,
        anneal_steps: int = 500_000_000,
        team_spirit_start: float = 0.0,
        team_spirit_end: float = 0.3,
    ):
        """Initialize reward annealing callback.

        Args:
            reward_fn: Combined reward function to anneal
            anneal_steps: Total steps for annealing
            team_spirit_start: Initial team spirit value
            team_spirit_end: Final team spirit value
        """
        self.reward_fn = reward_fn
        self.anneal_steps = anneal_steps
        self.team_spirit_start = team_spirit_start
        self.team_spirit_end = team_spirit_end

    def on_step(
        self,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        trainer: Optional[Any] = None,
    ) -> None:
        """Called after each training step.

        Args:
            step: Current global step
            metrics: Training metrics
            trainer: PPO trainer instance
        """
        # Update reward function's global step
        self.reward_fn.set_global_step(step)

        # Anneal team spirit
        progress = min(1.0, step / self.anneal_steps)
        team_spirit = (
            self.team_spirit_start
            + progress * (self.team_spirit_end - self.team_spirit_start)
        )
        self.reward_fn.set_team_spirit(team_spirit)

    def get_current_values(self, step: int) -> Dict[str, float]:
        """Get current annealing values for logging.

        Args:
            step: Current global step

        Returns:
            Dictionary of current values
        """
        progress = min(1.0, step / self.anneal_steps)
        team_spirit = (
            self.team_spirit_start
            + progress * (self.team_spirit_end - self.team_spirit_start)
        )

        return {
            "annealing/progress": progress,
            "annealing/team_spirit": team_spirit,
        }
