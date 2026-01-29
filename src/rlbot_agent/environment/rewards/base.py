"""Base reward function interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseReward(ABC):
    """Abstract base class for reward functions."""

    def __init__(self, weight: float = 1.0):
        """Initialize reward function.

        Args:
            weight: Multiplier for the reward
        """
        self.weight = weight
        self._global_step = 0
        self._team_spirit = 0.0

    def set_global_step(self, step: int) -> None:
        """Update global training step for annealing.

        Args:
            step: Current global training step
        """
        self._global_step = step

    def set_team_spirit(self, team_spirit: float) -> None:
        """Set team spirit coefficient for reward sharing.

        Args:
            team_spirit: Team spirit coefficient (0-1)
        """
        self._team_spirit = team_spirit

    def reset(self, initial_state: Any) -> None:
        """Reset reward function for a new episode.

        Args:
            initial_state: Initial game state
        """
        pass

    @abstractmethod
    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate reward for a single player.

        Args:
            player: Player data
            state: Current game state
            previous_action: Previous action taken

        Returns:
            Reward value (before weighting)
        """
        pass

    def get_weighted_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate weighted reward.

        Args:
            player: Player data
            state: Current game state
            previous_action: Previous action taken

        Returns:
            Weighted reward value
        """
        return self.weight * self.get_reward(player, state, previous_action)

    def get_final_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate final reward at episode end.

        Override this for rewards that need special handling at episode end.

        Args:
            player: Player data
            state: Final game state
            previous_action: Final action taken

        Returns:
            Final reward value (before weighting)
        """
        return self.get_reward(player, state, previous_action)
