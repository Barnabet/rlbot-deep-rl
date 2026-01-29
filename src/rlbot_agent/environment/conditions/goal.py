"""Goal terminal condition."""

from typing import Any

from ...core.registry import registry


@registry.register("condition", "goal")
class GoalCondition:
    """Terminal condition that triggers when a goal is scored."""

    def __init__(self):
        self._prev_blue_score = 0
        self._prev_orange_score = 0

    def reset(self, initial_state: Any) -> None:
        """Reset score tracking.

        Args:
            initial_state: Initial game state
        """
        self._prev_blue_score = getattr(initial_state, 'blue_score', 0)
        self._prev_orange_score = getattr(initial_state, 'orange_score', 0)

    def is_terminal(self, state: Any) -> bool:
        """Check if a goal was scored.

        Args:
            state: Current game state

        Returns:
            True if a goal was scored
        """
        blue_score = getattr(state, 'blue_score', 0)
        orange_score = getattr(state, 'orange_score', 0)

        if blue_score > self._prev_blue_score or orange_score > self._prev_orange_score:
            self._prev_blue_score = blue_score
            self._prev_orange_score = orange_score
            return True

        return False
