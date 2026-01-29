"""No touch terminal condition."""

from typing import Any

from ...core.registry import registry


@registry.register("condition", "no_touch")
class NoTouchCondition:
    """Terminal condition that triggers if no one touches the ball for too long.

    Prevents episodes from getting stuck with no progress.
    """

    def __init__(self, timeout_seconds: float = 30.0, tick_skip: int = 8):
        """Initialize no-touch condition.

        Args:
            timeout_seconds: Max time without touch before termination
            tick_skip: Number of physics ticks per step
        """
        self.timeout_seconds = timeout_seconds
        self.tick_skip = tick_skip
        self._max_steps_without_touch = int(timeout_seconds * 120 / tick_skip)
        self._steps_since_touch = 0

    def reset(self, initial_state: Any) -> None:
        """Reset touch counter.

        Args:
            initial_state: Initial game state
        """
        self._steps_since_touch = 0

    def is_terminal(self, state: Any) -> bool:
        """Check if no-touch timeout has been reached.

        Args:
            state: Current game state

        Returns:
            True if no touch for too long
        """
        # Check if anyone touched the ball
        ball_touched = False
        for player in state.players:
            if hasattr(player, 'ball_touched') and player.ball_touched:
                ball_touched = True
                break

        if ball_touched:
            self._steps_since_touch = 0
        else:
            self._steps_since_touch += 1

        return self._steps_since_touch >= self._max_steps_without_touch
