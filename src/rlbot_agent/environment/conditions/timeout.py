"""Timeout terminal condition."""

from typing import Any

from ...core.registry import registry


@registry.register("condition", "timeout")
class TimeoutCondition:
    """Terminal condition that triggers after a time limit."""

    def __init__(self, timeout_seconds: float = 300.0, tick_skip: int = 8):
        """Initialize timeout condition.

        Args:
            timeout_seconds: Maximum episode duration in seconds
            tick_skip: Number of physics ticks per step (for step counting)
        """
        self.timeout_seconds = timeout_seconds
        self.tick_skip = tick_skip
        self._max_steps = int(timeout_seconds * 120 / tick_skip)  # 120 Hz physics
        self._steps = 0

    def reset(self, initial_state: Any) -> None:
        """Reset step counter.

        Args:
            initial_state: Initial game state
        """
        self._steps = 0

    def is_terminal(self, state: Any) -> bool:
        """Check if timeout has been reached.

        Args:
            state: Current game state

        Returns:
            True if timeout reached
        """
        self._steps += 1
        return self._steps >= self._max_steps
