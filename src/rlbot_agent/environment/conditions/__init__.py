"""Terminal conditions for RLGym."""

from .goal import GoalCondition
from .no_touch import NoTouchCondition
from .timeout import TimeoutCondition

__all__ = ["GoalCondition", "TimeoutCondition", "NoTouchCondition"]
