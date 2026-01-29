"""Team-based reward functions."""

from typing import Any, Dict, List

import numpy as np

from ...core.registry import registry
from .base import BaseReward


@registry.register("reward", "team_spacing")
class TeamSpacing(BaseReward):
    """Penalty for being too close to teammates.

    Encourages proper spacing and rotation.
    """

    MIN_SPACING = 1500.0  # Minimum desired distance from teammate

    def __init__(self, weight: float = -0.1):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate team spacing penalty."""
        car_pos = np.array(player.car_data.position)
        my_team = player.team_num

        min_teammate_dist = float('inf')

        for other in state.players:
            if other.car_id == player.car_id:
                continue
            if other.team_num != my_team:
                continue

            other_pos = np.array(other.car_data.position)
            dist = np.linalg.norm(car_pos - other_pos)
            min_teammate_dist = min(min_teammate_dist, dist)

        if min_teammate_dist == float('inf'):
            # No teammates (1v1)
            return 0.0

        if min_teammate_dist < self.MIN_SPACING:
            # Penalty proportional to how close they are
            penalty = 1.0 - (min_teammate_dist / self.MIN_SPACING)
            return penalty  # Note: weight is negative, so this becomes a penalty

        return 0.0


@registry.register("reward", "passing")
class PassingReward(BaseReward):
    """Reward for passing to teammates.

    Detects when ball is touched by player, then touched by teammate.
    """

    PASS_TIMEOUT = 3.0  # Seconds for pass to count

    def __init__(self, weight: float = 5.0):
        super().__init__(weight)
        self._last_toucher: Dict[int, float] = {}  # player_id -> timestamp
        self._last_toucher_team: int = -1

    def reset(self, initial_state: Any) -> None:
        """Reset pass tracking."""
        self._last_toucher = {}
        self._last_toucher_team = -1

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate passing reward."""
        if not (hasattr(player, 'ball_touched') and player.ball_touched):
            return 0.0

        player_id = player.car_id
        my_team = player.team_num
        current_time = state.game_time if hasattr(state, 'game_time') else 0.0

        reward = 0.0

        # Check if this is a received pass (teammate touched recently)
        if self._last_toucher_team == my_team and player_id not in self._last_toucher:
            for toucher_id, touch_time in self._last_toucher.items():
                if current_time - touch_time < self.PASS_TIMEOUT:
                    # Successful pass received
                    reward = 1.0
                    break

        # Update last toucher
        self._last_toucher = {player_id: current_time}
        self._last_toucher_team = my_team

        return reward


@registry.register("reward", "rotation")
class RotationReward(BaseReward):
    """Reward for proper rotation (not double committing).

    Penalizes having multiple teammates closer to ball than you.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate rotation reward."""
        car_pos = np.array(player.car_data.position)
        ball_pos = np.array(state.ball.position)
        my_team = player.team_num

        my_dist = np.linalg.norm(ball_pos - car_pos)

        # Count teammates closer to ball
        teammates_closer = 0
        total_teammates = 0

        for other in state.players:
            if other.car_id == player.car_id:
                continue
            if other.team_num != my_team:
                continue

            total_teammates += 1
            other_pos = np.array(other.car_data.position)
            other_dist = np.linalg.norm(ball_pos - other_pos)

            if other_dist < my_dist:
                teammates_closer += 1

        if total_teammates == 0:
            return 0.0

        # Reward for proper positioning based on role
        # If closest, should be going for ball
        # If 2nd closest in 3v3, should be ready for pass
        # If furthest, should be back

        # Simple version: reward for being appropriately spaced
        if teammates_closer == 0:
            # We're closest - should be aggressive
            return 0.0  # No bonus, other rewards handle this
        elif teammates_closer == total_teammates:
            # We're furthest back - good defensive position
            return 1.0 if my_dist > 3000 else 0.0
        else:
            # Middle position - support role
            return 0.5

        return 0.0


@registry.register("reward", "team_goal_diff")
class TeamGoalDifferential(BaseReward):
    """Reward based on team goal differential.

    Uses team_spirit to share goal rewards among teammates.
    """

    def __init__(self, weight: float = 50.0):
        super().__init__(weight)
        self._prev_goal_diff: int = 0

    def reset(self, initial_state: Any) -> None:
        """Reset goal differential tracking."""
        blue = initial_state.blue_score if hasattr(initial_state, 'blue_score') else 0
        orange = initial_state.orange_score if hasattr(initial_state, 'orange_score') else 0
        self._prev_goal_diff = blue - orange

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate team goal differential reward."""
        blue = state.blue_score if hasattr(state, 'blue_score') else 0
        orange = state.orange_score if hasattr(state, 'orange_score') else 0

        current_diff = blue - orange
        change = current_diff - self._prev_goal_diff
        self._prev_goal_diff = current_diff

        # Adjust sign based on team
        if player.team_num == 1:  # Orange
            change = -change

        return float(change)


@registry.register("reward", "possession")
class PossessionReward(BaseReward):
    """Reward for team possession (ball closer to opponent's half)."""

    FIELD_Y = 5120.0

    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate possession reward."""
        ball_y = state.ball.position[1]

        # Normalize ball position to [-1, 1]
        normalized_y = ball_y / self.FIELD_Y

        # Blue team wants ball at +Y, Orange at -Y
        if player.team_num == 0:
            return normalized_y
        else:
            return -normalized_y
