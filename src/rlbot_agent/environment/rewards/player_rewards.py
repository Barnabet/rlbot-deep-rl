"""Player-related reward functions."""

from typing import Any

import numpy as np

from ...core.registry import registry
from .base import BaseReward


@registry.register("reward", "save_boost")
class SaveBoost(BaseReward):
    """Reward for having boost available.

    Formula: sqrt(boost_amount / 100)

    Uses sqrt to encourage maintaining some boost rather than full boost.
    """

    def __init__(self, weight: float = 2.0):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate boost conservation reward."""
        boost = player.boost_amount / 100.0
        return np.sqrt(boost)


@registry.register("reward", "in_air")
class InAir(BaseReward):
    """Reward for being in the air (aerial training).

    Returns 1 if in air, 0 if on ground.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate in-air reward."""
        return 0.0 if player.on_ground else 1.0


@registry.register("reward", "on_ground")
class OnGround(BaseReward):
    """Reward for staying on the ground.

    Returns 1 if on ground, 0 if in air.
    Positive-only reward to encourage ground control.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate on-ground reward."""
        return 1.0 if player.on_ground else 0.0


@registry.register("reward", "ball_proximity")
class BallProximity(BaseReward):
    """Reward based on proximity to ball.

    Formula: 1 - (distance / max_distance)

    Clamped to [0, 1].
    """

    MAX_DISTANCE = 10000.0  # Approximate field diagonal

    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate ball proximity reward."""
        car_pos = np.array(player.car_data.position)
        ball_pos = np.array(state.ball.position)

        distance = np.linalg.norm(ball_pos - car_pos)
        proximity = 1.0 - (distance / self.MAX_DISTANCE)

        return max(0.0, proximity)


@registry.register("reward", "aerial_height")
class AerialHeight(BaseReward):
    """Reward for touching ball at height.

    Formula: ball_z / max_height on touch

    Only rewards aerial touches.
    """

    MAX_HEIGHT = 2048.0
    MIN_HEIGHT = 300.0  # Below this is not aerial

    def __init__(self, weight: float = 0.5):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate aerial height reward."""
        if not (hasattr(player, 'ball_touched') and player.ball_touched):
            return 0.0

        ball_z = state.ball.position[2]

        if ball_z < self.MIN_HEIGHT:
            return 0.0

        return ball_z / self.MAX_HEIGHT


@registry.register("reward", "speed")
class Speed(BaseReward):
    """Reward for car speed.

    Formula: car_speed / max_speed

    Encourages fast movement.
    """

    MAX_SPEED = 2300.0

    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate speed reward."""
        car_vel = np.array(player.car_data.linear_velocity)
        speed = np.linalg.norm(car_vel)
        return speed / self.MAX_SPEED


@registry.register("reward", "flip_reset")
class FlipReset(BaseReward):
    """Reward for getting a flip reset (touching ball with wheels while airborne).

    Only rewards if has_flip becomes true while not on ground.
    """

    def __init__(self, weight: float = 5.0):
        super().__init__(weight)
        self._prev_has_flip = {}

    def reset(self, initial_state: Any) -> None:
        """Reset flip tracking."""
        self._prev_has_flip = {}

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate flip reset reward."""
        player_id = player.car_id

        # Get previous flip state
        prev_flip = self._prev_has_flip.get(player_id, player.has_flip)
        curr_flip = player.has_flip

        # Update state
        self._prev_has_flip[player_id] = curr_flip

        # Check for flip reset (gained flip while in air)
        if not player.on_ground and curr_flip and not prev_flip:
            return 1.0

        return 0.0


@registry.register("reward", "align_ball")
class AlignBall(BaseReward):
    """Reward for positioning between ball and own goal.

    Encourages defensive positioning.
    """

    GOAL_Y = 5120.0

    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate alignment reward."""
        car_pos = np.array(player.car_data.position)
        ball_pos = np.array(state.ball.position)

        # Own goal position
        if player.team_num == 0:  # Blue
            own_goal = np.array([0.0, -self.GOAL_Y, 0.0])
        else:  # Orange
            own_goal = np.array([0.0, self.GOAL_Y, 0.0])

        # Vector from own goal to ball
        goal_to_ball = ball_pos - own_goal
        goal_to_ball_dist = np.linalg.norm(goal_to_ball[:2])  # 2D distance

        if goal_to_ball_dist < 1e-6:
            return 0.0

        # Vector from own goal to car
        goal_to_car = car_pos - own_goal

        # Project car position onto goal-to-ball line
        t = np.dot(goal_to_car[:2], goal_to_ball[:2]) / (goal_to_ball_dist ** 2)

        # Reward being on the line between goal and ball (0 < t < 1)
        if 0 < t < 1:
            # Calculate perpendicular distance from line
            projected = own_goal + t * goal_to_ball
            perp_dist = np.linalg.norm(car_pos[:2] - projected[:2])

            # Reward inversely proportional to perpendicular distance
            max_perp = 2000.0
            return max(0.0, 1.0 - perp_dist / max_perp)

        return 0.0
