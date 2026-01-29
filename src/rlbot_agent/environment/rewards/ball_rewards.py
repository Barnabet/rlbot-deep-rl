"""Ball-related reward functions."""

from typing import Any, Optional

import numpy as np

from ...core.registry import registry
from .base import BaseReward


@registry.register("reward", "touch_velocity")
class TouchVelocity(BaseReward):
    """Reward for hitting the ball with high velocity.

    Formula: min(Δball_vel / max_speed, 1.0) on touch

    Encourages powerful hits.
    """

    MAX_SPEED = 2300.0

    def __init__(self, weight: float = 50.0):
        super().__init__(weight)
        self._prev_ball_vel = None

    def reset(self, initial_state: Any) -> None:
        """Reset previous ball velocity."""
        self._prev_ball_vel = np.array(initial_state.ball.linear_velocity)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate touch velocity reward."""
        current_vel = np.array(state.ball.linear_velocity)

        # Check if this player touched the ball
        if hasattr(player, 'ball_touched') and player.ball_touched:
            if self._prev_ball_vel is not None:
                # Calculate velocity change
                delta_vel = np.linalg.norm(current_vel - self._prev_ball_vel)
                reward = min(delta_vel / self.MAX_SPEED, 1.0)
            else:
                reward = 0.0
        else:
            reward = 0.0

        # Update previous velocity
        self._prev_ball_vel = current_vel.copy()

        return reward


@registry.register("reward", "velocity_ball_to_goal")
class VelocityBallToGoal(BaseReward):
    """Reward for ball velocity toward opponent's goal.

    Formula: dot(ball_vel, goal_dir) / max_speed

    Positive when ball moves toward opponent goal, negative when toward own goal.
    """

    MAX_SPEED = 2300.0
    GOAL_Y = 5120.0

    def __init__(self, weight: float = 10.0):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate ball-to-goal velocity reward."""
        ball_pos = np.array(state.ball.position)
        ball_vel = np.array(state.ball.linear_velocity)

        # Determine opponent's goal position based on team
        if player.team_num == 0:  # Blue team
            goal_pos = np.array([0.0, self.GOAL_Y, 0.0])
        else:  # Orange team
            goal_pos = np.array([0.0, -self.GOAL_Y, 0.0])

        # Direction from ball to goal
        ball_to_goal = goal_pos - ball_pos
        ball_to_goal_dist = np.linalg.norm(ball_to_goal)

        if ball_to_goal_dist < 1e-6:
            return 0.0

        goal_dir = ball_to_goal / ball_to_goal_dist

        # Velocity component toward goal
        vel_toward_goal = np.dot(ball_vel, goal_dir)

        return vel_toward_goal / self.MAX_SPEED


@registry.register("reward", "speed_toward_ball")
class SpeedTowardBall(BaseReward):
    """Reward for car velocity toward the ball.

    Formula: max(0, dot(car_vel, ball_dir)) / max_speed

    Only positive when moving toward ball.
    """

    MAX_SPEED = 2300.0

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate speed toward ball reward."""
        car_pos = np.array(player.car_data.position)
        car_vel = np.array(player.car_data.linear_velocity)
        ball_pos = np.array(state.ball.position)

        # Direction from car to ball
        car_to_ball = ball_pos - car_pos
        distance = np.linalg.norm(car_to_ball)

        if distance < 1e-6:
            return 0.0

        ball_dir = car_to_ball / distance

        # Velocity component toward ball
        vel_toward_ball = np.dot(car_vel, ball_dir)

        # Only reward positive velocity
        return max(0.0, vel_toward_ball) / self.MAX_SPEED


@registry.register("reward", "ball_height")
class BallHeight(BaseReward):
    """Reward based on ball height (useful for aerial training).

    Formula: ball_z / max_height

    Encourages keeping ball in the air.
    """

    MAX_HEIGHT = 2048.0

    def __init__(self, weight: float = 0.5):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate ball height reward."""
        ball_z = state.ball.position[2]
        return ball_z / self.MAX_HEIGHT


@registry.register("reward", "face_ball")
class FaceBall(BaseReward):
    """Reward for facing toward the ball.

    Formula: dot(car_forward, ball_dir)

    Range [-1, 1], 1 when facing ball directly.
    """

    def __init__(self, weight: float = 0.5):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate face ball reward."""
        car = player.car_data
        car_pos = np.array(car.position)
        ball_pos = np.array(state.ball.position)

        # Direction to ball
        to_ball = ball_pos - car_pos
        distance = np.linalg.norm(to_ball)

        if distance < 1e-6:
            return 0.0

        ball_dir = to_ball / distance

        # Car's forward direction
        forward = np.array(car.forward())

        return np.dot(forward, ball_dir)
