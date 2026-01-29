"""Game event reward functions."""

from typing import Any, Dict

import numpy as np

from ...core.registry import registry
from .base import BaseReward


@registry.register("reward", "goal")
class GoalReward(BaseReward):
    """Reward for scoring/conceding goals.

    +1 for scoring, -1 for conceding.
    """

    def __init__(self, weight: float = 100.0):
        super().__init__(weight)
        self._prev_scores: Dict[int, int] = {}

    def reset(self, initial_state: Any) -> None:
        """Reset score tracking."""
        self._prev_scores = {
            0: initial_state.blue_score if hasattr(initial_state, 'blue_score') else 0,
            1: initial_state.orange_score if hasattr(initial_state, 'orange_score') else 0,
        }

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate goal reward."""
        # Get current scores
        blue_score = state.blue_score if hasattr(state, 'blue_score') else 0
        orange_score = state.orange_score if hasattr(state, 'orange_score') else 0

        # Calculate score changes
        blue_delta = blue_score - self._prev_scores.get(0, 0)
        orange_delta = orange_score - self._prev_scores.get(1, 0)

        # Update tracked scores
        self._prev_scores[0] = blue_score
        self._prev_scores[1] = orange_score

        # Determine reward based on team
        if player.team_num == 0:  # Blue team
            return float(blue_delta - orange_delta)
        else:  # Orange team
            return float(orange_delta - blue_delta)


@registry.register("reward", "save")
class SaveReward(BaseReward):
    """Reward for making saves (clearing ball from own goal area).

    Rewards touching ball when it's heading toward own goal and near goal.
    """

    GOAL_Y = 5120.0
    SAVE_ZONE_Y = 4000.0  # Y distance from goal to consider a save
    SAVE_ZONE_X = 1000.0  # X distance (goal width ~892)
    MAX_SPEED = 2300.0

    def __init__(self, weight: float = 10.0):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate save reward."""
        # Must have touched ball
        if not (hasattr(player, 'ball_touched') and player.ball_touched):
            return 0.0

        ball_pos = np.array(state.ball.position)
        ball_vel = np.array(state.ball.linear_velocity)

        # Check if ball is in save zone (near own goal)
        if player.team_num == 0:  # Blue team, own goal at -Y
            in_save_zone = (
                ball_pos[1] < -self.SAVE_ZONE_Y
                and abs(ball_pos[0]) < self.SAVE_ZONE_X
            )
            ball_toward_goal = ball_vel[1] < 0
        else:  # Orange team, own goal at +Y
            in_save_zone = (
                ball_pos[1] > self.SAVE_ZONE_Y
                and abs(ball_pos[0]) < self.SAVE_ZONE_X
            )
            ball_toward_goal = ball_vel[1] > 0

        if in_save_zone and ball_toward_goal:
            # Reward proportional to ball speed toward goal
            speed_toward_goal = abs(ball_vel[1])
            return min(speed_toward_goal / self.MAX_SPEED, 1.0)

        return 0.0


@registry.register("reward", "demo")
class DemoReward(BaseReward):
    """Reward for demolishing opponents / penalty for being demoed.

    +1 for demoing, -1 for being demoed.
    """

    def __init__(self, weight: float = 5.0):
        super().__init__(weight)
        self._prev_demo_state: Dict[int, bool] = {}
        self._demo_counts: Dict[int, int] = {}

    def reset(self, initial_state: Any) -> None:
        """Reset demo tracking."""
        self._prev_demo_state = {}
        self._demo_counts = {}

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate demo reward."""
        reward = 0.0
        player_id = player.car_id

        # Check if player was demoed
        is_demoed = player.is_demoed if hasattr(player, 'is_demoed') else False
        was_demoed = self._prev_demo_state.get(player_id, False)

        if is_demoed and not was_demoed:
            # Got demoed
            reward -= 1.0

        self._prev_demo_state[player_id] = is_demoed

        # Check if player demoed someone (requires demo counter or event)
        if hasattr(player, 'match_demolishes'):
            current_demos = player.match_demolishes
            prev_demos = self._demo_counts.get(player_id, 0)

            if current_demos > prev_demos:
                # Demoed someone
                reward += 1.0

            self._demo_counts[player_id] = current_demos

        return reward


@registry.register("reward", "assist")
class AssistReward(BaseReward):
    """Reward for assists (passing to a teammate who scores)."""

    def __init__(self, weight: float = 30.0):
        super().__init__(weight)
        self._prev_assists: Dict[int, int] = {}

    def reset(self, initial_state: Any) -> None:
        """Reset assist tracking."""
        self._prev_assists = {}

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate assist reward."""
        if not hasattr(player, 'match_assists'):
            return 0.0

        player_id = player.car_id
        current_assists = player.match_assists
        prev_assists = self._prev_assists.get(player_id, 0)

        self._prev_assists[player_id] = current_assists

        if current_assists > prev_assists:
            return 1.0

        return 0.0


@registry.register("reward", "shot_on_goal")
class ShotOnGoal(BaseReward):
    """Reward for shots on goal (ball heading toward goal after touch)."""

    GOAL_Y = 5120.0
    GOAL_WIDTH = 892.0
    GOAL_HEIGHT = 642.0
    MAX_SPEED = 2300.0

    def __init__(self, weight: float = 5.0):
        super().__init__(weight)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate shot on goal reward."""
        if not (hasattr(player, 'ball_touched') and player.ball_touched):
            return 0.0

        ball_pos = np.array(state.ball.position)
        ball_vel = np.array(state.ball.linear_velocity)

        # Determine opponent's goal
        if player.team_num == 0:  # Blue attacks +Y
            goal_y = self.GOAL_Y
            moving_toward = ball_vel[1] > 0
        else:  # Orange attacks -Y
            goal_y = -self.GOAL_Y
            moving_toward = ball_vel[1] < 0

        if not moving_toward:
            return 0.0

        # Predict where ball crosses goal line
        if abs(ball_vel[1]) < 1e-6:
            return 0.0

        time_to_goal = (goal_y - ball_pos[1]) / ball_vel[1]

        if time_to_goal < 0:
            return 0.0

        pred_x = ball_pos[0] + ball_vel[0] * time_to_goal
        pred_z = ball_pos[2] + ball_vel[2] * time_to_goal - 0.5 * 650 * time_to_goal ** 2

        # Check if predicted position is within goal
        if abs(pred_x) < self.GOAL_WIDTH and 0 < pred_z < self.GOAL_HEIGHT:
            # Reward proportional to ball speed
            speed = np.linalg.norm(ball_vel)
            return min(speed / self.MAX_SPEED, 1.0)

        return 0.0
