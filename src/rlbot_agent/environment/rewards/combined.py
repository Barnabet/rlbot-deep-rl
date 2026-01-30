"""Combined reward function with annealing support."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.config import RewardConfig
from ...core.registry import registry
from .base import BaseReward


@registry.register("reward", "combined")
class CombinedReward(BaseReward):
    """Combines multiple reward functions with weighted sum and annealing.

    Features:
    - Weighted combination of reward functions
    - Reward weight annealing over training
    - Team spirit for reward sharing between teammates
    - Per-reward normalization options
    """

    # Map class names to config field names
    CLASS_TO_CONFIG = {
        'TouchVelocity': 'touch_velocity',
        'VelocityBallToGoal': 'velocity_ball_to_goal',
        'SpeedTowardBall': 'speed_toward_ball',
        'GoalReward': 'goal',
        'SaveBoost': 'save_boost',
        'DemoReward': 'demo',
        'AerialHeight': 'aerial_height',
        'TeamSpacing': 'team_spacing_penalty',
        'OnGround': 'on_ground',
    }

    def __init__(
        self,
        rewards: List[Tuple[BaseReward, float]],
        config: Optional[RewardConfig] = None,
        normalize: bool = False,
    ):
        """Initialize combined reward.

        Args:
            rewards: List of (reward_function, initial_weight) tuples
            config: Reward configuration for annealing
            normalize: Whether to normalize rewards by their running statistics
        """
        super().__init__(weight=1.0)
        self.rewards = rewards
        self.config = config or RewardConfig()
        self.normalize = normalize

        # Store reward names for config lookup
        self._reward_names = [r.__class__.__name__ for r, _ in rewards]

        # Running statistics for normalization
        self._reward_means = [0.0] * len(rewards)
        self._reward_vars = [1.0] * len(rewards)
        self._reward_count = 0

    def set_global_step(self, step: int) -> None:
        """Update global step and interpolate weights from config."""
        super().set_global_step(step)

        # Update child rewards and their weights from config schedule
        for i, (reward, _) in enumerate(self.rewards):
            reward.set_global_step(step)

            # Get interpolated weight from config
            class_name = self._reward_names[i]
            config_name = self.CLASS_TO_CONFIG.get(class_name)
            if config_name:
                weight = self.config.get_weight(config_name, step)
                self.rewards[i] = (reward, weight)

    def set_team_spirit(self, team_spirit: float) -> None:
        """Set team spirit for all child rewards."""
        super().set_team_spirit(team_spirit)
        for reward, _ in self.rewards:
            reward.set_team_spirit(team_spirit)

    def reset(self, initial_state: Any) -> None:
        """Reset all child rewards."""
        for reward, _ in self.rewards:
            reward.reset(initial_state)

    def get_reward(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> float:
        """Calculate combined reward.

        Args:
            player: Player data
            state: Current game state
            previous_action: Previous action taken

        Returns:
            Weighted sum of all rewards
        """
        total_reward = 0.0
        individual_rewards = []

        # Cache per-player breakdown for get_individual_rewards()
        # Use player.car_id as key to support multi-agent
        player_id = getattr(player, 'car_id', id(player))
        if not hasattr(self, '_last_individual_rewards'):
            self._last_individual_rewards = {}
        self._last_individual_rewards[player_id] = {}

        for reward, weight in self.rewards:
            r = reward.get_reward(player, state, previous_action)
            individual_rewards.append(r)
            total_reward += weight * r

            # Cache for breakdown logging
            name = reward.__class__.__name__
            self._last_individual_rewards[player_id][f"{name}/raw"] = r
            self._last_individual_rewards[player_id][f"{name}/weighted"] = weight * r
            self._last_individual_rewards[player_id][f"{name}/weight"] = weight

        # Update normalization statistics if enabled
        if self.normalize:
            self._update_statistics(individual_rewards)

        return total_reward

    def get_individual_rewards(
        self,
        player: Any,
        state: Any,
        previous_action: Any,
    ) -> Dict[str, float]:
        """Get individual reward values for logging.

        Returns cached values from get_reward() to avoid double-calling
        stateful rewards (which would return 0 on second call).

        Args:
            player: Player data
            state: Current game state (unused, for API compat)
            previous_action: Previous action taken (unused, for API compat)

        Returns:
            Dictionary mapping reward names to values
        """
        player_id = getattr(player, 'car_id', id(player))
        if hasattr(self, '_last_individual_rewards') and player_id in self._last_individual_rewards:
            return self._last_individual_rewards[player_id]

        # Fallback: compute fresh (shouldn't happen if get_reward was called first)
        result = {}
        for reward, weight in self.rewards:
            r = reward.get_reward(player, state, previous_action)
            name = reward.__class__.__name__
            result[f"{name}/raw"] = r
            result[f"{name}/weighted"] = weight * r
            result[f"{name}/weight"] = weight
        return result

    def get_team_reward(
        self,
        players: List[Any],
        state: Any,
        previous_actions: List[Any],
        team: int,
    ) -> float:
        """Calculate team reward with team spirit.

        Team spirit blends individual and team average rewards:
        final_reward = (1 - team_spirit) * individual + team_spirit * team_avg

        Args:
            players: All players in the game
            state: Current game state
            previous_actions: Previous actions for all players
            team: Team number (0 = blue, 1 = orange)

        Returns:
            Team-blended reward
        """
        # Get team members
        team_players = [(p, a) for p, a in zip(players, previous_actions) if p.team_num == team]

        if not team_players:
            return 0.0

        # Calculate individual rewards
        individual_rewards = []
        for player, action in team_players:
            r = self.get_reward(player, state, action)
            individual_rewards.append(r)

        # Calculate team average
        team_avg = np.mean(individual_rewards)

        # Blend with team spirit
        blended_rewards = []
        for individual in individual_rewards:
            blended = (1 - self._team_spirit) * individual + self._team_spirit * team_avg
            blended_rewards.append(blended)

        return blended_rewards

    def _update_statistics(self, rewards: List[float]) -> None:
        """Update running mean and variance for normalization.

        Uses Welford's online algorithm.
        """
        self._reward_count += 1

        for i, r in enumerate(rewards):
            delta = r - self._reward_means[i]
            self._reward_means[i] += delta / self._reward_count
            delta2 = r - self._reward_means[i]
            self._reward_vars[i] += delta * delta2


@registry.register("reward", "scheduled")
class ScheduledReward(CombinedReward):
    """Combined reward with curriculum-based scheduling.

    Enables/disables rewards based on training progress.
    """

    def __init__(
        self,
        rewards: List[Tuple[BaseReward, float]],
        schedule: Dict[str, Tuple[int, int]],  # reward_name -> (start_step, end_step)
        config: Optional[RewardConfig] = None,
    ):
        """Initialize scheduled reward.

        Args:
            rewards: List of (reward_function, max_weight) tuples
            schedule: Dict mapping reward names to (start_step, end_step)
            config: Reward configuration
        """
        super().__init__(rewards, config)
        self.schedule = schedule
        self._reward_names = [r.__class__.__name__ for r, _ in rewards]

    def set_global_step(self, step: int) -> None:
        """Update weights based on schedule."""
        self._global_step = step

        for i, (reward, max_weight) in enumerate(self.rewards):
            name = self._reward_names[i]
            reward.set_global_step(step)

            if name in self.schedule:
                start_step, end_step = self.schedule[name]

                if step < start_step:
                    # Not yet enabled
                    weight = 0.0
                elif step >= end_step:
                    # Fully enabled
                    weight = max_weight
                else:
                    # Ramping up
                    progress = (step - start_step) / (end_step - start_step)
                    weight = max_weight * progress
            else:
                # Always enabled
                weight = max_weight

            self.rewards[i] = (reward, weight)
