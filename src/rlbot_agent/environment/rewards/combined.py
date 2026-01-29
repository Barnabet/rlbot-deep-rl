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

        # Annealing state
        self._initial_weights = [w for _, w in rewards]
        self._target_weights: Optional[List[float]] = None

        # Running statistics for normalization
        self._reward_means = [0.0] * len(rewards)
        self._reward_vars = [1.0] * len(rewards)
        self._reward_count = 0

    def set_target_weights(self, weights: List[float]) -> None:
        """Set target weights for annealing.

        Args:
            weights: Target weights to anneal toward
        """
        assert len(weights) == len(self.rewards)
        self._target_weights = weights

    def set_global_step(self, step: int) -> None:
        """Update global step and anneal weights accordingly."""
        super().set_global_step(step)

        # Update child rewards
        for reward, _ in self.rewards:
            reward.set_global_step(step)

        # Anneal weights if targets are set
        if self._target_weights is not None:
            progress = min(1.0, step / self.config.anneal_steps)
            for i, ((reward, _), initial, target) in enumerate(
                zip(self.rewards, self._initial_weights, self._target_weights)
            ):
                current_weight = initial + progress * (target - initial)
                self.rewards[i] = (reward, current_weight)

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

        for reward, weight in self.rewards:
            r = reward.get_reward(player, state, previous_action)
            individual_rewards.append(r)
            total_reward += weight * r

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

        Args:
            player: Player data
            state: Current game state
            previous_action: Previous action taken

        Returns:
            Dictionary mapping reward names to values
        """
        result = {}
        for (reward, weight), r in zip(
            self.rewards,
            [reward.get_reward(player, state, previous_action) for reward, _ in self.rewards]
        ):
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
