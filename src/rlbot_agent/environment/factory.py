"""Environment factory for creating RLGym environments."""

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from ..core.config import EnvironmentConfig, ObservationConfig, ActionConfig, RewardConfig
from ..core.registry import registry
from .action_parsers import MultiDiscreteActionParser
from .conditions import GoalCondition, NoTouchCondition, TimeoutCondition
from .obs_builders import AdvancedObsBuilder
from .rewards import (
    CombinedReward,
    DemoReward,
    GoalReward,
    SpeedTowardBall,
    TouchVelocity,
    VelocityBallToGoal,
    SaveBoost,
    AerialHeight,
    TeamSpacing,
)
from .state_mutators import KickoffMutator, RandomStateMutator, ReplayStateMutator


def create_environment(
    env_config: Optional[EnvironmentConfig] = None,
    obs_config: Optional[ObservationConfig] = None,
    action_config: Optional[ActionConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    render: bool = False,
    self_play: bool = True,
) -> Any:
    """Create an RLGym environment with the specified configuration.

    Args:
        env_config: Environment configuration
        obs_config: Observation configuration
        action_config: Action configuration
        reward_config: Reward configuration
        render: Whether to render the environment
        self_play: Whether to use self-play (all agents use same policy)

    Returns:
        RLGym environment instance
    """
    env_config = env_config or EnvironmentConfig()
    obs_config = obs_config or ObservationConfig()
    action_config = action_config or ActionConfig()
    reward_config = reward_config or RewardConfig()

    # Create observation builder
    obs_builder = AdvancedObsBuilder(
        config=obs_config,
        max_players=env_config.max_players,
    )

    # Create action parser
    action_parser = MultiDiscreteActionParser(config=action_config)

    # Create reward function
    reward_fn = _create_reward_function(reward_config)

    # Create terminal conditions
    terminal_conditions = _create_terminal_conditions(env_config)

    # Create state mutator
    state_mutator = _create_state_mutator(env_config)

    # Build environment configuration dict for RLGym
    rlgym_config = {
        "tick_skip": env_config.tick_skip,
        "spawn_opponents": env_config.spawn_opponents,
        "team_size": env_config.team_size,
        "gravity": env_config.gravity,
        "boost_consumption": env_config.boost_consumption,
        "obs_builder": obs_builder,
        "action_parser": action_parser,
        "reward_fn": reward_fn,
        "terminal_conditions": terminal_conditions,
        "state_setter": state_mutator,
    }

    # Import RLGym and create environment
    try:
        import rlgym
        env = rlgym.make(**rlgym_config)
    except ImportError:
        # Fallback to a mock environment for testing
        env = MockEnvironment(
            obs_builder=obs_builder,
            action_parser=action_parser,
            reward_fn=reward_fn,
        )

    return env


def _create_reward_function(config: RewardConfig) -> CombinedReward:
    """Create combined reward function from config.

    Args:
        config: Reward configuration

    Returns:
        Combined reward function
    """
    rewards = [
        (TouchVelocity(), config.touch_velocity),
        (VelocityBallToGoal(), config.velocity_ball_to_goal),
        (SpeedTowardBall(), config.speed_toward_ball),
        (GoalReward(), config.goal),
        (SaveBoost(), config.save_boost),
        (DemoReward(), config.demo),
        (AerialHeight(), config.aerial_height),
        (TeamSpacing(), config.team_spacing_penalty),
    ]

    return CombinedReward(rewards=rewards, config=config)


def _create_terminal_conditions(config: EnvironmentConfig) -> List[Any]:
    """Create terminal conditions from config.

    Args:
        config: Environment configuration

    Returns:
        List of terminal conditions
    """
    conditions = []

    for cond_name in config.terminal_conditions:
        if cond_name == "goal":
            conditions.append(GoalCondition())
        elif cond_name == "timeout":
            conditions.append(TimeoutCondition(
                timeout_seconds=config.timeout_seconds,
                tick_skip=config.tick_skip,
            ))
        elif cond_name == "no_touch":
            conditions.append(NoTouchCondition(
                timeout_seconds=config.no_touch_timeout_seconds,
                tick_skip=config.tick_skip,
            ))

    return conditions


def _create_state_mutator(config: EnvironmentConfig) -> Any:
    """Create combined state mutator from config.

    Args:
        config: Environment configuration

    Returns:
        State mutator that randomly selects between kickoff, random, and replay states
    """
    kickoff = KickoffMutator()
    random_state = RandomStateMutator()
    replay_state = ReplayStateMutator()

    probs = [config.kickoff_prob, config.random_prob, config.replay_prob]
    mutators = [kickoff, random_state, replay_state]

    return CombinedStateMutator(mutators, probs)


class CombinedStateMutator:
    """State mutator that randomly selects between multiple mutators."""

    def __init__(self, mutators: List[Any], probabilities: List[float]):
        """Initialize combined state mutator.

        Args:
            mutators: List of state mutators
            probabilities: Selection probabilities for each mutator
        """
        self.mutators = mutators
        # Normalize probabilities
        total = sum(probabilities)
        self.probabilities = [p / total for p in probabilities]

    def __call__(self, state: Any) -> Any:
        """Apply a randomly selected mutator.

        Args:
            state: Game state to modify

        Returns:
            Modified game state
        """
        # Count players by team
        num_blue = sum(1 for p in state.players if p.team_num == 0)
        num_orange = sum(1 for p in state.players if p.team_num == 1)

        # Select mutator
        idx = np.random.choice(len(self.mutators), p=self.probabilities)
        mutator = self.mutators[idx]

        return mutator.apply(state, num_blue, num_orange)


class MockEnvironment:
    """Mock environment for testing without RLGym."""

    def __init__(
        self,
        obs_builder: Any,
        action_parser: Any,
        reward_fn: Any,
    ):
        """Initialize mock environment."""
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.reward_fn = reward_fn

        self.observation_space_size = obs_builder.get_obs_space_size()
        self.action_space_size = action_parser.get_action_space_size()

    def reset(self) -> np.ndarray:
        """Reset environment."""
        return np.zeros(self.observation_space_size, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        obs = np.zeros(self.observation_space_size, dtype=np.float32)
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def close(self) -> None:
        """Close environment."""
        pass
