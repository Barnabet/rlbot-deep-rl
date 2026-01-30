"""Environment factory for creating RLGym v2 environments."""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import EnvironmentConfig, ObservationConfig, ActionConfig, RewardConfig
from .obs_builders import AdvancedObsBuilder
from .action_parsers import MultiDiscreteActionParser
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
    OnGround,
)


# Field dimensions
FIELD_X = 4096  # Half-width
FIELD_Y = 5120  # Half-length (to goal line)
FIELD_Z = 2044  # Ceiling height
BALL_RADIUS = 92.75
CAR_HEIGHT = 17  # Resting height on ground


class RandomStateMutator:
    """Randomizes ball and car positions for diverse training scenarios.

    Ball starts stationary or slow so rewards reflect agent actions, not luck.
    """

    def apply(self, state, shared_info: Dict[str, Any]) -> None:
        """Apply random state mutation."""
        # Random ball position (keep away from walls/ceiling)
        ball_x = random.uniform(-FIELD_X * 0.8, FIELD_X * 0.8)
        ball_y = random.uniform(-FIELD_Y * 0.8, FIELD_Y * 0.8)
        ball_z = random.uniform(BALL_RADIUS, 300)  # Mostly on/near ground

        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)

        # Ball starts stationary or very slow - rewards should come from agent actions
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        # Random car positions - on their respective side
        ball_pos = state.ball.position

        for car in state.cars.values():
            car_x = random.uniform(-FIELD_X * 0.8, FIELD_X * 0.8)

            if car.team_num == 0:  # Blue team - spawns on blue half (negative Y)
                car_y = random.uniform(-FIELD_Y * 0.9, -500)
            else:  # Orange team - spawns on orange half (positive Y)
                car_y = random.uniform(500, FIELD_Y * 0.9)

            car_z = CAR_HEIGHT  # On ground

            car.physics.position = np.array([car_x, car_y, car_z], dtype=np.float32)

            # Face roughly toward the ball (± 45 degrees randomness)
            to_ball = ball_pos[:2] - np.array([car_x, car_y])
            yaw = np.arctan2(to_ball[1], to_ball[0])
            yaw += random.uniform(-np.pi/4, np.pi/4)  # ± 45 degrees
            car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)

            # Car starts stationary - cleaner credit assignment
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)

            # Random boost
            car.boost_amount = random.uniform(20, 100)


class WeightedStateMutator:
    """Randomly selects between multiple mutators based on weights."""

    def __init__(self, mutators: List[Tuple[Any, float]]):
        """Initialize with list of (mutator, weight) tuples."""
        self.mutators = [m for m, w in mutators]
        self.weights = [w for m, w in mutators]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]  # Normalize

    def apply(self, state, shared_info: Dict[str, Any]) -> None:
        """Apply a randomly selected mutator."""
        mutator = random.choices(self.mutators, weights=self.weights, k=1)[0]
        mutator.apply(state, shared_info)


def create_environment(
    env_config: Optional[EnvironmentConfig] = None,
    obs_config: Optional[ObservationConfig] = None,
    action_config: Optional[ActionConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    render: bool = False,
    self_play: bool = True,
) -> Any:
    """Create an RLGym v2 environment with the specified configuration.

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

    try:
        from rlgym.api import RLGym
        from rlgym.rocket_league.sim import RocketSimEngine
        from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
        from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator, FixedTeamSizeMutator

        # Create RocketSim engine
        engine = RocketSimEngine()

        # Create observation builder (wrapped for v2 API)
        obs_builder = ObsBuilderV2Wrapper(
            AdvancedObsBuilder(config=obs_config, max_players=env_config.max_players)
        )

        # Create action parser (wrapped for v2 API)
        action_parser = ActionParserV2Wrapper(
            MultiDiscreteActionParser(config=action_config)
        )

        # Create reward function (wrapped for v2 API)
        reward_fn = RewardFunctionV2Wrapper(
            _create_reward_function(reward_config)
        )

        # Create termination condition (goal scored)
        termination_cond = GoalCondition()

        # Create truncation conditions
        truncation_cond = AnyCondition([
            TimeoutCondition(timeout_seconds=env_config.timeout_seconds),
            NoTouchTimeoutCondition(timeout_seconds=env_config.no_touch_timeout_seconds),
        ])

        # Create state mutator - must include FixedTeamSizeMutator to create cars!
        blue_size = env_config.team_size
        orange_size = env_config.team_size if env_config.spawn_opponents else 0

        # Weighted mutator: 30% kickoffs, 70% random positions
        position_mutator = WeightedStateMutator([
            (KickoffMutator(), 0.3),
            (RandomStateMutator(), 0.7),
        ])

        state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size),
            position_mutator,
        )

        # Create environment
        env = RLGym(
            state_mutator=state_mutator,
            obs_builder=obs_builder,
            action_parser=action_parser,
            reward_fn=reward_fn,
            termination_cond=termination_cond,
            truncation_cond=truncation_cond,
            transition_engine=engine,
        )

        # Wrap for multi-agent training
        wrapped_env = MultiAgentWrapper(env, reward_wrapper=reward_fn)

        # Wrap for fixed episode length (goals reset state but don't end episode)
        if env_config.episode_steps and env_config.episode_steps > 0:
            wrapped_env = FixedEpisodeLengthWrapper(wrapped_env, episode_steps=env_config.episode_steps)

        return wrapped_env

    except ImportError as e:
        print(f"RLGym v2 import failed: {e}")
        print("Falling back to mock environment")
        # Fallback to mock environment for testing
        return MockEnvironment(
            obs_dim=obs_config.self_car_dim + obs_config.ball_dim + 5 * obs_config.other_car_dim,
            n_actions=action_config.n_actions,
        )


def _create_reward_function(config: RewardConfig) -> CombinedReward:
    """Create combined reward function from config."""
    # Use config.get_weight() to get initial weights (step=0)
    # CombinedReward.set_global_step() will handle interpolation during training
    rewards = [
        (TouchVelocity(), config.get_weight('touch_velocity', 0)),
        (VelocityBallToGoal(), config.get_weight('velocity_ball_to_goal', 0)),
        (SpeedTowardBall(), config.get_weight('speed_toward_ball', 0)),
        (GoalReward(), config.get_weight('goal', 0)),
        (SaveBoost(), config.get_weight('save_boost', 0)),
        (DemoReward(), config.get_weight('demo', 0)),
        (AerialHeight(), config.get_weight('aerial_height', 0)),
        (TeamSpacing(), config.get_weight('team_spacing_penalty', 0)),
        (OnGround(), config.get_weight('on_ground', 0)),
    ]
    return CombinedReward(rewards=rewards, config=config)


class ObsBuilderV2Wrapper:
    """Wraps our observation builder to RLGym v2 API."""

    def __init__(self, obs_builder):
        self.obs_builder = obs_builder

    def get_obs_space(self, agent):
        """Return observation space for agent."""
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_builder.get_obs_space_size(),),
            dtype=np.float32
        )

    def reset(self, agents, initial_state, shared_info):
        """Reset the observation builder."""
        adapted_state = StateAdapter(initial_state)
        self.obs_builder.reset(adapted_state)

    def build_obs(self, agents, state, shared_info):
        """Build observations for all agents."""
        obs_dict = {}
        adapted_state = StateAdapter(state)

        for agent in agents:
            car = state.cars.get(agent)
            if car is not None:
                player = PlayerAdapter(car, agent)
                obs = self.obs_builder.build_obs(
                    player=player,
                    state=adapted_state,
                    previous_action=np.zeros(8, dtype=np.float32),
                )
                obs_dict[agent] = obs
        return obs_dict


class ActionParserV2Wrapper:
    """Wraps our action parser to RLGym v2 API."""

    def __init__(self, action_parser):
        self.action_parser = action_parser

    def get_action_space(self, agent):
        """Return action space for agent."""
        import gymnasium as gym
        return gym.spaces.Discrete(self.action_parser.get_action_space_size())

    def reset(self, agents, initial_state, shared_info):
        """Reset the action parser."""
        pass

    def parse_actions(self, actions, state, shared_info):
        """Parse actions for all agents."""
        # Convert dict of actions to array for batch processing
        agents_list = list(actions.keys())

        # Handle both flat actions (scalar) and multi-discrete ([8] array)
        first_action = actions[agents_list[0]]
        if np.asarray(first_action).ndim == 0 or (hasattr(first_action, '__len__') and len(first_action) == 1):
            # Flat action (scalar or single-element array)
            action_array = np.array([int(actions[a]) for a in agents_list], dtype=np.int64)
        else:
            # Multi-discrete action ([8] array per agent)
            action_array = np.array([actions[a] for a in agents_list], dtype=np.int64)

        # Parse all actions at once - returns shape (n_agents, 8)
        controls_array = self.action_parser.parse_actions(action_array, state)

        # Convert back to dict with shape (1, 8) per agent
        # RocketSim expects (N, 8) where N is number of physics steps
        parsed = {}
        for i, agent in enumerate(agents_list):
            # Reshape from (8,) to (1, 8) - single step of controls
            parsed[agent] = controls_array[i].reshape(1, 8)
        return parsed


class PlayerAdapter:
    """Adapts RLGym v2 Car to our expected player interface."""

    def __init__(self, car, agent_id, prev_ball_touches=0):
        self._car = car
        self._agent_id = agent_id
        self._prev_ball_touches = prev_ball_touches

    @property
    def car_id(self):
        """Return agent ID as car_id for compatibility."""
        return self._agent_id

    @property
    def car_data(self):
        """Return physics as car_data for compatibility."""
        return self._car.physics

    @property
    def team_num(self):
        return self._car.team_num

    @property
    def ball_touched(self):
        """Return True if ball was touched this frame."""
        return self._car.ball_touches > self._prev_ball_touches

    @property
    def ball_touches(self):
        return self._car.ball_touches

    @property
    def on_ground(self):
        return self._car.on_ground

    @property
    def has_flip(self):
        return self._car.has_flip

    @property
    def boost_amount(self):
        return self._car.boost_amount

    @property
    def is_demoed(self):
        return self._car.is_demoed

    @property
    def demo_respawn_timer(self):
        return self._car.demo_respawn_timer

    @property
    def is_supersonic(self):
        return self._car.is_supersonic

    @property
    def has_jumped(self):
        return self._car.has_jumped

    @property
    def has_double_jumped(self):
        return self._car.has_double_jumped


class StateAdapter:
    """Adapts RLGym v2 GameState to our expected state interface."""

    def __init__(self, state, prev_ball_touches=None):
        self._state = state
        self._prev_ball_touches = prev_ball_touches or {}

    @property
    def ball(self):
        return self._state.ball

    @property
    def players(self):
        """Return cars as a list of PlayerAdapters for compatibility."""
        adapted_players = []
        for agent_id, car in self._state.cars.items():
            prev_touches = self._prev_ball_touches.get(agent_id, 0)
            player = PlayerAdapter(car, agent_id, prev_touches)
            adapted_players.append(player)
        return adapted_players

    @property
    def cars(self):
        return self._state.cars

    @property
    def goal_scored(self):
        """Whether a goal was scored this step."""
        return getattr(self._state, 'goal_scored', False)

    @property
    def scoring_team(self):
        """Team that scored (0=blue, 1=orange, None if no goal)."""
        return getattr(self._state, 'scoring_team', None)

    @property
    def blue_score(self):
        """Blue team score from RLGym state."""
        return getattr(self._state, 'blue_score', 0)

    @property
    def orange_score(self):
        """Orange team score from RLGym state."""
        return getattr(self._state, 'orange_score', 0)


class RewardFunctionV2Wrapper:
    """Wraps our reward function to RLGym v2 API."""

    def __init__(self, reward_fn):
        self.reward_fn = reward_fn
        self._prev_state = None
        self._prev_ball_touches = {}  # Track ball touches per agent
        self._last_reward_breakdown = {}  # Per-agent reward breakdown

    def reset(self, agents, initial_state, shared_info):
        """Reset the reward function."""
        self._prev_state = initial_state
        self._prev_ball_touches = {agent: 0 for agent in agents}
        self._last_reward_breakdown = {}
        # Adapt state for our reward function
        adapted_state = StateAdapter(initial_state)
        if hasattr(self.reward_fn, 'reset'):
            self.reward_fn.reset(adapted_state)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        """Get rewards for all agents."""
        rewards = {}
        adapted_state = StateAdapter(state)

        for agent in agents:
            car = state.cars.get(agent)
            if car is not None:
                prev_touches = self._prev_ball_touches.get(agent, 0)
                player = PlayerAdapter(car, agent, prev_touches)

                reward = self.reward_fn.get_reward(
                    player=player,
                    state=adapted_state,
                    previous_action=np.zeros(8, dtype=np.float32),
                )
                rewards[agent] = float(reward)

                # Get individual reward breakdown for debugging
                if hasattr(self.reward_fn, 'get_individual_rewards'):
                    self._last_reward_breakdown[agent] = self.reward_fn.get_individual_rewards(
                        player=player,
                        state=adapted_state,
                        previous_action=np.zeros(8, dtype=np.float32),
                    )

                # Update ball touch tracking
                self._prev_ball_touches[agent] = car.ball_touches
            else:
                rewards[agent] = 0.0

        self._prev_state = state
        return rewards

    def get_reward_breakdown(self, agent):
        """Get the last reward breakdown for an agent."""
        return self._last_reward_breakdown.get(agent, {})


class AnyCondition:
    """Truncation condition that triggers if any sub-condition is met."""

    def __init__(self, conditions):
        self.conditions = conditions

    def reset(self, agents, initial_state, shared_info):
        for cond in self.conditions:
            if hasattr(cond, 'reset'):
                cond.reset(agents, initial_state, shared_info)

    def is_done(self, agents, state, shared_info):
        """Check if any condition is done (RLGym v2 API)."""
        result = {agent: False for agent in agents}
        for cond in self.conditions:
            cond_result = cond.is_done(agents, state, shared_info)
            for agent in agents:
                if cond_result.get(agent, False):
                    result[agent] = True
        return result


class MultiAgentWrapper:
    """Wraps RLGym v2 multi-agent env for multi-agent training.

    Each player gets their own observation and takes their own action.
    All players' experiences are collected for training (true self-play).

    Returns arrays of shape [n_agents, ...] for obs, rewards, dones.
    """

    def __init__(self, env, reward_wrapper=None):
        self.env = env
        self.reward_wrapper = reward_wrapper
        self._all_agents = []
        self._prev_ball_touches = {}
        self.n_agents = 0

    def reset(self):
        """Reset and return observations for all agents.

        Returns:
            observations: [n_agents, obs_dim]
        """
        obs_dict = self.env.reset()
        self._all_agents = list(obs_dict.keys())
        self._prev_ball_touches = {agent: 0 for agent in self._all_agents}
        self.n_agents = len(self._all_agents)

        if self._all_agents:
            return np.stack([obs_dict[agent] for agent in self._all_agents])
        return np.zeros((1, 113), dtype=np.float32)

    def step(self, actions):
        """Step with per-agent actions, return per-agent values.

        Args:
            actions: [n_agents, action_dim] or [n_agents] array of actions

        Returns:
            observations: [n_agents, obs_dim]
            rewards: [n_agents]
            dones: [n_agents] (all same since episode ends for everyone)
            infos: list of info dicts per agent
        """
        if not self._all_agents:
            obs = self.reset()
            return obs, np.zeros(self.n_agents), np.zeros(self.n_agents), [{} for _ in range(self.n_agents)]

        # Build action dict for each agent
        action_dict = {agent: actions[i] for i, agent in enumerate(self._all_agents)}

        # RLGym v2 step
        obs_dict, reward_dict, terminated_dict, truncated_dict = self.env.step(action_dict)

        # Episode ends for all agents together
        terminated = any(terminated_dict.values())
        truncated = any(truncated_dict.values())
        done = terminated or truncated

        # Collect per-agent data
        obs_list = []
        reward_list = []
        info_list = []

        try:
            state = self.env.state
            adapted_state = StateAdapter(state, self._prev_ball_touches)
        except:
            state = None
            adapted_state = None

        for agent in self._all_agents:
            obs = obs_dict.get(agent, np.zeros(113, dtype=np.float32))
            reward = reward_dict.get(agent, 0.0)

            obs_list.append(obs)
            reward_list.append(float(reward))

            # Build info dict with state/player data for stats tracking
            info = {}
            try:
                if state is not None:
                    car = state.cars.get(agent)
                    if car is not None:
                        prev_touches = self._prev_ball_touches.get(agent, 0)
                        player = PlayerAdapter(car, agent, prev_touches)

                        info['state'] = adapted_state
                        info['player'] = player

                        # Get per-reward breakdown
                        if self.reward_wrapper is not None:
                            breakdown = self.reward_wrapper.get_reward_breakdown(agent)
                            if breakdown:
                                info['reward_breakdown'] = breakdown

                        # Update ball touches tracking
                        self._prev_ball_touches[agent] = car.ball_touches
            except Exception:
                pass

            info_list.append(info)

        # All agents share the same done flag (episode ends together)
        dones = np.full(self.n_agents, float(done))

        return np.stack(obs_list), np.array(reward_list), dones, info_list

    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


# Keep SingleAgentWrapper as alias for backwards compatibility
SingleAgentWrapper = MultiAgentWrapper


class FixedEpisodeLengthWrapper:
    """Wrapper for fixed-length episodes with goal resets.

    Goals don't end the episode - they just reset state and continue.
    Episode ends only when step count reaches episode_steps.

    Benefits:
    - Consistent episode lengths for stable metrics
    - No variable-length episode tracking complexity
    - More goals per episode = more learning signal
    """

    def __init__(self, env, episode_steps: int = 300):
        """Initialize wrapper.

        Args:
            env: Underlying MultiAgentWrapper environment
            episode_steps: Fixed number of steps per episode
        """
        self.env = env
        self.episode_steps = episode_steps
        self.current_step = 0
        self.goals_this_episode = 0
        self.n_agents = getattr(env, 'n_agents', 2)

    def reset(self):
        """Reset environment and step counter."""
        self.current_step = 0
        self.goals_this_episode = 0
        obs = self.env.reset()
        self.n_agents = getattr(self.env, 'n_agents', len(obs))
        return obs

    def step(self, actions):
        """Step with goal reset handling.

        If underlying env returns done (goal), reset internally but continue.
        Only return done=True when we reach episode_steps.
        """
        obs, rewards, dones, infos = self.env.step(actions)
        self.current_step += 1

        # Check if goal was scored (underlying env returned done)
        goal_scored = dones[0] > 0.5 if len(dones) > 0 else False

        if goal_scored:
            self.goals_this_episode += 1
            # Add goal info to all agents
            for info in infos:
                info['goal_scored_this_step'] = True
                info['goals_this_episode'] = self.goals_this_episode

            # Check if we've reached episode length
            if self.current_step >= self.episode_steps:
                # Episode truly ends
                for info in infos:
                    info['episode_goals'] = self.goals_this_episode
                return obs, rewards, dones, infos
            else:
                # Goal scored but episode continues - reset state internally
                obs = self.env.reset()
                # Return the rewards from the goal but done=False
                dones = np.zeros(self.n_agents)
                return obs, rewards, dones, infos

        # No goal - check if we've reached episode length
        if self.current_step >= self.episode_steps:
            dones = np.ones(self.n_agents)
            for info in infos:
                info['episode_goals'] = self.goals_this_episode
        else:
            dones = np.zeros(self.n_agents)

        return obs, rewards, dones, infos

    def close(self):
        """Close underlying environment."""
        self.env.close()

    @property
    def n_agents(self):
        return getattr(self.env, 'n_agents', 2)

    @n_agents.setter
    def n_agents(self, value):
        pass  # Read-only, derived from env


class MockEnvironment:
    """Mock environment for testing without RLGym."""

    def __init__(self, obs_dim: int = 113, n_actions: int = 1944):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self._step_count = 0

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self._step_count = 0
        return np.random.randn(self.obs_dim).astype(np.float32) * 0.1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        self._step_count += 1
        obs = np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        reward = np.random.randn() * 0.1
        done = self._step_count >= 1000  # Episode ends after 1000 steps
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def close(self) -> None:
        """Close environment."""
        pass
