"""Self-play environment wrapper for training against historical checkpoints."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..training.policy_pool import PolicyPool


class SelfPlayEnvWrapper:
    """Wraps RLGym v2 environment for self-play with historical opponents.

    Routes actions to different policies:
    - Primary team agents: receive actions from the training policy
    - Opponent team agents: receive actions from historical checkpoints

    Features:
    - Selects new opponent each episode
    - Tracks win/loss against opponents
    - Supports team-based play (1v1, 2v2, 3v3)
    """

    def __init__(
        self,
        env: Any,
        policy_pool: PolicyPool,
        primary_team: str = 'blue',
        device: str = 'cpu',
    ):
        """Initialize self-play environment wrapper.

        Args:
            env: RLGym v2 multi-agent environment (unwrapped)
            policy_pool: Pool of historical checkpoints
            primary_team: Team controlled by training policy ('blue' or 'orange')
            device: PyTorch device for opponent inference
        """
        self.env = env
        self.policy_pool = policy_pool
        self.primary_team = primary_team
        self.device = torch.device(device)

        # Agent tracking
        self._all_agents: List[str] = []
        self._primary_agents: List[str] = []
        self._opponent_agents: List[str] = []
        self._primary_agent: Optional[str] = None

        # Observation tracking for opponent inference
        self._last_obs_dict: Dict[str, np.ndarray] = {}

        # Episode tracking for win/loss
        self._episode_goals_for = 0
        self._episode_goals_against = 0

    def reset(self) -> np.ndarray:
        """Reset environment and select new opponent.

        Returns:
            Observation for primary agent
        """
        # Select new opponent from pool
        self.policy_pool.select_opponent()

        # Reset environment
        obs_dict = self.env.reset()
        self._all_agents = list(obs_dict.keys())
        self._last_obs_dict = dict(obs_dict)

        # Categorize agents by team
        self._primary_agents = []
        self._opponent_agents = []

        for agent in self._all_agents:
            agent_str = str(agent)
            if agent_str.startswith(self.primary_team):
                self._primary_agents.append(agent)
            else:
                self._opponent_agents.append(agent)

        # Primary agent for single-agent interface
        self._primary_agent = self._primary_agents[0] if self._primary_agents else None

        # Reset episode tracking
        self._episode_goals_for = 0
        self._episode_goals_against = 0

        if self._primary_agent is not None:
            return obs_dict[self._primary_agent]
        return np.zeros(113, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with action for primary agents, opponents use historical policy.

        Args:
            action: Action for primary agents (same action for all primary team members)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._primary_agent is None:
            return self.reset(), 0.0, False, False, {}

        # Build action dict
        actions = {}

        # Primary agents get the given action
        for agent in self._primary_agents:
            actions[agent] = action

        # Opponent agents get actions from historical policy
        opponent_engine = self.policy_pool.get_current_opponent()
        for agent in self._opponent_agents:
            if opponent_engine is not None and agent in self._last_obs_dict:
                obs = self._last_obs_dict[agent]
                opponent_action, _ = opponent_engine.get_action(obs, deterministic=True)
                actions[agent] = opponent_action
            else:
                # Fallback: same action as primary (like SingleAgentWrapper)
                actions[agent] = action

        # Step environment
        obs_dict, reward_dict, terminated_dict, truncated_dict = self.env.step(actions)

        # Store observations for next step
        self._last_obs_dict = dict(obs_dict)

        # Get primary agent's results
        obs = obs_dict.get(self._primary_agent, np.zeros(113, dtype=np.float32))
        reward = reward_dict.get(self._primary_agent, 0.0)
        terminated = any(terminated_dict.values())
        truncated = any(truncated_dict.values())

        # Build info dict
        info = self._build_info()

        # Track goals for win/loss tracking
        if 'goal_scored' in info:
            self._episode_goals_for += info['goal_scored']
        if 'goal_conceded' in info:
            self._episode_goals_against += info['goal_conceded']

        # Update opponent stats on episode end
        if terminated or truncated:
            won = self._episode_goals_for > self._episode_goals_against
            self.policy_pool.update_result(won)
            info['episode_won'] = won
            info['episode_goals_for'] = self._episode_goals_for
            info['episode_goals_against'] = self._episode_goals_against

        return obs, reward, terminated, truncated, info

    def _build_info(self) -> Dict:
        """Build info dict with state/player data.

        Returns:
            Info dictionary
        """
        info = {}
        try:
            state = self.env.state
            car = state.cars.get(self._primary_agent)
            if car is not None:
                # Import adapters from factory (avoid circular import)
                from .factory import PlayerAdapter, StateAdapter

                player = PlayerAdapter(car, self._primary_agent)
                adapted_state = StateAdapter(state)

                info['state'] = adapted_state
                info['player'] = player
        except Exception:
            pass

        return info

    def get_opponent_stats(self) -> Dict[str, float]:
        """Get opponent statistics for logging.

        Returns:
            Dictionary of opponent statistics
        """
        return self.policy_pool.get_stats()

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


class SelfPlaySingleAgentWrapper:
    """Wraps SingleAgentWrapper to add self-play against historical opponents.

    This is a simpler alternative that wraps the existing SingleAgentWrapper
    and intercepts actions for opponent agents.
    """

    def __init__(
        self,
        single_agent_env: Any,
        policy_pool: PolicyPool,
        device: str = 'cpu',
    ):
        """Initialize self-play wrapper.

        Args:
            single_agent_env: SingleAgentWrapper instance
            policy_pool: Pool of historical checkpoints
            device: PyTorch device for opponent inference
        """
        self.wrapped_env = single_agent_env
        self.policy_pool = policy_pool
        self.device = torch.device(device)

        # We need access to the underlying multi-agent env
        self.multi_agent_env = single_agent_env.env

        # Episode tracking
        self._episode_goals_for = 0
        self._episode_goals_against = 0

    def reset(self) -> np.ndarray:
        """Reset and select new opponent.

        Returns:
            Observation for primary agent
        """
        # Select new opponent
        self.policy_pool.select_opponent()

        # Reset episode tracking
        self._episode_goals_for = 0
        self._episode_goals_against = 0

        return self.wrapped_env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with self-play.

        Note: This version still sends same action to all agents but provides
        the framework for future enhancement where we could intercept and
        modify opponent actions.

        Args:
            action: Action for primary agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.wrapped_env.step(action)

        # Track goals
        if 'goal_scored' in info:
            self._episode_goals_for += info['goal_scored']
        if 'goal_conceded' in info:
            self._episode_goals_against += info['goal_conceded']

        # Update opponent stats on episode end
        if terminated or truncated:
            won = self._episode_goals_for > self._episode_goals_against
            self.policy_pool.update_result(won)
            info['episode_won'] = won

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the environment."""
        self.wrapped_env.close()
