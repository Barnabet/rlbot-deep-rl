"""Dataset classes for replay data."""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..environment.obs_builders import AdvancedObsBuilder
from ..environment.action_parsers import MultiDiscreteActionParser
from ..core.config import ObservationConfig, ActionConfig


class ReplayDataset(Dataset):
    """PyTorch dataset for replay data.

    Provides (observation, action) pairs for behavioral cloning.
    """

    def __init__(
        self,
        states: List[dict],
        actions: List[dict],
        obs_config: Optional[ObservationConfig] = None,
        action_config: Optional[ActionConfig] = None,
    ):
        """Initialize replay dataset.

        Args:
            states: List of game states from parsed replays
            actions: List of corresponding actions
            obs_config: Observation configuration
            action_config: Action configuration
        """
        self.states = states
        self.actions = actions

        self.obs_config = obs_config or ObservationConfig()
        self.action_config = action_config or ActionConfig()

        # Create observation builder for converting states
        self.obs_builder = AdvancedObsBuilder(config=self.obs_config)

        # Create action parser for discretizing actions
        self.action_parser = MultiDiscreteActionParser(config=self.action_config)

        # Pre-process data
        self._observations: Optional[np.ndarray] = None
        self._action_indices: Optional[np.ndarray] = None

    def process(self) -> None:
        """Pre-process all states and actions into arrays."""
        observations = []
        action_indices = []

        for state, action_list in zip(self.states, self.actions):
            if action_list is None:
                continue

            # For each player in the state
            for player_action in action_list:
                # Convert state to observation (simplified - would need full impl)
                obs = self._state_to_observation(state, player_action['team'])
                observations.append(obs)

                # Convert action to discrete index
                action_idx = self._action_to_index(player_action)
                action_indices.append(action_idx)

        self._observations = np.array(observations, dtype=np.float32)
        self._action_indices = np.array(action_indices, dtype=np.int64)

        print(f"Processed {len(self._observations)} samples")

    def _state_to_observation(
        self,
        state: dict,
        team: int,
    ) -> np.ndarray:
        """Convert game state to observation array.

        Args:
            state: Game state dictionary
            team: Team to build observation for

        Returns:
            Observation array
        """
        obs_dim = self.obs_builder.get_obs_space_size()
        obs = np.zeros(obs_dim, dtype=np.float32)

        # Extract ball state
        ball = state['ball']
        ball_pos = np.array(ball['position']) / np.array([4096, 5120, 2048])
        ball_vel = np.array(ball['velocity']) / 2300.0

        # Find self player
        players = state['players']
        self_player = None
        for p in players:
            if p['team'] == team:
                self_player = p
                break

        if self_player is None:
            return obs

        # Build self observation (simplified)
        pos = np.array(self_player['position']) / np.array([4096, 5120, 2048])
        vel = np.array(self_player['velocity']) / 2300.0

        # Fill observation (simplified version)
        idx = 0
        obs[idx:idx+3] = pos
        idx += 3
        obs[idx:idx+3] = vel
        idx += 6  # Skip angular vel
        idx += 6  # Skip rotation sin/cos
        obs[idx] = self_player['boost']
        idx += 4  # Skip other state

        # Ball
        obs[idx:idx+3] = ball_pos
        idx += 3
        obs[idx:idx+3] = ball_vel

        return obs

    def _action_to_index(self, action: dict) -> int:
        """Convert continuous action to discrete index.

        Args:
            action: Action dictionary with throttle, steer, etc.

        Returns:
            Discrete action index
        """
        return self.action_parser.get_action_from_controls(
            throttle=action.get('throttle', 0),
            steer=action.get('steer', 0),
            pitch=action.get('pitch', 0),
            yaw=action.get('yaw', 0),
            roll=action.get('roll', 0),
            jump=bool(action.get('jump', 0)),
            boost=bool(action.get('boost', 0)),
            handbrake=bool(action.get('handbrake', 0)),
        )

    def __len__(self) -> int:
        if self._observations is None:
            return len(self.states)
        return len(self._observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._observations is None:
            self.process()

        obs = torch.tensor(self._observations[idx], dtype=torch.float32)
        action = torch.tensor(self._action_indices[idx], dtype=torch.long)

        return obs, action

    @staticmethod
    def from_npz(path: str) -> "ReplayDataset":
        """Load dataset from npz file.

        Args:
            path: Path to npz file

        Returns:
            ReplayDataset instance
        """
        data = np.load(path, allow_pickle=True)

        states = data['states'].tolist()
        actions = data['actions'].tolist()

        dataset = ReplayDataset(states, actions)
        dataset.process()

        return dataset

    def save(self, path: str) -> None:
        """Save processed data to npz file.

        Args:
            path: Path to save
        """
        if self._observations is None:
            self.process()

        np.savez(
            path,
            observations=self._observations,
            action_indices=self._action_indices,
        )


class StreamingReplayDataset(IterableDataset):
    """Iterable dataset for streaming large replay datasets.

    Useful when the full dataset doesn't fit in memory.
    """

    def __init__(
        self,
        replay_dir: str,
        obs_config: Optional[ObservationConfig] = None,
        action_config: Optional[ActionConfig] = None,
    ):
        """Initialize streaming dataset.

        Args:
            replay_dir: Directory containing processed replay .npz files
            obs_config: Observation configuration
            action_config: Action configuration
        """
        self.replay_dir = Path(replay_dir)
        self.obs_config = obs_config or ObservationConfig()
        self.action_config = action_config or ActionConfig()

        # Find all npz files
        self.npz_files = list(self.replay_dir.glob("*.npz"))
        print(f"Found {len(self.npz_files)} replay files")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over all samples from all files."""
        for npz_file in self.npz_files:
            try:
                data = np.load(npz_file)
                observations = data['observations']
                actions = data['action_indices']

                for obs, action in zip(observations, actions):
                    yield (
                        torch.tensor(obs, dtype=torch.float32),
                        torch.tensor(action, dtype=torch.long),
                    )
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
                continue
