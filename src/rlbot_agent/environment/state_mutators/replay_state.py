"""Replay-based state mutator."""

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ...core.registry import registry


@registry.register("state_mutator", "replay")
class ReplayStateMutator:
    """State mutator that loads states from parsed replays.

    Useful for training on realistic game situations.
    """

    def __init__(
        self,
        replay_data_path: Optional[str] = None,
        states: Optional[List[dict]] = None,
    ):
        """Initialize replay state mutator.

        Args:
            replay_data_path: Path to directory containing parsed replay data
            states: Pre-loaded list of state dictionaries
        """
        self.states: List[dict] = states or []

        if replay_data_path is not None:
            self._load_replays(Path(replay_data_path))

    def _load_replays(self, path: Path) -> None:
        """Load parsed replay data from directory.

        Args:
            path: Path to replay data directory
        """
        if not path.exists():
            print(f"Warning: Replay data path {path} does not exist")
            return

        # Load .npz files containing parsed states
        for npz_file in path.glob("*.npz"):
            try:
                data = np.load(npz_file, allow_pickle=True)
                if 'states' in data:
                    self.states.extend(data['states'].tolist())
            except Exception as e:
                print(f"Warning: Failed to load {npz_file}: {e}")

        print(f"Loaded {len(self.states)} states from replays")

    def add_states(self, states: List[dict]) -> None:
        """Add states to the pool.

        Args:
            states: List of state dictionaries
        """
        self.states.extend(states)

    def apply(self, state: Any, num_blue: int, num_orange: int) -> Any:
        """Apply a random replay state to the game.

        Args:
            state: Game state to modify
            num_blue: Number of blue players
            num_orange: Number of orange players

        Returns:
            Modified game state
        """
        if not self.states:
            # Fall back to default state
            return state

        # Choose a random state
        replay_state = self.states[np.random.randint(len(self.states))]

        # Apply ball state
        if 'ball' in replay_state:
            ball = replay_state['ball']
            state.ball.position = np.array(ball['position'], dtype=np.float32)
            state.ball.linear_velocity = np.array(ball['velocity'], dtype=np.float32)
            state.ball.angular_velocity = np.array(
                ball.get('angular_velocity', [0, 0, 0]), dtype=np.float32
            )

        # Apply player states
        if 'players' in replay_state:
            replay_players = replay_state['players']

            # Match players by team
            blue_replay = [p for p in replay_players if p.get('team', 0) == 0]
            orange_replay = [p for p in replay_players if p.get('team', 1) == 1]

            for player in state.players:
                if player.team_num == 0 and blue_replay:
                    self._apply_player_state(player, blue_replay.pop(0))
                elif player.team_num == 1 and orange_replay:
                    self._apply_player_state(player, orange_replay.pop(0))

        return state

    def _apply_player_state(self, player: Any, replay_player: dict) -> None:
        """Apply replay player state to a game player.

        Args:
            player: Game player to modify
            replay_player: Replay player state dictionary
        """
        car = player.car_data

        # Position
        if 'position' in replay_player:
            car.position = np.array(replay_player['position'], dtype=np.float32)

        # Velocity
        if 'velocity' in replay_player:
            car.linear_velocity = np.array(replay_player['velocity'], dtype=np.float32)

        # Angular velocity
        if 'angular_velocity' in replay_player:
            car.angular_velocity = np.array(replay_player['angular_velocity'], dtype=np.float32)

        # Rotation
        if 'rotation' in replay_player:
            # Handle different rotation formats
            rot = replay_player['rotation']
            if len(rot) == 3:
                # Euler angles
                car.euler_angles = np.array(rot, dtype=np.float32)
            elif len(rot) == 4:
                # Quaternion - convert to euler
                car.euler_angles = self._quat_to_euler(rot)

        # Boost
        if 'boost' in replay_player:
            player.boost_amount = float(replay_player['boost']) * 100.0

        # State flags
        player.on_ground = replay_player.get('on_ground', True)
        player.has_flip = replay_player.get('has_flip', True)
        player.is_demoed = replay_player.get('is_demoed', False)

    def _quat_to_euler(self, quat: List[float]) -> np.ndarray:
        """Convert quaternion to euler angles (pitch, yaw, roll).

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            Euler angles [pitch, yaw, roll]
        """
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([pitch, yaw, roll], dtype=np.float32)
