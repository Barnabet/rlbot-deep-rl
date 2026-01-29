"""Replay file parser using carball."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ReplayParser:
    """Parser for Rocket League replay files.

    Uses carball library to extract game states and actions
    from .replay files.
    """

    def __init__(self, tick_skip: int = 8):
        """Initialize replay parser.

        Args:
            tick_skip: Number of ticks to skip between samples
        """
        self.tick_skip = tick_skip

        try:
            import carball
            self._carball_available = True
        except ImportError:
            print("Warning: carball not installed. Replay parsing disabled.")
            self._carball_available = False

    def parse_replay(self, replay_path: str) -> Optional[Dict[str, Any]]:
        """Parse a single replay file.

        Args:
            replay_path: Path to .replay file

        Returns:
            Dictionary containing parsed game data, or None on failure
        """
        if not self._carball_available:
            return None

        import carball

        try:
            analysis = carball.analyze_replay_file(replay_path)
            return self._extract_data(analysis)
        except Exception as e:
            print(f"Failed to parse {replay_path}: {e}")
            return None

    def parse_directory(self, replay_dir: str) -> List[Dict[str, Any]]:
        """Parse all replays in a directory.

        Args:
            replay_dir: Path to directory containing .replay files

        Returns:
            List of parsed replay data
        """
        replay_path = Path(replay_dir)
        replay_files = list(replay_path.glob("*.replay"))

        print(f"Found {len(replay_files)} replay files")

        parsed_replays = []
        for i, replay_file in enumerate(replay_files):
            if (i + 1) % 10 == 0:
                print(f"  Parsing {i + 1}/{len(replay_files)}")

            data = self.parse_replay(str(replay_file))
            if data is not None:
                parsed_replays.append(data)

        print(f"Successfully parsed {len(parsed_replays)} replays")
        return parsed_replays

    def _extract_data(self, analysis: Any) -> Dict[str, Any]:
        """Extract relevant data from carball analysis.

        Args:
            analysis: Carball analysis object

        Returns:
            Dictionary with extracted data
        """
        frames = analysis.get_data_frame()

        # Get game data
        game_data = frames.game

        # Get ball data
        ball_df = frames.ball

        # Get player data
        players = []
        for player_id in frames.players:
            player_df = frames.players[player_id]
            players.append({
                'id': player_id,
                'team': player_df.team.iloc[0] if 'team' in player_df.columns else 0,
                'data': player_df,
            })

        # Sample at tick_skip intervals
        n_frames = len(ball_df)
        sample_indices = range(0, n_frames, self.tick_skip)

        states = []
        actions = []

        for idx in sample_indices:
            state = self._extract_state(ball_df, players, idx)
            states.append(state)

            # Extract action (controls at this frame)
            action = self._extract_action(players, idx)
            if action is not None:
                actions.append(action)

        return {
            'states': states,
            'actions': actions,
            'metadata': {
                'n_frames': n_frames,
                'n_samples': len(states),
                'n_players': len(players),
            }
        }

    def _extract_state(
        self,
        ball_df: Any,
        players: List[Dict],
        idx: int,
    ) -> Dict[str, Any]:
        """Extract game state at a specific frame.

        Args:
            ball_df: Ball dataframe
            players: List of player data
            idx: Frame index

        Returns:
            State dictionary
        """
        # Ball state
        ball_state = {
            'position': [
                ball_df.pos_x.iloc[idx] if 'pos_x' in ball_df.columns else 0,
                ball_df.pos_y.iloc[idx] if 'pos_y' in ball_df.columns else 0,
                ball_df.pos_z.iloc[idx] if 'pos_z' in ball_df.columns else 0,
            ],
            'velocity': [
                ball_df.vel_x.iloc[idx] if 'vel_x' in ball_df.columns else 0,
                ball_df.vel_y.iloc[idx] if 'vel_y' in ball_df.columns else 0,
                ball_df.vel_z.iloc[idx] if 'vel_z' in ball_df.columns else 0,
            ],
        }

        # Player states
        player_states = []
        for player in players:
            df = player['data']
            if idx >= len(df):
                continue

            player_state = {
                'team': player['team'],
                'position': [
                    df.pos_x.iloc[idx] if 'pos_x' in df.columns else 0,
                    df.pos_y.iloc[idx] if 'pos_y' in df.columns else 0,
                    df.pos_z.iloc[idx] if 'pos_z' in df.columns else 0,
                ],
                'velocity': [
                    df.vel_x.iloc[idx] if 'vel_x' in df.columns else 0,
                    df.vel_y.iloc[idx] if 'vel_y' in df.columns else 0,
                    df.vel_z.iloc[idx] if 'vel_z' in df.columns else 0,
                ],
                'rotation': [
                    df.rot_x.iloc[idx] if 'rot_x' in df.columns else 0,
                    df.rot_y.iloc[idx] if 'rot_y' in df.columns else 0,
                    df.rot_z.iloc[idx] if 'rot_z' in df.columns else 0,
                ],
                'boost': df.boost.iloc[idx] / 100.0 if 'boost' in df.columns else 0.33,
            }
            player_states.append(player_state)

        return {
            'ball': ball_state,
            'players': player_states,
        }

    def _extract_action(
        self,
        players: List[Dict],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Extract actions at a specific frame.

        Args:
            players: List of player data
            idx: Frame index

        Returns:
            Action dictionary or None
        """
        actions = []

        for player in players:
            df = player['data']
            if idx >= len(df):
                continue

            # Extract controls
            action = {
                'player_id': player['id'],
                'team': player['team'],
                'throttle': df.throttle.iloc[idx] if 'throttle' in df.columns else 0,
                'steer': df.steer.iloc[idx] if 'steer' in df.columns else 0,
                'pitch': df.pitch.iloc[idx] if 'pitch' in df.columns else 0,
                'yaw': df.yaw.iloc[idx] if 'yaw' in df.columns else 0,
                'roll': df.roll.iloc[idx] if 'roll' in df.columns else 0,
                'jump': df.jump.iloc[idx] if 'jump' in df.columns else 0,
                'boost': df.boost_active.iloc[idx] if 'boost_active' in df.columns else 0,
                'handbrake': df.handbrake.iloc[idx] if 'handbrake' in df.columns else 0,
            }
            actions.append(action)

        return actions if actions else None
