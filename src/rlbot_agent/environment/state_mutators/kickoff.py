"""Kickoff state mutator."""

from typing import Any, List, Tuple

import numpy as np

from ...core.registry import registry


@registry.register("state_mutator", "kickoff")
class KickoffMutator:
    """State mutator that sets up standard kickoff positions.

    5 standard kickoff positions:
    1. Center (both teams)
    2. Left diagonal
    3. Right diagonal
    4. Left corner
    5. Right corner
    """

    # Standard kickoff positions (blue team, orange team mirrors)
    # (x, y, yaw) - yaw is rotation toward ball
    KICKOFF_POSITIONS = [
        # Position 1: Center
        (0.0, -4608.0, np.pi / 2),
        # Position 2: Left diagonal
        (-2048.0, -2560.0, 0.25 * np.pi),
        # Position 3: Right diagonal
        (2048.0, -2560.0, 0.75 * np.pi),
        # Position 4: Left corner
        (-256.0, -3840.0, np.pi / 2),
        # Position 5: Right corner
        (256.0, -3840.0, np.pi / 2),
    ]

    # Common Z position for all cars
    CAR_Z = 17.0

    def __init__(self, spawn_blue_chance: float = 0.5):
        """Initialize kickoff mutator.

        Args:
            spawn_blue_chance: Probability of blue team starting with ball advantage
        """
        self.spawn_blue_chance = spawn_blue_chance

    def apply(self, state: Any, num_blue: int, num_orange: int) -> Any:
        """Apply kickoff state to the game.

        Args:
            state: Game state to modify
            num_blue: Number of blue players
            num_orange: Number of orange players

        Returns:
            Modified game state
        """
        # Reset ball to center
        state.ball.position = np.array([0.0, 0.0, 93.0], dtype=np.float32)
        state.ball.linear_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        state.ball.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Choose positions for each team
        blue_positions = self._select_positions(num_blue)
        orange_positions = self._select_positions(num_orange)

        # Assign positions to players
        blue_idx = 0
        orange_idx = 0

        for player in state.players:
            if player.team_num == 0:  # Blue
                pos = blue_positions[blue_idx]
                blue_idx += 1
            else:  # Orange
                pos = orange_positions[orange_idx]
                orange_idx += 1

            self._set_player_position(player, pos, player.team_num)

        return state

    def _select_positions(self, num_players: int) -> List[Tuple[float, float, float]]:
        """Select appropriate kickoff positions based on team size.

        Args:
            num_players: Number of players on team

        Returns:
            List of (x, y, yaw) positions
        """
        if num_players == 1:
            # 1v1: Random position
            idx = np.random.randint(len(self.KICKOFF_POSITIONS))
            return [self.KICKOFF_POSITIONS[idx]]

        elif num_players == 2:
            # 2v2: Diagonal positions or corner + diagonal
            configs = [
                [1, 2],  # Both diagonals
                [0, 1],  # Center + left diagonal
                [0, 2],  # Center + right diagonal
                [3, 4],  # Both corners
            ]
            chosen = configs[np.random.randint(len(configs))]
            return [self.KICKOFF_POSITIONS[i] for i in chosen]

        else:  # 3v3
            # 3v3: Standard configurations
            configs = [
                [1, 2, 0],  # Diagonals + center
                [1, 2, 3],  # Diagonals + left corner
                [1, 2, 4],  # Diagonals + right corner
                [3, 4, 0],  # Corners + center
            ]
            chosen = configs[np.random.randint(len(configs))]
            return [self.KICKOFF_POSITIONS[i] for i in chosen]

    def _set_player_position(
        self,
        player: Any,
        position: Tuple[float, float, float],
        team: int,
    ) -> None:
        """Set player to a kickoff position.

        Args:
            player: Player to position
            position: (x, y, yaw) position
            team: Team number (0=blue, 1=orange)
        """
        x, y, yaw = position

        # Mirror for orange team
        if team == 1:
            y = -y
            yaw = -yaw + np.pi

        # Set position
        player.car_data.position = np.array([x, y, self.CAR_Z], dtype=np.float32)
        player.car_data.linear_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        player.car_data.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Set rotation (pitch=0, yaw=facing ball, roll=0)
        player.car_data.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)

        # Reset other state
        player.boost_amount = 33.0  # Standard kickoff boost
        player.on_ground = True
        player.has_flip = True
        player.is_demoed = False
