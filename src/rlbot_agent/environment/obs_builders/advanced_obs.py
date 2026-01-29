"""Advanced observation builder with team-invariant relative observations."""

from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ...core.config import ObservationConfig
from ...core.registry import registry
from .base import BaseObsBuilder


@registry.register("obs_builder", "advanced")
class AdvancedObsBuilder(BaseObsBuilder):
    """Advanced observation builder with normalized, team-invariant observations.

    Features:
    - Normalized positions (X/4096, Y/5120, Z/2048)
    - Normalized velocities (scale by 2300)
    - Rotations as sin/cos of pitch, yaw, roll
    - Boost level (0-1), on_ground, has_flip, demo_timer
    - Relative ball position/velocity in car's local frame
    - Other players sorted by team (teammates first) then distance
    - Team-side invariance: flip X/Y coords for orange team
    """

    # Field dimensions for normalization
    FIELD_X = 4096.0
    FIELD_Y = 5120.0
    FIELD_Z = 2048.0
    MAX_SPEED = 2300.0
    MAX_ANG_VEL = 5.5
    GOAL_Y = 5120.0  # Y position of goals

    def __init__(
        self,
        config: Optional[ObservationConfig] = None,
        max_players: int = 6,
    ):
        """Initialize the observation builder.

        Args:
            config: Observation configuration
            max_players: Maximum number of players in a game
        """
        self.config = config or ObservationConfig()
        self.max_players = max_players

        # Calculate observation sizes
        self.self_car_size = self.config.self_car_dim  # 19
        self.other_car_size = self.config.other_car_dim  # 14
        self.ball_size = self.config.ball_dim  # 15

        # Total observation size
        # Self car + ball + (max_players - 1) other cars
        self._obs_size = (
            self.self_car_size
            + self.ball_size
            + (self.max_players - 1) * self.other_car_size
        )

    def reset(self, initial_state: Any) -> None:
        """Reset for a new episode."""
        pass

    def get_obs_space_size(self) -> int:
        """Get the observation space size."""
        return self._obs_size

    def build_obs(
        self, player: Any, state: Any, previous_action: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Build observation for a single player.

        Args:
            player: Player data with car state
            state: Current game state with ball and all players
            previous_action: Previous action (8 floats)

        Returns:
            Flattened observation array
        """
        # Determine if we need to flip for orange team
        flip = self.config.flip_for_orange and player.team_num == 1

        # Build self car observation
        self_obs = self._build_self_car_obs(player, previous_action, flip)

        # Build ball observation relative to self car
        ball_obs = self._build_ball_obs(player, state.ball, flip)

        # Build other players observations
        others_obs = self._build_others_obs(player, state.players, flip)

        # Concatenate all observations
        obs = np.concatenate([self_obs, ball_obs, others_obs])

        return obs.astype(np.float32)

    def _build_self_car_obs(
        self,
        player: Any,
        previous_action: NDArray[np.float32],
        flip: bool,
    ) -> NDArray[np.float32]:
        """Build observation for self car.

        Returns 19-dimensional vector:
        - Position (3): normalized x, y, z
        - Velocity (3): normalized vx, vy, vz
        - Angular velocity (3): normalized wx, wy, wz
        - Rotation (6): sin/cos of pitch, yaw, roll
        - Boost (1): 0-1
        - On ground (1): binary
        - Has flip (1): binary
        - Demo timer (1): 0-3 seconds normalized
        """
        car = player.car_data

        # Position - normalized
        pos = self._normalize_position(car.position, flip)

        # Velocity - normalized
        vel = self._normalize_velocity(car.linear_velocity, flip)

        # Angular velocity - normalized
        ang_vel = self._normalize_angular_velocity(car.angular_velocity, flip)

        # Rotation as sin/cos
        rot = self._rotation_to_sincos(car.pitch(), car.yaw(), car.roll(), flip)

        # Other state
        boost = player.boost_amount / 100.0
        on_ground = float(player.on_ground)
        has_flip = float(player.has_flip)
        demo_timer = player.demo_respawn_timer / 3.0 if hasattr(player, 'demo_respawn_timer') else 0.0

        return np.array(
            [*pos, *vel, *ang_vel, *rot, boost, on_ground, has_flip, demo_timer],
            dtype=np.float32,
        )

    def _build_ball_obs(
        self,
        player: Any,
        ball: Any,
        flip: bool,
    ) -> NDArray[np.float32]:
        """Build observation for ball relative to player.

        Returns 15-dimensional vector:
        - Absolute position (3): normalized
        - Absolute velocity (3): normalized
        - Angular velocity (3): normalized
        - Relative position in car's local frame (3)
        - Relative velocity in car's local frame (3)
        """
        car = player.car_data

        # Absolute position and velocity
        ball_pos = self._normalize_position(ball.position, flip)
        ball_vel = self._normalize_velocity(ball.linear_velocity, flip)
        ball_ang_vel = self._normalize_angular_velocity(ball.angular_velocity, flip)

        # Relative position in world frame
        rel_pos = ball.position - car.position
        rel_vel = ball.linear_velocity - car.linear_velocity

        # Transform to car's local frame
        rotation_matrix = self._get_rotation_matrix(car, flip)
        local_rel_pos = rotation_matrix @ rel_pos
        local_rel_vel = rotation_matrix @ rel_vel

        # Normalize relative values
        local_rel_pos = local_rel_pos / np.array([self.FIELD_X, self.FIELD_Y, self.FIELD_Z])
        local_rel_vel = local_rel_vel / self.MAX_SPEED

        return np.array(
            [*ball_pos, *ball_vel, *ball_ang_vel, *local_rel_pos, *local_rel_vel],
            dtype=np.float32,
        )

    def _build_others_obs(
        self,
        player: Any,
        all_players: List[Any],
        flip: bool,
    ) -> NDArray[np.float32]:
        """Build observations for other players.

        Players are sorted: teammates first (by distance), then opponents (by distance).
        Each player is represented by 14 values.

        Returns flattened array of shape ((max_players - 1) * 14,)
        """
        car = player.car_data
        my_team = player.team_num

        # Separate teammates and opponents
        teammates = []
        opponents = []

        for p in all_players:
            if p.car_id == player.car_id:
                continue
            dist = np.linalg.norm(p.car_data.position - car.position)
            if p.team_num == my_team:
                teammates.append((dist, p))
            else:
                opponents.append((dist, p))

        # Sort by distance
        teammates.sort(key=lambda x: x[0])
        opponents.sort(key=lambda x: x[0])

        # Build observation for each player
        others = []
        for _, p in teammates + opponents:
            others.append(self._build_other_car_obs(player, p, flip))

        # Pad with zeros if fewer players than max
        while len(others) < self.max_players - 1:
            others.append(np.zeros(self.other_car_size, dtype=np.float32))

        return np.concatenate(others)

    def _build_other_car_obs(
        self,
        self_player: Any,
        other_player: Any,
        flip: bool,
    ) -> NDArray[np.float32]:
        """Build observation for another car relative to self.

        Returns 14-dimensional vector:
        - Relative position in self's local frame (3)
        - Relative velocity in self's local frame (3)
        - Rotation as sin/cos (6): pitch, yaw, roll
        - Boost (1): 0-1
        - Is teammate (1): binary
        """
        self_car = self_player.car_data
        other_car = other_player.car_data

        # Relative position and velocity in world frame
        rel_pos = other_car.position - self_car.position
        rel_vel = other_car.linear_velocity - self_car.linear_velocity

        # Transform to self's local frame
        rotation_matrix = self._get_rotation_matrix(self_car, flip)
        local_rel_pos = rotation_matrix @ rel_pos
        local_rel_vel = rotation_matrix @ rel_vel

        # Normalize
        local_rel_pos = local_rel_pos / np.array([self.FIELD_X, self.FIELD_Y, self.FIELD_Z])
        local_rel_vel = local_rel_vel / self.MAX_SPEED

        # Other car's rotation
        rot = self._rotation_to_sincos(
            other_car.pitch(), other_car.yaw(), other_car.roll(), flip
        )

        # Boost and team
        boost = other_player.boost_amount / 100.0
        is_teammate = float(other_player.team_num == self_player.team_num)

        return np.array(
            [*local_rel_pos, *local_rel_vel, *rot, boost, is_teammate],
            dtype=np.float32,
        )

    def _normalize_position(
        self, pos: NDArray[np.float32], flip: bool
    ) -> NDArray[np.float32]:
        """Normalize position to [-1, 1] range."""
        x, y, z = pos
        if flip:
            x, y = -x, -y
        return np.array(
            [x / self.FIELD_X, y / self.FIELD_Y, z / self.FIELD_Z],
            dtype=np.float32,
        )

    def _normalize_velocity(
        self, vel: NDArray[np.float32], flip: bool
    ) -> NDArray[np.float32]:
        """Normalize velocity by max speed."""
        vx, vy, vz = vel
        if flip:
            vx, vy = -vx, -vy
        return np.array([vx, vy, vz], dtype=np.float32) / self.MAX_SPEED

    def _normalize_angular_velocity(
        self, ang_vel: NDArray[np.float32], flip: bool
    ) -> NDArray[np.float32]:
        """Normalize angular velocity."""
        wx, wy, wz = ang_vel
        if flip:
            wx, wy = -wx, -wy
        return np.array([wx, wy, wz], dtype=np.float32) / self.MAX_ANG_VEL

    def _rotation_to_sincos(
        self, pitch: float, yaw: float, roll: float, flip: bool
    ) -> NDArray[np.float32]:
        """Convert rotation angles to sin/cos representation."""
        if flip:
            yaw = yaw + np.pi  # Flip yaw by 180 degrees

        return np.array(
            [
                np.sin(pitch), np.cos(pitch),
                np.sin(yaw), np.cos(yaw),
                np.sin(roll), np.cos(roll),
            ],
            dtype=np.float32,
        )

    def _get_rotation_matrix(self, car: Any, flip: bool) -> NDArray[np.float32]:
        """Get rotation matrix to transform from world to car's local frame."""
        # Get car's forward, right, up vectors
        forward = car.forward()
        right = car.right()
        up = car.up()

        if flip:
            forward = -forward
            right = -right
            # up stays the same

        # Rotation matrix: rows are the car's axes
        return np.array([forward, right, up], dtype=np.float32)
