"""Random state mutator."""

from typing import Any, Optional

import numpy as np

from ...core.registry import registry


@registry.register("state_mutator", "random")
class RandomStateMutator:
    """State mutator that generates random game states.

    Useful for training on diverse situations.
    """

    # Field boundaries
    FIELD_X = 4096.0
    FIELD_Y = 5120.0
    FIELD_Z = 2048.0

    # Ball parameters
    BALL_RADIUS = 92.75
    MAX_BALL_SPEED = 6000.0
    MAX_BALL_ANG_VEL = 6.0

    # Car parameters
    CAR_Z_GROUND = 17.0
    MAX_CAR_SPEED = 2300.0
    MAX_CAR_ANG_VEL = 5.5

    def __init__(
        self,
        ball_on_ground_prob: float = 0.5,
        car_on_ground_prob: float = 0.8,
        random_boost: bool = True,
        min_boost: float = 0.0,
        max_boost: float = 100.0,
    ):
        """Initialize random state mutator.

        Args:
            ball_on_ground_prob: Probability ball starts on ground
            car_on_ground_prob: Probability each car starts on ground
            random_boost: Whether to randomize boost amounts
            min_boost: Minimum boost amount
            max_boost: Maximum boost amount
        """
        self.ball_on_ground_prob = ball_on_ground_prob
        self.car_on_ground_prob = car_on_ground_prob
        self.random_boost = random_boost
        self.min_boost = min_boost
        self.max_boost = max_boost

    def apply(self, state: Any, num_blue: int, num_orange: int) -> Any:
        """Apply random state to the game.

        Args:
            state: Game state to modify
            num_blue: Number of blue players
            num_orange: Number of orange players

        Returns:
            Modified game state
        """
        # Randomize ball
        self._randomize_ball(state)

        # Randomize each player
        for player in state.players:
            self._randomize_car(player)

        return state

    def _randomize_ball(self, state: Any) -> None:
        """Randomize ball position and velocity."""
        # Position
        x = np.random.uniform(-self.FIELD_X * 0.9, self.FIELD_X * 0.9)
        y = np.random.uniform(-self.FIELD_Y * 0.9, self.FIELD_Y * 0.9)

        if np.random.random() < self.ball_on_ground_prob:
            z = self.BALL_RADIUS
        else:
            z = np.random.uniform(self.BALL_RADIUS, self.FIELD_Z * 0.8)

        state.ball.position = np.array([x, y, z], dtype=np.float32)

        # Velocity (biased toward lower speeds)
        speed = np.random.exponential(self.MAX_BALL_SPEED / 4)
        speed = min(speed, self.MAX_BALL_SPEED)
        direction = self._random_unit_vector()
        state.ball.linear_velocity = (direction * speed).astype(np.float32)

        # Angular velocity
        ang_vel = np.random.uniform(-self.MAX_BALL_ANG_VEL, self.MAX_BALL_ANG_VEL, size=3)
        state.ball.angular_velocity = ang_vel.astype(np.float32)

    def _randomize_car(self, player: Any) -> None:
        """Randomize car position, velocity, and rotation."""
        car = player.car_data

        # Position - keep some distance from walls
        x = np.random.uniform(-self.FIELD_X * 0.85, self.FIELD_X * 0.85)
        y = np.random.uniform(-self.FIELD_Y * 0.85, self.FIELD_Y * 0.85)

        on_ground = np.random.random() < self.car_on_ground_prob

        if on_ground:
            z = self.CAR_Z_GROUND
        else:
            z = np.random.uniform(self.CAR_Z_GROUND, self.FIELD_Z * 0.7)

        car.position = np.array([x, y, z], dtype=np.float32)

        # Velocity
        speed = np.random.exponential(self.MAX_CAR_SPEED / 3)
        speed = min(speed, self.MAX_CAR_SPEED)
        direction = self._random_unit_vector()
        car.linear_velocity = (direction * speed).astype(np.float32)

        # Angular velocity
        if on_ground:
            # Mostly yaw rotation on ground
            ang_vel = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-self.MAX_CAR_ANG_VEL, self.MAX_CAR_ANG_VEL),
            ])
        else:
            ang_vel = np.random.uniform(-self.MAX_CAR_ANG_VEL, self.MAX_CAR_ANG_VEL, size=3)

        car.angular_velocity = ang_vel.astype(np.float32)

        # Rotation
        if on_ground:
            pitch = 0.0
            roll = 0.0
            yaw = np.random.uniform(-np.pi, np.pi)
        else:
            pitch = np.random.uniform(-np.pi, np.pi)
            yaw = np.random.uniform(-np.pi, np.pi)
            roll = np.random.uniform(-np.pi, np.pi)

        car.euler_angles = np.array([pitch, yaw, roll], dtype=np.float32)

        # State
        player.on_ground = on_ground
        player.has_flip = on_ground or np.random.random() < 0.5
        player.is_demoed = False

        # Boost
        if self.random_boost:
            player.boost_amount = np.random.uniform(self.min_boost, self.max_boost)
        else:
            player.boost_amount = 33.0

    def _random_unit_vector(self) -> np.ndarray:
        """Generate a random 3D unit vector."""
        vec = np.random.randn(3)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return vec / norm
