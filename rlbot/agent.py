"""RLBot agent wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

# Try to import RLBot dependencies
try:
    from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
    from rlbot.utils.structures.game_data_struct import GameTickPacket
    RLBOT_AVAILABLE = True
except ImportError:
    RLBOT_AVAILABLE = False
    BaseAgent = object
    SimpleControllerState = None
    GameTickPacket = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlbot_agent.deployment import RLBotAgent


class CompetitiveAgent(BaseAgent if RLBOT_AVAILABLE else object):
    """RLBot agent using trained PPO model.

    This class interfaces with the RLBot framework to run the
    trained deep RL model in actual Rocket League games.
    """

    # Path to trained model checkpoint
    CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "checkpoints" / "latest.pt"

    # Observation normalization constants
    FIELD_X = 4096.0
    FIELD_Y = 5120.0
    FIELD_Z = 2048.0
    MAX_SPEED = 2300.0
    MAX_ANG_VEL = 5.5

    def __init__(self):
        """Initialize the competitive agent."""
        if RLBOT_AVAILABLE:
            super().__init__()

        self.agent: Optional[RLBotAgent] = None
        self._tick_count = 0
        self._cached_controls = None

    def initialize_agent(self):
        """Called once when the agent is started."""
        # Load trained model
        if self.CHECKPOINT_PATH.exists():
            self.agent = RLBotAgent(
                checkpoint_path=str(self.CHECKPOINT_PATH),
                device="cpu",
                deterministic=True,
            )
            print(f"Loaded model from {self.CHECKPOINT_PATH}")
        else:
            print(f"Warning: No checkpoint found at {self.CHECKPOINT_PATH}")
            print("Using random policy")
            self.agent = None

    def get_output(self, packet: 'GameTickPacket') -> 'SimpleControllerState':
        """Called at 120Hz to get controller output.

        Args:
            packet: Current game state from RLBot

        Returns:
            Controller state for this tick
        """
        # Use cached controls if within tick skip
        if self._tick_count < 8 and self._cached_controls is not None:
            self._tick_count += 1
            return self._cached_controls

        self._tick_count = 0

        # Build observation from packet
        obs = self._build_observation(packet)

        # Get action
        if self.agent is not None:
            action_idx = self._get_model_action(obs)
        else:
            action_idx = np.random.randint(1944)

        # Convert to controller state
        controls = self._action_to_controller(action_idx)

        # Cache
        self._cached_controls = controls
        self._tick_count = 1

        return controls

    def _build_observation(self, packet: 'GameTickPacket') -> np.ndarray:
        """Build observation array from game packet.

        Args:
            packet: RLBot game packet

        Returns:
            Observation array matching training format
        """
        # Get own car
        my_car = packet.game_cars[self.index]
        my_team = my_car.team

        # Get ball
        ball = packet.game_ball.physics

        # Build self car observation (19 dims)
        self_obs = self._build_car_obs(my_car, my_team)

        # Build ball observation (15 dims)
        ball_obs = self._build_ball_obs(ball, my_car, my_team)

        # Build other cars observation (5 cars * 14 dims = 70 dims)
        others_obs = self._build_others_obs(packet, my_car, my_team)

        # Concatenate all
        obs = np.concatenate([self_obs, ball_obs, others_obs])

        return obs.astype(np.float32)

    def _build_car_obs(self, car, my_team: int) -> np.ndarray:
        """Build observation for a car."""
        pos = car.physics.location
        vel = car.physics.velocity
        ang_vel = car.physics.angular_velocity
        rot = car.physics.rotation

        # Apply team flipping for orange team
        flip = my_team == 1

        # Position (normalized)
        x = -pos.x / self.FIELD_X if flip else pos.x / self.FIELD_X
        y = -pos.y / self.FIELD_Y if flip else pos.y / self.FIELD_Y
        z = pos.z / self.FIELD_Z

        # Velocity (normalized)
        vx = -vel.x / self.MAX_SPEED if flip else vel.x / self.MAX_SPEED
        vy = -vel.y / self.MAX_SPEED if flip else vel.y / self.MAX_SPEED
        vz = vel.z / self.MAX_SPEED

        # Angular velocity (normalized)
        wx = -ang_vel.x / self.MAX_ANG_VEL if flip else ang_vel.x / self.MAX_ANG_VEL
        wy = -ang_vel.y / self.MAX_ANG_VEL if flip else ang_vel.y / self.MAX_ANG_VEL
        wz = ang_vel.z / self.MAX_ANG_VEL

        # Rotation as sin/cos
        pitch, yaw, roll = rot.pitch, rot.yaw, rot.roll
        if flip:
            yaw = yaw + np.pi

        sin_cos = [
            np.sin(pitch), np.cos(pitch),
            np.sin(yaw), np.cos(yaw),
            np.sin(roll), np.cos(roll),
        ]

        # Other state
        boost = car.boost / 100.0
        on_ground = float(car.has_wheel_contact)
        has_flip = float(not car.double_jumped)
        demo_timer = 0.0  # Not easily available

        return np.array([
            x, y, z, vx, vy, vz, wx, wy, wz,
            *sin_cos, boost, on_ground, has_flip, demo_timer
        ])

    def _build_ball_obs(self, ball, my_car, my_team: int) -> np.ndarray:
        """Build observation for the ball."""
        flip = my_team == 1

        # Absolute position/velocity
        bx = -ball.location.x / self.FIELD_X if flip else ball.location.x / self.FIELD_X
        by = -ball.location.y / self.FIELD_Y if flip else ball.location.y / self.FIELD_Y
        bz = ball.location.z / self.FIELD_Z

        bvx = -ball.velocity.x / self.MAX_SPEED if flip else ball.velocity.x / self.MAX_SPEED
        bvy = -ball.velocity.y / self.MAX_SPEED if flip else ball.velocity.y / self.MAX_SPEED
        bvz = ball.velocity.z / self.MAX_SPEED

        bwx = -ball.angular_velocity.x / self.MAX_ANG_VEL if flip else ball.angular_velocity.x / self.MAX_ANG_VEL
        bwy = -ball.angular_velocity.y / self.MAX_ANG_VEL if flip else ball.angular_velocity.y / self.MAX_ANG_VEL
        bwz = ball.angular_velocity.z / self.MAX_ANG_VEL

        # Relative position/velocity (simplified - not in car's frame)
        car_pos = my_car.physics.location
        car_vel = my_car.physics.velocity

        rel_x = (ball.location.x - car_pos.x) / self.FIELD_X
        rel_y = (ball.location.y - car_pos.y) / self.FIELD_Y
        rel_z = (ball.location.z - car_pos.z) / self.FIELD_Z

        rel_vx = (ball.velocity.x - car_vel.x) / self.MAX_SPEED
        rel_vy = (ball.velocity.y - car_vel.y) / self.MAX_SPEED
        rel_vz = (ball.velocity.z - car_vel.z) / self.MAX_SPEED

        if flip:
            rel_x, rel_y = -rel_x, -rel_y
            rel_vx, rel_vy = -rel_vx, -rel_vy

        return np.array([
            bx, by, bz, bvx, bvy, bvz, bwx, bwy, bwz,
            rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz
        ])

    def _build_others_obs(self, packet, my_car, my_team: int) -> np.ndarray:
        """Build observation for other cars."""
        # Collect teammates and opponents
        teammates = []
        opponents = []

        for i, car in enumerate(packet.game_cars):
            if i == self.index:
                continue
            if not car.is_spawned:
                continue

            car_obs = self._build_other_car_obs(car, my_car, my_team)

            if car.team == my_team:
                teammates.append(car_obs)
            else:
                opponents.append(car_obs)

        # Sort by distance (simplified)
        # Combine: teammates first, then opponents
        all_others = teammates + opponents

        # Pad to 5 cars
        while len(all_others) < 5:
            all_others.append(np.zeros(14))

        # Truncate if more than 5
        all_others = all_others[:5]

        return np.concatenate(all_others)

    def _build_other_car_obs(self, car, my_car, my_team: int) -> np.ndarray:
        """Build observation for another car (14 dims)."""
        flip = my_team == 1

        # Relative position
        my_pos = my_car.physics.location
        rel_x = (car.physics.location.x - my_pos.x) / self.FIELD_X
        rel_y = (car.physics.location.y - my_pos.y) / self.FIELD_Y
        rel_z = (car.physics.location.z - my_pos.z) / self.FIELD_Z

        # Relative velocity
        my_vel = my_car.physics.velocity
        rel_vx = (car.physics.velocity.x - my_vel.x) / self.MAX_SPEED
        rel_vy = (car.physics.velocity.y - my_vel.y) / self.MAX_SPEED
        rel_vz = (car.physics.velocity.z - my_vel.z) / self.MAX_SPEED

        if flip:
            rel_x, rel_y = -rel_x, -rel_y
            rel_vx, rel_vy = -rel_vx, -rel_vy

        # Rotation as sin/cos
        rot = car.physics.rotation
        pitch, yaw, roll = rot.pitch, rot.yaw, rot.roll
        if flip:
            yaw = yaw + np.pi

        sin_cos = [
            np.sin(pitch), np.cos(pitch),
            np.sin(yaw), np.cos(yaw),
            np.sin(roll), np.cos(roll),
        ]

        # Boost and team
        boost = car.boost / 100.0
        is_teammate = float(car.team == my_team)

        return np.array([
            rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz,
            *sin_cos, boost, is_teammate
        ])

    def _get_model_action(self, obs: np.ndarray) -> int:
        """Get action from trained model.

        Args:
            obs: Observation array

        Returns:
            Action index
        """
        import torch

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _, _ = self.agent.model.get_action(
                obs_tensor.to(self.agent.device),
                deterministic=True
            )
            return action.item()

    def _action_to_controller(self, action_idx: int) -> 'SimpleControllerState':
        """Convert action index to controller state.

        Args:
            action_idx: Action index (0-1943)

        Returns:
            RLBot SimpleControllerState
        """
        # Get controls from action parser
        if self.agent is not None:
            controls = self.agent.action_parser._action_table[action_idx]
        else:
            # Random fallback
            controls = np.zeros(8)

        controller = SimpleControllerState()
        controller.throttle = float(controls[0])
        controller.steer = float(controls[1])
        controller.pitch = float(controls[2])
        controller.yaw = float(controls[3])
        controller.roll = float(controls[4])
        controller.jump = bool(controls[5])
        controller.boost = bool(controls[6])
        controller.handbrake = bool(controls[7])

        return controller
