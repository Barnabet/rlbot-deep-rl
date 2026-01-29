"""Type definitions for the RL bot."""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray


# Type aliases
Observation = NDArray[np.float32]
Action = Union[int, NDArray[np.int64]]
Reward = float
Done = bool
Info = Dict[str, any]

# Tensor types
TensorObs = torch.Tensor
TensorAction = torch.Tensor
TensorReward = torch.Tensor


@dataclass
class CarState:
    """State of a car in the game."""

    position: NDArray[np.float32]  # (3,) - x, y, z
    velocity: NDArray[np.float32]  # (3,) - vx, vy, vz
    angular_velocity: NDArray[np.float32]  # (3,) - wx, wy, wz
    rotation: NDArray[np.float32]  # (3,) - pitch, yaw, roll
    boost_amount: float  # 0-100
    on_ground: bool
    has_flip: bool
    is_demoed: bool
    demo_respawn_timer: float
    team: int  # 0 = blue, 1 = orange


@dataclass
class BallState:
    """State of the ball."""

    position: NDArray[np.float32]  # (3,)
    velocity: NDArray[np.float32]  # (3,)
    angular_velocity: NDArray[np.float32]  # (3,)


@dataclass
class GameState:
    """Full game state."""

    ball: BallState
    cars: List[CarState]
    boost_pads: NDArray[np.bool_]  # (34,) - availability of each boost pad
    time_remaining: float
    blue_score: int
    orange_score: int


class StepResult(NamedTuple):
    """Result of an environment step."""

    observation: Observation
    reward: Reward
    done: Done
    truncated: bool
    info: Info


class RolloutBatch(NamedTuple):
    """Batch of rollout data for training."""

    observations: torch.Tensor  # (batch, obs_dim)
    actions: torch.Tensor  # (batch,)
    log_probs: torch.Tensor  # (batch,)
    values: torch.Tensor  # (batch,)
    rewards: torch.Tensor  # (batch,)
    dones: torch.Tensor  # (batch,)
    advantages: torch.Tensor  # (batch,)
    returns: torch.Tensor  # (batch,)
    action_masks: Optional[torch.Tensor] = None  # (batch, n_actions)


class TrainingMetrics(NamedTuple):
    """Metrics from a training iteration."""

    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    clip_fraction: float
    approx_kl: float
    explained_variance: float
    learning_rate: float


@dataclass
class ControllerInput:
    """Controller input to send to the game."""

    throttle: float = 0.0  # -1 to 1
    steer: float = 0.0  # -1 to 1
    pitch: float = 0.0  # -1 to 1
    yaw: float = 0.0  # -1 to 1
    roll: float = 0.0  # -1 to 1
    jump: bool = False
    boost: bool = False
    handbrake: bool = False

    def to_array(self) -> NDArray[np.float32]:
        """Convert to numpy array."""
        return np.array(
            [
                self.throttle,
                self.steer,
                self.pitch,
                self.yaw,
                self.roll,
                float(self.jump),
                float(self.boost),
                float(self.handbrake),
            ],
            dtype=np.float32,
        )
