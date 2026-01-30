"""Dataclass configurations for the RL bot."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class ObservationConfig:
    """Configuration for observation builder."""

    # Normalization constants
    pos_norm: Tuple[float, float, float] = (4096.0, 5120.0, 2048.0)
    vel_norm: float = 2300.0
    ang_vel_norm: float = 5.5

    # Feature dimensions (with goal features and velocity magnitudes)
    # self_car: 19 base + 1 speed + 4 goal features = 24
    # ball: 15 base + 1 speed + 3 goal features = 19
    self_car_dim: int = 24
    other_car_dim: int = 14
    ball_dim: int = 19

    # Team-side invariance
    flip_for_orange: bool = True


@dataclass
class ActionConfig:
    """Configuration for action parser."""

    n_actions: int = 1944

    # Discrete options for each control
    throttle_options: Tuple[float, ...] = (-1.0, 0.0, 1.0)
    steer_options: Tuple[float, ...] = (-1.0, 0.0, 1.0)
    pitch_options: Tuple[float, ...] = (-1.0, 0.0, 1.0)
    yaw_options: Tuple[float, ...] = (-1.0, 0.0, 1.0)
    roll_options: Tuple[float, ...] = (-1.0, 0.0, 1.0)
    jump_options: Tuple[int, ...] = (0, 1)
    boost_options: Tuple[int, ...] = (0, 1)
    handbrake_options: Tuple[int, ...] = (0, 1)

    # Action masking
    mask_jump_in_air: bool = True
    mask_boost_empty: bool = True

    # Multi-discrete action space (8 independent heads vs 1944 flat)
    use_multi_discrete: bool = True


@dataclass
class EncoderConfig:
    """Configuration for neural network encoders."""

    hidden_dims: Tuple[int, ...] = (256, 256)
    output_dim: int = 128
    activation: str = "relu"


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism."""

    n_heads: int = 4
    n_layers: int = 2
    embed_dim: int = 128
    ff_dim: int = 512
    dropout: float = 0.0


@dataclass
class LSTMConfig:
    """Configuration for LSTM temporal memory layer."""

    use_lstm: bool = True
    hidden_size: int = 128
    num_layers: int = 1
    sequence_length: int = 16  # Truncated BPTT sequence length


@dataclass
class NetworkConfig:
    """Configuration for the full neural network architecture."""

    car_encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig())
    ball_encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(output_dim=64)
    )
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)

    # Policy head
    policy_hidden_dims: Tuple[int, ...] = (512, 512, 256)
    policy_activation: str = "relu"

    # Value head
    value_hidden_dims: Tuple[int, ...] = (512, 512, 256)
    value_activation: str = "relu"


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""

    # Learning rate
    learning_rate: float = 1e-4
    lr_end: float = 1e-5
    lr_anneal_steps: int = 100_000_000
    lr_warmup_steps: int = 3  # Number of updates (not env steps) to warmup LR

    # Batch sizes
    # batch_size: Fixed total samples per training update (independent of worker count)
    # More workers = faster collection, same data-to-gradient ratio
    batch_size: int = 65_536
    minibatch_size: int = 4096
    experience_buffer_size: int = 200_000

    # GAE parameters
    gamma: float = 0.995
    gae_lambda: float = 0.95

    # PPO parameters
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Training parameters
    n_epochs: int = 3
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


@dataclass
class RewardConfig:
    """Configuration for reward functions.

    Reward weights can be either:
    - A single float: constant weight throughout training
    - A tuple (start, end): linearly interpolated over schedule_steps
    """

    # Reward weights - each can be float or (start, end) tuple
    # Union type allows both constant values and scheduled interpolation
    touch_velocity: Union[float, Tuple[float, float]] = (10.0, 30.0)
    velocity_ball_to_goal: Union[float, Tuple[float, float]] = (1.0, 15.0)
    speed_toward_ball: Union[float, Tuple[float, float]] = (2.0, 0.5)
    goal: Union[float, Tuple[float, float]] = (5.0, 100.0)
    save_boost: Union[float, Tuple[float, float]] = (0.5, 3.0)
    demo: Union[float, Tuple[float, float]] = (0.1, 5.0)
    aerial_height: Union[float, Tuple[float, float]] = (0.1, 2.0)
    team_spacing_penalty: Union[float, Tuple[float, float]] = (0.0, -0.5)
    on_ground: Union[float, Tuple[float, float]] = 0.0  # Test reward

    # Schedule: number of steps over which to interpolate from start to end weights
    schedule_steps: int = 5_000_000_000

    # Team spirit annealing (separate from reward weights)
    team_spirit: float = 0.0
    team_spirit_end: float = 0.5

    def get_weight(self, name: str, step: int) -> float:
        """Get interpolated weight for a reward at given step.

        Args:
            name: Reward name (e.g., 'touch_velocity')
            step: Current training step

        Returns:
            Interpolated weight value
        """
        value = getattr(self, name, 0.0)

        # Handle constant float
        if isinstance(value, (int, float)):
            return float(value)

        # Handle (start, end) tuple
        if isinstance(value, (tuple, list)) and len(value) == 2:
            start, end = value
            progress = min(1.0, step / max(1, self.schedule_steps))
            return start + (end - start) * progress

        return 0.0

    def get_all_weights(self, step: int) -> dict:
        """Get all reward weights at given step.

        Args:
            step: Current training step

        Returns:
            Dict of reward name -> weight
        """
        reward_names = [
            'touch_velocity', 'velocity_ball_to_goal', 'speed_toward_ball',
            'goal', 'save_boost', 'demo', 'aerial_height', 'team_spacing_penalty',
            'on_ground',
        ]
        return {name: self.get_weight(name, step) for name in reward_names}


@dataclass
class EnvironmentConfig:
    """Configuration for the RLGym environment."""

    tick_skip: int = 8
    max_players: int = 6
    spawn_opponents: bool = True
    team_size: int = 1
    gravity: float = -650.0
    boost_consumption: float = 33.3

    # Fixed episode length (steps, not seconds)
    # Goals reset state but don't end episode - episode ends at fixed step count
    episode_steps: int = 300  # ~20 seconds of game time at tick_skip=8

    # Legacy terminal conditions (only used if episode_steps is None)
    terminal_conditions: Tuple[str, ...] = ("goal", "timeout", "no_touch")
    timeout_seconds: float = 300.0
    no_touch_timeout_seconds: float = 30.0

    # State mutator probabilities
    kickoff_prob: float = 0.3
    random_prob: float = 0.5
    replay_prob: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    total_steps: int = 10_000_000_000
    n_workers: int = 32
    n_envs_per_worker: int = 4

    # Checkpointing
    checkpoint_interval: int = 10_000_000
    checkpoint_dir: str = "data/checkpoints"

    # Evaluation
    eval_interval: int = 5_000_000
    eval_episodes: int = 100

    # Logging
    log_interval: int = 10_000
    wandb_project: str = "rlbot-competitive"
    wandb_entity: Optional[str] = None

    # Device
    device: str = "auto"
    seed: int = 42
    deterministic: bool = False


@dataclass
class CurriculumPhase:
    """Configuration for a single curriculum phase."""

    name: str
    end_step: int
    rewards: List[str]
    team_size: int = 1
    team_spirit: float = 0.0
    use_historical_checkpoints: bool = False


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    enabled: bool = True
    phases: List[CurriculumPhase] = field(default_factory=list)
