#!/usr/bin/env python3
"""Main training script with Hydra configuration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf

from rlbot_agent.core.config import (
    EnvironmentConfig,
    NetworkConfig,
    ObservationConfig,
    PPOConfig,
    RewardConfig,
    TrainingConfig,
    EncoderConfig,
    AttentionConfig,
    LSTMConfig,
    CurriculumConfig,
    CurriculumPhase,
)
from rlbot_agent.environment import create_environment
from rlbot_agent.environment.rewards import CombinedReward
from rlbot_agent.training import TrainingCoordinator


def config_to_dataclass(cfg: DictConfig) -> dict:
    """Convert Hydra config to dataclasses.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary of configuration dataclasses
    """
    # Environment config
    env_config = EnvironmentConfig(
        tick_skip=cfg.environment.tick_skip,
        max_players=cfg.environment.max_players,
        spawn_opponents=cfg.environment.spawn_opponents,
        team_size=cfg.environment.team_size,
        gravity=cfg.environment.gravity,
        boost_consumption=cfg.environment.boost_consumption,
        episode_steps=cfg.environment.episode_steps,  # Fixed episode length
        terminal_conditions=tuple(cfg.environment.terminal_conditions),
        timeout_seconds=cfg.environment.timeout_seconds,
        no_touch_timeout_seconds=cfg.environment.no_touch_timeout_seconds,
    )

    # Observation config
    obs_config = ObservationConfig(
        pos_norm=tuple(cfg.observation.pos_norm),
        vel_norm=cfg.observation.vel_norm,
        ang_vel_norm=cfg.observation.ang_vel_norm,
        self_car_dim=cfg.observation.self_car_dim,
        other_car_dim=cfg.observation.other_car_dim,
        ball_dim=cfg.observation.ball_dim,
        flip_for_orange=cfg.observation.flip_for_orange,
    )

    # Network config
    car_encoder = EncoderConfig(
        hidden_dims=tuple(cfg.network.car_encoder.hidden_dims),
        output_dim=cfg.network.car_encoder.output_dim,
        activation=cfg.network.car_encoder.activation,
    )
    ball_encoder = EncoderConfig(
        hidden_dims=tuple(cfg.network.ball_encoder.hidden_dims),
        output_dim=cfg.network.ball_encoder.output_dim,
        activation=cfg.network.ball_encoder.activation,
    )
    attention = AttentionConfig(
        n_heads=cfg.network.attention.n_heads,
        n_layers=cfg.network.attention.n_layers,
        embed_dim=cfg.network.attention.embed_dim,
        ff_dim=cfg.network.attention.ff_dim,
        dropout=cfg.network.attention.dropout,
    )
    lstm = LSTMConfig(
        use_lstm=cfg.network.lstm.use_lstm,
        hidden_size=cfg.network.lstm.hidden_size,
        num_layers=cfg.network.lstm.num_layers,
        sequence_length=cfg.network.lstm.sequence_length,
    )
    network_config = NetworkConfig(
        car_encoder=car_encoder,
        ball_encoder=ball_encoder,
        attention=attention,
        lstm=lstm,
        policy_hidden_dims=tuple(cfg.network.policy_head.hidden_dims),
        policy_activation=cfg.network.policy_head.activation,
        value_hidden_dims=tuple(cfg.network.value_head.hidden_dims),
        value_activation=cfg.network.value_head.activation,
    )

    # PPO config
    ppo_config = PPOConfig(
        learning_rate=cfg.ppo.learning_rate,
        lr_end=cfg.ppo.lr_end,
        lr_anneal_steps=cfg.ppo.lr_anneal_steps,
        lr_warmup_steps=cfg.ppo.get('lr_warmup_steps', 3),
        batch_size=cfg.ppo.batch_size,
        minibatch_size=cfg.ppo.minibatch_size,
        experience_buffer_size=cfg.ppo.experience_buffer_size,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_epsilon=cfg.ppo.clip_epsilon,
        entropy_coef=cfg.ppo.entropy_coef,
        value_coef=cfg.ppo.value_coef,
        n_epochs=cfg.ppo.n_epochs,
        max_grad_norm=cfg.ppo.max_grad_norm,
        normalize_advantages=cfg.ppo.normalize_advantages,
    )

    # Reward config - convert lists to tuples for schedule support
    def parse_weight(val):
        """Convert YAML value to weight (float or tuple)."""
        # OmegaConf returns ListConfig for lists
        if hasattr(val, '__iter__') and not isinstance(val, str):
            val_list = list(val)
            if len(val_list) == 2:
                return (float(val_list[0]), float(val_list[1]))
        return float(val)

    reward_config = RewardConfig(
        touch_velocity=parse_weight(cfg.rewards.touch_velocity),
        velocity_ball_to_goal=parse_weight(cfg.rewards.velocity_ball_to_goal),
        speed_toward_ball=parse_weight(cfg.rewards.speed_toward_ball),
        goal=parse_weight(cfg.rewards.goal),
        save_boost=parse_weight(cfg.rewards.save_boost),
        demo=parse_weight(cfg.rewards.demo),
        aerial_height=parse_weight(cfg.rewards.aerial_height),
        team_spacing_penalty=parse_weight(cfg.rewards.team_spacing_penalty),
        on_ground=parse_weight(cfg.rewards.get('on_ground', 0.0)),
        schedule_steps=cfg.rewards.get('schedule_steps', 5_000_000_000),
        team_spirit=cfg.rewards.team_spirit,
        team_spirit_end=cfg.rewards.team_spirit_end,
    )

    # Training config
    training_config = TrainingConfig(
        total_steps=cfg.training.total_steps,
        n_workers=cfg.training.n_workers,
        n_envs_per_worker=cfg.training.n_envs_per_worker,
        checkpoint_interval=cfg.training.checkpoint_interval,
        checkpoint_dir=cfg.training.checkpoint_dir,
        eval_interval=cfg.training.eval_interval,
        eval_episodes=cfg.training.eval_episodes,
        log_interval=cfg.training.log_interval,
        wandb_project=cfg.training.wandb_project,
        wandb_entity=cfg.training.wandb_entity,
        device=cfg.device,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )

    # Curriculum config
    curriculum_config = None
    if hasattr(cfg, 'curriculum') and cfg.curriculum.enabled:
        curriculum_phases = []
        for phase in cfg.curriculum.phases:
            curriculum_phases.append(CurriculumPhase(
                name=phase.name,
                end_step=phase.end_step,
                rewards=list(phase.rewards),
                team_size=phase.get('team_size', 1),
                team_spirit=phase.get('team_spirit', 0.0),
                use_historical_checkpoints=phase.get('use_historical_checkpoints', False),
            ))
        curriculum_config = CurriculumConfig(
            enabled=cfg.curriculum.enabled,
            phases=curriculum_phases,
        )

    return {
        'env': env_config,
        'obs': obs_config,
        'network': network_config,
        'ppo': ppo_config,
        'reward': reward_config,
        'training': training_config,
        'curriculum': curriculum_config,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration
    """
    print("RL Rocket League Bot - Training")
    print("-" * 40)

    # Convert to dataclasses
    configs = config_to_dataclass(cfg)

    # Create reward function
    from rlbot_agent.environment.rewards import (
        TouchVelocity,
        VelocityBallToGoal,
        SpeedTowardBall,
        GoalReward,
        SaveBoost,
        DemoReward,
        AerialHeight,
        TeamSpacing,
        OnGround,
    )

    reward_config = configs['reward']

    # Initial weights (start values for scheduled rewards)
    def get_initial_weight(val):
        """Get initial weight (start value for tuples)."""
        if isinstance(val, tuple):
            return val[0]
        return val

    rewards = [
        (TouchVelocity(), get_initial_weight(reward_config.touch_velocity)),
        (VelocityBallToGoal(), get_initial_weight(reward_config.velocity_ball_to_goal)),
        (SpeedTowardBall(), get_initial_weight(reward_config.speed_toward_ball)),
        (GoalReward(), get_initial_weight(reward_config.goal)),
        (SaveBoost(), get_initial_weight(reward_config.save_boost)),
        (DemoReward(), get_initial_weight(reward_config.demo)),
        (AerialHeight(), get_initial_weight(reward_config.aerial_height)),
        (TeamSpacing(), get_initial_weight(reward_config.team_spacing_penalty)),
        (OnGround(), get_initial_weight(reward_config.on_ground)),
    ]
    reward_fn = CombinedReward(rewards=rewards, config=reward_config)

    # Print reward schedule
    print(f"\nReward schedule (over {reward_config.schedule_steps:,} steps):")
    reward_names = [
        ('TouchVelocity', reward_config.touch_velocity),
        ('VelocityBallToGoal', reward_config.velocity_ball_to_goal),
        ('SpeedTowardBall', reward_config.speed_toward_ball),
        ('GoalReward', reward_config.goal),
        ('SaveBoost', reward_config.save_boost),
        ('DemoReward', reward_config.demo),
        ('AerialHeight', reward_config.aerial_height),
        ('TeamSpacing', reward_config.team_spacing_penalty),
        ('OnGround', reward_config.on_ground),
    ]
    for name, weight in reward_names:
        if isinstance(weight, tuple):
            print(f"  {name}: {weight[0]} → {weight[1]}")
        elif weight != 0:
            print(f"  {name}: {weight} (constant)")
        else:
            print(f"  {name}: off")

    # Create coordinator
    coordinator = TrainingCoordinator(
        env_config=configs['env'],
        obs_config=configs['obs'],
        network_config=configs['network'],
        ppo_config=configs['ppo'],
        training_config=configs['training'],
        reward_fn=reward_fn,
        reward_config=configs['reward'],
        curriculum_config=configs['curriculum'],
    )

    # Resume from checkpoint if available (unless fresh_start is set)
    fresh_start = cfg.get('fresh_start', False)
    checkpoint_path = Path(configs['training'].checkpoint_dir) / "latest.pt"
    if checkpoint_path.exists() and not fresh_start:
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        coordinator.load(str(checkpoint_path))
    elif fresh_start:
        print("\nStarting fresh (ignoring existing checkpoints)")

    # Train
    print("\nStarting training...")
    coordinator.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
