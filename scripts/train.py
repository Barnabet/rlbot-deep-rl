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
    network_config = NetworkConfig(
        car_encoder=car_encoder,
        ball_encoder=ball_encoder,
        attention=attention,
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

    # Reward config
    reward_config = RewardConfig(
        touch_velocity=cfg.rewards.touch_velocity,
        velocity_ball_to_goal=cfg.rewards.velocity_ball_to_goal,
        speed_toward_ball=cfg.rewards.speed_toward_ball,
        goal=cfg.rewards.goal,
        save_boost=cfg.rewards.save_boost,
        demo=cfg.rewards.demo,
        aerial_height=cfg.rewards.aerial_height,
        team_spacing_penalty=cfg.rewards.team_spacing_penalty,
        anneal_steps=cfg.rewards.anneal_steps,
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

    return {
        'env': env_config,
        'obs': obs_config,
        'network': network_config,
        'ppo': ppo_config,
        'reward': reward_config,
        'training': training_config,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("RL Rocket League Bot - Training")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

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
    )

    reward_config = configs['reward']
    rewards = [
        (TouchVelocity(), reward_config.touch_velocity),
        (VelocityBallToGoal(), reward_config.velocity_ball_to_goal),
        (SpeedTowardBall(), reward_config.speed_toward_ball),
        (GoalReward(), reward_config.goal),
        (SaveBoost(), reward_config.save_boost),
        (DemoReward(), reward_config.demo),
        (AerialHeight(), reward_config.aerial_height),
        (TeamSpacing(), reward_config.team_spacing_penalty),
    ]
    reward_fn = CombinedReward(rewards=rewards, config=reward_config)

    # Create coordinator
    coordinator = TrainingCoordinator(
        env_config=configs['env'],
        obs_config=configs['obs'],
        network_config=configs['network'],
        ppo_config=configs['ppo'],
        training_config=configs['training'],
        reward_fn=reward_fn,
    )

    # Resume from checkpoint if available
    checkpoint_path = Path(configs['training'].checkpoint_dir) / "latest.pt"
    if checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        coordinator.load(str(checkpoint_path))

    # Train
    print("\nStarting training...")
    coordinator.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
