#!/usr/bin/env python3
"""Evaluation script for testing trained agents."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlbot_agent.deployment import InferenceEngine
from rlbot_agent.environment import create_environment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL bot")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/latest.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Opponent bot name (nexto, necto, etc.)",
    )

    return parser.parse_args()


def evaluate_agent(
    checkpoint_path: str,
    n_episodes: int = 100,
    render: bool = False,
    deterministic: bool = True,
    device: str = "cpu",
):
    """Evaluate a trained agent.

    Args:
        checkpoint_path: Path to model checkpoint
        n_episodes: Number of episodes
        render: Whether to render
        deterministic: Use deterministic policy
        device: PyTorch device

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load inference engine
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Create environment
    env = create_environment(render=render)

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    goals_scored = []
    goals_conceded = []

    print(f"\nEvaluating for {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        length = 0
        scored = 0
        conceded = 0

        while not done:
            # Get action
            action, _ = engine.get_action(obs, deterministic=deterministic)

            # Environment step
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            total_reward += reward
            length += 1

            # Track goals from info
            if 'goal_scored' in info:
                scored += info['goal_scored']
            if 'goal_conceded' in info:
                conceded += info['goal_conceded']

        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        goals_scored.append(scored)
        goals_conceded.append(conceded)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: "
                  f"reward={total_reward:.2f}, length={length}")

    env.close()

    # Compute statistics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_goals_scored": np.mean(goals_scored),
        "mean_goals_conceded": np.mean(goals_conceded),
        "goal_differential": np.mean(goals_scored) - np.mean(goals_conceded),
        "win_rate": np.mean([s > c for s, c in zip(goals_scored, goals_conceded)]),
    }

    return metrics


def main():
    """Main evaluation entry point."""
    args = parse_args()

    print("=" * 60)
    print("RL Rocket League Bot - Evaluation")
    print("=" * 60)

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Run evaluation
    metrics = evaluate_agent(
        checkpoint_path=str(checkpoint_path),
        n_episodes=args.n_episodes,
        render=args.render,
        deterministic=args.deterministic,
        device=args.device,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    main()
