#!/usr/bin/env python3
"""Watch a trained bot play with visualization (no Rocket League needed)."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch


def watch_bot(
    checkpoint_path: str = None,
    n_episodes: int = 5,
    speed: float = 1.0,
    deterministic: bool = True,
):
    """Watch a bot play with RLViser visualization.

    Args:
        checkpoint_path: Path to checkpoint (uses latest if None)
        n_episodes: Number of episodes to watch
        speed: Playback speed multiplier (1.0 = real-time)
        deterministic: Use deterministic actions (no sampling)
    """
    # Import visualization
    try:
        import rlviser_py as vis
        import RocketSim as rsim
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: pip install rlviser-py rocketsim")
        return

    from rlbot_agent.core.config import EnvironmentConfig, ObservationConfig, NetworkConfig
    from rlbot_agent.models import ActorAttentionCritic

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_dir = Path("data/checkpoints")
        if checkpoint_dir.exists():
            latest = checkpoint_dir / "latest.pt"
            if latest.exists():
                checkpoint_path = str(latest)
            else:
                checkpoints = sorted(checkpoint_dir.glob("*.pt"))
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])

    if checkpoint_path is None:
        print("No checkpoint found. Run training first or specify --checkpoint")
        return

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model
    obs_config = ObservationConfig()
    network_config = NetworkConfig()

    obs_dim = obs_config.self_car_dim + obs_config.ball_dim + 5 * obs_config.other_car_dim
    model = ActorAttentionCritic(
        obs_dim=obs_dim,
        n_actions=1944,
        max_players=6,
        obs_config=obs_config,
        network_config=network_config,
        use_multi_discrete=True,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded. Device: {device}")

    # Create RocketSim arena directly for visualization
    game_mode = rsim.GameMode.SOCCAR
    arena = rsim.Arena(game_mode)

    # Set boost pad locations for RLViser
    vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])

    # Add cars
    blue_car = arena.add_car(rsim.Team.BLUE)
    orange_car = arena.add_car(rsim.Team.ORANGE)

    # Create environment for observation building
    from rlbot_agent.environment.obs_builders import AdvancedObsBuilder
    from rlbot_agent.environment.action_parsers import MultiDiscreteActionParser

    obs_builder = AdvancedObsBuilder(config=obs_config, max_players=6)
    action_parser = MultiDiscreteActionParser()

    tick_skip = 8
    tick_rate = 120
    step_time = tick_skip / tick_rate / speed

    print(f"\nWatching {n_episodes} episodes at {speed}x speed")
    print("RLViser window should open automatically...")
    print("Press Ctrl+C to exit\n")

    # Initialize LSTM hidden state
    use_lstm = hasattr(model, 'use_lstm') and model.use_lstm
    if use_lstm:
        hidden = model.get_initial_hidden(2, device)  # 2 agents
    else:
        hidden = None

    packet_id = 0

    for episode in range(n_episodes):
        # Reset arena to kickoff
        arena.reset_to_random_kickoff()

        episode_reward = 0
        step = 0
        max_steps = 512  # ~34 seconds

        print(f"Episode {episode + 1}/{n_episodes}")

        while step < max_steps:
            # Build game state from arena
            ball_state = arena.ball.get_state()
            cars = arena.get_cars()

            # Build observations for both cars using a simplified approach
            # For proper obs, we'd need full RLGym state, but this is for visualization
            blue_obs = _build_simple_obs(blue_car, orange_car, ball_state, obs_config, team=0)
            orange_obs = _build_simple_obs(orange_car, blue_car, ball_state, obs_config, team=1)

            # Get actions from model
            with torch.no_grad():
                blue_tensor = torch.tensor(blue_obs, dtype=torch.float32, device=device).unsqueeze(0)
                orange_tensor = torch.tensor(orange_obs, dtype=torch.float32, device=device).unsqueeze(0)

                blue_action, _, _, _, _ = model.get_action(blue_tensor, None, deterministic=deterministic)
                orange_action, _, _, _, _ = model.get_action(orange_tensor, None, deterministic=deterministic)

            # Convert actions to controls
            blue_controls = _action_to_controls(blue_action.cpu().numpy()[0], action_parser)
            orange_controls = _action_to_controls(orange_action.cpu().numpy()[0], action_parser)

            blue_car.set_controls(blue_controls)
            orange_car.set_controls(orange_controls)

            # Step simulation
            arena.step(tick_skip)
            step += 1
            packet_id += 1

            # Render
            pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
            ball = arena.ball.get_state()
            car_data = [
                (car.id, car.team, car.get_config(), car.get_state())
                for car in arena.get_cars()
            ]
            vis.render(packet_id, tick_rate, game_mode, pad_states, ball, car_data)

            # Check for goal
            if ball.pos.y > 5120 + 92.75 or ball.pos.y < -5120 - 92.75:
                team = "Blue" if ball.pos.y > 0 else "Orange"
                print(f"  GOAL by {team}! (step {step})")
                arena.reset_to_random_kickoff()

            # Sleep for real-time playback
            time.sleep(step_time)

        print(f"  Episode finished at step {step}")

    print("\nDone! Close the RLViser window to exit.")


def _build_simple_obs(our_car, opponent_car, ball_state, obs_config, team):
    """Build a simplified observation vector.

    This is a quick approximation - full obs would use AdvancedObsBuilder.
    """
    obs = np.zeros(obs_config.self_car_dim + obs_config.ball_dim + 5 * obs_config.other_car_dim, dtype=np.float32)

    # Normalization
    pos_norm = np.array(obs_config.pos_norm)
    vel_norm = obs_config.vel_norm
    ang_vel_norm = obs_config.ang_vel_norm

    # Team-side flip for orange
    flip = -1 if team == 1 else 1

    our_state = our_car.get_state()
    opp_state = opponent_car.get_state()

    # Self car (first 24 dims)
    idx = 0
    # Position
    obs[idx:idx+3] = np.array([our_state.pos.x, our_state.pos.y * flip, our_state.pos.z]) / pos_norm
    idx += 3
    # Velocity
    obs[idx:idx+3] = np.array([our_state.vel.x, our_state.vel.y * flip, our_state.vel.z]) / vel_norm
    idx += 3
    # Angular velocity
    obs[idx:idx+3] = np.array([our_state.ang_vel.x, our_state.ang_vel.y * flip, our_state.ang_vel.z]) / ang_vel_norm
    idx += 3
    # Rotation (forward, up vectors simplified)
    obs[idx:idx+6] = 0  # Placeholder for rotation vectors
    idx += 6
    # Boost, on_ground, has_flip, is_demoed
    obs[idx] = our_state.boost / 100.0
    obs[idx+1] = float(our_state.on_ground)
    obs[idx+2] = float(our_state.has_flip)
    obs[idx+3] = 0  # is_demoed
    idx += 4
    # Speed magnitude
    obs[idx] = np.linalg.norm([our_state.vel.x, our_state.vel.y, our_state.vel.z]) / vel_norm
    idx += 1
    # Goal features (placeholder)
    obs[idx:idx+4] = 0
    idx += 4

    # Ball (next 19 dims)
    obs[idx:idx+3] = np.array([ball_state.pos.x, ball_state.pos.y * flip, ball_state.pos.z]) / pos_norm
    idx += 3
    obs[idx:idx+3] = np.array([ball_state.vel.x, ball_state.vel.y * flip, ball_state.vel.z]) / vel_norm
    idx += 3
    obs[idx:idx+3] = np.array([ball_state.ang_vel.x, ball_state.ang_vel.y * flip, ball_state.ang_vel.z]) / ang_vel_norm
    idx += 3
    # Ball rotation (placeholder)
    obs[idx:idx+6] = 0
    idx += 6
    # Ball speed
    obs[idx] = np.linalg.norm([ball_state.vel.x, ball_state.vel.y, ball_state.vel.z]) / vel_norm
    idx += 1
    # Goal features (placeholder)
    obs[idx:idx+3] = 0
    idx += 3

    # Opponent car (first other car slot, 14 dims)
    obs[idx:idx+3] = np.array([opp_state.pos.x, opp_state.pos.y * flip, opp_state.pos.z]) / pos_norm
    idx += 3
    obs[idx:idx+3] = np.array([opp_state.vel.x, opp_state.vel.y * flip, opp_state.vel.z]) / vel_norm
    idx += 3
    obs[idx:idx+3] = np.array([opp_state.ang_vel.x, opp_state.ang_vel.y * flip, opp_state.ang_vel.z]) / ang_vel_norm
    idx += 3
    # Rotation placeholder
    obs[idx:idx+3] = 0
    idx += 3
    # Boost, on_ground
    obs[idx] = opp_state.boost / 100.0
    obs[idx+1] = float(opp_state.on_ground)
    idx += 2

    return obs


def _action_to_controls(action, action_parser):
    """Convert discrete action to RocketSim controls."""
    import RocketSim as rsim

    controls = rsim.CarControls()

    if isinstance(action, np.ndarray) and len(action) == 8:
        # Multi-discrete action
        throttle_idx, steer_idx, pitch_idx, yaw_idx, roll_idx, jump_idx, boost_idx, handbrake_idx = action

        throttle_opts = [-1.0, 0.0, 1.0]
        steer_opts = [-1.0, 0.0, 1.0]
        pitch_opts = [-1.0, 0.0, 1.0]
        yaw_opts = [-1.0, 0.0, 1.0]
        roll_opts = [-1.0, 0.0, 1.0]

        controls.throttle = throttle_opts[int(throttle_idx)]
        controls.steer = steer_opts[int(steer_idx)]
        controls.pitch = pitch_opts[int(pitch_idx)]
        controls.yaw = yaw_opts[int(yaw_idx)]
        controls.roll = roll_opts[int(roll_idx)]
        controls.jump = bool(jump_idx)
        controls.boost = bool(boost_idx)
        controls.handbrake = bool(handbrake_idx)
    else:
        # Flat action - decode
        action_int = int(action)
        controls.throttle = [-1.0, 0.0, 1.0][action_int % 3]
        action_int //= 3
        controls.steer = [-1.0, 0.0, 1.0][action_int % 3]
        action_int //= 3
        controls.pitch = [-1.0, 0.0, 1.0][action_int % 3]
        action_int //= 3
        controls.yaw = [-1.0, 0.0, 1.0][action_int % 3]
        action_int //= 3
        controls.roll = [-1.0, 0.0, 1.0][action_int % 3]
        action_int //= 3
        controls.jump = bool(action_int % 2)
        action_int //= 2
        controls.boost = bool(action_int % 2)
        action_int //= 2
        controls.handbrake = bool(action_int % 2)

    return controls


def main():
    parser = argparse.ArgumentParser(description="Watch trained bot play")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to checkpoint (uses latest if not specified)",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=5,
        help="Number of episodes to watch",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (sample from policy)",
    )

    args = parser.parse_args()

    watch_bot(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        speed=args.speed,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
