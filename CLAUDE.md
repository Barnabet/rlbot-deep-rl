# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep reinforcement learning framework for training competitive Rocket League bots using PPO with attention-based neural networks. Integrates with RLGym 2.0 for simulation and RLBot for deployment.

## Commands

### CLI Tool (Recommended)

```bash
# Install
pip install -e .

# Training
python scripts/cli.py train                    # Start with defaults
python scripts/cli.py train -w 32 -s 1000000  # 32 workers, 1M steps
python scripts/cli.py train -r latest.pt      # Resume from checkpoint
python scripts/cli.py train --wandb my-proj   # Log to W&B

# Evaluation
python scripts/cli.py eval                     # Evaluate latest checkpoint
python scripts/cli.py eval checkpoint.pt -n 50 # 50 episodes on specific checkpoint

# Configuration
python scripts/cli.py config                   # View base config
python scripts/cli.py config -s network       # View network section only
python scripts/cli.py config --edit           # Edit config in $EDITOR

# Management
python scripts/cli.py list                     # List checkpoints
python scripts/cli.py clean -k 5              # Keep 5 most recent
python scripts/cli.py info                     # System info
```

### Direct Scripts (Advanced)

```bash
# Training with Hydra (full control)
python scripts/train.py                                          # Default config
python scripts/train.py training=ppo_1v1 rewards=early_stage     # Override configs
python scripts/train.py ppo.learning_rate=5e-5 training.n_workers=16  # Override params
python scripts/train.py render=true training.n_workers=1         # Visual debugging

# Evaluation
python scripts/evaluate.py --checkpoint data/checkpoints/latest.pt --n-episodes 100

# Testing
python -m pytest tests/ -v
python -m pytest tests/test_models/test_actor_attention_critic.py -v  # Single file

# Linting
black src/ --line-length 100
isort src/ --profile black
mypy src/

# Deploy to RLBot
rlbot run rlbot/bot.toml
```

## Architecture

```
Training Coordinator (Distributed PPO)
        │
        ├── Workers (16-64x) ──► Collect experience via RLGym environments
        │                         - AdvancedObsBuilder: 104-dim team-invariant obs
        │                         - MultiDiscreteAction: 1944 discrete actions
        │                         - Composite reward functions
        │
        └── Learner ◄───────────► Actor-Attention-Critic Network
                                    - CarEncoder/BallEncoder (MLPs)
                                    - Multi-head attention (4 heads, 2 layers)
                                    - Policy head (1944 actions) + Value head
```

**Key data flow:**
1. `scripts/train.py` → Hydra loads `configs/base.yaml` → `TrainingCoordinator`
2. Workers create RLGym envs via `environment/factory.py`
3. `AdvancedObsBuilder` produces 104-dim observations (19 self + 15 ball + 5×14 others)
4. `ActorAttentionCritic` encodes → attends → outputs action logits + value
5. PPO updates centrally, broadcasts weights to workers

## Key Files

- **`configs/base.yaml`**: All hyperparameters (191 lines) - training, network, rewards, curriculum
- **`src/rlbot_agent/models/actor_attention_critic.py`**: Main neural network with attention
- **`src/rlbot_agent/environment/obs_builders/advanced_obs.py`**: Observation construction
- **`src/rlbot_agent/algorithms/ppo/ppo.py`**: PPO trainer with GAE
- **`src/rlbot_agent/training/coordinator.py`**: Distributed training orchestration
- **`src/rlbot_agent/core/config.py`**: Dataclass configs (ObservationConfig, PPOConfig, etc.)
- **`src/rlbot_agent/core/types.py`**: Core types (CarState, BallState, RolloutBatch)

## Configuration System

Uses Hydra with YAML configs. Override hierarchy:
1. `configs/base.yaml` (defaults)
2. `configs/training/*.yaml` (match modes: 1v1, 2v2, 3v3)
3. `configs/rewards/*.yaml` (reward schedules)
4. CLI overrides (`key=value`)

Config dataclasses in `core/config.py` mirror YAML structure.

## Environment Components

All in `src/rlbot_agent/environment/`:
- **Rewards** (`rewards/`): TouchVelocity, VelocityBallToGoal, GoalReward, etc. - combined via `CombinedReward`
- **State Mutators** (`state_mutators/`): KickoffMutator, RandomStateMutator, ReplayStateMutator
- **Conditions** (`conditions/`): GoalCondition, TimeoutCondition, NoTouchCondition

## Curriculum Learning

Training progresses through 5 phases with different rewards and complexity:

| Phase | Steps | Active Rewards | Team Size |
|-------|-------|----------------|-----------|
| basic_movement | 0-100M | speed_toward_ball, touch_velocity | 1v1 |
| scoring | 100M-500M | + velocity_ball_to_goal, goal | 1v1 |
| advanced | 500M-1B | + aerial_height, demo | 1v1 |
| team_play | 1B-5B | + team_spacing_penalty | 2v2 |
| self_play | 5B-10B | goal only + historical checkpoints | 3v3 |

Curriculum is implemented in `training/callbacks/curriculum.py` and enabled by default.

## Self-Play

When the `self_play` curriculum phase is reached:
- `PolicyPool` (`training/policy_pool.py`) manages historical checkpoints
- Opponents are selected from recent checkpoints
- Win/loss statistics are tracked per checkpoint
- Environment wrapper (`environment/self_play_env.py`) routes actions to different policies

## Technical Details

- **Observation**: 104 dims, team-invariant (flipped for orange), normalized by field dims
- **Action space**: 3^5 × 2^3 = 1944 discrete (throttle/steer/pitch/yaw/roll × jump/boost/handbrake)
- **Tick skip**: 8 (15Hz decisions at 120Hz physics)
- **PPO**: clip=0.2, gamma=0.995, GAE lambda=0.95, entropy=0.01
- **Network**: 256 embed dim, 3 attention layers, 256 LSTM hidden
- **Scale**: 64 workers × 8 envs = 512 parallel environments
