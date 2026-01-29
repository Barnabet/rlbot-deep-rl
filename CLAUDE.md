# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep reinforcement learning framework for training competitive Rocket League bots using PPO with attention-based neural networks. Integrates with RLGym 2.0 for simulation and RLBot for deployment.

## Commands

```bash
# Install (editable mode)
pip install -e .

# Training
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

## Technical Details

- **Observation**: 104 dims, team-invariant (flipped for orange), normalized by field dims
- **Action space**: 3^5 × 2^3 = 1944 discrete (throttle/steer/pitch/yaw/roll × jump/boost/handbrake)
- **Tick skip**: 8 (15Hz decisions at 120Hz physics)
- **PPO**: clip=0.2, gamma=0.995, GAE lambda=0.95, entropy=0.01
