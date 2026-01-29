# Deep RL Rocket League Bot

A competitive deep reinforcement learning bot for Rocket League using PPO with Actor-Attention-Critic architecture.

## Features

- **Actor-Attention-Critic Network**: Multi-head self-attention for handling variable numbers of players
- **1944 Discrete Actions**: Full action space with throttle, steer, pitch, yaw, roll, jump, boost, handbrake
- **Team-Invariant Observations**: Normalized, relative observations that work for both blue and orange teams
- **Curriculum Learning**: Progressive reward shaping from basic movement to team play
- **Distributed Training**: Multi-worker experience collection with central learner
- **RLBot Integration**: Ready to deploy in RLBot framework

## Architecture

```
Observation (104-dim) → Car/Ball Encoders → Self-Attention (4 heads, 2 layers)
                                                    ↓
                                    Policy Head → 1944 action logits
                                    Value Head  → state value
```

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/rlbot-deep-rl.git
cd rlbot-deep-rl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Training

```bash
# Basic training
python scripts/train.py

# With specific configuration
python scripts/train.py training=ppo_1v1 rewards=early_stage

# Override hyperparameters
python scripts/train.py ppo.learning_rate=5e-5 training.n_workers=16

# Single worker with rendering
python scripts/train.py render=true training.n_workers=1
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint data/checkpoints/latest.pt --n-episodes 100
```

### Deploy to RLBot

```bash
# Run with RLBot
rlbot run rlbot/bot.toml
```

## Project Structure

```
├── configs/                    # Hydra configuration files
│   ├── base.yaml              # Base configuration
│   ├── training/              # Training configs (1v1, 2v2, 3v3)
│   └── rewards/               # Reward schedules
├── src/rlbot_agent/
│   ├── core/                  # Config dataclasses, types, registry
│   ├── environment/
│   │   ├── obs_builders/      # Observation construction
│   │   ├── action_parsers/    # Action space handling
│   │   ├── rewards/           # Reward functions
│   │   ├── state_mutators/    # State initialization
│   │   └── conditions/        # Terminal conditions
│   ├── models/
│   │   ├── encoders/          # Car/ball state encoders
│   │   ├── attention/         # Multi-head attention
│   │   └── actor_attention_critic.py
│   ├── algorithms/
│   │   ├── ppo/               # PPO implementation
│   │   └── imitation/         # Behavioral cloning
│   ├── training/              # Training infrastructure
│   └── deployment/            # RLBot integration
├── scripts/
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   └── parse_replays.py       # Replay parsing
├── rlbot/                     # RLBot bot files
└── tests/                     # Unit tests
```

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-4 → 1e-5 | Linear annealing over 100M steps |
| Batch Size | 100,000 | Experience per update |
| Gamma | 0.995 | Discount factor |
| GAE Lambda | 0.95 | Advantage estimation |
| Clip Epsilon | 0.2 | PPO clipping |
| Entropy Coef | 0.01 | Exploration bonus |
| Tick Skip | 8 | 15Hz decisions at 120Hz physics |
| Attention | 4 heads, 2 layers | Transformer architecture |

## Training Curriculum

1. **Phase 1 (0-100M)**: 1v1, focus on ball interaction
2. **Phase 2 (100M-500M)**: Add scoring rewards
3. **Phase 3 (500M-1B)**: Aerials, demos, advanced mechanics
4. **Phase 4 (1B-5B)**: 2v2/3v3 with team spirit
5. **Phase 5 (5B-10B)**: Self-play with historical checkpoints

## Requirements

- Python 3.10+
- PyTorch 2.0+
- rlgym >= 2.0
- rlgym-tools >= 2.6
- CUDA (recommended for training)

## Cloud Training

For training on cloud GPUs (RunPod, Lambda Labs, etc.):

```bash
# Install with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e .

# Start training with more workers
python scripts/train.py training.n_workers=64 device=cuda
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT

## Acknowledgments

- [RLGym](https://rlgym.org/) - Rocket League Gym environment
- [RocketSim](https://github.com/ZealanL/RocketSim) - Rocket League physics simulation
- Inspired by Nexto, Necto, and other ML RLBot projects
