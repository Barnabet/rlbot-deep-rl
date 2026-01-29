"""Pytest fixtures for testing."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlbot_agent.core.config import (
    ActionConfig,
    AttentionConfig,
    EncoderConfig,
    EnvironmentConfig,
    NetworkConfig,
    ObservationConfig,
    PPOConfig,
    RewardConfig,
)
from rlbot_agent.models import ActorAttentionCritic


@pytest.fixture
def obs_config():
    """Default observation configuration."""
    return ObservationConfig()


@pytest.fixture
def action_config():
    """Default action configuration."""
    return ActionConfig()


@pytest.fixture
def network_config():
    """Default network configuration."""
    return NetworkConfig()


@pytest.fixture
def ppo_config():
    """Default PPO configuration."""
    return PPOConfig()


@pytest.fixture
def reward_config():
    """Default reward configuration."""
    return RewardConfig()


@pytest.fixture
def env_config():
    """Default environment configuration."""
    return EnvironmentConfig()


@pytest.fixture
def device():
    """PyTorch device for testing."""
    return torch.device("cpu")


@pytest.fixture
def model(obs_config, network_config, device):
    """Create a test model."""
    model = ActorAttentionCritic(
        obs_config=obs_config,
        network_config=network_config,
        n_actions=1944,
        max_players=6,
    ).to(device)
    return model


@pytest.fixture
def sample_observation(obs_config):
    """Create a sample observation."""
    obs_dim = (
        obs_config.self_car_dim
        + obs_config.ball_dim
        + 5 * obs_config.other_car_dim
    )
    return np.random.randn(obs_dim).astype(np.float32)


@pytest.fixture
def batch_observations(obs_config):
    """Create a batch of observations."""
    obs_dim = (
        obs_config.self_car_dim
        + obs_config.ball_dim
        + 5 * obs_config.other_car_dim
    )
    batch_size = 32
    return np.random.randn(batch_size, obs_dim).astype(np.float32)


@pytest.fixture
def sample_action_mask():
    """Create a sample action mask."""
    mask = np.ones(1944, dtype=np.bool_)
    # Disable some random actions
    mask[np.random.choice(1944, 100, replace=False)] = False
    return mask
