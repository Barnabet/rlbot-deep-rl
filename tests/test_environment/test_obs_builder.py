"""Tests for observation builder."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rlbot_agent.environment.obs_builders import AdvancedObsBuilder
from rlbot_agent.core.config import ObservationConfig


class TestAdvancedObsBuilder:
    """Tests for AdvancedObsBuilder."""

    def test_init(self, obs_config):
        """Test initialization."""
        builder = AdvancedObsBuilder(config=obs_config, max_players=6)

        assert builder.self_car_size == obs_config.self_car_dim
        assert builder.other_car_size == obs_config.other_car_dim
        assert builder.ball_size == obs_config.ball_dim

    def test_obs_space_size(self, obs_config):
        """Test observation space size calculation."""
        builder = AdvancedObsBuilder(config=obs_config, max_players=6)

        expected_size = (
            obs_config.self_car_dim  # 24 (with goal features + velocity magnitude)
            + obs_config.ball_dim  # 19 (with goal features + velocity magnitude)
            + 5 * obs_config.other_car_dim  # 5 * 14 = 70
        )

        assert builder.get_obs_space_size() == expected_size
        # Total: 24 + 19 + 70 = 113
        assert builder.get_obs_space_size() == 113

    def test_position_normalization(self, obs_config):
        """Test position normalization."""
        builder = AdvancedObsBuilder(config=obs_config)

        # Test position at field boundaries
        pos = np.array([4096.0, 5120.0, 2048.0])
        normalized = builder._normalize_position(pos, flip=False)

        assert np.allclose(normalized, [1.0, 1.0, 1.0])

        # Test position at center
        pos_center = np.array([0.0, 0.0, 1024.0])
        normalized_center = builder._normalize_position(pos_center, flip=False)

        assert np.allclose(normalized_center, [0.0, 0.0, 0.5])

    def test_velocity_normalization(self, obs_config):
        """Test velocity normalization."""
        builder = AdvancedObsBuilder(config=obs_config)

        vel = np.array([2300.0, 0.0, 0.0])
        normalized = builder._normalize_velocity(vel, flip=False)

        assert np.allclose(normalized, [1.0, 0.0, 0.0])

    def test_flip_for_orange(self, obs_config):
        """Test team-side invariance (flipping for orange team)."""
        builder = AdvancedObsBuilder(config=obs_config)

        pos = np.array([1000.0, 2000.0, 500.0])

        # Blue team (no flip)
        norm_blue = builder._normalize_position(pos, flip=False)

        # Orange team (flip)
        norm_orange = builder._normalize_position(pos, flip=True)

        # X and Y should be negated
        assert np.isclose(norm_blue[0], -norm_orange[0])
        assert np.isclose(norm_blue[1], -norm_orange[1])
        # Z should be same
        assert np.isclose(norm_blue[2], norm_orange[2])

    def test_rotation_sincos(self, obs_config):
        """Test rotation to sin/cos conversion."""
        builder = AdvancedObsBuilder(config=obs_config)

        # Test zero rotation
        sincos = builder._rotation_to_sincos(0.0, 0.0, 0.0, flip=False)

        expected = np.array([
            np.sin(0), np.cos(0),  # pitch: 0, 1
            np.sin(0), np.cos(0),  # yaw: 0, 1
            np.sin(0), np.cos(0),  # roll: 0, 1
        ])

        assert np.allclose(sincos, expected)

        # Test 90 degree yaw
        sincos_90 = builder._rotation_to_sincos(0.0, np.pi/2, 0.0, flip=False)

        assert np.isclose(sincos_90[2], 1.0)  # sin(pi/2)
        assert np.isclose(sincos_90[3], 0.0, atol=1e-6)  # cos(pi/2)


class TestObsBuilderConfig:
    """Tests for observation configuration."""

    def test_default_dims(self):
        """Test default dimension values (with goal features + velocity magnitudes)."""
        config = ObservationConfig()

        # New dimensions include goal features and velocity magnitudes:
        # self_car: 19 base + 1 speed + 4 goal features = 24
        # ball: 15 base + 1 speed + 3 goal features = 19
        assert config.self_car_dim == 24
        assert config.other_car_dim == 14
        assert config.ball_dim == 19

    def test_custom_config(self):
        """Test custom configuration."""
        config = ObservationConfig(
            pos_norm=(4000.0, 5000.0, 2000.0),
            vel_norm=2500.0,
        )

        builder = AdvancedObsBuilder(config=config)

        assert builder.config.vel_norm == 2500.0
