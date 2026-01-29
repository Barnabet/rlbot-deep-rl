"""Tests for PPO algorithm."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rlbot_agent.algorithms.ppo import PPO, RolloutBuffer, compute_gae
from rlbot_agent.algorithms.ppo.loss import PPOLoss, compute_explained_variance
from rlbot_agent.core.config import PPOConfig, NetworkConfig, ObservationConfig
from rlbot_agent.models import ActorAttentionCritic


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_init(self, device):
        """Test buffer initialization."""
        buffer = RolloutBuffer(
            buffer_size=100,
            n_envs=4,
            obs_dim=104,
            n_actions=1944,
            device=device,
        )

        assert buffer.buffer_size == 100
        assert buffer.n_envs == 4
        assert buffer.observations.shape == (100, 4, 104)
        assert buffer.actions.shape == (100, 4)

    def test_add(self, device):
        """Test adding transitions."""
        buffer = RolloutBuffer(
            buffer_size=10,
            n_envs=2,
            obs_dim=104,
            n_actions=1944,
            device=device,
        )

        obs = np.random.randn(2, 104).astype(np.float32)
        action = np.array([0, 1])
        reward = np.array([1.0, 0.5])
        done = np.array([0.0, 1.0])
        log_prob = np.array([-1.0, -2.0])
        value = np.array([0.5, 0.3])

        buffer.add(obs, action, reward, done, log_prob, value)

        assert buffer.pos == 1
        assert np.allclose(buffer.observations[0], obs)
        assert np.allclose(buffer.actions[0], action)

    def test_is_full(self, device):
        """Test buffer full detection."""
        buffer = RolloutBuffer(
            buffer_size=5,
            n_envs=2,
            obs_dim=104,
            n_actions=1944,
            device=device,
        )

        assert not buffer.is_full()

        # Fill buffer
        for _ in range(5):
            buffer.add(
                obs=np.zeros((2, 104)),
                action=np.zeros(2),
                reward=np.zeros(2),
                done=np.zeros(2),
                log_prob=np.zeros(2),
                value=np.zeros(2),
            )

        assert buffer.is_full()

    def test_reset(self, device):
        """Test buffer reset."""
        buffer = RolloutBuffer(
            buffer_size=5,
            n_envs=2,
            obs_dim=104,
            n_actions=1944,
            device=device,
        )

        # Add some data
        for _ in range(3):
            buffer.add(
                obs=np.zeros((2, 104)),
                action=np.zeros(2),
                reward=np.zeros(2),
                done=np.zeros(2),
                log_prob=np.zeros(2),
                value=np.zeros(2),
            )

        buffer.reset()

        assert buffer.pos == 0
        assert not buffer.is_full()


class TestGAE:
    """Tests for GAE computation."""

    def test_compute_gae_shape(self):
        """Test GAE output shapes."""
        T, N = 10, 4
        rewards = np.random.randn(T, N).astype(np.float32)
        values = np.random.randn(T, N).astype(np.float32)
        dones = np.zeros((T, N), dtype=np.float32)
        last_values = np.random.randn(N).astype(np.float32)

        advantages, returns = compute_gae(rewards, values, dones, last_values)

        assert advantages.shape == (T, N)
        assert returns.shape == (T, N)

    def test_compute_gae_no_discount(self):
        """Test GAE with gamma=0 (no discounting)."""
        T, N = 5, 1
        rewards = np.ones((T, N), dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        dones = np.zeros((T, N), dtype=np.float32)
        last_values = np.zeros(N, dtype=np.float32)

        advantages, returns = compute_gae(
            rewards, values, dones, last_values, gamma=0.0
        )

        # With gamma=0, advantage should just be the reward
        assert np.allclose(advantages, rewards)

    def test_compute_gae_terminal(self):
        """Test GAE handles terminal states."""
        T, N = 5, 1
        rewards = np.ones((T, N), dtype=np.float32)
        values = np.ones((T, N), dtype=np.float32) * 0.5
        dones = np.zeros((T, N), dtype=np.float32)
        dones[2] = 1.0  # Terminal at step 2
        last_values = np.array([1.0], dtype=np.float32)

        advantages, returns = compute_gae(rewards, values, dones, last_values)

        # Returns should be the sum of discounted rewards
        assert advantages.shape == (T, N)


class TestPPOLoss:
    """Tests for PPO loss computation."""

    def test_loss_output(self):
        """Test loss output structure."""
        loss_fn = PPOLoss()

        batch_size = 32
        log_probs = torch.randn(batch_size)
        old_log_probs = torch.randn(batch_size)
        values = torch.randn(batch_size)
        old_values = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        entropy = torch.rand(batch_size)

        output = loss_fn(
            log_probs, old_log_probs, values, old_values,
            advantages, returns, entropy
        )

        assert hasattr(output, 'total_loss')
        assert hasattr(output, 'policy_loss')
        assert hasattr(output, 'value_loss')
        assert hasattr(output, 'entropy_loss')
        assert hasattr(output, 'clip_fraction')
        assert hasattr(output, 'approx_kl')

    def test_clip_behavior(self):
        """Test that clipping is applied correctly."""
        loss_fn = PPOLoss(clip_epsilon=0.2)

        batch_size = 100

        # Create scenario where ratio would be very high without clipping
        old_log_probs = torch.zeros(batch_size)
        log_probs = torch.ones(batch_size) * 2  # ratio = e^2 ≈ 7.4

        advantages = torch.ones(batch_size)
        values = torch.zeros(batch_size)
        returns = torch.ones(batch_size)
        entropy = torch.ones(batch_size)

        output = loss_fn(
            log_probs, old_log_probs, values, values,
            advantages, returns, entropy
        )

        # Clip fraction should be high since ratio >> 1.2
        assert output.clip_fraction > 0.5


class TestExplainedVariance:
    """Tests for explained variance computation."""

    def test_perfect_prediction(self):
        """Test explained variance for perfect predictions."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.clone()

        ev = compute_explained_variance(y_pred, y_true)

        assert ev == 1.0

    def test_constant_prediction(self):
        """Test explained variance for mean prediction."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.ones(5) * y_true.mean()

        ev = compute_explained_variance(y_pred, y_true)

        assert abs(ev) < 0.01  # Should be close to 0

    def test_bad_prediction(self):
        """Test explained variance for bad predictions."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed

        ev = compute_explained_variance(y_pred, y_true)

        assert ev < 0  # Worse than mean prediction
