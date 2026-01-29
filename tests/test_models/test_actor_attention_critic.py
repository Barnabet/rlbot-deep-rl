"""Tests for Actor-Attention-Critic model."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rlbot_agent.models import ActorAttentionCritic
from rlbot_agent.core.config import NetworkConfig, ObservationConfig


class TestActorAttentionCritic:
    """Tests for ActorAttentionCritic model."""

    def test_init(self, obs_config, network_config):
        """Test model initialization."""
        model = ActorAttentionCritic(
            obs_config=obs_config,
            network_config=network_config,
            n_actions=1944,
            max_players=6,
        )

        assert model.n_actions == 1944
        assert model.max_players == 6

    def test_obs_dim(self, obs_config, network_config):
        """Test observation dimension calculation."""
        model = ActorAttentionCritic(
            obs_config=obs_config,
            network_config=network_config,
        )

        expected_dim = (
            obs_config.self_car_dim
            + obs_config.ball_dim
            + 5 * obs_config.other_car_dim
        )

        assert model.obs_dim == expected_dim

    def test_forward_pass(self, model, sample_observation, device):
        """Test forward pass produces correct output shapes."""
        obs = torch.tensor(sample_observation, device=device).unsqueeze(0)

        logits, value = model(obs)

        assert logits.shape == (1, 1944)
        assert value.shape == (1, 1)

    def test_batch_forward(self, model, batch_observations, device):
        """Test forward pass with batch."""
        obs = torch.tensor(batch_observations, device=device)
        batch_size = obs.shape[0]

        logits, value = model(obs)

        assert logits.shape == (batch_size, 1944)
        assert value.shape == (batch_size, 1)

    def test_get_action(self, model, sample_observation, device):
        """Test action sampling."""
        obs = torch.tensor(sample_observation, device=device).unsqueeze(0)

        action, log_prob, entropy, value = model.get_action(obs)

        assert action.shape == (1,)
        assert 0 <= action.item() < 1944
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)

    def test_deterministic_action(self, model, sample_observation, device):
        """Test deterministic action selection."""
        obs = torch.tensor(sample_observation, device=device).unsqueeze(0)

        # Get deterministic action multiple times
        action1, _, _, _ = model.get_action(obs, deterministic=True)
        action2, _, _, _ = model.get_action(obs, deterministic=True)

        assert action1.item() == action2.item()

    def test_action_mask(self, model, sample_observation, sample_action_mask, device):
        """Test action masking."""
        obs = torch.tensor(sample_observation, device=device).unsqueeze(0)
        mask = torch.tensor(sample_action_mask, device=device).unsqueeze(0)

        # Get action with mask
        action, _, _, _ = model.get_action(obs, action_mask=mask)

        # Action should be valid according to mask
        assert sample_action_mask[action.item()]

    def test_evaluate_actions(self, model, batch_observations, device):
        """Test action evaluation."""
        obs = torch.tensor(batch_observations, device=device)
        batch_size = obs.shape[0]

        # Get some actions
        actions, _, _, _ = model.get_action(obs)

        # Evaluate them
        log_prob, entropy, value = model.evaluate_actions(obs, actions)

        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)

    def test_get_value(self, model, sample_observation, device):
        """Test value estimation only."""
        obs = torch.tensor(sample_observation, device=device).unsqueeze(0)

        value = model.get_value(obs)

        assert value.shape == (1,)

    def test_gradient_flow(self, model, batch_observations, device):
        """Test that gradients flow properly."""
        model.train()
        obs = torch.tensor(batch_observations, device=device)

        # Forward pass
        logits, value = model(obs)

        # Compute dummy loss
        loss = logits.mean() + value.mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parse_observation(self, model, sample_observation, device):
        """Test observation parsing."""
        obs = torch.tensor(sample_observation, device=device).unsqueeze(0)

        self_car, ball, other_cars = model._parse_observation(obs)

        assert self_car.shape == (1, 19)
        assert ball.shape == (1, 15)
        assert other_cars.shape == (1, 5, 14)


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_load_weights(self, model, tmp_path, device):
        """Test saving and loading model weights."""
        save_path = tmp_path / "test_model.pt"

        # Save
        torch.save(model.state_dict(), save_path)

        # Create new model and load
        new_model = ActorAttentionCritic(
            obs_config=ObservationConfig(),
            network_config=NetworkConfig(),
        ).to(device)

        new_model.load_state_dict(torch.load(save_path))

        # Check weights are the same
        for (name1, p1), (name2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(p1, p2)
