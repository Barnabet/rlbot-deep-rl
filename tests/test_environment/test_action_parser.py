"""Tests for action parser."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rlbot_agent.environment.action_parsers import MultiDiscreteActionParser
from rlbot_agent.core.config import ActionConfig


class TestMultiDiscreteActionParser:
    """Tests for MultiDiscreteActionParser."""

    def test_action_space_size(self, action_config):
        """Test that action space is 1944."""
        parser = MultiDiscreteActionParser(config=action_config)

        assert parser.get_action_space_size() == 1944

    def test_action_table_shape(self, action_config):
        """Test action lookup table shape."""
        parser = MultiDiscreteActionParser(config=action_config)

        assert parser._action_table.shape == (1944, 8)

    def test_action_table_values(self, action_config):
        """Test that action table contains valid values."""
        parser = MultiDiscreteActionParser(config=action_config)

        # Throttle, steer, pitch, yaw, roll should be in [-1, 0, 1]
        for i in range(5):
            unique_vals = np.unique(parser._action_table[:, i])
            assert set(unique_vals) == {-1.0, 0.0, 1.0}

        # Jump, boost, handbrake should be in [0, 1]
        for i in range(5, 8):
            unique_vals = np.unique(parser._action_table[:, i])
            assert set(unique_vals) == {0.0, 1.0}

    def test_parse_actions(self, action_config):
        """Test action parsing."""
        parser = MultiDiscreteActionParser(config=action_config)

        # Test first action
        actions = np.array([0])
        controls = parser.parse_actions(actions, state=None)

        assert controls.shape == (1, 8)
        assert controls[0, 0] == -1.0  # First throttle option

    def test_action_to_indices(self, action_config):
        """Test action index decomposition."""
        parser = MultiDiscreteActionParser(config=action_config)

        # Test roundtrip
        for action in [0, 100, 500, 1000, 1943]:
            indices = parser.action_to_indices(action)
            reconstructed = parser.indices_to_action(indices)
            assert reconstructed == action

    def test_get_action_from_controls(self, action_config):
        """Test finding closest action from continuous controls."""
        parser = MultiDiscreteActionParser(config=action_config)

        # Test exact match
        action = parser.get_action_from_controls(
            throttle=1.0,
            steer=0.0,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            jump=False,
            boost=True,
            handbrake=False,
        )

        controls = parser._action_table[action]
        assert controls[0] == 1.0  # throttle
        assert controls[1] == 0.0  # steer
        assert controls[6] == 1.0  # boost

    def test_action_mask_jump(self, action_config):
        """Test action masking for jump."""
        parser = MultiDiscreteActionParser(config=action_config)

        # Create mock player that can jump
        class MockPlayer:
            on_ground = True
            has_flip = True
            boost_amount = 100

        player = MockPlayer()
        mask = parser.get_action_mask(player, state=None)

        # All actions should be valid
        assert mask.all()

        # Now test player in air without flip
        player.on_ground = False
        player.has_flip = False
        mask = parser.get_action_mask(player, state=None)

        # Jump actions should be disabled
        assert not mask[parser._jump_actions].all()
        # Non-jump actions should still be valid
        assert mask[~parser._jump_actions].all()

    def test_action_mask_boost(self, action_config):
        """Test action masking for empty boost."""
        parser = MultiDiscreteActionParser(config=action_config)

        class MockPlayer:
            on_ground = True
            has_flip = True
            boost_amount = 0

        player = MockPlayer()
        mask = parser.get_action_mask(player, state=None)

        # Boost actions should be disabled
        assert not mask[parser._boost_actions].all()
        # Non-boost actions should still be valid
        assert mask[~parser._boost_actions].all()
