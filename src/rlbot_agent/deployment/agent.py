"""RLBot agent for deployment."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..core.config import NetworkConfig, ObservationConfig
from ..core.types import ControllerInput
from ..environment.action_parsers import MultiDiscreteActionParser
from ..environment.obs_builders import AdvancedObsBuilder
from ..models import ActorAttentionCritic


class RLBotAgent:
    """Agent for deployment in RLBot framework.

    Handles:
    - Loading trained model checkpoints
    - Converting game state to observations
    - Running inference at 120Hz
    - Converting actions to controller inputs
    """

    # Action repeat (tick skip) - cache action for this many ticks
    TICK_SKIP = 8

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        deterministic: bool = True,
    ):
        """Initialize RLBot agent.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: PyTorch device ('cpu' recommended for RLBot)
            deterministic: If True, use mode instead of sampling
        """
        self.device = torch.device(device)
        self.deterministic = deterministic

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract config from checkpoint or use defaults
        config = checkpoint.get('config', {})

        self.obs_config = ObservationConfig()
        self.network_config = NetworkConfig()

        # Create model
        self.model = ActorAttentionCritic(
            obs_config=self.obs_config,
            network_config=self.network_config,
            n_actions=1944,
            max_players=6,
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Create observation builder and action parser
        self.obs_builder = AdvancedObsBuilder(config=self.obs_config, max_players=6)
        self.action_parser = MultiDiscreteActionParser()

        # Action caching
        self._tick_count = 0
        self._cached_action: Optional[int] = None
        self._cached_controls: Optional[ControllerInput] = None

    def get_output(self, game_state: any) -> ControllerInput:
        """Get controller output for current game state.

        Called at 120Hz by RLBot.

        Args:
            game_state: RLBot GameTickPacket

        Returns:
            Controller input for this tick
        """
        # Use cached action if within tick skip
        if self._tick_count < self.TICK_SKIP and self._cached_controls is not None:
            self._tick_count += 1
            return self._cached_controls

        # Time to compute new action
        self._tick_count = 0

        # Convert game state to observation
        obs = self._build_observation(game_state)

        # Run inference
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
            action, _, _, _ = self.model.get_action(
                obs_tensor, deterministic=self.deterministic
            )
            action = action.item()

        # Convert action to controls
        controls = self._action_to_controls(action)

        # Cache
        self._cached_action = action
        self._cached_controls = controls
        self._tick_count = 1

        return controls

    def _build_observation(self, game_state: any) -> np.ndarray:
        """Build observation from RLBot game state.

        Args:
            game_state: RLBot GameTickPacket

        Returns:
            Flattened observation array
        """
        # This would need to be implemented based on RLBot's API
        # For now, return a placeholder
        return np.zeros(self.obs_builder.get_obs_space_size(), dtype=np.float32)

    def _action_to_controls(self, action: int) -> ControllerInput:
        """Convert discrete action to controller input.

        Args:
            action: Discrete action index (0-1943)

        Returns:
            Controller input
        """
        # Get controller values from action table
        controls_array = self.action_parser._action_table[action]

        return ControllerInput(
            throttle=float(controls_array[0]),
            steer=float(controls_array[1]),
            pitch=float(controls_array[2]),
            yaw=float(controls_array[3]),
            roll=float(controls_array[4]),
            jump=bool(controls_array[5]),
            boost=bool(controls_array[6]),
            handbrake=bool(controls_array[7]),
        )

    @staticmethod
    def load_from_checkpoint(
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "RLBotAgent":
        """Load agent from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint
            device: PyTorch device

        Returns:
            Loaded agent
        """
        return RLBotAgent(checkpoint_path=checkpoint_path, device=device)


class RLBotAgentV2(RLBotAgent):
    """Extended agent with RLBot v5 API support."""

    def initialize_agent(self) -> None:
        """Called once when agent is loaded."""
        print(f"Agent initialized on device: {self.device}")
        print(f"Model loaded: {self.model.__class__.__name__}")

    def get_controls(self, game_state: any) -> any:
        """Get controller state for RLBot v5.

        Args:
            game_state: Game state from RLBot

        Returns:
            ControllerState for RLBot
        """
        controls = self.get_output(game_state)

        # Convert to RLBot ControllerState format
        # This would use RLBot's actual ControllerState class
        return {
            'throttle': controls.throttle,
            'steer': controls.steer,
            'pitch': controls.pitch,
            'yaw': controls.yaw,
            'roll': controls.roll,
            'jump': controls.jump,
            'boost': controls.boost,
            'handbrake': controls.handbrake,
        }
