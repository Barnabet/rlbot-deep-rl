"""Multi-discrete action parser with 1944-action lookup table."""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ...core.config import ActionConfig
from ...core.registry import registry
from .base import BaseActionParser


@registry.register("action_parser", "multi_discrete")
class MultiDiscreteActionParser(BaseActionParser):
    """Action parser using a 1944-action lookup table.

    Action space breakdown:
    - Throttle: 3 options (-1, 0, 1)
    - Steer: 3 options (-1, 0, 1)
    - Pitch: 3 options (-1, 0, 1)
    - Yaw: 3 options (-1, 0, 1)
    - Roll: 3 options (-1, 0, 1)
    - Jump: 2 options (0, 1)
    - Boost: 2 options (0, 1)
    - Handbrake: 2 options (0, 1)

    Total: 3^5 × 2^3 = 243 × 8 = 1944 actions

    Action masking:
    - Disable jump if in air without flip available
    - Disable boost if boost amount is 0
    """

    def __init__(self, config: Optional[ActionConfig] = None):
        """Initialize the action parser.

        Args:
            config: Action configuration
        """
        self.config = config or ActionConfig()

        # Indices for jump and boost in the action table columns
        self._jump_idx = 5
        self._boost_idx = 6

        # Build action lookup table
        self._build_action_table()

    def _build_action_table(self) -> None:
        """Build the 1944-action lookup table."""
        throttle_opts = np.array(self.config.throttle_options, dtype=np.float32)
        steer_opts = np.array(self.config.steer_options, dtype=np.float32)
        pitch_opts = np.array(self.config.pitch_options, dtype=np.float32)
        yaw_opts = np.array(self.config.yaw_options, dtype=np.float32)
        roll_opts = np.array(self.config.roll_options, dtype=np.float32)
        jump_opts = np.array(self.config.jump_options, dtype=np.float32)
        boost_opts = np.array(self.config.boost_options, dtype=np.float32)
        handbrake_opts = np.array(self.config.handbrake_options, dtype=np.float32)

        # Store option sizes for action decomposition
        self._option_sizes = (
            len(throttle_opts),  # 3
            len(steer_opts),     # 3
            len(pitch_opts),     # 3
            len(yaw_opts),       # 3
            len(roll_opts),      # 3
            len(jump_opts),      # 2
            len(boost_opts),     # 2
            len(handbrake_opts), # 2
        )

        # Calculate total actions
        n_actions = 1
        for size in self._option_sizes:
            n_actions *= size
        assert n_actions == self.config.n_actions, f"Expected {self.config.n_actions}, got {n_actions}"

        # Build lookup table using meshgrid
        grids = np.meshgrid(
            throttle_opts, steer_opts, pitch_opts, yaw_opts, roll_opts,
            jump_opts, boost_opts, handbrake_opts,
            indexing='ij'
        )

        # Flatten and stack: shape (1944, 8)
        self._action_table = np.stack(
            [g.flatten() for g in grids], axis=1
        ).astype(np.float32)

        # Pre-compute which actions have jump=1 and boost=1 for masking
        self._jump_actions = self._action_table[:, self._jump_idx] == 1.0
        self._boost_actions = self._action_table[:, self._boost_idx] == 1.0

    def get_action_space_size(self) -> int:
        """Get the number of discrete actions."""
        return self.config.n_actions

    def parse_actions(
        self, actions: NDArray[np.int64], state: Any
    ) -> NDArray[np.float32]:
        """Convert discrete action indices to controller inputs.

        Args:
            actions: Array of action indices, shape (n_agents,)
            state: Current game state (unused, masking done separately)

        Returns:
            Array of controller inputs, shape (n_agents, 8)
        """
        return self._action_table[actions]

    def get_action_mask(
        self, player: Any, state: Any
    ) -> NDArray[np.bool_]:
        """Get action mask for valid actions.

        Args:
            player: Player data with car state
            state: Current game state

        Returns:
            Boolean mask where True = valid action
        """
        mask = np.ones(self.config.n_actions, dtype=np.bool_)

        # Mask jump if in air without flip
        if self.config.mask_jump_in_air:
            if not player.on_ground and not player.has_flip:
                mask[self._jump_actions] = False

        # Mask boost if empty
        if self.config.mask_boost_empty:
            if player.boost_amount <= 0:
                mask[self._boost_actions] = False

        return mask

    def action_to_indices(self, action: int) -> Tuple[int, ...]:
        """Decompose a flat action index into individual component indices.

        Args:
            action: Flat action index (0-1943)

        Returns:
            Tuple of indices for each action component
        """
        indices = []
        remaining = action

        for size in reversed(self._option_sizes):
            indices.append(remaining % size)
            remaining //= size

        return tuple(reversed(indices))

    def indices_to_action(self, indices: Tuple[int, ...]) -> int:
        """Compose individual component indices into a flat action index.

        Args:
            indices: Tuple of indices for each component

        Returns:
            Flat action index
        """
        action = 0
        multiplier = 1

        for idx, size in zip(reversed(indices), reversed(self._option_sizes)):
            action += idx * multiplier
            multiplier *= size

        return action

    def get_action_from_controls(
        self,
        throttle: float,
        steer: float,
        pitch: float,
        yaw: float,
        roll: float,
        jump: bool,
        boost: bool,
        handbrake: bool,
    ) -> int:
        """Find the closest discrete action to the given continuous controls.

        Args:
            throttle: -1 to 1
            steer: -1 to 1
            pitch: -1 to 1
            yaw: -1 to 1
            roll: -1 to 1
            jump: bool
            boost: bool
            handbrake: bool

        Returns:
            Closest discrete action index
        """
        def find_closest_idx(value: float, options: Tuple[float, ...]) -> int:
            return int(np.argmin(np.abs(np.array(options) - value)))

        indices = (
            find_closest_idx(throttle, self.config.throttle_options),
            find_closest_idx(steer, self.config.steer_options),
            find_closest_idx(pitch, self.config.pitch_options),
            find_closest_idx(yaw, self.config.yaw_options),
            find_closest_idx(roll, self.config.roll_options),
            int(jump),
            int(boost),
            int(handbrake),
        )

        return self.indices_to_action(indices)
