"""Base action parser interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseActionParser(ABC):
    """Abstract base class for action parsers."""

    @abstractmethod
    def get_action_space_size(self) -> int:
        """Get the size of the action space.

        Returns:
            Number of discrete actions
        """
        pass

    @abstractmethod
    def parse_actions(
        self, actions: NDArray[np.int64], state: Any
    ) -> NDArray[np.float32]:
        """Convert discrete action indices to controller inputs.

        Args:
            actions: Array of discrete action indices, shape (n_agents,)
            state: Current game state for action masking context

        Returns:
            Array of controller inputs, shape (n_agents, 8)
            [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        pass

    @abstractmethod
    def get_action_mask(
        self, player: Any, state: Any
    ) -> NDArray[np.bool_]:
        """Get action mask for valid actions.

        Args:
            player: Player data
            state: Current game state

        Returns:
            Boolean mask of valid actions, shape (n_actions,)
        """
        pass
