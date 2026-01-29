"""Base observation builder interface."""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from numpy.typing import NDArray


class BaseObsBuilder(ABC):
    """Abstract base class for observation builders."""

    @abstractmethod
    def reset(self, initial_state: Any) -> None:
        """Reset the observation builder for a new episode.

        Args:
            initial_state: Initial game state
        """
        pass

    @abstractmethod
    def build_obs(
        self, player: Any, state: Any, previous_action: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Build observation for a single player.

        Args:
            player: Player data
            state: Current game state
            previous_action: Previous action taken

        Returns:
            Observation array
        """
        pass

    @abstractmethod
    def get_obs_space_size(self) -> int:
        """Get the size of the observation space.

        Returns:
            Size of the flattened observation
        """
        pass
