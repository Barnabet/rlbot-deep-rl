"""Policy pool for self-play with historical checkpoints."""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..core.config import NetworkConfig, ObservationConfig
from ..deployment.inference import InferenceEngine
from ..models import ActorAttentionCritic


@dataclass
class PolicyMetadata:
    """Metadata for a policy checkpoint in the pool."""

    checkpoint_path: str
    step: int
    win_rate: float = 0.5
    games_played: int = 0
    wins: int = 0
    losses: int = 0

    def update_result(self, won: bool) -> None:
        """Update win/loss statistics.

        Args:
            won: Whether the policy won
        """
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.win_rate = self.wins / self.games_played if self.games_played > 0 else 0.5


class PolicyPool:
    """Manages a pool of historical policy checkpoints for self-play.

    Features:
    - Maintains pool of recent checkpoints
    - Selects opponents from most recent checkpoints
    - Tracks win rates against each checkpoint
    - Lazy loading of inference engines
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_pool_size: int = 10,
        device: str = "cpu",
        obs_config: Optional[ObservationConfig] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initialize policy pool.

        Args:
            checkpoint_dir: Directory containing checkpoints
            max_pool_size: Maximum number of policies in pool
            device: PyTorch device for inference
            obs_config: Observation config for loading models
            network_config: Network config for loading models
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_pool_size = max_pool_size
        self.device = device
        self.obs_config = obs_config or ObservationConfig()
        self.network_config = network_config or NetworkConfig()

        # Policy pool: ordered by step (newest last)
        self.policies: List[PolicyMetadata] = []

        # Loaded inference engines (lazy loading)
        self._loaded_engines: Dict[str, InferenceEngine] = {}

        # Current opponent
        self._current_opponent_path: Optional[str] = None

        # Scan for existing checkpoints
        self._scan_checkpoints()

    def _scan_checkpoints(self) -> None:
        """Scan checkpoint directory for existing checkpoints."""
        if not self.checkpoint_dir.exists():
            return

        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.split('_')[1]),
        )

        for checkpoint_path in checkpoints[-self.max_pool_size:]:
            step = int(checkpoint_path.stem.split('_')[1])
            self.policies.append(PolicyMetadata(
                checkpoint_path=str(checkpoint_path),
                step=step,
            ))

        if self.policies:
            print(f"PolicyPool: Found {len(self.policies)} existing checkpoints")

    def add_checkpoint(self, checkpoint_path: str, step: int) -> None:
        """Add a checkpoint to the pool.

        Args:
            checkpoint_path: Path to the checkpoint file
            step: Training step of the checkpoint
        """
        # Check if already in pool
        for policy in self.policies:
            if policy.checkpoint_path == checkpoint_path:
                return

        # Add new policy
        metadata = PolicyMetadata(
            checkpoint_path=checkpoint_path,
            step=step,
        )
        self.policies.append(metadata)

        # Sort by step
        self.policies.sort(key=lambda p: p.step)

        # Evict oldest if at capacity
        while len(self.policies) > self.max_pool_size:
            oldest = self.policies.pop(0)
            # Unload engine if loaded
            if oldest.checkpoint_path in self._loaded_engines:
                del self._loaded_engines[oldest.checkpoint_path]

        print(f"PolicyPool: Added checkpoint at step {step:,} (pool size: {len(self.policies)})")

    def select_opponent(self, strategy: str = 'recent') -> Optional[InferenceEngine]:
        """Select an opponent policy from the pool.

        Args:
            strategy: Selection strategy ('recent', 'uniform', 'weighted')
                - recent: Prefer most recent checkpoints
                - uniform: Random uniform selection
                - weighted: Weight by inverse win rate (harder opponents)

        Returns:
            Inference engine for the selected opponent, or None if pool is empty
        """
        if not self.policies:
            return None

        if strategy == 'recent':
            # Select from recent checkpoints with higher probability for newer ones
            # Use exponential weighting: recent checkpoints are more likely
            n = len(self.policies)
            weights = [2 ** i for i in range(n)]  # Exponential weights
            selected = random.choices(self.policies, weights=weights, k=1)[0]

        elif strategy == 'uniform':
            selected = random.choice(self.policies)

        elif strategy == 'weighted':
            # Weight by inverse win rate (prefer challenging opponents)
            # Add epsilon to avoid division by zero
            weights = [1.0 / (p.win_rate + 0.1) for p in self.policies]
            selected = random.choices(self.policies, weights=weights, k=1)[0]

        else:
            # Default to recent
            selected = self.policies[-1]

        self._current_opponent_path = selected.checkpoint_path
        return self._load_engine(selected.checkpoint_path)

    def get_current_opponent(self) -> Optional[InferenceEngine]:
        """Get the currently selected opponent.

        Returns:
            Current opponent's inference engine, or None if not selected
        """
        if self._current_opponent_path is None:
            return self.select_opponent()
        return self._load_engine(self._current_opponent_path)

    def _load_engine(self, checkpoint_path: str) -> InferenceEngine:
        """Load or retrieve cached inference engine.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Inference engine
        """
        if checkpoint_path not in self._loaded_engines:
            print(f"PolicyPool: Loading opponent from {checkpoint_path}")
            engine = self._create_engine(checkpoint_path)
            self._loaded_engines[checkpoint_path] = engine

        return self._loaded_engines[checkpoint_path]

    def _create_engine(self, checkpoint_path: str) -> InferenceEngine:
        """Create inference engine from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            New inference engine
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = ActorAttentionCritic(
            obs_config=self.obs_config,
            network_config=self.network_config,
            n_actions=1944,
            max_players=6,
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        return InferenceEngine(model, self.device, use_jit=False)

    def update_result(self, won: bool) -> None:
        """Update win/loss statistics for current opponent.

        Args:
            won: Whether the training policy won against the opponent
        """
        if self._current_opponent_path is None:
            return

        for policy in self.policies:
            if policy.checkpoint_path == self._current_opponent_path:
                # Note: opponent's perspective is inverted
                policy.update_result(not won)
                break

    def get_stats(self) -> Dict[str, float]:
        """Get pool statistics for logging.

        Returns:
            Dictionary of statistics
        """
        if not self.policies:
            return {
                'selfplay/pool_size': 0,
                'selfplay/avg_opponent_win_rate': 0.0,
            }

        avg_win_rate = sum(p.win_rate for p in self.policies) / len(self.policies)
        total_games = sum(p.games_played for p in self.policies)

        stats = {
            'selfplay/pool_size': float(len(self.policies)),
            'selfplay/avg_opponent_win_rate': avg_win_rate,
            'selfplay/total_opponent_games': float(total_games),
        }

        if self._current_opponent_path:
            for policy in self.policies:
                if policy.checkpoint_path == self._current_opponent_path:
                    stats['selfplay/current_opponent_step'] = float(policy.step)
                    stats['selfplay/current_opponent_win_rate'] = policy.win_rate
                    break

        return stats

    def get_newest_step(self) -> int:
        """Get the step of the newest checkpoint in pool.

        Returns:
            Step number or 0 if pool is empty
        """
        if not self.policies:
            return 0
        return self.policies[-1].step

    def clear(self) -> None:
        """Clear the policy pool and unload all engines."""
        self.policies.clear()
        self._loaded_engines.clear()
        self._current_opponent_path = None
