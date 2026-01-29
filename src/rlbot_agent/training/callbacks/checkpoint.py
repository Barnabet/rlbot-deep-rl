"""Checkpoint saving callback."""

import os
from pathlib import Path
from typing import Any, Optional

import torch


class CheckpointCallback:
    """Callback for saving model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 10_000_000,
        keep_last_n: int = 5,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Steps between checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_save_step = 0

    def on_step(
        self,
        step: int,
        trainer: Any,
        metrics: Optional[dict] = None,
    ) -> None:
        """Called after each training step.

        Args:
            step: Current global step
            trainer: PPO trainer instance
            metrics: Training metrics
        """
        if step - self._last_save_step >= self.save_interval:
            self.save_checkpoint(step, trainer, metrics)
            self._last_save_step = step
            self._cleanup_old_checkpoints()

    def save_checkpoint(
        self,
        step: int,
        trainer: Any,
        metrics: Optional[dict] = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            step: Current global step
            trainer: PPO trainer instance
            metrics: Training metrics

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:012d}.pt"

        checkpoint = {
            'step': step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'metrics': metrics or {},
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Also save as 'latest.pt' for easy resumption
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        trainer: Any,
        path: Optional[str] = None,
    ) -> int:
        """Load a checkpoint.

        Args:
            trainer: PPO trainer instance
            path: Path to checkpoint (uses latest if None)

        Returns:
            Step number from checkpoint
        """
        if path is None:
            path = self.checkpoint_dir / "latest.pt"

        if not Path(path).exists():
            print(f"No checkpoint found at {path}")
            return 0

        checkpoint = torch.load(path, map_location=trainer.device)

        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        step = checkpoint.get('step', 0)
        print(f"Loaded checkpoint from {path} (step {step})")

        return step

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.split('_')[1]),
        )

        while len(checkpoints) > self.keep_last_n:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"Removed old checkpoint: {oldest}")
