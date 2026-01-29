"""Behavioral cloning from expert demonstrations."""

from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ...models import ActorAttentionCritic


class BehavioralCloning:
    """Behavioral cloning trainer.

    Trains the policy to imitate expert actions using supervised learning.
    Useful for pre-training before RL fine-tuning.
    """

    def __init__(
        self,
        model: ActorAttentionCritic,
        learning_rate: float = 1e-4,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize behavioral cloning trainer.

        Args:
            model: Policy model to train
            learning_rate: Learning rate
            device: PyTorch device
        """
        self.model = model.to(device)
        self.device = device

        self.optimizer = Adam(model.parameters(), lr=learning_rate)

    def train_step(
        self,
        observations: torch.Tensor,
        expert_actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> dict:
        """Perform one training step.

        Args:
            observations: Observation batch [batch, obs_dim]
            expert_actions: Expert action indices [batch]
            action_masks: Optional action masks [batch, n_actions]

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Forward pass
        logits, _ = self.model(observations, action_mask=action_masks)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, expert_actions)

        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == expert_actions).float().mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict:
        """Train for one epoch.

        Args:
            dataloader: DataLoader with (observations, actions) batches

        Returns:
            Dictionary of epoch metrics
        """
        total_loss = 0
        total_accuracy = 0
        n_batches = 0

        for batch in dataloader:
            if len(batch) == 2:
                obs, actions = batch
                masks = None
            else:
                obs, actions, masks = batch

            obs = obs.to(self.device)
            actions = actions.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)

            metrics = self.train_step(obs, actions, masks)

            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "accuracy": total_accuracy / n_batches,
        }

    def train(
        self,
        observations: np.ndarray,
        expert_actions: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
    ) -> dict:
        """Full training loop.

        Args:
            observations: All observations [n_samples, obs_dim]
            expert_actions: All expert actions [n_samples]
            n_epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation

        Returns:
            Training history
        """
        # Split into train/val
        n_samples = len(observations)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        # Create datasets
        train_obs = torch.tensor(observations[train_idx], dtype=torch.float32)
        train_actions = torch.tensor(expert_actions[train_idx], dtype=torch.long)
        train_dataset = TensorDataset(train_obs, train_actions)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_obs = torch.tensor(observations[val_idx], dtype=torch.float32)
        val_actions = torch.tensor(expert_actions[val_idx], dtype=torch.long)

        # Training loop
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        print(f"Training behavioral cloning for {n_epochs} epochs")
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")

        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.evaluate(val_obs, val_actions)

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )

        return history

    def evaluate(
        self,
        observations: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> dict:
        """Evaluate on a dataset.

        Args:
            observations: Observations [n_samples, obs_dim]
            expert_actions: Expert actions [n_samples]

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        observations = observations.to(self.device)
        expert_actions = expert_actions.to(self.device)

        with torch.no_grad():
            logits, _ = self.model(observations)
            loss = F.cross_entropy(logits, expert_actions)

            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == expert_actions).float().mean()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
