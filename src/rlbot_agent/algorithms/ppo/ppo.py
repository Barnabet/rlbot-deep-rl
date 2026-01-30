"""PPO (Proximal Policy Optimization) trainer."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import warnings
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from ...core.config import PPOConfig
from ...core.types import TrainingMetrics
from ...models import ActorAttentionCritic
from .buffer import RolloutBuffer
from .gae import GAE
from .loss import PPOLoss, compute_explained_variance


class PPO:
    """Proximal Policy Optimization algorithm.

    Implements the PPO-Clip algorithm with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus
    - Learning rate annealing
    """

    def __init__(
        self,
        model: ActorAttentionCritic,
        config: PPOConfig,
        device: torch.device,
    ):
        """Initialize PPO trainer.

        Args:
            model: Actor-Attention-Critic model
            config: PPO configuration
            device: PyTorch device
        """
        self.model = model
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

        # Learning rate scheduler with warmup then linear annealing
        # Warmup: 0.01 * LR -> LR over first few updates (prevents extreme first-step dynamics)
        # Anneal: LR -> lr_end over training
        warmup_steps = getattr(config, 'lr_warmup_steps', 3)

        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,  # Start at 1% of LR
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            anneal_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=config.lr_end / config.learning_rate,
                total_iters=config.lr_anneal_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, anneal_scheduler],
                milestones=[warmup_steps],
            )
        else:
            # No warmup, just annealing
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=config.lr_end / config.learning_rate,
                total_iters=config.lr_anneal_steps,
            )

        # Loss function
        self.loss_fn = PPOLoss(
            clip_epsilon=config.clip_epsilon,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
        )

        # GAE calculator
        self.gae = GAE(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize=config.normalize_advantages,
        )

        # Training state
        self.global_step = 0

    def collect_rollouts(
        self,
        env,
        buffer: RolloutBuffer,
        n_steps: int,
    ) -> Dict[str, float]:
        """Collect experience from environment.

        Args:
            env: Vectorized environment
            buffer: Rollout buffer to fill
            n_steps: Number of steps to collect

        Returns:
            Dictionary of collection metrics
        """
        buffer.reset()

        obs = env.reset() if buffer.pos == 0 else buffer._last_obs
        dones = np.zeros(env.num_envs, dtype=np.float32)

        total_rewards = []
        episode_rewards = np.zeros(env.num_envs)

        # Initialize hidden state for LSTM
        use_lstm = hasattr(self.model, 'use_lstm') and self.model.use_lstm
        if use_lstm:
            hidden = self.model.get_initial_hidden(env.num_envs, self.device)
        else:
            hidden = None

        self.model.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                # Get action from policy
                obs_tensor = torch.tensor(obs, device=self.device)
                action, log_prob, _, value, new_hidden = self.model.get_action(
                    obs_tensor, hidden
                )

                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

                # Store hidden state before updating
                if use_lstm and hidden is not None:
                    hidden_np = (
                        hidden[0].cpu().numpy(),
                        hidden[1].cpu().numpy(),
                    )
                else:
                    hidden_np = None

                # Environment step
                next_obs, rewards, next_dones, truncated, infos = env.step(action)
                dones_float = np.logical_or(next_dones, truncated).astype(np.float32)

                # Store transition
                buffer.add(
                    obs=obs,
                    action=action,
                    reward=rewards,
                    done=dones_float,
                    log_prob=log_prob,
                    value=value,
                    hidden_state=hidden_np,
                )

                # Update hidden state
                if use_lstm:
                    hidden = new_hidden
                    # Reset hidden for done environments
                    for i, done in enumerate(dones_float):
                        if done > 0.5:
                            hidden[0][:, i, :] = 0
                            hidden[1][:, i, :] = 0

                # Track episode rewards
                episode_rewards += rewards
                for i, done in enumerate(next_dones):
                    if done:
                        total_rewards.append(episode_rewards[i])
                        episode_rewards[i] = 0

                obs = next_obs
                dones = dones_float

        # Store last observation for bootstrapping
        buffer.set_last_obs(obs, dones)

        # Compute last values for GAE
        with torch.no_grad():
            last_values = self.model.get_value(
                torch.tensor(obs, device=self.device), hidden
            ).cpu().numpy()

        # Compute advantages and returns
        buffer.compute_returns_and_advantages(last_values)

        # Collection metrics
        metrics = {
            "rollout/mean_reward": np.mean(total_rewards) if total_rewards else 0.0,
            "rollout/episodes_completed": len(total_rewards),
        }

        return metrics

    def train_step(self, buffer: RolloutBuffer) -> TrainingMetrics:
        """Perform one training update.

        Args:
            buffer: Filled rollout buffer

        Returns:
            Training metrics
        """
        self.model.train()

        # Track metrics across epochs
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_clip_fractions = []
        all_approx_kl = []

        # Check if LSTM training mode
        use_lstm = hasattr(self.model, 'use_lstm') and self.model.use_lstm

        for epoch in range(self.config.n_epochs):
            if use_lstm and buffer.lstm_config is not None:
                # LSTM: Use sequence-based batching
                seq_len = buffer.lstm_config.sequence_length
                batch_size = max(1, self.config.minibatch_size // seq_len)

                for batch in buffer.get_sequences(batch_size, seq_len):
                    # batch.observations: [batch, seq_len, obs_dim]
                    # batch.hidden_states: (h, c) each [num_layers, batch, hidden]

                    # Evaluate actions with sequences
                    log_probs, entropy, values, _ = self.model.evaluate_actions(
                        batch.observations,
                        batch.actions,
                        batch.hidden_states,
                        batch.action_masks,
                    )
                    # log_probs, entropy, values: [batch, seq_len]

                    # Apply mask for episode boundaries
                    mask = batch.masks  # [batch, seq_len]
                    mask_flat = mask.view(-1)
                    valid_mask = mask_flat > 0.5

                    # Flatten for loss computation
                    log_probs_flat = log_probs.view(-1)[valid_mask]
                    old_log_probs_flat = batch.old_log_probs.view(-1)[valid_mask]
                    values_flat = values.view(-1)[valid_mask]
                    old_values_flat = batch.old_values.view(-1)[valid_mask]
                    advantages_flat = batch.advantages.view(-1)[valid_mask]
                    returns_flat = batch.returns.view(-1)[valid_mask]
                    entropy_flat = entropy.view(-1)[valid_mask]

                    # Compute loss
                    loss_output = self.loss_fn(
                        log_probs=log_probs_flat,
                        old_log_probs=old_log_probs_flat,
                        values=values_flat,
                        old_values=old_values_flat,
                        advantages=advantages_flat,
                        returns=returns_flat,
                        entropy=entropy_flat,
                    )

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss_output.total_loss.backward()

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    self.optimizer.step()

                    # Track metrics
                    all_policy_losses.append(loss_output.policy_loss.item())
                    all_value_losses.append(loss_output.value_loss.item())
                    all_entropy_losses.append(loss_output.entropy_loss.item())
                    all_clip_fractions.append(loss_output.clip_fraction)
                    all_approx_kl.append(loss_output.approx_kl)
            else:
                # Non-LSTM: Use standard random batching
                for batch in buffer.get(self.config.minibatch_size):
                    # Evaluate actions with current policy
                    log_probs, entropy, values, _ = self.model.evaluate_actions(
                        batch.observations,
                        batch.actions,
                        None,  # No hidden state
                        batch.action_masks,
                    )

                    # Compute loss
                    loss_output = self.loss_fn(
                        log_probs=log_probs,
                        old_log_probs=batch.old_log_probs,
                        values=values,
                        old_values=batch.old_values,
                        advantages=batch.advantages,
                        returns=batch.returns,
                        entropy=entropy,
                    )

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss_output.total_loss.backward()

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    self.optimizer.step()

                    # Track metrics
                    all_policy_losses.append(loss_output.policy_loss.item())
                    all_value_losses.append(loss_output.value_loss.item())
                    all_entropy_losses.append(loss_output.entropy_loss.item())
                    all_clip_fractions.append(loss_output.clip_fraction)
                    all_approx_kl.append(loss_output.approx_kl)

        # Update learning rate (suppress deprecation warning from SequentialLR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*epoch parameter.*")
            self.scheduler.step()
        self.global_step += buffer.total_samples

        # Compute explained variance
        with torch.no_grad():
            all_values = []
            all_returns = []
            for batch in buffer.get(buffer.total_samples):
                _, _, values, _ = self.model.evaluate_actions(
                    batch.observations, batch.actions, None, batch.action_masks
                )
                all_values.append(values)
                all_returns.append(batch.returns)

            values_tensor = torch.cat(all_values)
            returns_tensor = torch.cat(all_returns)
            explained_var = compute_explained_variance(values_tensor, returns_tensor)

        return TrainingMetrics(
            policy_loss=np.mean(all_policy_losses),
            value_loss=np.mean(all_value_losses),
            entropy_loss=np.mean(all_entropy_losses),
            total_loss=np.mean(all_policy_losses) + self.config.value_coef * np.mean(all_value_losses),
            clip_fraction=np.mean(all_clip_fractions),
            approx_kl=np.mean(all_approx_kl),
            explained_variance=explained_var,
            learning_rate=self.optimizer.param_groups[0]['lr'],
        )

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Try to load scheduler, but handle format changes gracefully
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not load scheduler state (format changed), using fresh scheduler")

        # Handle missing global_step (older checkpoints)
        self.global_step = checkpoint.get('global_step', 0)
        if 'global_step' not in checkpoint:
            print(f"Warning: Checkpoint missing global_step, starting from 0")

    def get_action(
        self,
        obs: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get action for inference.

        Args:
            obs: Observation [obs_dim] or [batch, obs_dim]
            hidden: Optional LSTM hidden state
            deterministic: If True, return mode

        Returns:
            Tuple of (action, log_prob, new_hidden)
        """
        self.model.eval()

        with torch.no_grad():
            if obs.ndim == 1:
                obs = obs[np.newaxis, :]

            obs_tensor = torch.tensor(obs, device=self.device)
            action, log_prob, _, _, new_hidden = self.model.get_action(
                obs_tensor, hidden, deterministic=deterministic
            )

            return action.cpu().numpy(), log_prob.cpu().numpy(), new_hidden
