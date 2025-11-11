"""PPO trainer for RL finetuning of VLA policy."""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class RolloutBatch:
    """Batch of rollout data for PPO training."""
    rgb_static: torch.Tensor  # (batch, 3, H, W)
    instructions: List[str]  # (batch,)
    proprio: torch.Tensor  # (batch, 8)
    action_history: torch.Tensor  # (batch, history_len, action_dim)
    actions: torch.Tensor  # (batch, action_dim)
    log_probs: torch.Tensor  # (batch,)
    values: torch.Tensor  # (batch, 1)
    rewards: torch.Tensor  # (batch,)
    dones: torch.Tensor  # (batch,)
    advantages: torch.Tensor  # (batch,)
    returns: torch.Tensor  # (batch,)


class RolloutBuffer:
    """Buffer for storing rollout trajectories."""

    def __init__(
        self,
        buffer_size: int,
        action_dim: int,
        history_length: int,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.action_dim = action_dim
        self.history_length = history_length
        self.device = device
        self.reset()

    def reset(self):
        """Reset buffer to empty state."""
        self.rgb_buffer = []
        self.instruction_buffer = []
        self.proprio_buffer = []
        self.action_history_buffer = []
        self.action_buffer = []
        self.log_prob_buffer = []
        self.value_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.pos = 0

    def add(
        self,
        rgb: torch.Tensor,
        instruction: str,
        proprio: torch.Tensor,
        action_history: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ):
        """Add a single transition to the buffer."""
        self.rgb_buffer.append(rgb.cpu())
        self.instruction_buffer.append(instruction)
        self.proprio_buffer.append(proprio.cpu())
        self.action_history_buffer.append(action_history.cpu())
        self.action_buffer.append(action.cpu())
        self.log_prob_buffer.append(log_prob.cpu())
        self.value_buffer.append(value.cpu())
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.pos += 1

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for the last state (for bootstrap)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Convert lists to tensors
        rewards = torch.tensor(self.reward_buffer, dtype=torch.float32)
        values = torch.cat(self.value_buffer, dim=0).squeeze(-1)
        dones = torch.tensor(self.done_buffer, dtype=torch.float32)

        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value.squeeze()  # Fully squeeze to scalar
            else:
                next_value = values[t + 1]

            # TD error: r + gamma * V(s') - V(s)
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

            # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae_lambda
            last_gae_lambda = advantages[t]

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def get(self, advantages: torch.Tensor, returns: torch.Tensor) -> RolloutBatch:
        """Get all data from buffer as a batch.

        Args:
            advantages: Computed advantages
            returns: Computed returns

        Returns:
            RolloutBatch with all stored data
        """
        return RolloutBatch(
            rgb_static=torch.stack(self.rgb_buffer).to(self.device),
            instructions=self.instruction_buffer,
            proprio=torch.stack(self.proprio_buffer).to(self.device),
            action_history=torch.stack(self.action_history_buffer).to(self.device),
            actions=torch.stack(self.action_buffer).to(self.device),
            log_probs=torch.stack(self.log_prob_buffer).to(self.device),
            values=torch.cat(self.value_buffer, dim=0).to(self.device),
            rewards=torch.tensor(self.reward_buffer, dtype=torch.float32).to(self.device),
            dones=torch.tensor(self.done_buffer, dtype=torch.float32).to(self.device),
            advantages=advantages.to(self.device),
            returns=returns.to(self.device),
        )

    def __len__(self):
        return self.pos


class PPOTrainer:
    """PPO trainer for VLA policy."""

    def __init__(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        action_std: float = 0.1,
        target_kl: float = 0.01,
    ):
        """Initialize PPO trainer.

        Args:
            policy: VLA policy model with actor-critic heads
            optimizer: Optimizer for policy parameters
            clip_range: PPO clipping parameter
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            action_std: Standard deviation for action exploration
            target_kl: Target KL divergence for early stopping
        """
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.action_std = action_std
        self.target_kl = target_kl

    def compute_ppo_loss(
        self,
        batch: RolloutBatch,
        old_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute PPO loss.

        Args:
            batch: Batch of rollout data
            old_log_probs: Log probabilities from old policy

        Returns:
            Tuple of (total_loss, info_dict)
        """
        # Evaluate actions under current policy
        new_log_probs, new_values, entropy = self.policy.evaluate_actions(
            rgb_static=batch.rgb_static,
            instruction=batch.instructions,
            proprio=batch.proprio,
            actions=batch.actions,
            action_history=batch.action_history,
            action_std=self.action_std,
        )

        # Compute ratio for PPO
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Normalize advantages
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        new_values = new_values.squeeze(-1)
        value_pred_clipped = batch.values.squeeze(-1) + torch.clamp(
            new_values - batch.values.squeeze(-1),
            -self.clip_range,
            self.clip_range,
        )
        value_loss_unclipped = (new_values - batch.returns) ** 2
        value_loss_clipped = (value_pred_clipped - batch.returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # Entropy loss (for exploration)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # Compute approximate KL divergence for early stopping
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            clip_fraction = ((ratio - 1).abs() > self.clip_range).float().mean()

        info = {
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "loss/total": total_loss.item(),
            "policy/approx_kl": approx_kl.item(),
            "policy/clip_fraction": clip_fraction.item(),
            "policy/entropy": -entropy_loss.item(),
        }

        return total_loss, info, approx_kl

    def train_step(
        self,
        batch: RolloutBatch,
        num_epochs: int = 4,
        batch_size: int = 64,
    ) -> dict:
        """Perform PPO update on a batch of rollouts.

        Args:
            batch: Batch of rollout data
            num_epochs: Number of epochs to train on the batch
            batch_size: Minibatch size for updates

        Returns:
            Dictionary of training metrics
        """
        # Store old log probs (detached from graph)
        old_log_probs = batch.log_probs.detach()

        all_metrics = []

        # Train for multiple epochs on the same batch
        for epoch in range(num_epochs):
            # Create random minibatches
            indices = torch.randperm(len(batch.rgb_static))

            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]

                # Create minibatch
                mb = RolloutBatch(
                    rgb_static=batch.rgb_static[mb_indices],
                    instructions=[batch.instructions[i.item()] for i in mb_indices],
                    proprio=batch.proprio[mb_indices],
                    action_history=batch.action_history[mb_indices],
                    actions=batch.actions[mb_indices],
                    log_probs=batch.log_probs[mb_indices],
                    values=batch.values[mb_indices],
                    rewards=batch.rewards[mb_indices],
                    dones=batch.dones[mb_indices],
                    advantages=batch.advantages[mb_indices],
                    returns=batch.returns[mb_indices],
                )

                # Compute loss and update
                loss, info, approx_kl = self.compute_ppo_loss(mb, old_log_probs[mb_indices])

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                all_metrics.append(info)

                # Early stopping if KL divergence is too high
                if approx_kl > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch} due to high KL divergence: {approx_kl:.4f}")
                    break

            # Check for early stopping after each epoch
            if approx_kl > 1.5 * self.target_kl:
                break

        # Average metrics across all minibatches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics
