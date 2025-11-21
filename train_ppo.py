"""PPO training script for RL finetuning of VLA policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env.mujoco_env import FrankaPickPlaceEnv
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from rl.ppo_trainer import PPOTrainer, RolloutBuffer
from utils.config import load_config, save_config
from utils.logging import get_logger, setup_logging
from utils.seed import seed_everything


class ActionHistoryTracker:
    """Tracks action history for temporal context."""

    def __init__(self, history_length: int, action_dim: int, device: torch.device):
        self.history_length = history_length
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        """Reset history to zeros."""
        self.history = torch.zeros(self.history_length, self.action_dim, device=self.device)

    def update(self, action: torch.Tensor):
        """Update history with new action."""
        # Shift history and add new action
        self.history = torch.cat([self.history[1:], action.unsqueeze(0)], dim=0)

    def get(self) -> torch.Tensor:
        """Get current history."""
        return self.history.clone()


def collect_rollout(
    env: FrankaPickPlaceEnv,
    policy: VLADinoV2Policy,
    buffer: RolloutBuffer,
    rollout_length: int,
    device: torch.device,
    action_std: float,
    action_stats: Dict[str, Any],
    instruction: str = "Pick up the red sphere and place it in the goal bin.",
    render: bool = False,
) -> Dict[str, float]:
    """Collect a rollout using the current policy.

    Args:
        env: Environment to collect rollout in
        policy: Current policy
        buffer: Rollout buffer to store transitions
        rollout_length: Number of steps to collect
        device: Device to run policy on
        action_std: Standard deviation for action exploration
        action_stats: Action statistics for denormalization
        instruction: Language instruction for the task
        render: Whether to render the environment

    Returns:
        Dictionary of rollout metrics
    """
    policy.eval()

    # Action history tracker
    action_tracker = ActionHistoryTracker(
        history_length=policy.config.history_length,
        action_dim=policy.config.action_dim,
        device=device,
    )

    obs, info = env.reset()  # Environment returns (obs, info) tuple
    action_tracker.reset()

    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    successes = []

    with torch.no_grad():
        for step in range(rollout_length):
            # Prepare observation
            rgb = torch.from_numpy(obs["rgb_static"]).to(device).float().permute(2, 0, 1).unsqueeze(0)
            proprio = torch.from_numpy(obs["proprio"]).to(device).float()

            # Add timestep to proprio
            timestep = torch.tensor([step / env.max_steps], device=device, dtype=torch.float32)
            proprio = torch.cat([proprio, timestep], dim=-1).unsqueeze(0)

            # Get action history
            action_history = action_tracker.get().unsqueeze(0)

            # Get action, log prob, and value from policy
            action, log_prob, value = policy.get_action_and_value(
                rgb_static=rgb,
                instruction=[instruction],
                proprio=proprio,
                action_history=action_history,
                action_std=action_std,
            )

            # Denormalize action
            action_np = action.squeeze(0).cpu().numpy()
            if action_stats is not None:
                action_mean = np.array(action_stats["mean"])
                action_std_norm = np.array(action_stats["std"])
                action_denorm = action_np * action_std_norm + action_mean
            else:
                action_denorm = action_np

            # Step environment (returns StepResult object, not tuple)
            step_result = env.step(action_denorm)
            next_obs = step_result.observation
            reward = step_result.reward
            done = step_result.terminated or step_result.truncated
            info = step_result.info

            # Store in buffer (squeeze batch dimension)
            buffer.add(
                rgb=rgb.squeeze(0),
                instruction=instruction,
                proprio=proprio.squeeze(0),
                action_history=action_history.squeeze(0),
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                value=value.squeeze(0),
                reward=reward,
                done=done,
            )

            # Update action history
            action_tracker.update(action.squeeze(0))

            # Update episode metrics
            current_episode_reward += reward
            current_episode_length += 1

            if render:
                env.render()

            # Check if episode is done
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                successes.append(info.get("success", False))  # Fixed: use "success" not "is_success"

                # Reset environment and trackers
                obs, info = env.reset()  # Environment returns (obs, info) tuple
                action_tracker.reset()
                current_episode_reward = 0
                current_episode_length = 0
            else:
                obs = next_obs

    metrics = {
        "rollout/mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
        "rollout/mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "rollout/success_rate": np.mean(successes) if successes else 0.0,
        "rollout/num_episodes": len(episode_rewards),
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VLA policy with PPO.")
    parser.add_argument("--config", type=str, default="rl/ppo_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="BC checkpoint to load")
    parser.add_argument("--render", action="store_true", help="Render environment during rollouts")
    parser.add_argument("--quick-test", action="store_true", 
                        help="Run quick test with reduced epochs/rollouts (uses ppo_config_quick_test.yaml)")
    args = parser.parse_args()
    
    # Use quick test config if flag is set
    if args.quick_test:
        args.config = "rl/ppo_config_quick_test.yaml"
        print("=" * 60)
        print("QUICK TEST MODE ENABLED")
        print("=" * 60)
        print("Running reduced training for quick validation:")
        print("  - 3 epochs instead of 100")
        print("  - 256 steps/rollout instead of 2048")
        print("  - Expected runtime: ~5-10 minutes")
        print("=" * 60)
        print()

    # Load config
    config = load_config(args.config)
    policy_cfg = config["policy"]
    env_cfg = config["environment"]
    logging_cfg = config["logging"]

    setup_logging()
    logger = get_logger("train_ppo")

    # Set random seed
    seed = policy_cfg.get("seed", 0)
    seed_everything(seed)

    # Device setup - support M1/M2/M3 Mac GPU
    device_name = policy_cfg.get("device", "auto")
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(logging_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Load action statistics
    action_stats_path = Path("dataset/action_stats.json")
    if action_stats_path.exists():
        with open(action_stats_path, "r") as f:
            action_stats = json.load(f)
        logger.info("Loaded action statistics for denormalization")
    else:
        action_stats = None
        logger.warning("No action statistics found - actions will not be denormalized")

    # Load BC checkpoint
    checkpoint_path = args.checkpoint or policy_cfg["checkpoint"]
    logger.info(f"Loading BC checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model_config = VLADinoV2Config(**checkpoint["config"])
    policy = VLADinoV2Policy(model_config)

    # Load BC weights (actor head)
    state_dict = checkpoint["model_state"]
    if "proprio_projection.0.weight" in state_dict:
        # Slice to keep only first 7 dimensions (remove timestep)
        state_dict["proprio_projection.0.weight"] = state_dict["proprio_projection.0.weight"][:, :7]

    # Load BC weights (actor head)
    policy.load_state_dict(state_dict, strict=False)
    logger.info("Loaded BC weights (sliced proprio_projection to 7D, value_head initialized randomly)")

    policy.to(device)
    policy.train()

    # Create environment
    logger.info("Creating environment")
    env = FrankaPickPlaceEnv(
        asset_root=env_cfg.get("asset_root", "./env/mujoco_assets"),
        gui=args.render,  # Use gui parameter instead of render_mode
        seed=policy_cfg.get("seed", 42),
        reward_type=env_cfg.get("reward_type", "dense"),  # Now configurable!
    )
    logger.info(f"Using reward type: {env.reward_type}")
    # Note: max_steps is hardcoded to 340 in the environment

    # Create optimizer (convert to float in case YAML parses scientific notation as string)
    learning_rate = float(policy_cfg.get("learning_rate", 5e-6))
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        policy=policy,
        optimizer=optimizer,
        clip_range=float(policy_cfg.get("clip_range", 0.2)),
        value_coef=float(policy_cfg.get("value_coef", 0.5)),
        entropy_coef=float(policy_cfg.get("entropy_coef", 0.01)),
        max_grad_norm=float(policy_cfg.get("max_grad_norm", 0.5)),
        action_std=float(policy_cfg.get("action_std", 0.1)),
        target_kl=float(policy_cfg.get("target_kl", 0.01)),
    )

    # Create rollout buffer
    rollout_buffer = RolloutBuffer(
        buffer_size=int(policy_cfg.get("rollout_length", 2048)),
        action_dim=model_config.action_dim,
        history_length=model_config.history_length,
        device=device,
    )

    # Training loop (ensure all numeric types are correct)
    num_epochs = int(policy_cfg.get("num_epochs", 10))
    rollout_length = int(policy_cfg.get("rollout_length", 2048))
    ppo_epochs = int(policy_cfg.get("ppo_epochs", 4))
    batch_size = int(policy_cfg.get("batch_size", 64))
    action_std = float(policy_cfg.get("action_std", 0.1))
    gamma = float(policy_cfg.get("gamma", 0.99))
    gae_lambda = float(policy_cfg.get("gae_lambda", 0.95))

    logger.info("Starting PPO training")
    logger.info(f"Epochs: {num_epochs}, Rollout length: {rollout_length}")

    global_step = 0
    best_success_rate = -1.0  # Track best success rate
    best_checkpoint_path = output_dir / "ppo_best.pt"

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Collect rollouts
        logger.info("Collecting rollouts...")
        rollout_buffer.reset()

        rollout_metrics = collect_rollout(
            env=env,
            policy=policy,
            buffer=rollout_buffer,
            rollout_length=rollout_length,
            device=device,
            action_std=action_std,
            action_stats=action_stats,
            render=args.render,
        )

        # Get last value for GAE
        # Use the last observation from the buffer
        with torch.no_grad():
            last_rgb = rollout_buffer.rgb_buffer[-1].unsqueeze(0).to(device)
            last_proprio = rollout_buffer.proprio_buffer[-1].unsqueeze(0).to(device)
            last_action_history = rollout_buffer.action_history_buffer[-1].unsqueeze(0).to(device)
            last_instruction = [rollout_buffer.instruction_buffer[-1]]

            last_value = policy.get_value(
                rgb_static=last_rgb,
                instruction=last_instruction,
                proprio=last_proprio,
                action_history=last_action_history,
            )

        # Compute returns and advantages
        advantages, returns = rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Get batch for training
        batch = rollout_buffer.get(advantages, returns)

        # Train with PPO
        logger.info("Training with PPO...")
        policy.train()
        train_metrics = ppo_trainer.train_step(
            batch=batch,
            num_epochs=ppo_epochs,
            batch_size=batch_size,
        )

        # Log metrics
        global_step += rollout_length

        for key, value in rollout_metrics.items():
            writer.add_scalar(key, value, global_step)
            logger.info(f"  {key}: {value:.4f}")

        for key, value in train_metrics.items():
            writer.add_scalar(key, value, global_step)
            logger.info(f"  {key}: {value:.4f}")

        # Check for divergence (NaN or extreme values)
        if np.isnan(train_metrics["loss/total"]) or np.isinf(train_metrics["loss/total"]):
            logger.error(f"Training diverged! Loss is {train_metrics['loss/total']}")
            logger.error("Stopping training and saving checkpoint...")
            checkpoint_path = output_dir / f"ppo_diverged_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": model_config.__dict__,
                "global_step": global_step,
            }, checkpoint_path)
            break

        # Check for extreme losses (potential divergence)
        if train_metrics["loss/total"] > 1000:
            logger.warning(f"Very high loss detected: {train_metrics['loss/total']:.2f}")
            logger.warning("Training may be diverging. Consider lowering learning rate.")

        # Save best checkpoint based on success rate
        current_success_rate = rollout_metrics.get("rollout/success_rate", 0.0)
        if current_success_rate > best_success_rate:
            best_success_rate = current_success_rate
            torch.save({
                "epoch": epoch + 1,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": model_config.__dict__,
                "global_step": global_step,
                "success_rate": best_success_rate,
            }, best_checkpoint_path)
            logger.info(f"✓ New best success rate: {best_success_rate:.2%} - Saved to {best_checkpoint_path}")

    # Save final checkpoint (last epoch)
    final_checkpoint_path = output_dir / "ppo_last.pt"
    torch.save({
        "epoch": num_epochs,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model_config.__dict__,
        "global_step": global_step,
    }, final_checkpoint_path)
    logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    logger.info(f"Best success rate achieved: {best_success_rate:.2%} (saved at {best_checkpoint_path})")

    writer.close()
    env.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
