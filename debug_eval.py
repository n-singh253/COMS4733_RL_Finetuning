"""Quick debug script to see what's happening during evaluation."""
import numpy as np
from pathlib import Path
from env.mujoco_env import FrankaPickPlaceEnv

# Create environment
env = FrankaPickPlaceEnv(gui=False, seed=0, asset_root=Path("env/mujoco_assets"))

# Run a simple episode and see what happens
obs, info = env.reset(hindered=False)
print("Episode started:")
print(f"  Target: {info['target_color']}")
print(f"  Instruction: {info['instruction']}")

# Get some sample actions from the dataset to compare
episode_dir = Path("dataset/episode_0000")
dataset_actions = np.load(episode_dir / "actions.npy")

print(f"\nDataset action statistics:")
print(f"  Mean joints: {dataset_actions[:, :7].mean(axis=0)}")
print(f"  Mean gripper: {dataset_actions[:, 7].mean():.4f}")
print(f"  Gripper values: {np.unique(dataset_actions[:, 7])}")

# Try running with dataset actions
print(f"\nRunning first 10 steps with dataset actions:")
for step in range(10):
    action = dataset_actions[step]
    result = env.step(action)
    if step % 5 == 0:
        print(f"  Step {step}: reward={result.reward:.3f}")

env.close()
print("\nEnvironment test complete!")
