"""Compute action normalization statistics from the training dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compute_action_statistics(dataset_root: Path, split: str = "train") -> dict:
    """Compute mean and std of actions across all episodes in the specified split.
    
    Args:
        dataset_root: Root directory containing episode folders
        split: Dataset split to use ('train' or 'val')
    
    Returns:
        Dictionary containing action statistics with separate stats for joints and gripper
    """
    # Load metadata to get train/val splits
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        splits = metadata.get("splits", {})
        if split in splits:
            episode_names = splits[split]
        else:
            print(f"Warning: Split '{split}' not found in metadata. Using all episodes.")
            episode_names = [d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    else:
        print("Warning: metadata.json not found. Using all episodes.")
        episode_names = [d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    
    # Collect all actions from training episodes
    all_actions = []
    for episode_name in episode_names:
        episode_dir = dataset_root / episode_name
        actions_path = episode_dir / "actions.npy"
        
        if not actions_path.exists():
            print(f"Warning: {actions_path} not found, skipping episode {episode_name}")
            continue
        
        actions = np.load(actions_path)
        all_actions.append(actions)
    
    if not all_actions:
        raise RuntimeError(f"No action data found in {dataset_root} for split '{split}'")
    
    # Concatenate all actions
    all_actions = np.concatenate(all_actions, axis=0)  # Shape: (total_timesteps, 8)
    
    print(f"Loaded {len(all_actions)} timesteps from {len(episode_names)} episodes")
    print(f"Action shape: {all_actions.shape}")
    
    # Compute statistics separately for joints (0-6) and gripper (7)
    joints_actions = all_actions[:, :7]  # First 7 dimensions (joints)
    gripper_actions = all_actions[:, 7:8]  # Last dimension (gripper)
    
    joints_mean = joints_actions.mean(axis=0)
    joints_std = joints_actions.std(axis=0)
    
    gripper_mean = gripper_actions.mean(axis=0)
    gripper_std = gripper_actions.std(axis=0)
    
    # Combine into single arrays
    action_mean = np.concatenate([joints_mean, gripper_mean], axis=0)
    action_std = np.concatenate([joints_std, gripper_std], axis=0)
    
    print("\nAction statistics (separate for joints and gripper):")
    print(f"  Joints (0-6) mean: {joints_mean}")
    print(f"  Joints (0-6) std:  {joints_std}")
    print(f"  Gripper (7) mean: {gripper_mean}")
    print(f"  Gripper (7) std:  {gripper_std}")
    
    # Return statistics
    stats = {
        "mean": action_mean.tolist(),
        "std": action_std.tolist(),
        "num_episodes": len(episode_names),
        "num_timesteps": len(all_actions),
        "split": split,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute action statistics from LeRobot dataset")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (train or val)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file (default: dataset/action_stats.json)")
    args = parser.parse_args()
    
    dataset_root = args.dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # Compute statistics
    stats = compute_action_statistics(dataset_root, split=args.split)
    
    # Save to JSON
    output_path = args.output or (dataset_root / "action_stats.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nAction statistics saved to {output_path}")


if __name__ == "__main__":
    main()

