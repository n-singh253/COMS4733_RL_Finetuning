"""Dataset integrity and temporal-alignment validator for LeRobot demos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _check_rgb_frames(rgb_dir: Path) -> Tuple[int, Tuple[int, int]]:
    image_paths = sorted(rgb_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    size = Image.open(image_paths[0]).size
    if size != (224, 224):
        raise ValueError(f"Expected 224x224 images, found {size} in {rgb_dir}")
    return len(image_paths), size


def _temporal_alignment(proprio: np.ndarray, actions: np.ndarray, timestamps: np.ndarray, dt_hint: float) -> Dict[str, float]:
    if proprio.shape[0] < 2:
        return {"dt": float(dt_hint), "forward_mse": 0.0, "backward_mse": 0.0, "alignment": "insufficient_data"}

    diffs = np.diff(timestamps)
    if np.any(diffs <= 0):
        raise ValueError("Timestamps must be strictly increasing")

    dt = float(np.mean(diffs)) if diffs.size else float(dt_hint)
    if dt_hint > 0:
        if abs(dt - dt_hint) > 1e-3:
            print(f"Warning: inferred dt {dt:.4f} differs from meta hint {dt_hint:.4f}")
    else:
        dt_hint = dt

    # Extract only arm joint actions (first 7 dimensions) for temporal alignment
    # Actions are 8D: [7 joint positions + 1 gripper position]
    # Proprio is 7D: [7 joint positions]
    arm_actions = actions[:, :7] if actions.ndim > 1 and actions.shape[1] > 7 else actions
    
    forward_pred = proprio[:-1] + arm_actions[:-1] * dt
    forward_mse = float(np.mean((forward_pred - proprio[1:]) ** 2))
    backward_pred = proprio[1:] - arm_actions[1:] * dt
    backward_mse = float(np.mean((backward_pred - proprio[:-1]) ** 2))

    if forward_mse <= backward_mse:
        alignment = "action_t_to_state_t+1"
    else:
        alignment = "action_t+1_to_state_t+1"
    return {"dt": dt, "forward_mse": forward_mse, "backward_mse": backward_mse, "alignment": alignment}


def load_episode(episode_dir: Path) -> Dict[str, object]:
    obs_dir = episode_dir / "obs"
    rgb_dir = obs_dir / "rgb_static"
    length, _ = _check_rgb_frames(rgb_dir)

    proprio = np.load(obs_dir / "proprio.npy")
    actions = np.load(episode_dir / "actions.npy")
    timestamps = np.load(episode_dir / "timestamps.npy")

    if proprio.shape[0] != length:
        raise ValueError(f"Mismatch proprio ({proprio.shape[0]}) vs RGB frames ({length}) in {episode_dir.name}")
    if actions.shape[0] != length:
        raise ValueError(f"Mismatch actions ({actions.shape[0]}) vs RGB frames ({length}) in {episode_dir.name}")
    if timestamps.shape[0] != length:
        raise ValueError(f"Mismatch timestamps ({timestamps.shape[0]}) vs RGB frames ({length}) in {episode_dir.name}")

    if not np.all(np.isfinite(proprio)):
        raise ValueError(f"Non-finite values detected in proprio for {episode_dir.name}")
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"Non-finite values detected in actions for {episode_dir.name}")
    
    # Validate dimensions: proprio is 7D (joint positions), actions is 8D (7 joints + gripper)
    if proprio.ndim != 2 or proprio.shape[1] != 7:
        raise ValueError(f"Expected proprio shape (N, 7), got {proprio.shape} in {episode_dir.name}")
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"Expected actions shape (N, 8), got {actions.shape} in {episode_dir.name}")

    instruction = (episode_dir / "instruction.txt").read_text().strip()
    meta = json.loads((episode_dir / "meta.json").read_text())
    dt_hint = float(meta.get("control_dt", 0.0))
    alignment = _temporal_alignment(proprio, actions, timestamps, dt_hint)

    meta_alignment = {
        "length": length,
        "instruction": instruction,
        "hindered": bool(meta.get("hindered", False)),
        "target_color": meta.get("target_color", "unknown"),
        "alignment": alignment,
    }
    return meta_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LeRobot dataset structure and temporal alignment.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset root directory.")
    parser.add_argument("--report", type=Path, help="Optional path to write JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_root}' does not exist")

    episodes = sorted(d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("episode_"))
    if not episodes:
        raise FileNotFoundError(f"No episode directories found in {dataset_root}")

    stats: List[Dict[str, object]] = []
    dt_values: List[float] = []

    for episode_dir in episodes:
        result = load_episode(episode_dir)
        stats.append(result)
        alignment = result["alignment"]
        dt_values.append(alignment["dt"])  # type: ignore[index]
        print(
            f"Validated {episode_dir.name}: steps={result['length']}, hindered={result['hindered']}, "
            f"alignment={alignment['alignment']} forward_mse={alignment['forward_mse']:.6f}"
        )

    hindered = sum(1 for entry in stats if entry["hindered"])
    summary = {
        "episodes": len(stats),
        "static": len(stats) - hindered,
        "hindered": hindered,
        "mean_dt": float(np.mean(dt_values) if dt_values else 0.0),
        "std_dt": float(np.std(dt_values, ddof=1) if len(dt_values) > 1 else 0.0),
        "alignments": {
            "action_t_to_state_t+1": sum(1 for entry in stats if entry["alignment"]["alignment"] == "action_t_to_state_t+1"),
            "action_t+1_to_state_t+1": sum(1 for entry in stats if entry["alignment"]["alignment"] == "action_t+1_to_state_t+1"),
        },
    }

    print("\nDataset summary")
    print("---------------")
    print(json.dumps(summary, indent=2))

    if args.report is not None:
        payload = {"episodes": stats, "summary": summary}
        args.report.write_text(json.dumps(payload, indent=2))
        print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
