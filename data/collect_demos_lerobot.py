"""Scripted demonstration collection for the Franka pick-and-place task.

This utility generates LeRobot-compatible episodes by running a simple
heuristic controller inside :class:`env.mujoco_env.FrankaPickPlaceEnv`.  The
resulting dataset can be used directly by ``train_bc.py`` and mirrors the
structure expected by the COMS4733 Milestone 1 baseline.

The script purposely keeps the policy trivial – it drives the gripper towards
the target object's site using a proportional controller and lifts it above the
table.  While this will not solve challenging scenes, it is sufficient for
smoke-testing the end-to-end data pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from env.mujoco_env import FrankaPickPlaceEnv


@dataclass(slots=True)
class EpisodeBuffer:
    """Stores trajectory information before writing to disk."""

    rgb_frames: List[np.ndarray]
    proprio: List[np.ndarray]
    actions: List[np.ndarray]
    timestamps: List[float]
    instruction: str
    meta: Dict[str, object]

    def extend(self, obs: Dict[str, np.ndarray], action: np.ndarray, timestamp: float) -> None:
        self.rgb_frames.append((obs["rgb_static"] * 255).astype(np.uint8))
        self.proprio.append(obs["proprio"].astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.timestamps.append(float(timestamp))

    def save(self, root: Path, episode_id: int) -> None:
        episode_dir = root / f"episode_{episode_id:04d}"
        image_dir = episode_dir / "obs" / "rgb_static"
        episode_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(self.rgb_frames):
            Image.fromarray(frame).save(image_dir / f"{idx:06d}.png")

        np.save(episode_dir / "obs" / "proprio.npy", np.stack(self.proprio, axis=0))
        np.save(episode_dir / "actions.npy", np.stack(self.actions, axis=0))
        np.save(episode_dir / "timestamps.npy", np.asarray(self.timestamps, dtype=np.float32))

        (episode_dir / "instruction.txt").write_text(self.instruction)
        (episode_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))


def scripted_policy(env: FrankaPickPlaceEnv, gain: float = 2.0) -> np.ndarray:
    """Proportional controller driving the gripper towards the target object."""

    target_site = env._object_site_ids[env.target_color]  # type: ignore[attr-defined]
    target_pos = env.data.site_xpos[target_site]
    gripper_pos = env.data.site_xpos[env._gripper_site_id]  # type: ignore[attr-defined]

    direction = target_pos - gripper_pos
    horizontal = direction.copy()
    horizontal[2] = 0.0
    vertical = np.array([0.0, 0.0, 1.0]) * max(direction[2], 0.0)

    action = gain * np.concatenate([horizontal, vertical[:1], np.zeros(3)])
    return np.clip(action, -0.25, 0.25)


def collect_episode(env: FrankaPickPlaceEnv, hindered: bool, max_steps: int) -> EpisodeBuffer:
    obs, info = env.reset(hindered=hindered)
    buffer = EpisodeBuffer(
        rgb_frames=[],
        proprio=[],
        actions=[],
        timestamps=[],
        instruction=info["instruction"],
        meta=info,
    )

    timestamp = 0.0
    for _ in range(max_steps):
        action = scripted_policy(env)
        buffer.extend(obs, action, timestamp)
        result = env.step(action)
        obs = result.observation
        timestamp += env.step_dt
        if result.terminated or result.truncated:
            break

    buffer.meta.update(
        {
            "episode_length": len(buffer.actions),
            "control_dt": env.step_dt,
        }
    )
    return buffer


def write_metadata(dataset_root: Path, metadata: List[Dict[str, object]]) -> None:
    payload = {
        "episodes": metadata,
        "num_static": sum(1 for item in metadata if not item.get("hindered", False)),
        "num_hindered": sum(1 for item in metadata if item.get("hindered", False)),
    }
    (dataset_root / "metadata.json").write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LeRobot demonstrations using MuJoCo.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Output directory for episodes.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to record.")
    parser.add_argument("--hindered-fraction", type=float, default=0.2, help="Fraction of episodes with hindered resets.")
    parser.add_argument("--max-steps", type=int, default=180, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--gui", action="store_true", help="Enable the interactive MuJoCo viewer.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("env/mujoco_assets"),
        help="Directory containing franka_scene.xml and associated assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    env = FrankaPickPlaceEnv(gui=args.gui, seed=args.seed, asset_root=args.asset_root)
    metadata: List[Dict[str, object]] = []

    rng = np.random.default_rng(args.seed)
    hindered_fraction = float(np.clip(args.hindered_fraction, 0.0, 1.0))

    for episode_idx in range(args.episodes):
        hindered = rng.random() < hindered_fraction
        buffer = collect_episode(env, hindered=hindered, max_steps=args.max_steps)
        buffer.save(dataset_root, episode_idx)
        metadata.append({
            "episode": f"episode_{episode_idx:04d}",
            "length": len(buffer.actions),
            "hindered": hindered,
            "instruction": buffer.instruction,
            "target_color": buffer.meta.get("target_color"),
        })
        print(f"Recorded episode {episode_idx:04d} | hindered={hindered} | steps={len(buffer.actions)}")

    write_metadata(dataset_root, metadata)
    env.close()
    print(f"Saved dataset with {len(metadata)} episodes to {dataset_root}")


if __name__ == "__main__":
    main()

