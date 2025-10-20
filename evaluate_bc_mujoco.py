"""Evaluate a trained behavioral cloning policy in MuJoCo."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoImageProcessor

from env.mujoco_env import FrankaPickPlaceEnv
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from utils.config import load_config
from utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a behavioral cloning policy in MuJoCo")
    parser.add_argument("--config", type=str, default="configs/openvla_dinov2_bc.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes-static", type=int, default=None,
                        help="Number of static episodes to run when evaluating both modes.")
    parser.add_argument("--episodes-hindered", type=int, default=None,
                        help="Number of hindered episodes to run when evaluating both modes.")
    parser.add_argument(
        "--mode",
        choices=["static", "hindered"],
        default=None,
        help="If provided, evaluate only the selected mode (static or hindered).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to run when --mode is specified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = get_logger("evaluate_bc")

    config = load_config(args.config)
    evaluation_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})

    episodes_static = args.episodes_static or evaluation_cfg.get("episodes_static", 50)
    episodes_hindered = args.episodes_hindered or evaluation_cfg.get("episodes_hindered", 50)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" in checkpoint:
        model_cfg = checkpoint["config"]

    model_config = VLADinoV2Config(**model_cfg)
    policy = VLADinoV2Policy(model_config)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    image_processor = AutoImageProcessor.from_pretrained(model_config.vision_encoder)

    asset_root = evaluation_cfg.get("asset_root", "./env/mujoco_assets")
    env = FrankaPickPlaceEnv(
        asset_root=asset_root,
        gui=evaluation_cfg.get("render", False),
        seed=evaluation_cfg.get("seed", 0),
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode is not None:
            selected_mode = args.mode
            default_episodes = episodes_static if selected_mode == "static" else episodes_hindered
            episodes = args.episodes or default_episodes
            if episodes is None:
                raise ValueError("Number of episodes must be specified via --episodes or the evaluation config.")
            results, success_rate = run_rollouts(
                env,
                policy,
                image_processor,
                device,
                episodes,
                mode=selected_mode,
            )
            save_results(results_dir / f"{selected_mode}_eval.csv", results)
            if selected_mode == "static":
                save_plot(results_dir / "comparison_plot.png", success_rate, None)
            logger.info(
                "%s success rate: %.2f%%",
                selected_mode.capitalize(),
                success_rate * 100 if success_rate is not None else float("nan"),
            )
            return

        static_results, static_rate = run_rollouts(
            env,
            policy,
            image_processor,
            device,
            episodes_static,
            mode="static",
        )
        hindered_results, hindered_rate = run_rollouts(
            env,
            policy,
            image_processor,
            device,
            episodes_hindered,
            mode="hindered",
        )

        save_results(results_dir / "static_eval.csv", static_results)
        save_results(results_dir / "hindered_eval.csv", hindered_results)
        save_plot(results_dir / "comparison_plot.png", static_rate, hindered_rate)

        logger.info("Static success rate: %.2f%%", static_rate * 100 if static_rate is not None else float("nan"))
        logger.info("Hindered success rate: %.2f%%", hindered_rate * 100 if hindered_rate is not None else float("nan"))
        if static_rate is not None and hindered_rate is not None:
            logger.info("Performance gap (static - hindered): %.2f%%", (static_rate - hindered_rate) * 100)
    finally:
        env.close()


def run_rollouts(
    env: FrankaPickPlaceEnv,
    policy: VLADinoV2Policy,
    image_processor,
    device: torch.device,
    episodes: int,
    mode: str,
) -> Tuple[List[Dict[str, float]], float | None]:
    logger = get_logger("evaluate_bc")
    results: List[Dict[str, float]] = []

    for episode in range(episodes):
        observation, info = env.reset(hindered=mode == "hindered")
        done = False
        total_reward = 0.0
        success = False

        while not done:
            rgb_tensor, proprio_tensor = preprocess_observation(observation, image_processor, device)
            with torch.no_grad():
                instructions = ["perform the task"] * rgb_tensor.size(0)
                action = policy(rgb_static=rgb_tensor, proprio=proprio_tensor, instruction=instructions).cpu().numpy()
            step_result = env.step(action.squeeze(0))
            observation = step_result.observation
            total_reward += float(step_result.reward)
            done = bool(step_result.terminated or step_result.truncated)
            success = bool(step_result.info.get("success", success))

        results.append({"episode": episode, "success": float(success), "reward": total_reward})

    if not results:
        return results, 0.0

    success_rate = sum(item["success"] for item in results) / len(results)
    return results, success_rate


def preprocess_observation(observation, image_processor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(observation, dict):
        rgb = observation.get("rgb_static") or observation.get("image")
        proprio = observation.get("proprio") or observation.get("robot_qpos")
    else:
        raise ValueError("Observation must be a dictionary containing rgb_static/image and proprio data")

    if rgb is None:
        raise ValueError("Observation missing rgb information")

    if isinstance(rgb, torch.Tensor):
        rgb_tensor = rgb.detach().cpu()
        if rgb_tensor.ndim == 3 and rgb_tensor.shape[0] in {1, 3}:
            rgb_array = rgb_tensor.permute(1, 2, 0).numpy()
        else:
            raise ValueError("Unsupported rgb tensor shape")
    else:
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            rgb_array = rgb.astype(np.float32)
        else:
            raise ValueError("Unsupported rgb format")
        if rgb_array.max() > 1.0:
            rgb_array = rgb_array / 255.0

    inputs = image_processor(images=rgb_array, return_tensors="pt")
    rgb_tensor = inputs["pixel_values"].to(device)

    if proprio is None:
        proprio_tensor = torch.zeros(1, 7, device=device)
    else:
        if isinstance(proprio, torch.Tensor):
            proprio_tensor = proprio.float().to(device).unsqueeze(0)
        else:
            proprio_array = np.asarray(proprio, dtype=np.float32)
            proprio_tensor = torch.from_numpy(proprio_array).float().to(device).unsqueeze(0)
        if proprio_tensor.ndim == 1:
            proprio_tensor = proprio_tensor.unsqueeze(0)

    return rgb_tensor, proprio_tensor


def save_results(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "success", "reward"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_plot(path: Path, static_rate: float | None, hindered_rate: float | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib optional
        get_logger("evaluate_bc").warning("matplotlib unavailable: %s", exc)
        return

    rates = []
    labels = []
    if static_rate is not None:
        labels.append("Static")
        rates.append(static_rate)
    if hindered_rate is not None:
        labels.append("Hindered")
        rates.append(hindered_rate)

    if not rates:
        return

    plt.figure(figsize=(4, 4))
    plt.bar(labels, rates, color=["#4CAF50", "#FF5722"][: len(rates)])
    plt.ylim(0, 1)
    plt.ylabel("Success Rate")
    plt.title("BC Policy Evaluation")
    for idx, rate in enumerate(rates):
        plt.text(idx, rate + 0.02, f"{rate*100:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
