"""Diagnose why evaluation is failing despite correct pipeline."""
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoImageProcessor

from env.mujoco_env import FrankaPickPlaceEnv
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from utils.config import load_config
from utils.logging import get_logger, setup_logging


def preprocess_observation(observation, image_processor, device, timestep, max_steps):
    """Same as evaluate_bc_mujoco.py"""
    rgb = observation.get("rgb_static")
    proprio = observation.get("proprio")
    
    if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
        rgb_array = rgb.astype(np.float32)
    else:
        raise ValueError("Unsupported rgb format")
    
    if rgb_array.max() > 1.0:
        rgb_array = rgb_array / 255.0
    
    inputs = image_processor(images=rgb_array, return_tensors="pt", do_rescale=False)
    rgb_tensor = inputs["pixel_values"].to(device)
    
    if proprio is None:
        proprio_tensor = torch.zeros(1, 7, device=device)  # 7 joints only (no timestep)
    else:
        proprio_array = np.asarray(proprio, dtype=np.float32)
        proprio_tensor = torch.from_numpy(proprio_array).float().to(device).unsqueeze(0)
        if proprio_tensor.ndim == 1:
            proprio_tensor = proprio_tensor.unsqueeze(0)
        
        # Normalize proprio
        joint_min = -2.8973
        joint_max = 2.8973
        proprio_tensor = 2.0 * (proprio_tensor - joint_min) / (joint_max - joint_min) - 1.0
    
    # REMOVED TIMESTEP: Testing hypothesis that timestep enables harmful open-loop behavior
    # Model now receives only joint positions (7 dims), forcing it to rely on vision + action history
    
    return rgb_tensor, proprio_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/openvla_dinov2_bc.yaml")
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger("diagnose")
    
    # Load everything
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    action_stats = checkpoint.get("action_stats")
    
    model_cfg = config.get("model", {})
    model_config = VLADinoV2Config(**model_cfg)
    policy = VLADinoV2Policy(model_config)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    policy.to(device)
    
    image_processor = AutoImageProcessor.from_pretrained(model_config.vision_encoder, use_fast=True, do_rescale=False)
    
    env = FrankaPickPlaceEnv(
        asset_root=config.get("evaluation", {}).get("asset_root", "./env/mujoco_assets"),
        gui=False,
        seed=0,
    )
    
    print("="*70)
    print("DETAILED EVALUATION DIAGNOSTIC")
    print("="*70)
    
    observation, info = env.reset(hindered=False)
    done = False
    step = 0
    
    action_mean = np.array(action_stats["mean"], dtype=np.float32)
    action_std = np.array(action_stats["std"], dtype=np.float32)
    
    # Critical checkpoints
    checkpoints = [1, 120, 140, 160, 320, 360]
    
    while not done and step < 400:
        step += 1
        
        # Get action
        rgb_tensor, proprio_tensor = preprocess_observation(
            observation, image_processor, device, timestep=step, max_steps=220
        )
        
        with torch.no_grad():
            instructions = [info.get("instruction", "Pick up the sphere and place it in the goal bin.")]
            action = policy(rgb_static=rgb_tensor, proprio=proprio_tensor, instruction=instructions).cpu().numpy()
        
        # Denormalize
        action[0, :7] = action[0, :7] * action_std[:7] + action_mean[:7]
        
        # Gripper threshold
        gripper_logit = action[0, 7]
        GRIPPER_LOGIT_THRESHOLD = -0.4
        action[0, 7] = 0.04 if gripper_logit > GRIPPER_LOGIT_THRESHOLD else 0.0
        
        # Step
        result = env.step(action.squeeze(0))
        observation = result.observation
        done = bool(result.terminated or result.truncated)
        
        # Detailed logging at checkpoints
        if step in checkpoints or done:
            site_id = env._object_site_ids[env._target_color]
            obj_pos = env.data.site_xpos[site_id]
            
            horizontal_dist = np.linalg.norm(obj_pos[:2] - env.bin_position[:2])
            
            print(f"\n{'='*70}")
            print(f"Step {step:3d} | Gripper: {action[0,7]:.2f} | Timestep: {step/340:.3f}")
            print(f"{'='*70}")
            print(f"  Object position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
            print(f"  Bin position:    [{env.bin_position[0]:.3f}, {env.bin_position[1]:.3f}, {env.bin_position[2]:.3f}]")
            print(f"  Horizontal dist to bin: {horizontal_dist:.3f}m (need < {env.bin_radius:.3f}m)")
            print(f"  Object height: {obj_pos[2]:.3f}m (need < 0.08m to be IN bin)")
            print(f"  Gripper logit: {gripper_logit:.3f}")
            print(f"  Reward: {result.reward:.3f}")
            print(f"  Success: {result.info['success']}")
            
            # Success condition breakdown
            cond1 = horizontal_dist < env.bin_radius
            cond2 = obj_pos[2] < 0.08
            print(f"\n  Success conditions:")
            print(f"    [{'✓' if cond1 else '✗'}] Horizontal distance < {env.bin_radius:.3f}m: {horizontal_dist:.3f}m")
            print(f"    [{'✓' if cond2 else '✗'}] Object height < 0.08m: {obj_pos[2]:.3f}m")
            
            if step in [140, 320, 360]:
                gripper_state = "CLOSED" if action[0, 7] < 0.02 else "OPEN"
                expected_state = "CLOSED" if step == 140 else "OPEN"
                status = "✓" if gripper_state == expected_state else "✗"
                print(f"\n  [{status}] Gripper {gripper_state} (expected {expected_state})")
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT")
    print(f"{'='*70}")
    print(f"  Steps: {step}")
    print(f"  Success: {result.info['success']}")
    print(f"  Final reward: {result.reward:.3f}")
    
    if not result.info['success']:
        print(f"\n  Why it failed:")
        site_id = env._object_site_ids[env._target_color]
        obj_pos = env.data.site_xpos[site_id]
        horizontal_dist = np.linalg.norm(obj_pos[:2] - env.bin_position[:2])
        
        if horizontal_dist >= env.bin_radius:
            print(f"    • Object not near bin (dist={horizontal_dist:.3f}m > {env.bin_radius:.3f}m)")
        if obj_pos[2] >= 0.08:
            print(f"    • Object not in bin (height={obj_pos[2]:.3f}m >= 0.08m)")
        if step >= 400:
            print(f"    • Timeout (took 400 steps)")
    
    env.close()


if __name__ == "__main__":
    main()

