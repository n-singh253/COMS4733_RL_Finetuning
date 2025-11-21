"""Debug evaluation script with verbose logging to diagnose 0% success rate."""
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoImageProcessor

from env.mujoco_env import FrankaPickPlaceEnv
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy


def preprocess_observation(observation, image_processor, device, timestep, max_steps=184):
    """Same as evaluate_bc_mujoco.py but with logging."""
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
        proprio_tensor = torch.zeros(1, 8, device=device)
    else:
        proprio_array = np.asarray(proprio, dtype=np.float32)
        proprio_tensor = torch.from_numpy(proprio_array).float().to(device).unsqueeze(0)
        if proprio_tensor.ndim == 1:
            proprio_tensor = proprio_tensor.unsqueeze(0)
        
        # Normalize proprio
        joint_min = -2.8973
        joint_max = 2.8973
        proprio_tensor = 2.0 * (proprio_tensor - joint_min) / (joint_max - joint_min) - 1.0
    
    # Add timestep
    timestep_normalized = timestep / max(max_steps, 1)
    timestep_tensor = torch.tensor([[timestep_normalized]], dtype=torch.float32, device=device)
    proprio_tensor = torch.cat([proprio_tensor, timestep_tensor], dim=-1)
    
    return rgb_tensor, proprio_tensor, timestep_normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    
    # Load checkpoint
    print("="*80)
    print("LOADING CHECKPOINT")
    print("="*80)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"  Has action_stats: {'action_stats' in checkpoint}")
    print(f"  Has uses_bce_gripper: {'uses_bce_gripper' in checkpoint}")
    print(f"  Action dim: {checkpoint['config']['action_dim']}")
    
    action_stats = checkpoint.get("action_stats", None)
    if action_stats:
        print(f"  Action mean (first 3): {action_stats['mean'][:3]}")
        print(f"  Action std (first 3): {action_stats['std'][:3]}")
        print(f"  Gripper mean: {action_stats['mean'][7]:.4f}")
        print(f"  Gripper std: {action_stats['std'][7]:.4f}")
    
    # Load model
    model_cfg = checkpoint['config']
    policy = VLADinoV2Policy(VLADinoV2Config(**model_cfg))
    policy.load_state_dict(checkpoint['model_state'])
    policy.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    print(f"  Device: {device}")
    
    # Setup
    image_processor = AutoImageProcessor.from_pretrained(
        model_cfg['vision_encoder'], use_fast=True, do_rescale=False
    )
    
    env = FrankaPickPlaceEnv(
        asset_root=Path("env/mujoco_assets"),
        gui=True,  # Enable GUI visualization
        seed=0,
    )
    
    # Run episodes
    for episode_num in range(args.episodes):
        print("\n" + "="*80)
        print(f"EPISODE {episode_num}")
        print("="*80)
        
        observation, info = env.reset(hindered=False)
        print(f"Target: {info['target_color']}")
        print(f"Instruction: {info['instruction']}")
        
        done = False
        total_reward = 0.0
        success = False
        step = 0
        instruction_text = info.get("instruction", "Pick up the sphere and place it in the goal bin.")
        
        while not done:
            step += 1
            
            # Preprocess
            rgb_tensor, proprio_tensor, timestep_norm = preprocess_observation(
                observation, image_processor, device, timestep=step, max_steps=184
            )
            
            # Predict
            with torch.no_grad():
                instructions = [instruction_text]
                action = policy(rgb_static=rgb_tensor, proprio=proprio_tensor, instruction=instructions).cpu().numpy()
            
            # Log raw prediction
            if step % 50 == 0 or step <= 5:
                print(f"\nStep {step} (t={timestep_norm:.3f}):")
                print(f"  Raw prediction (normalized):")
                print(f"    Joints [0-2]: {action[0, :3]}")
                print(f"    Gripper [7]: {action[0, 7]:.4f}")
            
            # Denormalize
            if action_stats is not None:
                action_mean = np.array(action_stats["mean"], dtype=np.float32)
                action_std = np.array(action_stats["std"], dtype=np.float32)
                action_before = action.copy()
                action[0, :7] = action[0, :7] * action_std[:7] + action_mean[:7]
                
                if step % 50 == 0 or step <= 5:
                    print(f"  Denormalized action:")
                    print(f"    Joints [0-2]: {action[0, :3]}")
                    print(f"    Change: {action[0, :3] - action_before[0, :3]}")
            
            # Gripper processing (MSE trained)
            action[0, 7] = 0.0 if action[0, 7] < 0.02 else 0.04
            
            if step % 50 == 0 or step <= 5:
                print(f"  Final action sent to env:")
                print(f"    Joints [0-2]: {action[0, :3]}")
                print(f"    Gripper: {action[0, 7]:.4f} ({'CLOSED' if action[0, 7] < 0.02 else 'OPEN'})")
            
            # Step environment
            step_result = env.step(action.squeeze(0))
            observation = step_result.observation
            total_reward += float(step_result.reward)
            done = bool(step_result.terminated or step_result.truncated)
            success = bool(step_result.info.get("success", success))
            
            # Render GUI (update visualization)
            env.render(mode="human")
            
            if step % 50 == 0 or step <= 5:
                print(f"  Reward this step: {step_result.reward:.3f}, Total: {total_reward:.3f}")
                print(f"  Done: {done}, Success: {success}")
            
            if step >= 400:  # Safety limit
                print(f"\nReached step limit {step}")
                break
        
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num} COMPLETE")
        print(f"{'='*80}")
        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Success: {success}")
        
        if not success:
            print(f"\n  FAILURE ANALYSIS:")
            site_id = env._object_site_ids[env._target_color]
            obj_pos = env.data.site_xpos[site_id]
            horizontal_dist = np.linalg.norm(obj_pos[:2] - env.bin_position[:2])
            print(f"    Object position: {obj_pos}")
            print(f"    Bin position: {env.bin_position}")
            print(f"    Horizontal distance to bin: {horizontal_dist:.3f}m (need < {env.bin_radius:.3f}m)")
            print(f"    Object height: {obj_pos[2]:.3f}m (need < 0.08m)")
            
            if horizontal_dist >= env.bin_radius:
                print(f"    ❌ Object not near bin")
            if obj_pos[2] >= 0.08:
                print(f"    ❌ Object not in bin (too high)")
    
    env.close()
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

