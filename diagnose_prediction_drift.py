"""Diagnose why model fails despite low validation loss.

This script:
1. Loads a training episode
2. Runs model predictions step-by-step
3. Compares predictions to demo actions
4. Tracks cumulative error
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from transformers import AutoImageProcessor
from env.mujoco_env import FrankaPickPlaceEnv
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy

def load_episode_data(episode_dir):
    """Load saved episode data."""
    episode_dir = Path(episode_dir)
    
    actions = np.load(episode_dir / "actions.npy")
    proprio = np.load(episode_dir / "obs" / "proprio.npy")  # In obs/ subdirectory
    timestamps = np.load(episode_dir / "timestamps.npy")
    
    with open(episode_dir / "instruction.txt", 'r') as f:
        instruction = f.read().strip()
    
    # Load images
    rgb_dir = episode_dir / "obs" / "rgb_static"
    images = []
    for i in range(len(actions)):
        img_path = rgb_dir / f"{i:06d}.png"
        if img_path.exists():
            images.append(np.array(Image.open(img_path)))
    
    return {
        'actions': actions,
        'proprio': proprio,
        'timestamps': timestamps,
        'instruction': instruction,
        'images': images if images else None
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episode", type=str, default="dataset/episode_0000")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to analyze")
    args = parser.parse_args()
    
    print("=" * 80)
    print("PREDICTION DRIFT ANALYSIS")
    print("=" * 80)
    print()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    action_mean = torch.tensor(checkpoint["action_stats"]["mean"], dtype=torch.float32)
    action_std = torch.tensor(checkpoint["action_stats"]["std"], dtype=torch.float32)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = checkpoint["config"]
    model_config = VLADinoV2Config(**model_cfg)
    model = VLADinoV2Policy(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(model_config.vision_encoder)
    
    # Load episode data
    print(f"Loading episode: {args.episode}")
    data = load_episode_data(args.episode)
    print(f"  Episode length: {len(data['actions'])}")
    print(f"  Instruction: {data['instruction']}")
    print()
    
    max_steps = min(args.steps, len(data['actions']))
    
    # Analyze predictions
    print("=" * 80)
    print("STEP-BY-STEP ANALYSIS")
    print("=" * 80)
    print()
    
    prediction_errors = []
    cumulative_error = np.zeros(8)
    
    for step in range(max_steps):
        # Get demo action (normalized)
        demo_action_raw = data['actions'][step]
        demo_action_norm = (demo_action_raw - action_mean.numpy()) / action_std.numpy()
        
        # Get observation
        if data['images'] is not None:
            rgb = Image.fromarray(data['images'][step])
        else:
            # If no images, skip this test
            print("No images found in episode - cannot test vision")
            return
        
        image_dict = image_processor(images=rgb, return_tensors="pt")
        image = image_dict["pixel_values"].to(device)
        
        # Proprio: qpos (7) + qvel (7) + timestep (1)
        qpos = torch.from_numpy(data['proprio'][step]).float()
        # Approximate qvel as zeros (not saved in episode)
        qvel = torch.zeros(7)
        timestep = torch.tensor([data['timestamps'][step] / 10.0])  # Normalize
        proprio = torch.cat([qpos, qvel, timestep]).unsqueeze(0).to(device)
        
        # Model prediction
        with torch.no_grad():
            pred_action_norm = model(image, proprio, [data['instruction']])
        
        pred_action_norm = pred_action_norm[0].cpu().numpy()
        
        # Compute error
        error = np.abs(pred_action_norm - demo_action_norm)
        prediction_errors.append(error)
        cumulative_error += error
        
        # Print every 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d}:")
            print(f"  Demo action (norm):  {demo_action_norm[:3]} ... gripper={demo_action_norm[7]:.3f}")
            print(f"  Pred action (norm):  {pred_action_norm[:3]} ... gripper={pred_action_norm[7]:.3f}")
            print(f"  Error (joints):      {error[:7].mean():.6f}")
            print(f"  Error (gripper):     {error[7]:.6f}")
            print(f"  Cumulative (joints): {cumulative_error[:7].mean():.6f}")
            print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    errors_array = np.array(prediction_errors)
    
    print(f"Analyzed {max_steps} steps")
    print()
    print("Per-step error:")
    print(f"  Joints (avg):   {errors_array[:, :7].mean():.6f}")
    print(f"  Joints (max):   {errors_array[:, :7].max():.6f}")
    print(f"  Gripper (avg):  {errors_array[:, 7].mean():.6f}")
    print(f"  Gripper (max):  {errors_array[:, 7].max():.6f}")
    print()
    
    print("Cumulative error over episode:")
    print(f"  Joints:  {cumulative_error[:7].mean():.6f}")
    print(f"  Gripper: {cumulative_error[7]:.6f}")
    print()
    
    # Denormalize to physical units
    avg_joint_error_rad = errors_array[:, :7].mean() * action_std[:7].mean().numpy()
    print(f"Average joint error in physical units: {avg_joint_error_rad:.6f} radians")
    print(f"                                        ({np.degrees(avg_joint_error_rad):.3f} degrees)")
    print()
    
    # Analysis
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    if errors_array[:, :7].mean() < 0.05:
        print("✓ Model predictions are VERY CLOSE to demo actions")
        print("  Small errors (<0.05 in normalized space)")
        print()
        print("But this still causes failure because:")
        print("  - Errors compound over 400 steps")
        print("  - Keyframe-based control is brittle")
        print("  - Model expects EXACT convergence at each keyframe")
        print()
        print("Solution: Model needs to be MORE ROBUST, not more accurate")
        print("  → Add noise during training")
        print("  → Use history/recurrence")
        print("  → Learn feedback control, not open-loop")
    elif errors_array[:, :7].mean() < 0.2:
        print("⚠ Model predictions are CLOSE but not perfect")
        print(f"  Error: ~{errors_array[:, :7].mean():.3f} per step")
        print()
        print("This level of error can cause failure because:")
        print("  - Over 400 steps, errors accumulate")
        print("  - Robot drifts from intended path")
        print()
        print("Solution: Improve model accuracy")
        print("  → More training data")
        print("  → Better visual features (unfreeze encoder)")
        print("  → Visual augmentation")
    else:
        print("❌ Model predictions are QUITE DIFFERENT from demos")
        print(f"  Error: ~{errors_array[:, :7].mean():.3f} per step")
        print()
        print("This is too large - model hasn't learned properly")
        print()
        print("Solution: Fix fundamental training issues")
        print("  → Check normalization")
        print("  → Verify data loading")
        print("  → Train longer")
    
    print()


if __name__ == "__main__":
    main()

