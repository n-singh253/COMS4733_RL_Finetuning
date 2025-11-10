"""Compare model predictions on training data vs evaluation scenes."""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from env.mujoco_env import FrankaPickPlaceEnv

def load_training_sample(episode_dir: Path):
    """Load a sample from training dataset."""
    # Load image
    rgb_files = sorted((episode_dir / "obs" / "rgb_static").glob("*.png"))
    image = Image.open(rgb_files[0])
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Load proprio
    proprio = np.load(episode_dir / "obs" / "proprio.npy")
    
    # Load action
    actions = np.load(episode_dir / "actions.npy")
    
    # Load instruction
    instruction = (episode_dir / "instruction.txt").read_text().strip()
    
    return {
        "rgb": image_array,
        "proprio": proprio[0],
        "action": actions[0],
        "instruction": instruction,
    }

def main():
    # Load checkpoint
    checkpoint_path = Path("runs/dinov2_bc_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    action_stats = checkpoint.get("action_stats")
    
    # Load model
    model_config = VLADinoV2Config(**checkpoint["config"])
    model = VLADinoV2Policy(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # Setup image processor
    image_processor = AutoImageProcessor.from_pretrained(
        model_config.vision_encoder, use_fast=True, do_rescale=False
    )
    
    print("=" * 70)
    print("COMPARING TRAIN VS EVAL PREDICTIONS")
    print("=" * 70)
    
    # Test on training sample
    print("\n1. Testing on TRAINING data (episode_0000):")
    print("-" * 70)
    
    train_sample = load_training_sample(Path("dataset/episode_0000"))
    
    # Prepare inputs
    rgb_train = torch.from_numpy(train_sample["rgb"]).permute(2, 0, 1).unsqueeze(0).float()
    proprio_train = torch.from_numpy(train_sample["proprio"]).unsqueeze(0).float()
    proprio_train_norm = proprio_train / 2.8973
    instruction_train = train_sample["instruction"]
    ground_truth_action = train_sample["action"]
    
    # Get prediction
    with torch.no_grad():
        pred_train = model(
            rgb_static=rgb_train,
            proprio=proprio_train_norm,
            instruction=[instruction_train]
        )
    
    pred_train_np = pred_train.cpu().numpy()[0]
    
    # Denormalize prediction
    pred_train_denorm = pred_train_np.copy()
    if action_stats is not None:
        action_mean = np.array(action_stats["mean"], dtype=np.float32)
        action_std = np.array(action_stats["std"], dtype=np.float32)
        pred_train_denorm[:7] = pred_train_denorm[:7] * action_std[:7] + action_mean[:7]
    
    print(f"Instruction: {instruction_train}")
    print(f"\nGround truth action:")
    print(f"  Joints: {ground_truth_action[:7]}")
    print(f"  Gripper: {ground_truth_action[7]:.6f}")
    
    print(f"\nModel prediction (raw, normalized):")
    print(f"  Joints: {pred_train_np[:7]}")
    print(f"  Gripper: {pred_train_np[7]:.6f}")
    
    print(f"\nModel prediction (denormalized):")
    print(f"  Joints: {pred_train_denorm[:7]}")
    print(f"  Gripper: {pred_train_denorm[7]:.6f}")
    
    # Compute error
    joint_error = np.abs(pred_train_denorm[:7] - ground_truth_action[:7])
    gripper_error = abs(pred_train_denorm[7] - ground_truth_action[7])
    
    print(f"\nPrediction error:")
    print(f"  Joint MAE: {np.mean(joint_error):.6f} rad")
    print(f"  Gripper error: {gripper_error:.6f}")
    
    # Test on new evaluation scene
    print("\n\n2. Testing on NEW EVAL scene:")
    print("-" * 70)
    
    env = FrankaPickPlaceEnv(gui=False, seed=999, asset_root="env/mujoco_assets")
    obs, info = env.reset(hindered=False)
    
    rgb_eval = torch.from_numpy(obs["rgb_static"]).unsqueeze(0).float()
    proprio_eval = torch.from_numpy(obs["proprio"]).unsqueeze(0).float()
    proprio_eval_norm = proprio_eval / 2.8973
    instruction_eval = info["instruction"]
    
    # Get prediction
    with torch.no_grad():
        pred_eval = model(
            rgb_static=rgb_eval,
            proprio=proprio_eval_norm,
            instruction=[instruction_eval]
        )
    
    pred_eval_np = pred_eval.cpu().numpy()[0]
    
    # Denormalize
    pred_eval_denorm = pred_eval_np.copy()
    if action_stats is not None:
        pred_eval_denorm[:7] = pred_eval_denorm[:7] * action_std[:7] + action_mean[:7]
    
    print(f"Instruction: {instruction_eval}")
    print(f"Target color: {info['target_color']}")
    print(f"Active objects: {env._active_objects}")
    
    print(f"\nModel prediction (raw, normalized):")
    print(f"  Joints: {pred_eval_np[:7]}")
    print(f"  Gripper: {pred_eval_np[7]:.6f}")
    
    print(f"\nModel prediction (denormalized):")
    print(f"  Joints: {pred_eval_denorm[:7]}")
    print(f"  Gripper: {pred_eval_denorm[7]:.6f}")
    
    # Compare train vs eval predictions
    print("\n\n3. COMPARISON:")
    print("-" * 70)
    
    joint_diff = np.abs(pred_train_denorm[:7] - pred_eval_denorm[:7])
    gripper_diff = abs(pred_train_denorm[7] - pred_eval_denorm[7])
    
    print(f"Difference between train and eval predictions:")
    print(f"  Joint difference (mean): {np.mean(joint_diff):.6f} rad")
    print(f"  Gripper difference: {gripper_diff:.6f}")
    
    if np.mean(joint_diff) < 0.1:
        print("\n⚠ WARNING: Predictions are very similar despite different scenes!")
        print("   Model may be ignoring visual input and just predicting mean actions.")
    else:
        print("\n✓ Predictions differ between scenes (model is vision-responsive)")
    
    # Check if predictions are close to mean
    if action_stats is not None:
        action_mean_arr = np.array(action_stats["mean"])
        distance_to_mean_train = np.mean(np.abs(pred_train_denorm - action_mean_arr))
        distance_to_mean_eval = np.mean(np.abs(pred_eval_denorm - action_mean_arr))
        
        print(f"\nDistance from dataset mean:")
        print(f"  Train prediction: {distance_to_mean_train:.6f}")
        print(f"  Eval prediction: {distance_to_mean_eval:.6f}")
        
        if distance_to_mean_train < 0.1 and distance_to_mean_eval < 0.1:
            print("\n⚠ WARNING: Both predictions very close to dataset mean!")
            print("   Model may not have learned meaningful patterns.")
    
    env.close()

if __name__ == "__main__":
    main()

