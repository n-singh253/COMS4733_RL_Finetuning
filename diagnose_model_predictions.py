"""Comprehensive model diagnostic to understand prediction failures."""
import torch
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor

from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from data.lerobot_dataset import LeRobotDataset
from data.augmentation import get_val_augmentation

def main():
    print('=' * 80)
    print('COMPREHENSIVE MODEL DIAGNOSTIC')
    print('=' * 80)
    
    # Load checkpoint
    checkpoint_path = Path('runs/dinov2_bc_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model_config = VLADinoV2Config(**checkpoint['config'])
    policy = VLADinoV2Policy(model_config)
    policy.load_state_dict(checkpoint['model_state'], strict=False)
    policy.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.to(device)
    
    print(f'\n✓ Model loaded')
    print(f'  Device: {device}')
    print(f'  Object detection: {model_config.use_object_detection}')
    print(f'  History length: {model_config.history_length}')
    
    # Load action stats
    action_stats = checkpoint.get('action_stats')
    if action_stats:
        action_mean = np.array(action_stats['mean'])
        action_std = np.array(action_stats['std'])
    else:
        print('  ✗ No action stats in checkpoint!')
        return
    
    # Load dataset
    dataset = LeRobotDataset(
        root='dataset',
        split='val',
        sequence_length=1,
        image_transform=get_val_augmentation(224),
        normalize_proprio=True,
        normalize_actions=True,
        action_stats=action_stats,
        history_length=model_config.history_length,
    )
    
    print(f'  Dataset: {len(dataset)} samples')
    
    # Test on a few samples
    print('\n' + '=' * 80)
    print('PREDICTION ANALYSIS')
    print('=' * 80)
    
    with torch.no_grad():
        for i in [0, 50, 100, 150, 200]:
            if i >= len(dataset):
                continue
                
            sample = dataset[i]
            
            # Prepare inputs
            rgb = sample['rgb_static'].unsqueeze(0).to(device)
            proprio = sample['proprio'].unsqueeze(0).to(device)
            instruction = [sample['instruction']]
            action_history = sample.get('action_history')
            if action_history is not None:
                action_history = action_history.unsqueeze(0).to(device)
            
            # Get prediction
            pred = policy(
                rgb_static=rgb,
                proprio=proprio,
                instruction=instruction,
                action_history=action_history
            )
            
            # Get object detection prediction if available
            obj_pos_pred = None
            if hasattr(policy, 'object_detection_head') and policy.object_detection_head is not None:
                obj_pos_pred = policy.predict_object_position(
                    rgb_static=rgb,
                    proprio=proprio,
                    instruction=instruction,
                    action_history=action_history
                )
            
            # Denormalize prediction
            pred_denorm = pred.cpu().numpy()[0] * action_std + action_mean
            
            # Ground truth (denormalized)
            gt = sample['action'].numpy() * action_std + action_mean
            obj_pos_gt = sample.get('object_position')
            
            print(f'\nSample {i}:')
            print(f'  Instruction: {instruction[0][:40]}...')
            print(f'  Joints prediction (first 3): [{pred_denorm[0]:.3f}, {pred_denorm[1]:.3f}, {pred_denorm[2]:.3f}]')
            print(f'  Joints ground truth:         [{gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}]')
            print(f'  Joint error: {np.abs(pred_denorm[:7] - gt[:7]).mean():.4f}')
            print(f'  Gripper prediction: {pred_denorm[7]:.4f}')
            print(f'  Gripper ground truth: {gt[7]:.4f}')
            
            if obj_pos_pred is not None and obj_pos_gt is not None:
                obj_pos_pred_np = obj_pos_pred.cpu().numpy()[0]
                obj_pos_gt_np = obj_pos_gt.numpy()
                print(f'  Object position prediction: [{obj_pos_pred_np[0]:.3f}, {obj_pos_pred_np[1]:.3f}]')
                print(f'  Object position ground truth: [{obj_pos_gt_np[0]:.3f}, {obj_pos_gt_np[1]:.3f}]')
                print(f'  Object detection error: {np.abs(obj_pos_pred_np - obj_pos_gt_np).mean():.4f}')
    
    # Check gripper variance
    print('\n' + '=' * 80)
    print('GRIPPER VARIANCE CHECK')
    print('=' * 80)
    
    gripper_predictions = []
    with torch.no_grad():
        for i in range(0, min(100, len(dataset)), 10):
            sample = dataset[i]
            rgb = sample['rgb_static'].unsqueeze(0).to(device)
            proprio = sample['proprio'].unsqueeze(0).to(device)
            instruction = [sample['instruction']]
            action_history = sample.get('action_history')
            if action_history is not None:
                action_history = action_history.unsqueeze(0).to(device)
            
            pred = policy(rgb_static=rgb, proprio=proprio, instruction=instruction, action_history=action_history)
            pred_denorm = pred.cpu().numpy()[0] * action_std + action_mean
            gripper_predictions.append(pred_denorm[7])
    
    gripper_predictions = np.array(gripper_predictions)
    print(f'\nGripper predictions (10 samples):')
    print(f'  Values: {gripper_predictions}')
    print(f'  Mean: {gripper_predictions.mean():.4f}')
    print(f'  Std: {gripper_predictions.std():.4f}')
    print(f'  Min: {gripper_predictions.min():.4f}')
    print(f'  Max: {gripper_predictions.max():.4f}')
    
    if gripper_predictions.std() < 0.005:
        print(f'\n❌ CRITICAL: Gripper is CONSTANT! Model not learning gripper control.')
    else:
        print(f'\n✓ Gripper varies (std={gripper_predictions.std():.4f})')
    
    print('\n' + '=' * 80)

if __name__ == '__main__':
    main()

