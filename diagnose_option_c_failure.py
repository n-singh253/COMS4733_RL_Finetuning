"""Diagnose why Option C got 0% success rate."""
import torch
import numpy as np
from pathlib import Path

print('=' * 80)
print('OPTION C FAILURE DIAGNOSIS')
print('=' * 80)

# Load checkpoint
checkpoint_path = Path('runs/dinov2_bc_best.pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu')

config = checkpoint['config']
model_state = checkpoint['model_state']
action_stats = checkpoint.get('action_stats', {})

print('\n1. CONFIGURATION:')
print(f'   use_object_detection: {config.get("use_object_detection")}')
print(f'   object_detection_weight: {config.get("object_detection_weight")}')
print(f'   history_length: {config.get("history_length")}')
print(f'   freeze_vision: {config.get("freeze_vision")}')
print(f'   freeze_language: {config.get("freeze_language")}')
print(f'   fusion_hidden_dim: {config.get("fusion_hidden_dim")}')
print(f'   fusion_layers: {config.get("fusion_layers")}')

print('\n2. MODEL STATE CHECK:')
# Count parameters
total_params = sum(p.numel() for p in model_state.values())
print(f'   Total parameters: {total_params:,}')

# Check specific components
vision_keys = [k for k in model_state.keys() if 'vision_encoder' in k]
lang_keys = [k for k in model_state.keys() if 'language_encoder' in k]
obj_det_keys = [k for k in model_state.keys() if 'object_detection_head' in k]
history_keys = [k for k in model_state.keys() if 'action_history_encoder' in k]
head_keys = [k for k in model_state.keys() if k.startswith('head.')]

print(f'   Vision encoder params: {len(vision_keys)}')
print(f'   Language encoder params: {len(lang_keys)}')
print(f'   Object detection head params: {len(obj_det_keys)} {"✅" if obj_det_keys else "❌"}')
print(f'   Action history encoder params: {len(history_keys)} {"✅" if history_keys else "❌"}')
print(f'   Action head params: {len(head_keys)} {"✅" if head_keys else "❌"}')

print('\n3. ACTION STATISTICS:')
mean = np.array(action_stats.get('mean', []))
std = np.array(action_stats.get('std', []))

if len(mean) > 0:
    print(f'   Mean: {mean}')
    print(f'   Std:  {std}')
    print(f'\n   Joint stats (dims 0-6):')
    print(f'      Mean range: [{mean[:7].min():.3f}, {mean[:7].max():.3f}]')
    print(f'      Std range:  [{std[:7].min():.3f}, {std[:7].max():.3f}]')
    print(f'   Gripper stats (dim 7):')
    print(f'      Mean: {mean[7]:.6f}')
    print(f'      Std:  {std[7]:.6f}')
    
    if std[7] < 0.001:
        print(f'      ⚠️  WARNING: Gripper std too low! May cause normalization issues')
    
print('\n4. DATASET COMPATIBILITY CHECK:')
# Check if dataset has object positions
from data.lerobot_dataset import LeRobotDataset

try:
    dataset = LeRobotDataset(
        root=Path('dataset'),
        split='val',
        modalities=['rgb_static', 'proprio', 'action'],
        history_length=config.get('history_length', 5),
    )
    
    sample = dataset[0]
    print(f'   ✅ Dataset loadable')
    print(f'   Keys in sample: {list(sample.keys())}')
    print(f'   Has object_position: {"✅" if "object_position" in sample else "❌"}')
    print(f'   Has action_history: {"✅" if "action_history" in sample else "❌"}')
    
    if 'object_position' in sample:
        obj_pos = sample['object_position']
        print(f'   Object position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}]')
    
    if 'action_history' in sample:
        history = sample['action_history']
        print(f'   Action history shape: {history.shape}')
        print(f'   Action history variance: {history.var(dim=0).mean():.6f}')
        
except Exception as e:
    print(f'   ❌ Dataset error: {e}')

print('\n5. POSSIBLE FAILURE MODES:')
print('\nBased on the analysis:')

issues = []

if not obj_det_keys:
    issues.append('❌ Object detection head missing from model')
elif config.get('use_object_detection') == False:
    issues.append('❌ Object detection disabled in config')

if not history_keys:
    issues.append('❌ Action history encoder missing from model')
elif config.get('history_length', 0) == 0:
    issues.append('❌ History length is 0 (open-loop control)')

if len(std) > 7 and std[7] < 0.001:
    issues.append(f'⚠️  Gripper std very low ({std[7]:.6f}) - may cause denorm issues')

if config.get('freeze_vision') == True:
    issues.append('⚠️  Vision encoder is frozen - may not learn task-specific features')

if not issues:
    issues.append('⚠️  Model architecture looks correct, but still failing')
    issues.append('   Possible causes:')
    issues.append('   1. Model not trained long enough')
    issues.append('   2. Object detection converged but didn\'t help actions')
    issues.append('   3. Color-based object detection too noisy')
    issues.append('   4. Model overfitting to average trajectories')
    issues.append('   5. Evaluation using wrong checkpoint (old one)')

for issue in issues:
    print(f'   {issue}')

print('\n6. RECOMMENDED ACTIONS:')
print('\n   Next steps to debug:')
print('   1. Check training logs for object detection loss convergence')
print('   2. Run visualize_object_detection.py to verify obj det accuracy')
print('   3. Check if model outputs have variance (not all zeros/constants)')
print('   4. Verify evaluation is loading the correct checkpoint')
print('   5. Try evaluate_debug.py with GUI to watch gripper behavior')

print('\n' + '=' * 80)
print('Run this with your environment activated:')
print('  source myenv/bin/activate')
print('  python3 diagnose_option_c_failure.py')
print('=' * 80)

