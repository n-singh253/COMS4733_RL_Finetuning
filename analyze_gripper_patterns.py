"""Analyze gripper patterns in training data to understand what model should learn."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("="*80)
print("ANALYZING GRIPPER PATTERNS IN TRAINING DATA")
print("="*80)

# Analyze a few episodes
for ep_idx in range(3):
    ep_path = Path(f"dataset/episode_{ep_idx:04d}")
    actions = np.load(ep_path / "actions.npy")
    gripper_vals = actions[:, 7]
    
    print(f"\nEpisode {ep_idx:04d}:")
    print(f"  Total steps: {len(gripper_vals)}")
    
    # Find transitions
    transitions = []
    for i in range(1, len(gripper_vals)):
        if gripper_vals[i] != gripper_vals[i-1]:
            transitions.append({
                'step': i,
                'from': gripper_vals[i-1],
                'to': gripper_vals[i]
            })
    
    print(f"  Number of transitions: {len(transitions)}")
    if transitions:
        print(f"  Transitions:")
        for t in transitions[:10]:  # Show first 10
            print(f"    Step {t['step']}: {t['from']:.2f} → {t['to']:.2f}")
    
    # Show gripper sequence
    print(f"  Gripper sequence (first 50 steps):")
    sequence_str = "    "
    for i in range(min(50, len(gripper_vals))):
        if i > 0 and gripper_vals[i] != gripper_vals[i-1]:
            sequence_str += f" [{gripper_vals[i]:.2f}] "
        else:
            sequence_str += f" {gripper_vals[i]:.2f} "
        if (i + 1) % 10 == 0:
            print(sequence_str)
            sequence_str = "    "
    if sequence_str.strip():
        print(sequence_str)
    
    # Count consecutive sequences
    current_val = gripper_vals[0]
    current_count = 1
    sequences = []
    
    for i in range(1, len(gripper_vals)):
        if gripper_vals[i] == current_val:
            current_count += 1
        else:
            sequences.append((current_val, current_count))
            current_val = gripper_vals[i]
            current_count = 1
    sequences.append((current_val, current_count))
    
    print(f"  Gripper state sequences:")
    open_lengths = [count for val, count in sequences if val == 0.04]
    closed_lengths = [count for val, count in sequences if val == 0.0]
    
    if open_lengths:
        print(f"    OPEN (0.04) sequences: {len(open_lengths)} total")
        print(f"      Average length: {np.mean(open_lengths):.1f} steps")
        print(f"      Range: {min(open_lengths)}-{max(open_lengths)} steps")
    
    if closed_lengths:
        print(f"    CLOSED (0.0) sequences: {len(closed_lengths)} total")
        print(f"      Average length: {np.mean(closed_lengths):.1f} steps")
        print(f"      Range: {min(closed_lengths)}-{max(closed_lengths)} steps")

# Visualize gripper pattern for one episode
print("\n" + "="*80)
print("VISUALIZING GRIPPER PATTERN")
print("="*80)

ep_path = Path("dataset/episode_0000")
actions = np.load(ep_path / "actions.npy")
proprio = np.load(ep_path / "obs/proprio.npy")
gripper_vals = actions[:, 7]

# Load metadata to know target
import json
with open(ep_path / "meta.json", "r") as f:
    meta = json.load(f)

print(f"\nEpisode 0000: {meta.get('instruction', 'N/A')}")

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Plot gripper
axes[0].plot(gripper_vals, 'b.-', linewidth=2, markersize=4)
axes[0].set_ylabel('Gripper State')
axes[0].set_title('Gripper Over Time (0.0=CLOSED, 0.04=OPEN)')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([-0.01, 0.05])

# Plot end-effector height (z-coordinate from proprio)
# Joint 0 is base rotation, joints 1-6 control arm
# We can't directly get end-effector height from proprio, but we can look at joint patterns
joint_sum = np.sum(np.abs(proprio), axis=1)
axes[1].plot(joint_sum, 'g-', linewidth=1)
axes[1].set_ylabel('Sum of |Joint Angles|')
axes[1].set_title('Robot Configuration (proxy for movement)')
axes[1].grid(True, alpha=0.3)

# Plot gripper with transitions highlighted
axes[2].plot(gripper_vals, 'b.-', linewidth=2, markersize=4)
for i in range(1, len(gripper_vals)):
    if gripper_vals[i] != gripper_vals[i-1]:
        axes[2].axvline(x=i, color='r', linestyle='--', alpha=0.5, linewidth=1)
        axes[2].text(i, 0.02, f'{i}', rotation=90, fontsize=8, alpha=0.7)
axes[2].set_ylabel('Gripper State')
axes[2].set_xlabel('Time Step')
axes[2].set_title('Gripper Transitions (red lines)')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([-0.01, 0.05])

plt.tight_layout()
plt.savefig('gripper_pattern_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to gripper_pattern_analysis.png")

# ==============================================================================
# KEY INSIGHT: What should model learn?
# ==============================================================================
print("\n" + "="*80)
print("WHAT THE MODEL SHOULD LEARN")
print("="*80)

print("""
Based on the analysis above, the model should learn:

1. **Temporal Pattern**: Gripper follows a pick-and-place sequence:
   - Start OPEN (approach object)
   - CLOSE (grasp object)  
   - Stay CLOSED (lift and transport)
   - OPEN (release object)
   - Repeat if needed

2. **Visual Cues**: Model should use vision to determine:
   - Is gripper close enough to object? → CLOSE
   - Is object lifted and at goal? → OPEN
   - Otherwise → stay in current state

3. **Proprio Cues**: Joint configuration indicates:
   - Low/reaching pose → likely CLOSE soon
   - High/lifted pose → likely stay CLOSED
   - At goal position → OPEN

PROBLEM: If model just predicts ~0.015 (mean), it's NOT learning the pattern!
  - Gripper needs to be BINARY (fully open OR fully closed)
  - Predicting the mean doesn't work for this task

SOLUTION: The model MUST learn to predict close to 0.0 OR 0.04, not the middle.
  - This requires strong enough gradient signal from the loss function
  - 200x weight should provide this, but maybe architectural issues prevent learning
""")

print("\nDEBUGGING QUESTIONS:")
print("1. Is 200x weight actually being used during training?")
print("   → Check training log for loss values")
print("2. Is the model architecture capable of learning this binary pattern?")
print("   → The fusion layer output might be too smooth/regularized")
print("3. Are gradients actually flowing to the gripper output?")
print("   → Check gradient magnitudes during training")

print("\nRECOMMENDATION:")
print("Try 1000x weight OR switch to classification (BCE) for gripper")

