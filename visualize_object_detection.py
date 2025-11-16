"""Visualize object detection predictions to verify model is learning visual grounding."""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from models.vla_dinov2 import VLADinoV2Policy, VLADinoV2Config
from data.lerobot_dataset import LeRobotDataset
from utils.object_detection import detect_object_position_from_rgb


def visualize_predictions(checkpoint_path: str, num_samples: int = 9):
    """Visualize object detection predictions vs ground truth.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to visualize (must be square number: 4, 9, 16, etc.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = VLADinoV2Config(**checkpoint["config"])
    model = VLADinoV2Policy(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset(
        root=Path("dataset"),
        split="val",  # Use validation set
        modalities=["rgb_static", "proprio", "action"],
        history_length=config.history_length,
    )
    
    # Sample random indices
    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    print(f"Visualizing {num_samples} predictions...")
    
    for idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        
        # Prepare inputs
        rgb = sample["rgb_static"].unsqueeze(0).to(device)
        proprio = sample["proprio"].unsqueeze(0).to(device)
        instruction = [sample["instruction"]]
        action_history = sample["action_history"].unsqueeze(0).to(device) if "action_history" in sample else None
        
        # Ground truth
        gt_pos = sample["object_position"].cpu().numpy()
        
        # Prediction
        with torch.no_grad():
            pred_pos = model.predict_object_position(rgb, instruction, proprio, action_history)
            pred_pos = pred_pos.cpu().numpy()[0]
        
        # Convert RGB tensor to numpy for display
        rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()
        
        # Plot
        ax = axes[idx]
        ax.imshow(rgb_np)
        
        # Convert normalized coords to pixel coords
        h, w = rgb_np.shape[:2]
        gt_x_px, gt_y_px = gt_pos[0] * w, gt_pos[1] * h
        pred_x_px, pred_y_px = pred_pos[0] * w, pred_pos[1] * h
        
        # Plot ground truth (green) and prediction (red)
        ax.plot(gt_x_px, gt_y_px, 'go', markersize=15, label='GT', markeredgecolor='white', markeredgewidth=2)
        ax.plot(pred_x_px, pred_y_px, 'rx', markersize=15, label='Pred', markeredgewidth=3)
        
        # Compute error
        error_px = np.sqrt((gt_x_px - pred_x_px)**2 + (gt_y_px - pred_y_px)**2)
        
        ax.set_title(f"Error: {error_px:.1f}px", fontsize=10)
        ax.axis('off')
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("results") / "object_detection_visualization.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    # Compute average error
    print("\nComputing average detection error...")
    total_error = 0.0
    for sample_idx in indices:
        sample = dataset[sample_idx]
        rgb = sample["rgb_static"].unsqueeze(0).to(device)
        proprio = sample["proprio"].unsqueeze(0).to(device)
        instruction = [sample["instruction"]]
        action_history = sample["action_history"].unsqueeze(0).to(device) if "action_history" in sample else None
        
        gt_pos = sample["object_position"].cpu().numpy()
        
        with torch.no_grad():
            pred_pos = model.predict_object_position(rgb, instruction, proprio, action_history)
            pred_pos = pred_pos.cpu().numpy()[0]
        
        h, w = 224, 224
        gt_x_px, gt_y_px = gt_pos[0] * w, gt_pos[1] * h
        pred_x_px, pred_y_px = pred_pos[0] * w, pred_pos[1] * h
        error_px = np.sqrt((gt_x_px - pred_x_px)**2 + (gt_y_px - pred_y_px)**2)
        total_error += error_px
    
    avg_error = total_error / len(indices)
    print(f"Average detection error: {avg_error:.2f} pixels (224x224 image)")
    print(f"Success criterion: < 20 pixels (10% of image width)")
    print(f"Status: {'✓ PASS' if avg_error < 20 else '✗ FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize object detection predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=9, help="Number of samples to visualize")
    args = parser.parse_args()
    
    visualize_predictions(args.checkpoint, args.num_samples)

