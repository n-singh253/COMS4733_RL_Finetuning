"""Object detection utilities for ground truth extraction."""
import numpy as np
import torch
from typing import Tuple


def detect_object_position_from_rgb(
    rgb: np.ndarray,
    target_color: str = "red",
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[float, float]:
    """Detect object position in image using color-based detection.
    
    Args:
        rgb: RGB image array (H, W, 3) in range [0, 1] or [0, 255]
        target_color: Color of target object ("red", "green", "blue")
        image_size: Expected image size for normalization
        
    Returns:
        (x, y) position in normalized [0, 1] coordinates
        Returns (0.5, 0.5) if object not detected
    """
    # Ensure image is in [0, 1] range
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    # Define color ranges (HSV-like, but using RGB for simplicity)
    # Red spheres have high R, low G, low B
    if target_color == "red":
        mask = (rgb[:, :, 0] > 0.4) & (rgb[:, :, 1] < 0.3) & (rgb[:, :, 2] < 0.3)
    elif target_color == "green":
        mask = (rgb[:, :, 1] > 0.4) & (rgb[:, :, 0] < 0.3) & (rgb[:, :, 2] < 0.3)
    elif target_color == "blue":
        mask = (rgb[:, :, 2] > 0.4) & (rgb[:, :, 0] < 0.3) & (rgb[:, :, 1] < 0.3)
    else:
        # Fallback: any bright object
        brightness = rgb.mean(axis=2)
        mask = brightness > 0.5
    
    # Find center of mass of detected pixels
    if mask.sum() > 0:
        y_coords, x_coords = np.where(mask)
        center_x = x_coords.mean()
        center_y = y_coords.mean()
        
        # Normalize to [0, 1]
        h, w = rgb.shape[:2]
        norm_x = center_x / w
        norm_y = center_y / h
        
        return float(norm_x), float(norm_y)
    else:
        # Object not detected - return center
        return 0.5, 0.5


def detect_batch_object_positions(
    rgb_batch: torch.Tensor,
    target_colors: list,
    image_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """Detect object positions for a batch of images.
    
    Args:
        rgb_batch: Batch of RGB images (B, C, H, W) in range [0, 1]
        target_colors: List of target colors for each image in batch
        image_size: Expected image size
        
    Returns:
        Tensor of shape (B, 2) with (x, y) positions in [0, 1]
    """
    batch_size = rgb_batch.shape[0]
    positions = []
    
    for i in range(batch_size):
        # Convert from (C, H, W) to (H, W, C)
        rgb = rgb_batch[i].permute(1, 2, 0).cpu().numpy()
        color = target_colors[i] if i < len(target_colors) else "red"
        x, y = detect_object_position_from_rgb(rgb, color, image_size)
        positions.append([x, y])
    
    return torch.tensor(positions, dtype=torch.float32, device=rgb_batch.device)

