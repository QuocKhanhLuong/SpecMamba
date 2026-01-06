"""
Visualization Utilities for Medical Image Segmentation.

Provides:
- Prediction overlays
- Confusion matrices
- Training curves
- Constellation/embedding visualization
- Energy map visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional, Dict, Tuple
import os


# Default colormap for segmentation classes
DEFAULT_COLORS = [
    [0, 0, 0],        # Background - Black
    [255, 0, 0],      # Class 1 - Red
    [0, 255, 0],      # Class 2 - Green
    [0, 0, 255],      # Class 3 - Blue
    [255, 255, 0],    # Class 4 - Yellow
    [255, 0, 255],    # Class 5 - Magenta
    [0, 255, 255],    # Class 6 - Cyan
    [128, 128, 128],  # Class 7 - Gray
]


def create_colormap(num_classes: int) -> np.ndarray:
    """Create a colormap for num_classes."""
    if num_classes <= len(DEFAULT_COLORS):
        return np.array(DEFAULT_COLORS[:num_classes], dtype=np.uint8)
    else:
        # Generate random colors for additional classes
        np.random.seed(42)
        colors = DEFAULT_COLORS.copy()
        for _ in range(num_classes - len(DEFAULT_COLORS)):
            colors.append(np.random.randint(0, 256, 3).tolist())
        return np.array(colors, dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class index mask to RGB image.
    
    Args:
        mask: (H, W) array of class indices
        num_classes: number of classes
        
    Returns:
        (H, W, 3) RGB image
    """
    colormap = create_colormap(num_classes)
    rgb = colormap[mask.astype(np.int32)]
    return rgb


def plot_segmentation_result(
    image: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Segmentation Result"
) -> plt.Figure:
    """
    Plot image, prediction, ground truth side by side.
    
    Args:
        image: (H, W) or (H, W, 3) input image
        pred: (H, W) predicted mask
        target: (H, W) ground truth mask
        num_classes: number of classes
        class_names: optional list of class names
        save_path: optional path to save figure
        title: figure title
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Image
    if image.ndim == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Ground Truth
    gt_rgb = mask_to_rgb(target, num_classes)
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    # Prediction
    pred_rgb = mask_to_rgb(pred, num_classes)
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    # Overlay
    if image.ndim == 3:
        overlay = image.copy()
    else:
        overlay = np.stack([image] * 3, axis=-1)
    
    # Normalize overlay to [0, 255]
    if overlay.max() <= 1:
        overlay = (overlay * 255).astype(np.uint8)
    
    # Blend prediction with image
    alpha = 0.5
    blended = (alpha * pred_rgb + (1 - alpha) * overlay).astype(np.uint8)
    axes[3].imshow(blended)
    axes[3].set_title("Overlay")
    axes[3].axis('off')
    
    # Legend
    colormap = create_colormap(num_classes)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    patches = [mpatches.Patch(color=colormap[i] / 255., label=class_names[i]) 
               for i in range(num_classes)]
    fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.0, 0.5))
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_dices: Optional[List[float]] = None,
    val_dices: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: list of training losses per epoch
        val_losses: list of validation losses per epoch
        train_dices: optional list of training Dice scores
        val_dices: optional list of validation Dice scores
        save_path: optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    num_plots = 1 + (train_dices is not None)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
    
    if num_plots == 1:
        axes = [axes]
    
    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice curve
    if train_dices is not None and val_dices is not None:
        axes[1].plot(epochs, train_dices, 'b-', label='Train Dice')
        axes[1].plot(epochs, val_dices, 'r-', label='Val Dice')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Training & Validation Dice')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: (num_classes, num_classes) confusion matrix
        class_names: optional list of class names
        normalize: whether to normalize rows to sum to 1
        save_path: optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            text = f'{val:.2f}' if normalize else f'{int(val)}'
            color = 'white' if val > cm.max() / 2 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_constellation_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 2D constellation embeddings (from RBF head).
    
    Args:
        embeddings: (N, 2) array of I/Q embeddings
        labels: (N,) array of class labels
        prototypes: optional (num_classes, 2) array of prototype locations
        class_names: optional list of class names
        save_path: optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    num_classes = int(labels.max()) + 1
    colormap = create_colormap(num_classes) / 255.
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot embeddings
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            ax.scatter(
                embeddings[mask, 0], 
                embeddings[mask, 1],
                c=[colormap[c]],
                label=class_names[c],
                alpha=0.3,
                s=5
            )
    
    # Plot prototypes
    if prototypes is not None:
        for c in range(num_classes):
            ax.scatter(
                prototypes[c, 0],
                prototypes[c, 1],
                c=[colormap[c]],
                marker='*',
                s=200,
                edgecolors='black',
                linewidths=1
            )
    
    ax.set_xlabel('I (In-phase)')
    ax.set_ylabel('Q (Quadrature)')
    ax.set_title('Constellation Embeddings')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Draw unit circle reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_energy_map(
    image: np.ndarray,
    energy: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot energy map overlay.
    
    Args:
        image: (H, W) input image
        energy: (H, W) energy map
        save_path: optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Energy Map
    im = axes[1].imshow(energy, cmap='hot')
    axes[1].set_title('Energy Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(energy, cmap='hot', alpha=0.5)
    axes[2].set_title('Energy Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_batch_predictions(
    images: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    output_dir: str,
    batch_idx: int = 0,
    class_names: Optional[List[str]] = None
):
    """
    Save visualization of a batch of predictions.
    
    Args:
        images: (B, C, H, W) input images
        preds: (B, H, W) predicted masks
        targets: (B, H, W) ground truth masks
        num_classes: number of classes
        output_dir: directory to save visualizations
        batch_idx: batch index for naming
        class_names: optional list of class names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    B = images.shape[0]
    for i in range(B):
        # Convert to numpy
        if images.shape[1] == 1:
            img = images[i, 0].cpu().numpy()
        else:
            img = images[i].permute(1, 2, 0).cpu().numpy()
        
        pred = preds[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        
        # Plot and save
        save_path = os.path.join(output_dir, f'batch{batch_idx:04d}_sample{i:02d}.png')
        plot_segmentation_result(
            img, pred, target, num_classes,
            class_names=class_names,
            save_path=save_path,
            title=f'Batch {batch_idx}, Sample {i}'
        )
        plt.close()
