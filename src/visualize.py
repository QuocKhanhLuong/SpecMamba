"""
Visualization Script for ACDC Segmentation
- 2D Slice Visualization: Image, GT, Prediction overlay
- 3D Volume Visualization: Multi-slice grid, volume rendering
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
from tqdm import tqdm
from datetime import datetime

from models.egm_net import EGMNet
from data.acdc_dataset import ACDCDataset2D


# ACDC Color Map
COLORS = {
    0: [0, 0, 0],       # Background - black
    1: [255, 0, 0],     # RV - red
    2: [0, 255, 0],     # MYO - green
    3: [0, 0, 255],     # LV - blue
}
CLASS_NAMES = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


def create_colormap():
    """Create colormap for segmentation overlay."""
    colors = np.array([[0, 0, 0, 0],      # BG - transparent
                       [1, 0, 0, 0.6],     # RV - red
                       [0, 1, 0, 0.6],     # MYO - green
                       [0, 0, 1, 0.6]])    # LV - blue
    return ListedColormap(colors)


def mask_to_rgb(mask, alpha=0.6):
    """Convert segmentation mask to RGB overlay."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 4), dtype=np.float32)
    
    for cls, color in COLORS.items():
        rgb[mask == cls, :3] = np.array(color) / 255.0
        if cls > 0:  # Non-background
            rgb[mask == cls, 3] = alpha
    
    return rgb


def visualize_2d_slice(image, gt_mask, pred_mask, save_path=None, show=True):
    """
    Visualize a single 2D slice with image, GT, and prediction.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Normalize image for display
    if image.ndim == 3:
        image = image[0]  # Take first channel
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # 1. Original Image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 2. Ground Truth
    gt_rgb = mask_to_rgb(gt_mask, alpha=1.0)
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(gt_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 3. Prediction
    pred_rgb = mask_to_rgb(pred_mask, alpha=1.0)
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(pred_rgb)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # 4. Difference (Error Map)
    error = (gt_mask != pred_mask).astype(np.float32)
    axes[3].imshow(image, cmap='gray')
    axes[3].imshow(error, cmap='Reds', alpha=0.5)
    axes[3].set_title('Error Map')
    axes[3].axis('off')
    
    # Legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=np.array(COLORS[i])/255, label=CLASS_NAMES[i]) 
                       for i in range(1, 4)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_2d_grid(images, gt_masks, pred_masks, save_path=None, max_samples=16):
    """
    Visualize multiple 2D slices in a grid.
    """
    n = min(len(images), max_samples)
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i in range(n):
        ax = axes[i]
        img = images[i]
        if img.ndim == 3:
            img = img[0]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        ax.imshow(img, cmap='gray')
        
        # Overlay GT and Pred
        gt_rgb = mask_to_rgb(gt_masks[i], alpha=0.4)
        pred_rgb = mask_to_rgb(pred_masks[i], alpha=0.4)
        
        # GT in solid, Pred as contours
        ax.imshow(gt_rgb)
        
        # Add contours for prediction
        for cls in [1, 2, 3]:
            mask = (pred_masks[i] == cls).astype(np.uint8)
            ax.contour(mask, colors=[np.array(COLORS[cls])/255], linewidths=1)
        
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    
    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('2D Slice Visualization (Overlay=GT, Contour=Pred)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def visualize_3d_volume(pred_3d, gt_3d, vol_name, save_path=None):
    """
    Visualize 3D volume as multi-slice grid.
    """
    n_slices = pred_3d.shape[0]
    
    # Select slices evenly across the volume
    n_show = min(16, n_slices)
    slice_indices = np.linspace(0, n_slices - 1, n_show, dtype=int)
    
    cols = 4
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    
    for i, slice_idx in enumerate(slice_indices):
        ax = axes[i]
        
        # Create comparison: GT | Pred
        gt_rgb = mask_to_rgb(gt_3d[slice_idx], alpha=1.0)[:, :, :3]
        pred_rgb = mask_to_rgb(pred_3d[slice_idx], alpha=1.0)[:, :, :3]
        
        combined = np.concatenate([gt_rgb, pred_rgb], axis=1)
        
        ax.imshow(combined)
        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')
    
    for i in range(len(slice_indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{vol_name}: GT (left) | Pred (right)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def visualize_3d_metrics(pred_3d, gt_3d, vol_name, save_path=None):
    """
    Visualize 3D volume with per-slice Dice scores.
    """
    n_slices = pred_3d.shape[0]
    
    # Calculate per-slice Dice
    dice_per_slice = {1: [], 2: [], 3: []}
    
    for s in range(n_slices):
        for c in [1, 2, 3]:
            pred_c = (pred_3d[s] == c)
            gt_c = (gt_3d[s] == c)
            inter = (pred_c & gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice = 2 * inter / (union + 1e-6)
            dice_per_slice[c].append(dice)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Per-slice Dice
    ax = axes[0]
    for c in [1, 2, 3]:
        ax.plot(dice_per_slice[c], label=CLASS_NAMES[c], color=np.array(COLORS[c])/255)
    ax.set_xlabel('Slice Index')
    ax.set_ylabel('Dice Score')
    ax.set_title(f'{vol_name}: Per-Slice Dice')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Right: Middle slice visualization
    ax = axes[1]
    mid_slice = n_slices // 2
    gt_rgb = mask_to_rgb(gt_3d[mid_slice], alpha=1.0)[:, :, :3]
    pred_rgb = mask_to_rgb(pred_3d[mid_slice], alpha=1.0)[:, :, :3]
    combined = np.concatenate([gt_rgb, pred_rgb], axis=1)
    ax.imshow(combined)
    ax.set_title(f'Middle Slice {mid_slice}: GT | Pred')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def run_inference(model, dataset, device, max_samples=None):
    """Run inference on dataset and collect predictions."""
    model.eval()
    
    images = []
    gt_masks = []
    pred_masks = []
    vol_indices = []
    slice_indices = []
    
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    
    with torch.no_grad():
        for i in tqdm(range(n), desc="Inference"):
            vol_idx, slice_idx = dataset.index_map[i]
            
            img, gt = dataset[i]
            img_tensor = img.unsqueeze(0).to(device)
            
            out = model(img_tensor)['output']
            pred = out.argmax(1).squeeze(0).cpu().numpy()
            
            images.append(img.numpy())
            gt_masks.append(gt.numpy())
            pred_masks.append(pred)
            vol_indices.append(vol_idx)
            slice_indices.append(slice_idx)
    
    return images, gt_masks, pred_masks, vol_indices, slice_indices


def main():
    parser = argparse.ArgumentParser(description="Visualize ACDC Segmentation")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC/testing')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Model
    parser.add_argument('--block_type', type=str, default='dcn')
    parser.add_argument('--use_dog', action='store_true')
    parser.add_argument('--fine_head_type', type=str, default='shearlet')
    
    # Visualization
    parser.add_argument('--mode', type=str, default='both', choices=['2d', '3d', 'both'])
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("ACDC Segmentation Visualization")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print(f"Mode: {args.mode}")
    
    # Load model
    model = EGMNet(
        in_channels=3,
        num_classes=4,
        img_size=224,
        use_hrnet=True,
        use_mamba=False,
        use_spectral=False,
        use_fine_head=True,
        use_dog=args.use_dog,
        fine_head_type=args.fine_head_type,
        block_type=args.block_type
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✓ Loaded model from {args.checkpoint}")
    
    # Load data
    dataset = ACDCDataset2D(args.data_dir, in_channels=3)
    print(f"✓ Loaded {len(dataset)} slices from {len(dataset.vol_paths)} volumes")
    
    # Run inference
    images, gt_masks, pred_masks, vol_indices, slice_indices = run_inference(
        model, dataset, device, args.max_samples
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # 2D Visualization
    if args.mode in ['2d', 'both']:
        print("\n--- 2D Visualization ---")
        
        # Grid of samples
        save_path = os.path.join(args.output_dir, f'2d_grid_{timestamp}.png')
        visualize_2d_grid(images, gt_masks, pred_masks, save_path, max_samples=16)
        
        # Individual worst cases (lowest Dice)
        dice_scores = []
        for i in range(len(pred_masks)):
            inter = ((pred_masks[i] > 0) & (gt_masks[i] > 0)).sum()
            union = (pred_masks[i] > 0).sum() + (gt_masks[i] > 0).sum()
            dice = 2 * inter / (union + 1e-6)
            dice_scores.append(dice)
        
        worst_indices = np.argsort(dice_scores)[:5]
        for i, idx in enumerate(worst_indices):
            save_path = os.path.join(args.output_dir, f'2d_worst_{i+1}_{timestamp}.png')
            visualize_2d_slice(images[idx], gt_masks[idx], pred_masks[idx], save_path, show=False)
    
    # 3D Visualization
    if args.mode in ['3d', 'both']:
        print("\n--- 3D Visualization ---")
        
        # Group by volume
        from collections import defaultdict
        vol_data = defaultdict(lambda: {'pred': [], 'gt': [], 'slice_idx': []})
        
        for i in range(len(pred_masks)):
            vol_idx = vol_indices[i]
            slice_idx = slice_indices[i]
            vol_data[vol_idx]['pred'].append((slice_idx, pred_masks[i]))
            vol_data[vol_idx]['gt'].append((slice_idx, gt_masks[i]))
        
        # Visualize each volume
        for vol_idx in list(vol_data.keys())[:5]:  # First 5 volumes
            data = vol_data[vol_idx]
            
            # Sort by slice index
            pred_sorted = [x[1] for x in sorted(data['pred'], key=lambda x: x[0])]
            gt_sorted = [x[1] for x in sorted(data['gt'], key=lambda x: x[0])]
            
            pred_3d = np.stack(pred_sorted, axis=0)
            gt_3d = np.stack(gt_sorted, axis=0)
            
            vol_name = os.path.basename(dataset.vol_paths[vol_idx]).replace('.npy', '')
            
            # Multi-slice grid
            save_path = os.path.join(args.output_dir, f'3d_vol_{vol_name}_{timestamp}.png')
            visualize_3d_volume(pred_3d, gt_3d, vol_name, save_path)
            
            # Per-slice Dice plot
            save_path = os.path.join(args.output_dir, f'3d_dice_{vol_name}_{timestamp}.png')
            visualize_3d_metrics(pred_3d, gt_3d, vol_name, save_path)
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
