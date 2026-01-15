"""
Plot Training History from JSON
Generates training curves for loss and metrics.
"""
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


def plot_training_history(history_path, output_dir=None):
    """
    Plot training history from JSON file.
    
    Args:
        history_path: Path to the *_history.json file
        output_dir: Optional directory to save plots. If None, saves next to JSON.
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(history_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exp_name = Path(history_path).stem.replace('_history', '')
    
    # =========================================================================
    # Plot 1: Loss and Learning Rate
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training Loss
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning Rate
    ax2 = axes[1]
    ax2.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_loss_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / f'{exp_name}_loss_lr.png'}")
    
    # =========================================================================
    # Plot 2: Main Metrics (Dice, HD95, Balanced Score)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Dice Score
    ax1 = axes[0]
    ax1.plot(epochs, history['val_dice'], 'b-', linewidth=2, label='Mean Dice')
    ax1.axhline(y=max(history['val_dice']), color='b', linestyle='--', alpha=0.5, 
                label=f'Best: {max(history["val_dice"]):.4f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Validation Dice Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # HD95
    ax2 = axes[1]
    ax2.plot(epochs, history['val_hd95'], 'r-', linewidth=2, label='Mean HD95')
    ax2.axhline(y=min(history['val_hd95']), color='r', linestyle='--', alpha=0.5,
                label=f'Best: {min(history["val_hd95"]):.2f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('HD95 (mm)')
    ax2.set_title('Validation HD95')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Balanced Score
    ax3 = axes[2]
    ax3.plot(epochs, history['balanced_score'], 'g-', linewidth=2, label='Balanced Score')
    ax3.axhline(y=max(history['balanced_score']), color='g', linestyle='--', alpha=0.5,
                label=f'Best: {max(history["balanced_score"]):.4f}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Balanced Score (Dice - 0.1×HD95)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / f'{exp_name}_metrics.png'}")
    
    # =========================================================================
    # Plot 3: Per-Class Dice
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history['dice_rv'], 'r-', linewidth=2, label=f"RV (best: {max(history['dice_rv']):.4f})")
    ax.plot(epochs, history['dice_myo'], 'g-', linewidth=2, label=f"MYO (best: {max(history['dice_myo']):.4f})")
    ax.plot(epochs, history['dice_lv'], 'b-', linewidth=2, label=f"LV (best: {max(history['dice_lv']):.4f})")
    ax.plot(epochs, history['val_dice'], 'k--', linewidth=1.5, alpha=0.7, label='Mean')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title('Per-Class Dice Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_dice_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / f'{exp_name}_dice_per_class.png'}")
    
    # =========================================================================
    # Plot 4: Per-Class HD95
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history['hd95_rv'], 'r-', linewidth=2, label=f"RV (best: {min(history['hd95_rv']):.2f})")
    ax.plot(epochs, history['hd95_myo'], 'g-', linewidth=2, label=f"MYO (best: {min(history['hd95_myo']):.2f})")
    ax.plot(epochs, history['hd95_lv'], 'b-', linewidth=2, label=f"LV (best: {min(history['hd95_lv']):.2f})")
    ax.plot(epochs, history['val_hd95'], 'k--', linewidth=1.5, alpha=0.7, label='Mean')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('HD95 (mm)')
    ax.set_title('Per-Class HD95')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_hd95_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / f'{exp_name}_hd95_per_class.png'}")
    
    # =========================================================================
    # Plot 5: All Metrics Combined (for paper)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('(a) Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Per-class Dice
    axes[0, 1].plot(epochs, history['dice_rv'], 'r-', linewidth=2, label='RV')
    axes[0, 1].plot(epochs, history['dice_myo'], 'g-', linewidth=2, label='MYO')
    axes[0, 1].plot(epochs, history['dice_lv'], 'b-', linewidth=2, label='LV')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('(b) Per-Class Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.5, 1.0])
    
    # Per-class HD95
    axes[1, 0].plot(epochs, history['hd95_rv'], 'r-', linewidth=2, label='RV')
    axes[1, 0].plot(epochs, history['hd95_myo'], 'g-', linewidth=2, label='MYO')
    axes[1, 0].plot(epochs, history['hd95_lv'], 'b-', linewidth=2, label='LV')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('HD95 (mm)')
    axes[1, 0].set_title('(c) Per-Class HD95')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Balanced Score
    axes[1, 1].plot(epochs, history['balanced_score'], 'purple', linewidth=2)
    axes[1, 1].axhline(y=max(history['balanced_score']), color='purple', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title(f"(d) Balanced Score (Best: {max(history['balanced_score']):.4f})")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / f'{exp_name}_all_metrics.png'}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Total Epochs: {len(epochs)}")
    print(f"Best Dice: {max(history['val_dice']):.4f} (epoch {epochs[np.argmax(history['val_dice'])]})")
    print(f"Best HD95: {min(history['val_hd95']):.2f} (epoch {epochs[np.argmin(history['val_hd95'])]})")
    print(f"Best Balanced: {max(history['balanced_score']):.4f} (epoch {epochs[np.argmax(history['balanced_score'])]})")
    print(f"Final LR: {history['lr'][-1]:.2e}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Plot training history from JSON')
    parser.add_argument('history_path', type=str, help='Path to *_history.json file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_path):
        print(f"Error: File not found: {args.history_path}")
        return
    
    plot_training_history(args.history_path, args.output_dir)


if __name__ == '__main__':
    main()
