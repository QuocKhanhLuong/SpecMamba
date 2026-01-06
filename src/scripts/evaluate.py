"""
Evaluation Script for EGM-Net (Energy-Gated Gabor Mamba Network).

Provides comprehensive evaluation including:
- Quantitative metrics (Dice, IoU, HD95, Precision, Recall, F1)
- Qualitative visualization (overlays, confusion matrix)
- Per-class and mean metrics
- Export to CSV/JSON

Usage:
    python evaluate.py --checkpoint path/to/model.pt --data_dir path/to/data --output_dir results/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import yaml
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.egm_net import EGMNet
from utils.metrics import SegmentationMetrics, count_parameters
from utils.visualize import (
    plot_segmentation_result,
    plot_confusion_matrix,
    plot_constellation_embeddings,
    plot_energy_map,
    save_batch_predictions
)


def load_model(checkpoint_path: str, config: dict, device: str = 'cuda') -> EGMNet:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration dict
        device: Device to load model on
        
    Returns:
        Loaded EGMNet model
    """
    model = EGMNet(
        in_channels=config.get('in_channels', 3),
        num_classes=config.get('num_classes', 4),
        img_size=config.get('img_size', 256),
        base_channels=config.get('base_channels', 64),
        num_stages=config.get('num_stages', 4),
        use_hrnet=config.get('use_hrnet', True),
        use_mamba=config.get('use_mamba', True),
        use_spectral=config.get('use_spectral', True),
        use_fine_head=config.get('use_fine_head', True),
        coarse_head_type=config.get('coarse_head_type', 'constellation'),
        fusion_type=config.get('fusion_type', 'energy_gated')
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    return model


def create_dataloader(data_dir: str, config: dict, split: str = 'test'):
    """
    Create dataloader for evaluation.
    
    Args:
        data_dir: Path to data directory
        config: Configuration dict
        split: Data split ('test', 'val')
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    # Try to import dataset class (adjust based on your project structure)
    try:
        from data.data_utils import MonogenicDataset
        
        dataset = MonogenicDataset(
            image_dir=os.path.join(data_dir, split, 'images'),
            mask_dir=os.path.join(data_dir, split, 'masks'),
            img_size=config.get('img_size', 256)
        )
    except (ImportError, FileNotFoundError):
        # Fallback: Try ACDC-style dataset
        try:
            from monai.data import CacheDataset, DataLoader as MONAIDataLoader
            from monai.transforms import (
                Compose, LoadImaged, EnsureChannelFirstd,
                ScaleIntensityd, Resized, ToTensord
            )
            
            # Assume data is in .npy format
            images = sorted(Path(data_dir).glob(f'{split}/*_image.npy'))
            masks = sorted(Path(data_dir).glob(f'{split}/*_mask.npy'))
            
            if not images:
                raise FileNotFoundError(f"No images found in {data_dir}/{split}")
            
            data_dicts = [{'image': str(img), 'label': str(lbl)} 
                          for img, lbl in zip(images, masks)]
            
            transforms = Compose([
                LoadImaged(keys=['image', 'label']),
                EnsureChannelFirstd(keys=['image', 'label']),
                ScaleIntensityd(keys=['image']),
                Resized(keys=['image', 'label'], 
                       spatial_size=(config.get('img_size', 256),) * 2),
                ToTensord(keys=['image', 'label'])
            ])
            
            dataset = CacheDataset(data_dicts, transforms, cache_rate=1.0)
            
        except Exception as e:
            print(f"Warning: Could not create dataset from {data_dir}: {e}")
            print("Creating dummy dataset for testing...")
            
            from torch.utils.data import TensorDataset
            images = torch.randn(50, config.get('in_channels', 3), 
                               config.get('img_size', 256), 
                               config.get('img_size', 256))
            masks = torch.randint(0, config.get('num_classes', 4),
                                (50, config.get('img_size', 256), 
                                 config.get('img_size', 256)))
            dataset = TensorDataset(images, masks)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"✓ Created dataloader with {len(dataset)} samples")
    return dataloader


@torch.no_grad()
def evaluate(
    model: EGMNet,
    dataloader,
    device: str,
    num_classes: int,
    output_dir: Optional[str] = None,
    save_predictions: bool = False,
    class_names: Optional[List[str]] = None,
    num_visualize: int = 10
) -> Dict:
    """
    Run full evaluation.
    
    Args:
        model: EGMNet model
        dataloader: Test dataloader
        device: Device
        num_classes: Number of classes
        output_dir: Directory to save results
        save_predictions: Whether to save prediction visualizations
        class_names: Optional list of class names
        num_visualize: Number of samples to visualize
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Initialize metrics
    metrics = SegmentationMetrics(num_classes=num_classes, device=device)
    
    # For confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    
    # For storing embeddings (if constellation head)
    all_embeddings = []
    all_labels = []
    
    # Progress bar
    progress = tqdm(dataloader, desc="Evaluating")
    
    visualized = 0
    
    for batch_idx, batch in enumerate(progress):
        # Handle different batch formats
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            targets = batch['label'].to(device).squeeze(1).long()
        else:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device).long()
        
        # Ensure targets are 2D (H, W)
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        
        # Forward pass
        outputs = model(images, return_intermediates=True)
        
        # Get predictions
        logits = outputs['output']
        preds = logits.argmax(dim=1)
        
        # Update metrics
        metrics.update(preds, targets)
        
        # Update confusion matrix
        for t, p in zip(targets.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        
        # Collect embeddings (if available)
        if outputs.get('embeddings') is not None:
            emb = outputs['embeddings']  # (B, 2, H, W)
            # Sample some pixels for visualization
            B, _, H, W = emb.shape
            emb_flat = emb.permute(0, 2, 3, 1).reshape(-1, 2)  # (B*H*W, 2)
            labels_flat = targets.view(-1)
            
            # Subsample to avoid memory issues
            step = max(1, len(emb_flat) // 10000)
            all_embeddings.append(emb_flat[::step].cpu())
            all_labels.append(labels_flat[::step].cpu())
        
        # Save visualizations
        if save_predictions and output_dir and visualized < num_visualize:
            save_batch_predictions(
                images, preds, targets, num_classes,
                output_dir=os.path.join(output_dir, 'predictions'),
                batch_idx=batch_idx,
                class_names=class_names
            )
            visualized += images.shape[0]
        
        # Update progress
        progress.set_postfix({'samples': (batch_idx + 1) * images.shape[0]})
    
    # Compute final metrics
    results = metrics.compute()
    
    # Add confusion matrix
    results['confusion_matrix'] = confusion_matrix.cpu().numpy()
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {results['accuracy']:>10.4f}")
    print(f"{'Mean Dice':<20} {results['mean_dice']:>10.4f}")
    print(f"{'Mean IoU':<20} {results['mean_iou']:>10.4f}")
    print(f"{'Mean HD95':<20} {results['mean_hd95']:>10.4f}")
    
    # Per-class results
    print("\n" + "-" * 60)
    print("Per-Class Metrics:")
    print("-" * 60)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    print(f"{'Class':<15} {'Dice':>8} {'IoU':>8} {'HD95':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 70)
    
    for i in range(num_classes):
        dice = results['dice_scores'][i]
        iou = results['iou'][i]
        hd95 = results['hd95'][i]
        prec = results['precision'][i]
        recall = results['recall'][i]
        f1 = results['f1_score'][i]
        
        hd95_str = f"{hd95:>8.2f}" if not np.isnan(hd95) else "    N/A "
        
        print(f"{class_names[i]:<15} {dice:>8.4f} {iou:>8.4f} {hd95_str} {prec:>8.4f} {recall:>8.4f} {f1:>8.4f}")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        json_results = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in results.items() if k != 'confusion_matrix'}
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n✓ Saved metrics to {output_dir / 'metrics.json'}")
        
        # Save metrics to CSV
        with open(output_dir / 'per_class_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Dice', 'IoU', 'HD95', 'Precision', 'Recall', 'F1'])
            for i in range(num_classes):
                writer.writerow([
                    class_names[i],
                    results['dice_scores'][i],
                    results['iou'][i],
                    results['hd95'][i],
                    results['precision'][i],
                    results['recall'][i],
                    results['f1_score'][i]
                ])
        print(f"✓ Saved per-class metrics to {output_dir / 'per_class_metrics.csv'}")
        
        # Plot and save confusion matrix
        plot_confusion_matrix(
            results['confusion_matrix'],
            class_names=class_names,
            normalize=True,
            save_path=str(output_dir / 'confusion_matrix.png')
        )
        print(f"✓ Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
        
        # Plot embeddings if available
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            labels = torch.cat(all_labels, dim=0).numpy()
            
            # Get prototypes if constellation head
            if hasattr(model, 'coarse_head') and hasattr(model.coarse_head, 'prototypes'):
                prototypes = model.coarse_head.prototypes.cpu().numpy().T
            else:
                prototypes = None
            
            plot_constellation_embeddings(
                embeddings, labels, prototypes,
                class_names=class_names,
                save_path=str(output_dir / 'constellation_embeddings.png')
            )
            print(f"✓ Saved constellation embeddings to {output_dir / 'constellation_embeddings.png'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate EGM-Net segmentation model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml (if not provided, uses defaults)')
    parser.add_argument('--split', type=str, default='test',
                       help='Data split to evaluate (test, val)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction visualizations')
    parser.add_argument('--num_visualize', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='Class names for visualization')
    
    # Model config overrides
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {
        'in_channels': args.in_channels,
        'num_classes': args.num_classes,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'use_hrnet': True,
        'use_mamba': True,
        'use_spectral': True,
        'use_fine_head': True,
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Merge YAML config
        if 'model' in yaml_config:
            config.update(yaml_config['model'])
        if 'heads' in yaml_config:
            config.update(yaml_config['heads'])
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create dataloader
    dataloader = create_dataloader(args.data_dir, config, args.split)
    
    # Run evaluation
    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=config['num_classes'],
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        class_names=args.class_names,
        num_visualize=args.num_visualize
    )
    
    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
