"""
Test 3D Script for HRNetDCN on ACDC
Evaluates model with 3D volumetric metrics: Dice, HD95, Precision, Recall, Accuracy
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import distance_transform_edt, binary_erosion
from datetime import datetime
import json

from data.acdc_dataset import ACDCDataset2D
from torch.utils.data import Subset


CLASS_NAMES = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


def evaluate_3d(model, dataset, device, num_classes=4, use_tta=False):
    """3D Volumetric Evaluation with full metrics."""
    model.eval()
    
    vol_preds = defaultdict(list)
    vol_targets = defaultdict(list)
    
    print("\nCollecting predictions...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Inference"):
            # Handle both Subset and regular Dataset
            if hasattr(dataset, 'dataset'):
                vol_idx, slice_idx = dataset.dataset.index_map[dataset.indices[i]]
            else:
                vol_idx, slice_idx = dataset.index_map[i]
            
            img, target = dataset[i]
            img = img.unsqueeze(0).to(device)
            
            if use_tta:
                # Test-Time Augmentation: original + flip
                out1 = model(img)['output']
                out2 = torch.flip(model(torch.flip(img, [-1]))['output'], [-1])
                out = (out1 + out2) / 2
            else:
                out = model(img)['output']
            
            pred = out.argmax(1).squeeze(0).cpu().numpy()
            target_np = target.numpy()
            
            vol_preds[vol_idx].append((slice_idx, pred))
            vol_targets[vol_idx].append((slice_idx, target_np))
    
    # Initialize metric storage
    dice_3d = {c: [] for c in range(1, num_classes)}
    hd95_3d = {c: [] for c in range(1, num_classes)}
    prec_3d = {c: [] for c in range(1, num_classes)}
    recall_3d = {c: [] for c in range(1, num_classes)}
    acc_3d = {c: [] for c in range(1, num_classes)}
    
    print(f"\nComputing 3D metrics for {len(vol_preds)} volumes...")
    
    for vol_idx in tqdm(vol_preds.keys(), desc="Metrics"):
        preds_sorted = sorted(vol_preds[vol_idx], key=lambda x: x[0])
        targets_sorted = sorted(vol_targets[vol_idx], key=lambda x: x[0])
        
        pred_3d = np.stack([p[1] for p in preds_sorted], axis=0)
        target_3d = np.stack([t[1] for t in targets_sorted], axis=0)
        
        for c in range(1, num_classes):
            pred_c = (pred_3d == c)
            target_c = (target_3d == c)
            
            # Basic counts
            tp = (pred_c & target_c).sum()
            fp = (pred_c & ~target_c).sum()
            fn = (~pred_c & target_c).sum()
            tn = (~pred_c & ~target_c).sum()
            
            # 3D Dice
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            dice_3d[c].append(dice)
            
            # Precision, Recall, Accuracy
            prec = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            
            prec_3d[c].append(prec)
            recall_3d[c].append(recall)
            acc_3d[c].append(acc)
            
            # 3D HD95
            if pred_c.any() and target_c.any():
                pred_dist = distance_transform_edt(~pred_c)
                target_dist = distance_transform_edt(~target_c)
                
                pred_border = pred_c ^ binary_erosion(pred_c)
                target_border = target_c ^ binary_erosion(target_c)
                
                if pred_border.any() and target_border.any():
                    d1 = target_dist[pred_border]
                    d2 = pred_dist[target_border]
                    all_d = np.concatenate([d1, d2])
                    hd95_3d[c].append(np.percentile(all_d, 95))
                else:
                    hd95_3d[c].append(0.0)
            elif not pred_c.any() and not target_c.any():
                hd95_3d[c].append(0.0)
            else:
                hd95_3d[c].append(100.0)
    
    # Calculate means
    mean_dice = np.mean([np.mean(dice_3d[c]) for c in range(1, num_classes)])
    mean_hd95 = np.mean([np.mean(hd95_3d[c]) for c in range(1, num_classes)])
    mean_prec = np.mean([np.mean(prec_3d[c]) for c in range(1, num_classes)])
    mean_recall = np.mean([np.mean(recall_3d[c]) for c in range(1, num_classes)])
    mean_acc = np.mean([np.mean(acc_3d[c]) for c in range(1, num_classes)])
    
    return {
        'mean_dice': mean_dice,
        'mean_hd95': mean_hd95,
        'mean_prec': mean_prec,
        'mean_recall': mean_recall,
        'mean_acc': mean_acc,
        'per_class_dice': {c: np.mean(dice_3d[c]) for c in range(1, num_classes)},
        'per_class_hd95': {c: np.mean(hd95_3d[c]) for c in range(1, num_classes)},
        'per_class_prec': {c: np.mean(prec_3d[c]) for c in range(1, num_classes)},
        'per_class_recall': {c: np.mean(recall_3d[c]) for c in range(1, num_classes)},
        'per_class_acc': {c: np.mean(acc_3d[c]) for c in range(1, num_classes)},
        'num_volumes': len(vol_preds)
    }


def main():
    parser = argparse.ArgumentParser(description="Test HRNetDCN with 3D metrics on ACDC")
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC/testing',
                        help='Path to test data directory')
    
    # Model config (should match training)
    parser.add_argument('--base_channels', type=int, default=48, help='HRNetDCN base channels (32/48/64)')
    parser.add_argument('--use_pointrend', action='store_true', help='Enable PointRend')
    parser.add_argument('--use_shearlet', action='store_true', help='Enable Shearlet head')
    parser.add_argument('--no_full_res', action='store_true', help='Disable full resolution mode')
    parser.add_argument('--deep_supervision', action='store_true', help='Enable deep supervision')
    
    # Evaluation options
    parser.add_argument('--use_tta', action='store_true', help='Use Test-Time Augmentation')
    parser.add_argument('--save_results', type=str, default=None, 
                        help='Path to save results JSON (optional)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("HRNetDCN 3D Test Evaluation - ACDC")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data_dir}")
    print(f"Device:     {device}")
    print(f"TTA:        {'✓' if args.use_tta else '✗'}")
    
    # Load model
    from models.hrnet_dcn import HRNetDCN
    
    num_classes = 4
    in_channels = 3
    
    model = HRNetDCN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=args.base_channels,
        use_pointrend=args.use_pointrend,
        full_resolution_mode=not args.no_full_res,
        deep_supervision=args.deep_supervision,
        use_shearlet=args.use_shearlet
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Config:")
    print(f"  Base Channels:  {args.base_channels}")
    print(f"  Parameters:     {params:,}")
    print(f"  PointRend:      {'✓' if args.use_pointrend else '✗'}")
    print(f"  Shearlet:       {'✓' if args.use_shearlet else '✗'}")
    print(f"  Full Res Mode:  {'✓' if not args.no_full_res else '✗'}")
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"\n✗ Checkpoint not found: {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load test data
    if not os.path.exists(args.data_dir):
        print(f"\n✗ Data directory not found: {args.data_dir}")
        return
        
    dataset = ACDCDataset2D(args.data_dir, in_channels=in_channels)
    
    # Create subset with all indices
    test_indices = list(range(len(dataset)))
    test_subset = Subset(dataset, test_indices)
    
    print(f"✓ Data loaded: {len(dataset)} slices from {len(dataset.vol_paths)} volumes")
    
    # Evaluate
    print(f"\n{'='*70}")
    print("Running 3D Evaluation...")
    print(f"{'='*70}")
    
    metrics = evaluate_3d(model, test_subset, device, num_classes, use_tta=args.use_tta)
    
    # Print Results
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    
    print(f"\n{'Class':<8} {'Dice':>8} {'HD95':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}")
    print("-" * 50)
    
    for c in range(1, num_classes):
        print(f"{CLASS_NAMES[c]:<8} "
              f"{metrics['per_class_dice'][c]:>8.4f} "
              f"{metrics['per_class_hd95'][c]:>8.2f} "
              f"{metrics['per_class_prec'][c]:>8.4f} "
              f"{metrics['per_class_recall'][c]:>8.4f} "
              f"{metrics['per_class_acc'][c]:>8.4f}")
    
    print("-" * 50)
    print(f"{'Mean':<8} "
          f"{metrics['mean_dice']:>8.4f} "
          f"{metrics['mean_hd95']:>8.2f} "
          f"{metrics['mean_prec']:>8.4f} "
          f"{metrics['mean_recall']:>8.4f} "
          f"{metrics['mean_acc']:>8.4f}")
    
    print(f"\nVolumes evaluated: {metrics['num_volumes']}")
    
    # Calculate balanced score
    balanced_score = metrics['mean_dice'] - 0.1 * metrics['mean_hd95']
    print(f"Balanced Score (Dice - 0.1*HD95): {balanced_score:.4f}")
    
    # Save results if requested
    if args.save_results:
        results = {
            'checkpoint': args.checkpoint,
            'data_dir': args.data_dir,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'base_channels': args.base_channels,
                'use_pointrend': args.use_pointrend,
                'use_shearlet': args.use_shearlet,
                'full_resolution_mode': not args.no_full_res,
                'use_tta': args.use_tta
            },
            'metrics': {
                'mean_dice': float(metrics['mean_dice']),
                'mean_hd95': float(metrics['mean_hd95']),
                'mean_prec': float(metrics['mean_prec']),
                'mean_recall': float(metrics['mean_recall']),
                'mean_acc': float(metrics['mean_acc']),
                'balanced_score': float(balanced_score),
                'per_class': {
                    CLASS_NAMES[c]: {
                        'dice': float(metrics['per_class_dice'][c]),
                        'hd95': float(metrics['per_class_hd95'][c]),
                        'prec': float(metrics['per_class_prec'][c]),
                        'recall': float(metrics['per_class_recall'][c]),
                        'acc': float(metrics['per_class_acc'][c])
                    } for c in range(1, num_classes)
                },
                'num_volumes': metrics['num_volumes']
            }
        }
        
        os.makedirs(os.path.dirname(args.save_results) or '.', exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.save_results}")
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
