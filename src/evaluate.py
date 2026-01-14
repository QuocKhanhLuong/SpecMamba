"""
Evaluate checkpoint with 3D volumetric metrics
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

from data.acdc_dataset import ACDCDataset2D
from torch.utils.data import Subset


CLASS_NAMES = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


def evaluate_3d(model, dataset, device, num_classes=4):
    """3D Volumetric Evaluation"""
    model.eval()
    
    vol_preds = defaultdict(list)
    vol_targets = defaultdict(list)
    
    print("\nCollecting predictions...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            vol_idx, slice_idx = dataset.dataset.index_map[dataset.indices[i]]
            
            img, target = dataset[i]
            img = img.unsqueeze(0).to(device)
            
            out = model(img)['output']
            pred = out.argmax(1).squeeze(0).cpu().numpy()
            target_np = target.numpy()
            
            vol_preds[vol_idx].append((slice_idx, pred))
            vol_targets[vol_idx].append((slice_idx, target_np))
    
    dice_3d = {c: [] for c in range(1, num_classes)}
    hd95_3d = {c: [] for c in range(1, num_classes)}
    
    print(f"\nComputing 3D metrics for {len(vol_preds)} volumes...")
    
    for vol_idx in tqdm(vol_preds.keys()):
        preds_sorted = sorted(vol_preds[vol_idx], key=lambda x: x[0])
        targets_sorted = sorted(vol_targets[vol_idx], key=lambda x: x[0])
        
        pred_3d = np.stack([p[1] for p in preds_sorted], axis=0)
        target_3d = np.stack([t[1] for t in targets_sorted], axis=0)
        
        for c in range(1, num_classes):
            pred_c = (pred_3d == c)
            target_c = (target_3d == c)
            
            # 3D Dice
            inter = (pred_c & target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2 * inter) / (union + 1e-6)
            dice_3d[c].append(dice)
            
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
    
    return {
        'mean_dice': np.mean([np.mean(dice_3d[c]) for c in range(1, num_classes)]),
        'mean_hd95': np.mean([np.mean(hd95_3d[c]) for c in range(1, num_classes)]),
        'per_class_dice': {c: np.mean(dice_3d[c]) for c in range(1, num_classes)},
        'per_class_hd95': {c: np.mean(hd95_3d[c]) for c in range(1, num_classes)},
        'num_volumes': len(vol_preds)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with 3D metrics")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC/testing')
    parser.add_argument('--model_type', type=str, default='hrnet_advanced',
                       choices=['egmnet', 'hrnet_advanced'])
    parser.add_argument('--base_channels', type=int, default=62)
    parser.add_argument('--use_pointrend', action='store_true')
    parser.add_argument('--full_resolution_mode', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("3D Volumetric Evaluation")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    
    # Load model
    if args.model_type == 'hrnet_advanced':
        from training.test_advanced_arch import HRNetAdvanced
        
        model = HRNetAdvanced(
            in_channels=3,
            base_channels=args.base_channels,
            img_size=224,
            stage_configs=[
                {'blocks': ['dcn'] * 2},
                {'blocks': ['dcn'] * 4},
                {'blocks': ['dcn'] * 6},
            ],
            use_pointrend=args.use_pointrend,
            num_classes=4,
            full_resolution_mode=args.full_resolution_mode
        ).to(device)
    else:
        from models.egm_net import EGMNet
        model = EGMNet(
            in_channels=3, num_classes=4, img_size=224,
            use_hrnet=True, use_mamba=False, use_spectral=False,
            use_fine_head=True, block_type='dcn'
        ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("✓ Model loaded")
    
    # Load data
    dataset = ACDCDataset2D(args.data_dir, in_channels=3)
    test_indices = list(range(len(dataset)))
    test_subset = Subset(dataset, test_indices)
    test_subset.dataset = dataset
    test_subset.indices = test_indices
    print(f"✓ Data loaded: {len(dataset)} slices from {len(dataset.vol_paths)} volumes")
    
    # Evaluate
    metrics = evaluate_3d(model, test_subset, device)
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Class':<8} {'3D Dice':>10} {'3D HD95':>10}")
    print("-"*30)
    for c in range(1, 4):
        print(f"{CLASS_NAMES[c]:<8} {metrics['per_class_dice'][c]:>10.4f} {metrics['per_class_hd95'][c]:>10.2f}")
    print("-"*30)
    print(f"{'Mean':<8} {metrics['mean_dice']:>10.4f} {metrics['mean_hd95']:>10.2f}")
    print(f"\nVolumes evaluated: {metrics['num_volumes']}")


if __name__ == '__main__':
    main()
