"""
ACDC Training Script - Unified for EGMNet and HRNetDCN
Supports: Boundary Loss, Deep Supervision, TTA, 3D Eval, PointRend, Mixed Precision
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from scipy.ndimage import distance_transform_edt, binary_erosion
from datetime import datetime

from data.acdc_dataset import ACDCDataset2D
from losses.sota_loss import CombinedSOTALoss, TTAInference

CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_3d(model, dataset, device, num_classes=4):
    """3D Volumetric evaluation."""
    model.eval()
    vol_preds = defaultdict(list)
    vol_targets = defaultdict(list)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            vol_idx, slice_idx = dataset.dataset.index_map[dataset.indices[i]]
            img, target = dataset[i]
            img = img.unsqueeze(0).to(device)
            pred = model(img)['output'].argmax(1).squeeze(0).cpu().numpy()
            vol_preds[vol_idx].append((slice_idx, pred))
            vol_targets[vol_idx].append((slice_idx, target.numpy()))
    
    dice_3d = {c: [] for c in range(1, num_classes)}
    hd95_3d = {c: [] for c in range(1, num_classes)}
    
    for vol_idx in vol_preds.keys():
        pred_3d = np.stack([p[1] for p in sorted(vol_preds[vol_idx], key=lambda x: x[0])], axis=0)
        target_3d = np.stack([t[1] for t in sorted(vol_targets[vol_idx], key=lambda x: x[0])], axis=0)
        
        for c in range(1, num_classes):
            pred_c = (pred_3d == c)
            target_c = (target_3d == c)
            inter = (pred_c & target_c).sum()
            dice = (2 * inter) / (pred_c.sum() + target_c.sum() + 1e-6)
            dice_3d[c].append(dice)
            
            if pred_c.any() and target_c.any():
                pred_border = pred_c ^ binary_erosion(pred_c)
                target_border = target_c ^ binary_erosion(target_c)
                if pred_border.any() and target_border.any():
                    d1 = distance_transform_edt(~target_c)[pred_border]
                    d2 = distance_transform_edt(~pred_c)[target_border]
                    hd95_3d[c].append(np.percentile(np.concatenate([d1, d2]), 95))
                else:
                    hd95_3d[c].append(0.0)
            else:
                hd95_3d[c].append(0.0 if not pred_c.any() and not target_c.any() else 100.0)
    
    return {
        'mean_dice': np.mean([np.mean(dice_3d[c]) for c in range(1, num_classes)]),
        'mean_hd95': np.mean([np.mean(hd95_3d[c]) for c in range(1, num_classes)]),
        'per_class_dice': {CLASS_MAP[c]: np.mean(dice_3d[c]) for c in range(1, num_classes)}
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None, use_amp=False):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"E{epoch+1}", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        
        if use_amp and scaler:
            with torch.amp.autocast('cuda'):
                out = model(imgs)['output']
                loss, loss_dict = criterion(out, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)['output']
            loss, loss_dict = criterion(out, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(loader)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ACDC Training - HRNetDCN / EGMNet")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC/training')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model Selection
    parser.add_argument('--model', type=str, default='hrnet_dcn', choices=['hrnet_dcn', 'egmnet'])
    parser.add_argument('--base_channels', type=int, default=48, help='HRNetDCN: 32/48/64')
    parser.add_argument('--use_pointrend', action='store_true', help='Enable PointRend')
    parser.add_argument('--full_res', action='store_true', help='Full resolution mode (stream1=224x224)')
    
    # EGMNet-specific (legacy)
    parser.add_argument('--block_type', type=str, default='dcn')
    parser.add_argument('--fine_head_type', type=str, default='shearlet', choices=['gabor', 'shearlet'])
    parser.add_argument('--use_spectral', action='store_true', help='Enable spectral processing')
    parser.add_argument('--use_dog', action='store_true', help='Enable DoG preprocessing')
    
    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--use_amp', action='store_true', help='Mixed precision')
    
    # Loss
    parser.add_argument('--boundary_weight', type=float, default=0.5)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--deep_supervision', action='store_true', help='Enable deep supervision')
    
    # Evaluation
    parser.add_argument('--eval_3d', action='store_true', help='3D volumetric evaluation')
    parser.add_argument('--use_tta', action='store_true', help='TTA for validation')
    parser.add_argument('--tta_test', action='store_true', help='8x TTA for test')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--exp_name', type=str, default=None)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.exp_name is None:
        args.exp_name = f"acdc_{args.model}_c{args.base_channels}_{datetime.now().strftime('%m%d_%H%M')}"
    
    # Model
    num_classes = 4
    in_channels = 3
    
    if args.model == 'hrnet_dcn':
        from models.hrnet_dcn import HRNetDCN
        model = HRNetDCN(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=args.base_channels,
            use_pointrend=args.use_pointrend,
            full_resolution_mode=args.full_res
        ).to(device)
    else:
        from models.egm_net import EGMNet
        model = EGMNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=224,
            use_hrnet=True,
            use_spectral=args.use_spectral,
            use_dog=args.use_dog,
            fine_head_type=args.fine_head_type,
            block_type=args.block_type
        ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*60}")
    print(f"ACDC Training - {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Model:      {args.model} | Base Ch: {args.base_channels} | Params: {params:,}")
    print(f"Training:   BS={args.batch_size} | LR={args.lr} | Epochs={args.epochs}")
    print(f"Loss:       Boundary={args.boundary_weight} | DeepSup={'✓' if args.deep_supervision else '✗'}")
    print(f"Eval:       {'3D' if args.eval_3d else '2D'} | TTA={'✓' if args.use_tta else '✗'}")
    print(f"Options:    AMP={'✓' if args.use_amp else '✗'} | PointRend={'✓' if args.use_pointrend else '✗'}")
    
    # Data
    dataset = ACDCDataset2D(args.data_dir, in_channels=in_channels)
    
    num_vols = len(dataset.vol_paths)
    vol_indices = list(range(num_vols))
    np.random.seed(42)
    np.random.shuffle(vol_indices)
    split = int(num_vols * 0.8)
    train_vols, val_vols = set(vol_indices[:split]), set(vol_indices[split:])
    
    train_idx = [i for i, (v, s) in enumerate(dataset.index_map) if v in train_vols]
    val_idx = [i for i, (v, s) in enumerate(dataset.index_map) if v in val_vols]
    
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    print(f"Data:       Train={len(train_ds)} | Val={len(val_ds)} slices")
    
    # Loss & Optimizer
    criterion = CombinedSOTALoss(
        num_classes=num_classes,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        boundary_weight=args.boundary_weight,
        warmup_epochs=args.warmup_epochs
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    best_dice = 0
    epochs_no_improve = 0
    
    print(f"\n{'='*60}")
    print("Training Started")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        criterion.current_epoch = epoch
        loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, args.use_amp)
        scheduler.step()
        
        # Evaluate
        if args.eval_3d:
            metrics = evaluate_3d(model, val_ds, device, num_classes)
            dice = metrics['mean_dice']
            hd95 = metrics['mean_hd95']
            print(f"E{epoch+1:03d} | Loss: {loss:.4f} | 3D Dice: {dice:.4f} | HD95: {hd95:.2f}")
        else:
            metrics = evaluate_2d(model, val_loader, device, num_classes, args.use_tta)
            dice = metrics['mean_dice']
            print(f"E{epoch+1:03d} | Loss: {loss:.4f} | 2D Dice: {dice:.4f}")
        
        # Save best
        if dice > best_dice:
            best_dice = dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.exp_name}_best.pt"))
            print(f"  ★ New best: {best_dice:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best Dice: {best_dice:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
