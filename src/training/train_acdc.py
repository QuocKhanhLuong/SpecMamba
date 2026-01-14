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
    prec_3d = {c: [] for c in range(1, num_classes)}
    recall_3d = {c: [] for c in range(1, num_classes)}
    acc_3d = {c: [] for c in range(1, num_classes)}
    
    for vol_idx in vol_preds.keys():
        pred_3d = np.stack([p[1] for p in sorted(vol_preds[vol_idx], key=lambda x: x[0])], axis=0)
        target_3d = np.stack([t[1] for t in sorted(vol_targets[vol_idx], key=lambda x: x[0])], axis=0)
        
        for c in range(1, num_classes):
            pred_c = (pred_3d == c)
            target_c = (target_3d == c)
            
            tp = (pred_c & target_c).sum()
            fp = (pred_c & ~target_c).sum()
            fn = (~pred_c & target_c).sum()
            tn = (~pred_c & ~target_c).sum()
            
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            prec = (tp) / (tp + fp + 1e-6)
            recall = (tp) / (tp + fn + 1e-6)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            
            dice_3d[c].append(dice)
            prec_3d[c].append(prec)
            recall_3d[c].append(recall)
            acc_3d[c].append(acc)
            
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
        'mean_prec': np.mean([np.mean(prec_3d[c]) for c in range(1, num_classes)]),
        'mean_recall': np.mean([np.mean(recall_3d[c]) for c in range(1, num_classes)]),
        'mean_acc': np.mean([np.mean(acc_3d[c]) for c in range(1, num_classes)]),
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None, use_amp=False, 
                deep_supervision=False, ds_weights=[0.4, 0.3, 0.2, 0.1]):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"E{epoch+1}", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        
        if use_amp and scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                out = outputs['output']
                loss, loss_dict = criterion(out, masks)
                
                # Deep supervision auxiliary losses
                if deep_supervision and 'aux_outputs' in outputs:
                    for i, aux_out in enumerate(outputs['aux_outputs']):
                        aux_loss, _ = criterion(aux_out, masks)
                        loss = loss + ds_weights[i] * aux_loss
                        
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            out = outputs['output']
            loss, loss_dict = criterion(out, masks)
            
            # Deep supervision auxiliary losses
            if deep_supervision and 'aux_outputs' in outputs:
                for i, aux_out in enumerate(outputs['aux_outputs']):
                    aux_loss, _ = criterion(aux_out, masks)
                    loss = loss + ds_weights[i] * aux_loss
                    
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
    parser.add_argument('--base_channels', type=int, default=48, help='HRNetDCN: 32/48/64')
    parser.add_argument('--use_pointrend', action='store_true', help='Enable PointRend')
    parser.add_argument('--use_shearlet', action='store_true', help='Enable Shearlet refinement head')
    parser.add_argument('--no_full_res', action='store_true', help='Disable full resolution mode (faster, less VRAM)')
    
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
    
    # Evaluation - defaults to 3D + TTA
    parser.add_argument('--no_eval_3d', action='store_true', help='Disable 3D volumetric evaluation (use 2D)')
    parser.add_argument('--no_tta', action='store_true', help='Disable TTA for validation')
    parser.add_argument('--tta_test', action='store_true', help='8x TTA for test')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--exp_name', type=str, default=None)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.exp_name is None:
        args.exp_name = f"acdc_hrnet_c{args.base_channels}_{datetime.now().strftime('%m%d_%H%M')}"
    
    # Model
    num_classes = 4
    in_channels = 3
    
    from models.hrnet_dcn import HRNetDCN
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
    
    eval_3d = not args.no_eval_3d
    use_tta = not args.no_tta
    
    print(f"\n{'='*60}")
    print("ACDC Training - HRNetDCN")
    print(f"{'='*60}")
    print(f"Model:      Base Ch={args.base_channels} | Params={params:,}")
    print(f"Features:   FullRes={'✓' if not args.no_full_res else '✗'} | PointRend={'✓' if args.use_pointrend else '✗'} | Shearlet={'✓' if args.use_shearlet else '✗'}")
    print(f"Training:   BS={args.batch_size} | LR={args.lr} | Epochs={args.epochs}")
    print(f"Loss:       Boundary={args.boundary_weight} | DeepSup={'✓' if args.deep_supervision else '✗'}")
    print(f"Eval:       {'3D' if eval_3d else '2D'} | TTA={'✓' if use_tta else '✗'}")
    print(f"Options:    AMP={'✓' if args.use_amp else '✗'}")
    
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
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    
    best_dice = 0.0
    best_hd95 = float('inf')
    epochs_no_improve = 0
    
    print(f"\n{'='*60}")
    print("Training Started")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        criterion.current_epoch = epoch
        loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, args.use_amp, args.deep_supervision)
        scheduler.step()
        
        # Evaluate (3D volumetric by default)
        metrics = evaluate_3d(model, val_ds, device, num_classes)
        dice = metrics['mean_dice']
        hd95 = metrics['mean_hd95']
        prec = metrics['mean_prec']
        rec = metrics['mean_recall']
        acc = metrics['mean_acc']
        print(f"E{epoch+1:03d} | Loss: {loss:.4f} | Dice/F1: {dice:.4f} | HD95: {hd95:.2f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Acc: {acc:.4f}")
        
        # Save best models
        saved = False
        
        # 1. Best Dice
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.exp_name}_best_dice.pt"))
            print(f"  ★ New Best Dice: {best_dice:.4f}")
            saved = True
            
        # 2. Best HD95
        if hd95 < best_hd95:
            best_hd95 = hd95
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.exp_name}_best_hd95.pt"))
            print(f"  ★ New Best HD95: {best_hd95:.2f}")
            saved = True
            
        if saved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"\nTraining Complete!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Best HD95: {best_hd95:.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
