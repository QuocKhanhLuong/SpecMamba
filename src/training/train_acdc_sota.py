"""
ACDC SOTA Training Script
- Boundary Loss with warmup
- Deep Supervision
- TTA for evaluation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.egm_net import EGMNet
from data.acdc_dataset import ACDCDataset2D
from losses.sota_loss import CombinedSOTALoss, TTAInference


CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


def evaluate(model, loader, device, num_classes=4, use_tta=False):
    model.eval()
    
    if use_tta:
        tta = TTAInference(model, device)
    
    dice_s = [0.]*num_classes
    iou_s = [0.]*num_classes
    batches = 0
    
    with torch.no_grad():
        for imgs, tgts in tqdm(loader, desc="Eval", leave=False):
            imgs, tgts = imgs.to(device), tgts.to(device)
            
            if use_tta:
                preds = tta.predict_8x(imgs)
            else:
                out = model(imgs)['output']
                preds = out.argmax(1)
            
            batches += 1
            
            for c in range(num_classes):
                pc = (preds == c).float().view(-1)
                tc = (tgts == c).float().view(-1)
                inter = (pc * tc).sum()
                dice_s[c] += ((2.*inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)).item()
                iou_s[c] += ((inter + 1e-6) / (pc.sum() + tc.sum() - inter + 1e-6)).item()
    
    m = {'dice': [], 'iou': []}
    for c in range(num_classes):
        m['dice'].append(dice_s[c] / max(batches, 1))
        m['iou'].append(iou_s[c] / max(batches, 1))
    
    m['mean_dice'] = np.mean(m['dice'][1:])
    m['mean_iou'] = np.mean(m['iou'][1:])
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--use_dog', action='store_true')
    parser.add_argument('--fine_head_type', type=str, default='shearlet', choices=['gabor', 'shearlet'])
    parser.add_argument('--early_stop', type=int, default=30)
    
    # Block type selection (thay thế Mamba)
    parser.add_argument('--block_type', type=str, default='convnext',
                       choices=['convnext', 'dcn', 'inverted_residual', 'swin', 'fno', 'wavelet', 'rwkv', 'none'],
                       help='Block type: convnext, dcn, inverted_residual, swin, fno, wavelet, rwkv, none')
    parser.add_argument('--block_depth', type=int, default=2, help='Number of blocks per stage')
    
    # SOTA options
    parser.add_argument('--boundary_weight', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--use_tta', action='store_true', help='Enable TTA for validation')
    parser.add_argument('--tta_test', action='store_true', help='Enable 8x TTA for test set')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Config
    in_channels = 3
    num_classes = 4
    img_size = 224
    
    print(f"\n{'='*70}")
    print("ACDC SOTA Training")
    print(f"{'='*70}")
    
    # Model
    model = EGMNet(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        use_hrnet=True,
        use_mamba=False,
        use_spectral=False,
        use_fine_head=True,
        use_dog=args.use_dog,
        fine_head_type=args.fine_head_type,
        block_type=args.block_type,
        block_depth=args.block_depth
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Model ---")
    print(f"  Parameters:     {params:,}")
    print(f"  Block Type:     {args.block_type} (depth={args.block_depth})")
    print(f"  DoG:            {'✓' if args.use_dog else '✗'}")
    print(f"  Fine Head:      {args.fine_head_type}")
    print(f"\n--- SOTA Strategies ---")
    print(f"  Boundary Loss:  weight={args.boundary_weight}, warmup={args.warmup_epochs}")
    print(f"  TTA Val:        {'✓' if args.use_tta else '✗'}")
    print(f"  TTA Test:       {'✓ 8x' if args.tta_test else '✗'}")
    
    # Data
    train_dir = os.path.join(args.data_dir, 'training')
    test_dir = os.path.join(args.data_dir, 'testing')
    
    train_dataset = ACDCDataset2D(train_dir, in_channels=in_channels)
    test_dataset = ACDCDataset2D(test_dir, in_channels=in_channels)
    
    # Volume-based split
    num_vols = len(train_dataset.vol_paths)
    vol_indices = list(range(num_vols))
    np.random.seed(42)
    np.random.shuffle(vol_indices)
    split = int(num_vols * 0.8)
    train_vols = set(vol_indices[:split])
    val_vols = set(vol_indices[split:])
    
    train_indices = [i for i, (v, s) in enumerate(train_dataset.index_map) if v in train_vols]
    val_indices = [i for i, (v, s) in enumerate(train_dataset.index_map) if v in val_vols]
    
    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\n--- Data ---")
    print(f"  Train: {len(train_ds)} slices ({len(train_vols)} vols)")
    print(f"  Val:   {len(val_ds)} slices ({len(val_vols)} vols)")
    print(f"  Test:  {len(test_dataset)} slices")
    
    # Loss & Optimizer
    criterion = CombinedSOTALoss(
        num_classes=num_classes,
        ce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=args.boundary_weight,
        warmup_epochs=args.warmup_epochs
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_dice = 0
    epochs_no_improve = 0
    
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}")
    
    for epoch in range(args.epochs):
        model.train()
        criterion.set_epoch(epoch)
        
        train_loss = 0
        loss_dict_sum = {}
        valid_batches = 0
        
        pbar = tqdm(train_loader, desc=f"E{epoch+1}", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            out = model(imgs)['output']
            
            if torch.isnan(out).any():
                continue
            
            loss, loss_dict = criterion(out, masks)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            valid_batches += 1
            
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        train_loss /= max(valid_batches, 1)
        
        # Eval
        torch.cuda.empty_cache()
        v = evaluate(model, val_loader, device, num_classes, use_tta=args.use_tta)
        
        # Print
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*70}")
        
        loss_str = " | ".join([f"{k}: {v/max(valid_batches,1):.4f}" for k, v in loss_dict_sum.items()])
        print(f"Loss: {loss_str}")
        
        print(f"\n{'Class':<6} {'Dice':>8} {'IoU':>8}")
        print("-"*30)
        for c in range(num_classes):
            print(f"{CLASS_MAP[c]:<6} {v['dice'][c]:>8.4f} {v['iou'][c]:>8.4f}")
        print("-"*30)
        print(f"{'AvgFG':<6} {v['mean_dice']:>8.4f} {v['mean_iou']:>8.4f}")
        
        # Save
        if v['mean_dice'] > best_dice:
            best_dice = v['mean_dice']
            torch.save(model.state_dict(), f"{args.save_dir}/best.pt")
            print(f"★ Best! Dice={best_dice:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.early_stop:
            print(f"\nEarly stop at epoch {epoch+1}")
            break
    
    # Test with TTA
    print(f"\n{'='*70}")
    print("TEST EVALUATION" + (" (8x TTA)" if args.tta_test else ""))
    print(f"{'='*70}")
    
    model.load_state_dict(torch.load(f"{args.save_dir}/best.pt", weights_only=True))
    t = evaluate(model, test_loader, device, num_classes, use_tta=args.tta_test)
    
    print(f"{'Class':<6} {'Dice':>8} {'IoU':>8}")
    print("-"*30)
    for c in range(num_classes):
        print(f"{CLASS_MAP[c]:<6} {t['dice'][c]:>8.4f} {t['iou'][c]:>8.4f}")
    print("-"*30)
    print(f"{'AvgFG':<6} {t['mean_dice']:>8.4f} {t['mean_iou']:>8.4f}")
    
    print(f"\n✓ Done! Best Val: {best_dice:.4f}, Test: {t['mean_dice']:.4f}")


if __name__ == '__main__':
    main()
