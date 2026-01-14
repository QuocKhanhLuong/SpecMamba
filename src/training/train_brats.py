"""
BraTS21 Training Script (Brain Tumor Segmentation)
4 modalities: T1, T1ce, T2, FLAIR
4 classes: BG, NCR, ED, ET
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from scipy.ndimage import distance_transform_edt, binary_erosion
from datetime import datetime

from models.hrnet_dcn import HRNetDCN
from data.brats_dataset import BraTSDataset2D
from losses.sota_loss import CombinedSOTALoss

CLASS_MAP = {0: 'BG', 1: 'NCR', 2: 'ED', 3: 'ET'}


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
    
    for vol_idx in vol_preds.keys():
        pred_3d = np.stack([p[1] for p in sorted(vol_preds[vol_idx], key=lambda x: x[0])], axis=0)
        target_3d = np.stack([t[1] for t in sorted(vol_targets[vol_idx], key=lambda x: x[0])], axis=0)
        
        for c in range(1, num_classes):
            pred_c = (pred_3d == c)
            target_c = (target_3d == c)
            inter = (pred_c & target_c).sum()
            dice = (2 * inter) / (pred_c.sum() + target_c.sum() + 1e-6)
            dice_3d[c].append(dice)
    
    return {'mean_dice': np.mean([np.mean(dice_3d[c]) for c in range(1, num_classes)])}


def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None, use_amp=False):
    model.train()
    total_loss = 0
    
    for imgs, masks in tqdm(loader, desc=f"E{epoch+1}", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(imgs)['output']
                loss, _ = criterion(out, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)['output']
            loss, _ = criterion(out, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="BraTS21 Training")
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/BraTS21/training')
    parser.add_argument('--batch_size', type=int, default=4)  # Smaller due to 4 channels
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_channels', type=int, default=32)  # Smaller for memory
    parser.add_argument('--use_pointrend', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--exp_name', type=str, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.exp_name is None:
        args.exp_name = f"brats_hrnet_dcn_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print(f"\n{'='*60}")
    print("BraTS21 Training - HRNet DCN")
    print(f"{'='*60}")
    
    # Model - 4 input channels for 4 modalities
    model = HRNetDCN(
        in_channels=4, num_classes=4, base_channels=args.base_channels,
        use_pointrend=args.use_pointrend
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data - use all 4 modalities
    dataset = BraTSDataset2D(args.data_dir, in_channels=4)
    
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
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_ds)} slices | Val: {len(val_ds)} slices")
    
    # Training
    criterion = CombinedSOTALoss(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    best_dice = 0
    
    for epoch in range(args.epochs):
        criterion.current_epoch = epoch
        loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, args.use_amp)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_3d(model, val_ds, device)
            print(f"E{epoch+1:03d} | Loss: {loss:.4f} | 3D Dice: {metrics['mean_dice']:.4f}")
            
            if metrics['mean_dice'] > best_dice:
                best_dice = metrics['mean_dice']
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.exp_name}_best.pt"))
    
    print(f"\nBest 3D Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
