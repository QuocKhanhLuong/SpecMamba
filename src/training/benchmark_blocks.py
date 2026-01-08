"""
Block Benchmark Script
Test các block types lần lượt trên HRNet thuần (không fine-head, không spectral)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime

from models.egm_net import EGMNet
from data.acdc_dataset import ACDCDataset2D


CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}

BLOCK_TYPES = ['none', 'convnext', 'dcn', 'inverted_residual', 'swin', 'fno', 'wavelet', 'rwkv']


def evaluate(model, loader, device, num_classes=4):
    model.eval()
    dice_s = [0.]*num_classes
    batches = 0
    
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            out = model(imgs)['output']
            preds = out.argmax(1)
            batches += 1
            
            for c in range(num_classes):
                pc = (preds == c).float().view(-1)
                tc = (tgts == c).float().view(-1)
                inter = (pc * tc).sum()
                dice_s[c] += ((2.*inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)).item()
    
    return np.mean([dice_s[c] / batches for c in range(1, num_classes)])


def train_one_config(block_type, train_loader, val_loader, device, epochs=20, lr=1e-4):
    """Train một config và trả về best val dice"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {block_type.upper()}")
    print(f"{'='*60}")
    
    try:
        model = EGMNet(
            in_channels=3,
            num_classes=4,
            img_size=224,
            use_hrnet=True,
            use_mamba=False,
            use_spectral=False,
            use_fine_head=False,
            use_dog=False,
            coarse_head_type="linear",
            block_type=block_type,
            block_depth=2
        ).to(device)
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        return None, None
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_dice = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        valid_batches = 0
        
        for imgs, masks in tqdm(train_loader, desc=f"E{epoch+1}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            out = model(imgs)['output']
            
            if torch.isnan(out).any():
                continue
            
            loss = criterion(out, masks)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            valid_batches += 1
        
        val_dice = evaluate(model, val_loader, device)
        
        if val_dice > best_dice:
            best_dice = val_dice
        
        print(f"  E{epoch+1}: Loss={train_loss/max(valid_batches,1):.4f}, Val Dice={val_dice:.4f}")
    
    print(f"  Best Dice: {best_dice:.4f}")
    return best_dice, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--blocks', type=str, nargs='+', default=BLOCK_TYPES)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print("Block Benchmark - Pure HRNet (No Fine Head, No Spectral)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs per block: {args.epochs}")
    print(f"Blocks to test: {args.blocks}")
    
    train_dir = os.path.join(args.data_dir, 'training')
    train_dataset = ACDCDataset2D(train_dir, in_channels=3)
    
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
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)
    
    print(f"\nTrain: {len(train_ds)} slices, Val: {len(val_ds)} slices")
    
    results = {}
    
    for block_type in args.blocks:
        torch.cuda.empty_cache()
        best_dice, params = train_one_config(
            block_type, train_loader, val_loader, device, 
            epochs=args.epochs, lr=args.lr
        )
        results[block_type] = {'dice': best_dice, 'params': params}
    
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Block':<20} {'Params':>12} {'Best Dice':>12}")
    print("-"*50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['dice'] or 0, reverse=True)
    
    for block_type, res in sorted_results:
        dice = f"{res['dice']:.4f}" if res['dice'] else "FAILED"
        params = f"{res['params']:,}" if res['params'] else "N/A"
        print(f"{block_type:<20} {params:>12} {dice:>12}")
    
    print("-"*50)
    
    if sorted_results[0][1]['dice']:
        print(f"\n★ Best: {sorted_results[0][0]} (Dice={sorted_results[0][1]['dice']:.4f})")


if __name__ == '__main__':
    main()
