"""
Block Benchmark Script
Test các block types lần lượt trên HRNet thuần (không fine-head, không spectral)
Full metrics: Dice, IoU, Precision, Recall for each class
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


def compute_hd95(pred, target):
    """Compute HD95 for 2D binary masks"""
    from scipy.ndimage import distance_transform_edt
    
    pred_np = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)
    
    if not pred_np.any() or not target_np.any():
        return float('inf')
    
    # Distance transform
    pred_dist = distance_transform_edt(~pred_np)
    target_dist = distance_transform_edt(~target_np)
    
    # Surface distances
    pred_surface = pred_np & ~np.roll(pred_np, 1, axis=0)
    target_surface = target_np & ~np.roll(target_np, 1, axis=0)
    
    if not pred_surface.any() or not target_surface.any():
        return 0.0
    
    # Hausdorff distances
    d_pred_to_target = target_dist[pred_surface]
    d_target_to_pred = pred_dist[target_surface]
    
    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    hd95 = np.percentile(all_distances, 95) if len(all_distances) > 0 else 0.0
    return hd95


def evaluate_full(model, loader, device, num_classes=4):
    """Full evaluation with Dice, IoU, Precision, Recall, HD95 per class"""
    model.eval()
    
    tp = [0]*num_classes
    fp = [0]*num_classes
    fn = [0]*num_classes
    dice_sum = [0.]*num_classes
    iou_sum = [0.]*num_classes
    hd95_sum = [0.]*num_classes
    hd95_count = [0]*num_classes
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
                
                dice_sum[c] += ((2.*inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)).item()
                iou_sum[c] += ((inter + 1e-6) / (pc.sum() + tc.sum() - inter + 1e-6)).item()
                tp[c] += inter.item()
                fp[c] += (pc.sum() - inter).item()
                fn[c] += (tc.sum() - inter).item()
                
                # HD95 per sample
                for b in range(preds.shape[0]):
                    pred_c = (preds[b] == c)
                    tgt_c = (tgts[b] == c)
                    if pred_c.any() and tgt_c.any():
                        hd = compute_hd95(pred_c, tgt_c)
                        if hd != float('inf'):
                            hd95_sum[c] += hd
                            hd95_count[c] += 1
    
    metrics = {
        'dice': [dice_sum[c] / batches for c in range(num_classes)],
        'iou': [iou_sum[c] / batches for c in range(num_classes)],
        'precision': [tp[c] / (tp[c] + fp[c] + 1e-6) for c in range(num_classes)],
        'recall': [tp[c] / (tp[c] + fn[c] + 1e-6) for c in range(num_classes)],
        'hd95': [hd95_sum[c] / max(hd95_count[c], 1) for c in range(num_classes)],
    }
    
    # Mean foreground (exclude BG)
    metrics['mean_dice'] = np.mean(metrics['dice'][1:])
    metrics['mean_iou'] = np.mean(metrics['iou'][1:])
    metrics['mean_prec'] = np.mean(metrics['precision'][1:])
    metrics['mean_rec'] = np.mean(metrics['recall'][1:])
    metrics['mean_hd95'] = np.mean(metrics['hd95'][1:])
    
    return metrics


def train_one_config(block_type, train_loader, val_loader, device, epochs=20, lr=1e-4, num_classes=4):
    """Train một config và trả về best metrics"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {block_type.upper()}")
    print(f"{'='*70}")
    
    try:
        model = EGMNet(
            in_channels=3,
            num_classes=num_classes,
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
    best_metrics = None
    best_model_state = None
    
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
        
        # Full evaluation
        metrics = evaluate_full(model, val_loader, device, num_classes)
        
        if metrics['mean_dice'] > best_dice:
            best_dice = metrics['mean_dice']
            best_metrics = metrics.copy()
            best_model_state = model.state_dict().copy()
        
        # Per-epoch summary
        print(f"  E{epoch+1}: Loss={train_loss/max(valid_batches,1):.4f} | "
              f"Dice={metrics['mean_dice']:.4f} | IoU={metrics['mean_iou']:.4f}")
    
    # Save best weights
    if best_metrics and best_model_state:
        save_path = f"weights/benchmark_{block_type}_best.pt"
        os.makedirs("weights", exist_ok=True)
        torch.save(best_model_state, save_path)
        print(f"  ✓ Saved: {save_path}")
    
    # Print final per-class metrics
    if best_metrics:
        print(f"\n  --- Best Results for {block_type.upper()} ---")
        print(f"  {'Class':<6} {'Dice':>8} {'IoU':>8} {'HD95':>8} {'Prec':>8} {'Rec':>8}")
        print(f"  {'-'*55}")
        for c in range(num_classes):
            print(f"  {CLASS_MAP[c]:<6} {best_metrics['dice'][c]:>8.4f} {best_metrics['iou'][c]:>8.4f} "
                  f"{best_metrics['hd95'][c]:>8.2f} {best_metrics['precision'][c]:>8.4f} {best_metrics['recall'][c]:>8.4f}")
        print(f"  {'-'*55}")
        print(f"  {'AvgFG':<6} {best_metrics['mean_dice']:>8.4f} {best_metrics['mean_iou']:>8.4f} "
              f"{best_metrics['mean_hd95']:>8.2f} {best_metrics['mean_prec']:>8.4f} {best_metrics['mean_rec']:>8.4f}")
    
    return best_metrics, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--blocks', type=str, nargs='+', default=BLOCK_TYPES)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("Block Benchmark - Pure HRNet (No Fine Head, No Spectral)")
    print(f"{'='*70}")
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
        metrics, params = train_one_config(
            block_type, train_loader, val_loader, device, 
            epochs=args.epochs, lr=args.lr
        )
        results[block_type] = {'metrics': metrics, 'params': params}
    
    # Final Summary Table
    print(f"\n{'='*85}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*85}")
    print(f"{'Block':<20} {'Params':>12} {'Dice':>8} {'IoU':>8} {'HD95':>8} {'Prec':>8} {'Rec':>8}")
    print("-"*85)
    
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]['metrics']['mean_dice'] if x[1]['metrics'] else 0, 
        reverse=True
    )
    
    for block_type, res in sorted_results:
        if res['metrics']:
            m = res['metrics']
            print(f"{block_type:<20} {res['params']:>12,} {m['mean_dice']:>8.4f} {m['mean_iou']:>8.4f} "
                  f"{m['mean_hd95']:>8.2f} {m['mean_prec']:>8.4f} {m['mean_rec']:>8.4f}")
        else:
            print(f"{block_type:<20} {'FAILED':>12} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
    
    print("-"*85)
    
    if sorted_results[0][1]['metrics']:
        best = sorted_results[0]
        print(f"\n★ Best: {best[0]} (Dice={best[1]['metrics']['mean_dice']:.4f}, HD95={best[1]['metrics']['mean_hd95']:.2f})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"benchmark_results_{timestamp}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Block Benchmark Results - {timestamp}\n")
        f.write(f"Epochs: {args.epochs}, LR: {args.lr}\n\n")
        f.write(f"{'Block':<20} {'Params':>12} {'Dice':>8} {'IoU':>8} {'HD95':>8} {'Prec':>8} {'Rec':>8}\n")
        f.write("-"*85 + "\n")
        for block_type, res in sorted_results:
            if res['metrics']:
                m = res['metrics']
                f.write(f"{block_type:<20} {res['params']:>12,} {m['mean_dice']:>8.4f} {m['mean_iou']:>8.4f} "
                        f"{m['mean_hd95']:>8.2f} {m['mean_prec']:>8.4f} {m['mean_rec']:>8.4f}\n")
            else:
                f.write(f"{block_type:<20} FAILED\n")
    print(f"\nResults saved to: {result_file}")


if __name__ == '__main__':
    main()
