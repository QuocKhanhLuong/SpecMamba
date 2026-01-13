"""
Advanced Architecture Test Script
Test 2 ideas:
1. Asymmetric Depth: Different block counts per stage (2,3,4,6) + PointRend
2. Hybrid Blocks: Alternating inverted_residual + dcn
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
from torch.utils.data import DataLoader
from datetime import datetime

from data.acdc_dataset import ACDCDataset2D

CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


# =============================================================================
# CUSTOM HRNET WITH CONFIGURABLE BLOCKS
# =============================================================================

class HRNetAdvanced(nn.Module):
    """HRNet with configurable block types and depths per stage."""
    
    def __init__(self, in_channels=3, base_channels=32, img_size=224,
                 stage_configs=None, use_pointrend=False, num_classes=4):
        """
        Args:
            stage_configs: List of stage configs, each is dict with:
                - 'blocks': List of block types e.g. ['dcn', 'dcn'] or ['inverted_residual', 'dcn']
            use_pointrend: Whether to use PointRend for boundary refinement
        """
        super().__init__()
        
        # Imports
        from models.blocks import get_block
        from models.hrnet_mamba import HRNetStem, Bottleneck, FuseLayer
        
        self.get_block = get_block
        self.num_classes = num_classes
        
        # Default: asymmetric DCN
        if stage_configs is None:
            stage_configs = [
                {'blocks': ['dcn'] * 2},      # Stage 2: 2 blocks
                {'blocks': ['dcn'] * 3},      # Stage 3: 3 blocks
                {'blocks': ['dcn'] * 4},      # Stage 4: 4 blocks (originally 6, reduced for memory)
            ]
        
        self.stage_configs = stage_configs
        
        # Stem
        self.stem = HRNetStem(in_channels, 64)
        
        # Layer 1 (Bottleneck)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=nn.Sequential(
                nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256)
            )),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        
        C = base_channels
        s = img_size // 4
        
        # Transitions and Stages
        self.transition1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, C, 3, 1, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(256, C*2, 3, 2, 1, bias=False), nn.BatchNorm2d(C*2), nn.ReLU(True))
        ])
        
        self.stage2 = self._make_stage([C, C*2], stage_configs[0]['blocks'], [(s, s), (s//2, s//2)], FuseLayer)
        
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(nn.Conv2d(C*2, C*4, 3, 2, 1, bias=False), nn.BatchNorm2d(C*4), nn.ReLU(True))
        ])
        
        self.stage3 = self._make_stage([C, C*2, C*4], stage_configs[1]['blocks'], 
                                        [(s, s), (s//2, s//2), (s//4, s//4)], FuseLayer)
        
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(nn.Conv2d(C*4, C*8, 3, 2, 1, bias=False), nn.BatchNorm2d(C*8), nn.ReLU(True))
        ])
        
        self.stage4 = self._make_stage([C, C*2, C*4, C*8], stage_configs[2]['blocks'],
                                        [(s, s), (s//2, s//2), (s//4, s//4), (s//8, s//8)], FuseLayer)
        
        # Output channels
        self.out_channels = C + C*2 + C*4 + C*8
        
        # Segmentation head
        self.seg_head = nn.Conv2d(self.out_channels, num_classes, 1)
        
        # Optional PointRend
        self.use_pointrend = use_pointrend
        if use_pointrend:
            from layers.pointrend import PointRend
            self.pointrend = PointRend(
                in_channels=self.out_channels,
                num_classes=num_classes,
                num_points=1024,
                hidden_dim=128
            )
    
    def _make_stage(self, channels_list, block_types, sizes, FuseLayer):
        """Create a stage with configurable block types."""
        branches = nn.ModuleList()
        
        for idx, ch in enumerate(channels_list):
            blocks = []
            for bt in block_types:
                blocks.append(self.get_block(bt, ch))
            branches.append(nn.Sequential(*blocks))
        
        fuse = FuseLayer(channels_list, channels_list)
        
        return nn.ModuleDict({
            'branches': branches,
            'fuse': fuse
        })
    
    def _forward_stage(self, stage, x_list):
        out = [stage['branches'][i](x_list[i]) for i in range(len(x_list))]
        return stage['fuse'](out)
    
    def forward(self, x):
        target_size = x.shape[2:]
        
        x = self.stem(x)
        x = self.layer1(x)
        
        x_list = [self.transition1[0](x), self.transition1[1](x)]
        x_list = self._forward_stage(self.stage2, x_list)
        
        x_list = [
            self.transition2[0](x_list[0]),
            self.transition2[1](x_list[1]),
            self.transition2[2](x_list[1])
        ]
        x_list = self._forward_stage(self.stage3, x_list)
        
        x_list = [
            self.transition3[0](x_list[0]),
            self.transition3[1](x_list[1]),
            self.transition3[2](x_list[2]),
            self.transition3[3](x_list[2])
        ]
        x_list = self._forward_stage(self.stage4, x_list)
        
        # Aggregate multi-scale features
        x0_h, x0_w = x_list[0].shape[2:]
        feats = [
            x_list[0],
            F.interpolate(x_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
            F.interpolate(x_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
            F.interpolate(x_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
        ]
        features = torch.cat(feats, dim=1)
        
        # Segmentation
        logits = self.seg_head(features)
        
        if self.use_pointrend:
            logits = self.pointrend(logits, features, target_size)
        else:
            logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=True)
        
        return {'output': logits}


# =============================================================================
# EVALUATION
# =============================================================================

def compute_hd95(pred, target):
    from scipy.ndimage import distance_transform_edt
    pred_np = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)
    
    if not pred_np.any() or not target_np.any():
        return float('inf')
    
    pred_dist = distance_transform_edt(~pred_np)
    target_dist = distance_transform_edt(~target_np)
    
    pred_surface = pred_np & ~np.roll(pred_np, 1, axis=0)
    target_surface = target_np & ~np.roll(target_np, 1, axis=0)
    
    if not pred_surface.any() or not target_surface.any():
        return 0.0
    
    d_pred_to_target = target_dist[pred_surface]
    d_target_to_pred = pred_dist[target_surface]
    
    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return np.percentile(all_distances, 95) if len(all_distances) > 0 else 0.0


def evaluate(model, loader, device, num_classes=4):
    model.eval()
    
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
                
                for b in range(preds.shape[0]):
                    pred_c = (preds[b] == c)
                    tgt_c = (tgts[b] == c)
                    if pred_c.any() and tgt_c.any():
                        hd = compute_hd95(pred_c, tgt_c)
                        if hd != float('inf'):
                            hd95_sum[c] += hd
                            hd95_count[c] += 1
    
    return {
        'mean_dice': np.mean([dice_sum[c] / batches for c in range(1, num_classes)]),
        'mean_iou': np.mean([iou_sum[c] / batches for c in range(1, num_classes)]),
        'mean_hd95': np.mean([hd95_sum[c] / max(hd95_count[c], 1) for c in range(1, num_classes)])
    }


def train_config(name, model, train_loader, val_loader, device, epochs=50, lr=1e-4):
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_dice = 0
    best_metrics = None
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        valid_batches = 0
        
        for imgs, masks in tqdm(train_loader, desc=f"E{epoch+1}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            out = model(imgs)['output']
            loss = criterion(out, masks)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            valid_batches += 1
        
        metrics = evaluate(model, val_loader, device)
        
        if metrics['mean_dice'] > best_dice:
            best_dice = metrics['mean_dice']
            best_metrics = metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        print(f"  E{epoch+1}: Loss={train_loss/max(valid_batches,1):.4f} | "
              f"Dice={metrics['mean_dice']:.4f} | HD95={metrics['mean_hd95']:.2f}")
    
    # Save best
    os.makedirs("weights", exist_ok=True)
    save_name = name.lower().replace(" ", "_").replace("+", "_")
    torch.save(best_state, f"weights/advanced_{save_name}_best.pt")
    print(f"  ✓ Saved: weights/advanced_{save_name}_best.pt")
    
    return best_metrics, params


# =============================================================================
# CONFIGURATIONS
# =============================================================================

CONFIGS = {
    # Baseline: BasicBlock
    "Baseline (Basic)": {
        'stage_configs': [
            {'blocks': ['basic'] * 2},
            {'blocks': ['basic'] * 2},
            {'blocks': ['basic'] * 2},
        ],
        'use_pointrend': False
    },
    
    # DCN Baseline
    "DCN Uniform": {
        'stage_configs': [
            {'blocks': ['dcn'] * 2},
            {'blocks': ['dcn'] * 2},
            {'blocks': ['dcn'] * 2},
        ],
        'use_pointrend': False
    },
    
    # IDEA 1: Asymmetric DCN Depth
    "Asymmetric DCN (2-3-4)": {
        'stage_configs': [
            {'blocks': ['dcn'] * 2},
            {'blocks': ['dcn'] * 4},
            {'blocks': ['dcn'] * 6},
        ],
        'use_pointrend': False
    },
    
    # IDEA 1 + PointRend
    "Asymmetric DCN + PointRend": {
        'stage_configs': [
            {'blocks': ['dcn'] * 2},
            {'blocks': ['dcn'] * 4},
            {'blocks': ['dcn'] * 6},
        ],
        'use_pointrend': True
    },
    
    # IDEA 2: Hybrid Inverted Residual + DCN
    "Hybrid IR-DCN (4 blocks)": {
        'stage_configs': [
            {'blocks': ['inverted_residual', 'dcn']},
            {'blocks': ['inverted_residual', 'dcn', 'inverted_residual', 'dcn']},
            {'blocks': ['inverted_residual', 'dcn', 'inverted_residual', 'dcn', 'inverted_residual', 'dcn']},
        ],
        'use_pointrend': False
    },
    
    # IDEA 2 + PointRend
    "Hybrid IR-DCN + PointRend": {
        'stage_configs': [
            {'blocks': ['inverted_residual', 'dcn']},
            {'blocks': ['inverted_residual', 'dcn', 'inverted_residual', 'dcn']},
            {'blocks': ['inverted_residual', 'dcn', 'inverted_residual', 'dcn', 'inverted_residual', 'dcn']},
        ],
        'use_pointrend': True
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configs to test (default: all)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("Advanced Architecture Benchmark")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    
    # Data
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
    
    print(f"Train: {len(train_ds)} slices, Val: {len(val_ds)} slices")
    
    # Select configs
    configs_to_test = CONFIGS
    if args.configs:
        configs_to_test = {k: v for k, v in CONFIGS.items() if k in args.configs}
    
    results = {}
    
    for name, cfg in configs_to_test.items():
        torch.cuda.empty_cache()
        
        try:
            model = HRNetAdvanced(
                in_channels=3,
                base_channels=64,  
                img_size=224,
                stage_configs=cfg['stage_configs'],
                use_pointrend=cfg['use_pointrend'],
                num_classes=4
            ).to(device)
            
            metrics, params = train_config(name, model, train_loader, val_loader, device,
                                           epochs=args.epochs, lr=args.lr)
            results[name] = {'metrics': metrics, 'params': params}
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[name] = {'metrics': None, 'params': None}
    
    # Summary
    print(f"\n{'='*85}")
    print("RESULTS SUMMARY")
    print(f"{'='*85}")
    print(f"{'Config':<30} {'Params':>12} {'Dice':>8} {'IoU':>8} {'HD95':>8}")
    print("-"*85)
    
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['metrics']['mean_dice'] if x[1]['metrics'] else 0,
        reverse=True
    )
    
    for name, res in sorted_results:
        if res['metrics']:
            m = res['metrics']
            print(f"{name:<30} {res['params']:>12,} {m['mean_dice']:>8.4f} {m['mean_iou']:>8.4f} {m['mean_hd95']:>8.2f}")
        else:
            print(f"{name:<30} {'FAILED':>12}")
    
    print("-"*85)
    
    if sorted_results[0][1]['metrics']:
        best = sorted_results[0]
        print(f"\n★ Best: {best[0]} (Dice={best[1]['metrics']['mean_dice']:.4f}, HD95={best[1]['metrics']['mean_hd95']:.2f})")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"advanced_results_{timestamp}.txt", 'w') as f:
        f.write(f"Advanced Architecture Results - {timestamp}\n\n")
        for name, res in sorted_results:
            if res['metrics']:
                m = res['metrics']
                f.write(f"{name}: Dice={m['mean_dice']:.4f}, IoU={m['mean_iou']:.4f}, HD95={m['mean_hd95']:.2f}, Params={res['params']:,}\n")


if __name__ == '__main__':
    main()
