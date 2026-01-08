
import sys
import os
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from tqdm import tqdm
import math
import logging
import yaml

try:
    from src.models.egm_net import EGMNet, EGMNetLite
    from src.losses.physics_loss import DiceLoss, FocalLoss
except ImportError:

    from models.egm_net import EGMNet, EGMNetLite
    from losses.physics_loss import DiceLoss, FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class EGMTrainingConfig:

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    batch_size: int = 8
    num_points: int = 4096
    boundary_ratio: float = 0.5
    spatial_weight: float = 1.0
    point_weight: float = 1.0
    consistency_weight: float = 0.1
    energy_weight: float = 0.5
    save_interval: int = 10
    checkpoint_dir: str = './checkpoints'
    device: str = 'cuda'

class CoordinateSampler:

    def __init__(self, num_points: int = 4096,
                 boundary_ratio: float = 0.5,
                 jitter_scale: float = 0.01):

        self.num_points = num_points
        self.boundary_ratio = boundary_ratio
        self.jitter_scale = jitter_scale

    def sample(self, mask: torch.Tensor,
               energy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        B, H, W = mask.shape
        device = mask.device

        num_boundary = int(self.num_points * self.boundary_ratio)
        num_uniform = self.num_points - num_boundary

        all_coords = []
        all_labels = []

        for b in range(B):
            coords_list = []

            if num_uniform > 0:
                y_uniform = torch.rand(num_uniform, device=device) * 2 - 1
                x_uniform = torch.rand(num_uniform, device=device) * 2 - 1
                uniform_coords = torch.stack([x_uniform, y_uniform], dim=-1)
                coords_list.append(uniform_coords)

            if num_boundary > 0:
                if energy is not None:

                    energy_b = energy[b, 0]
                    energy_flat = energy_b.view(-1)
                    probs = energy_flat / (energy_flat.sum() + 1e-8)

                    indices = torch.multinomial(probs, num_boundary, replacement=True)
                    y_idx = indices // W
                    x_idx = indices % W
                else:

                    mask_b = mask[b].float()
                    grad_y = F.conv2d(mask_b.unsqueeze(0).unsqueeze(0),
                                     torch.tensor([[-1], [1]], dtype=mask_b.dtype,
                                                  device=device).view(1, 1, 2, 1),
                                     padding=(1, 0))[:, :, :-1, :]
                    grad_x = F.conv2d(mask_b.unsqueeze(0).unsqueeze(0),
                                     torch.tensor([[-1, 1]], dtype=mask_b.dtype,
                                                  device=device).view(1, 1, 1, 2),
                                     padding=(0, 1))[:, :, :, :-1]
                    grad_mag = torch.sqrt(grad_y**2 + grad_x**2 + 1e-8)
                    grad_flat = grad_mag.view(-1)
                    probs = grad_flat / (grad_flat.sum() + 1e-8)

                    indices = torch.multinomial(probs, num_boundary, replacement=True)
                    y_idx = indices // W
                    x_idx = indices % W

                y_norm = (y_idx.float() / (H - 1)) * 2 - 1
                x_norm = (x_idx.float() / (W - 1)) * 2 - 1

                y_norm = y_norm + torch.randn_like(y_norm) * self.jitter_scale
                x_norm = x_norm + torch.randn_like(x_norm) * self.jitter_scale

                y_norm = torch.clamp(y_norm, -1, 1)
                x_norm = torch.clamp(x_norm, -1, 1)

                boundary_coords = torch.stack([x_norm, y_norm], dim=-1)
                coords_list.append(boundary_coords)

            coords = torch.cat(coords_list, dim=0)

            x_px = ((coords[:, 0] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
            y_px = ((coords[:, 1] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
            labels = mask[b, y_px, x_px]

            all_coords.append(coords)
            all_labels.append(labels)

        coords = torch.stack(all_coords, dim=0)
        labels = torch.stack(all_labels, dim=0)

        return coords, labels

class EGMNetLoss(nn.Module):

    def __init__(self, spatial_weight: float = 1.0,
                 point_weight: float = 1.0,
                 consistency_weight: float = 0.1,
                 energy_weight: float = 0.5):

        super().__init__()
        self.spatial_weight = spatial_weight
        self.point_weight = point_weight
        self.consistency_weight = consistency_weight
        self.energy_weight = energy_weight

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs: Dict[str, torch.Tensor],
                target: torch.Tensor,
                point_coords: Optional[torch.Tensor] = None,
                point_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        coarse = outputs['coarse']
        fine = outputs['fine']
        energy = outputs['energy']
        fused = outputs['output']

        B, C, H, W = coarse.shape

        losses = {}

        dice = self.dice_loss(coarse, target)

        energy_resized = F.interpolate(energy, size=(H, W),
                                       mode='bilinear', align_corners=True)

        ce = self.ce_loss(coarse, target.long())
        weight = 1.0 + self.energy_weight * energy_resized.squeeze(1)
        weighted_ce = (ce * weight).mean()

        losses['coarse_dice'] = dice
        losses['coarse_ce'] = weighted_ce
        losses['spatial'] = self.spatial_weight * (dice + weighted_ce)

        if point_coords is not None and point_labels is not None:

            if fine.dim() == 4:

                grid = point_coords.unsqueeze(2)
                fine_points = F.grid_sample(fine, grid, mode='bilinear',
                                           align_corners=True)
                fine_points = fine_points.squeeze(-1).permute(0, 2, 1)
            else:
                fine_points = fine

            fine_points_flat = fine_points.reshape(-1, C)
            point_labels_flat = point_labels.reshape(-1)
            point_ce = F.cross_entropy(fine_points_flat, point_labels_flat.long())

            losses['point_ce'] = point_ce
            losses['point'] = self.point_weight * point_ce
        else:

            fine_dice = self.dice_loss(fine, target)
            losses['fine_dice'] = fine_dice
            losses['point'] = self.point_weight * fine_dice

        coarse_probs = F.softmax(coarse, dim=1)
        fine_probs = F.softmax(fine, dim=1)
        kl_div = F.kl_div(coarse_probs.log(), fine_probs, reduction='batchmean')

        losses['consistency'] = self.consistency_weight * kl_div

        fused_dice = self.dice_loss(fused, target)
        fused_ce = F.cross_entropy(fused, target.long())
        losses['fused'] = fused_dice + fused_ce

        losses['total'] = (losses['spatial'] + losses['point'] +
                          losses['consistency'] + losses['fused'])

        return losses

class EGMNetTrainer:

    def __init__(self, model: EGMNet, config: dict, device: str = 'cuda'):

        self.model = model.to(device)
        self.device = device
        self.config = config

        self.loss_fn = EGMNetLoss(
            spatial_weight=config.get('spatial_weight', 1.0),
            point_weight=config.get('point_weight', 1.0),
            consistency_weight=config.get('consistency_weight', 0.1),
            energy_weight=config.get('energy_weight', 0.5)
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100)
        )

        self.sampler = CoordinateSampler(
            num_points=config.get('num_points', 4096),
            boundary_ratio=config.get('boundary_ratio', 0.5)
        )

        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader,
                    use_point_sampling: bool = True) -> Dict[str, float]:

        self.model.train()

        total_losses = {}
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            if use_point_sampling:
                coords, labels = self.sampler.sample(masks, outputs['energy'])
                coords = coords.to(self.device)
                labels = labels.to(self.device)
            else:
                coords, labels = None, None

            losses = self.loss_fn(outputs, masks, coords, labels)

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': losses['total'].item()})

        for k in total_losses:
            total_losses[k] /= num_batches

        return total_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:

        self.model.eval()

        total_losses = {}
        num_batches = 0

        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            losses = self.loss_fn(outputs, masks)

            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1

        for k in total_losses:
            total_losses[k] /= num_batches

        return total_losses

    def train(self, train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None):

        num_epochs = num_epochs or self.config.get('num_epochs', 100)

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            train_losses = self.train_epoch(train_loader, use_point_sampling=True)
            print(f"Train - Total: {train_losses['total']:.4f}, "
                  f"Spatial: {train_losses['spatial']:.4f}, "
                  f"Point: {train_losses['point']:.4f}")

            if val_loader is not None:
                val_losses = self.validate(val_loader)
                print(f"Val   - Total: {val_losses['total']:.4f}, "
                      f"Fused: {val_losses['fused']:.4f}")

            self.scheduler.step()

            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch: int):

        path = self.checkpoint_dir / f"egm_net_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {path}")
        return checkpoint['epoch']

def create_dummy_dataset(num_samples: int = 100, img_size: int = 256,
                        num_classes: int = 3, in_channels: int = 3) -> TensorDataset:

    images = torch.randn(num_samples, in_channels, img_size, img_size)
    masks = torch.randint(0, num_classes, (num_samples, img_size, img_size))
    return TensorDataset(images, masks)

def parse_args():

    parser = argparse.ArgumentParser(
        description='Train EGM-Net for medical image segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')

    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_points', type=int, default=4096,
                       help='Number of points to sample per image')

    parser.add_argument('--in_channels', type=int, default=3,
                       help='Input channels (1=gray, 3=monogenic)')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of segmentation classes')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--use_lite', action='store_true',
                       help='Use EGMNetLite instead of full model')

    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    parser.add_argument('--test', action='store_true',
                       help='Run quick test with dummy data')

    parser.add_argument('--use_mamba', action='store_true', default=True)
    parser.add_argument('--no_mamba', dest='use_mamba', action='store_false')
    parser.add_argument('--use_spectral', action='store_true', default=True)
    parser.add_argument('--no_spectral', dest='use_spectral', action='store_false')
    parser.add_argument('--use_fine_head', action='store_true', default=True)
    parser.add_argument('--no_fine_head', dest='use_fine_head', action='store_false')
    
    parser.add_argument('--use_dog', action='store_true', default=False)
    parser.add_argument('--fine_head_type', type=str, default='gabor', choices=['gabor', 'shearlet'])

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 60)
    print("EGM-Net Training")
    print("=" * 60)

    config = {
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_points': args.num_points,
        'boundary_ratio': 0.5,
        'spatial_weight': 1.0,
        'point_weight': 1.0,
        'consistency_weight': 0.1,
        'energy_weight': 0.5,
        'save_interval': 10,
        'checkpoint_dir': args.checkpoint_dir
    }

    if args.config:
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if 'training' in yaml_config:
            config.update(yaml_config['training'])

        config['learning_rate'] = args.lr
        config['num_epochs'] = args.epochs
        config['batch_size'] = args.batch_size

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nCreating model...")
    if args.use_lite:
        model = EGMNetLite(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            img_size=args.img_size
        )
    else:
        model = EGMNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            img_size=args.img_size,
            use_hrnet=True,
            use_mamba=args.use_mamba,
            use_spectral=args.use_spectral,
            use_fine_head=args.use_fine_head,
            use_dog=args.use_dog,
            fine_head_type=args.fine_head_type
        )

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    if args.test:
        print("\nRunning test with dummy data...")
        dataset = create_dummy_dataset(
            num_samples=32,
            img_size=args.img_size,
            num_classes=args.num_classes,
            in_channels=args.in_channels
        )
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        config['num_epochs'] = 3
    else:
        print(f"\nLoading data from {args.data_dir}")
        
        from data.acdc_dataset import ACDCDataset2D
        from sklearn.model_selection import train_test_split
        import glob
        import os
        
        train_dir = os.path.join(args.data_dir, 'training')
        
        dataset = ACDCDataset2D(train_dir, use_memmap=True, in_channels=args.in_channels)
        
        num_vols = len(dataset.vol_paths) if hasattr(dataset, 'vol_paths') else len(dataset)
        train_size = int(num_vols * 0.8)
        val_size = num_vols - train_size
        
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    print("\nInitializing trainer...")
    trainer = EGMNetTrainer(model, config, device=device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    print("\nStarting training...")
    trainer.train(train_loader, num_epochs=config['num_epochs'])

    print("\n" + "=" * 60)
    print("✓ Training completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
