"""
Training Script for EGM-Net (Energy-Gated Gabor Mamba Network).

Implements specialized training procedure for implicit neural representations
with energy-gated boundary refinement for medical image segmentation.

Key Features:
    - Point sampling instead of full-image training
    - Multi-scale coordinate sampling for resolution-free learning
    - Energy-aware loss weighting for boundary emphasis
    - Progressive training (coarse-to-fine)

References:
    [1] O. Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac
        Multi-structures Segmentation and Diagnosis," IEEE TMI, 2018.
    [2] Sitzmann et al., "Implicit Neural Representations with Periodic
        Activation Functions," NeurIPS, 2020.
    [3] Liu et al., "VMamba: Visual State Space Model," arXiv, 2024.
"""

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

from egm_net import EGMNet, EGMNetLite
from physics_loss import DiceLoss, FocalLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EGMTrainingConfig:
    """Configuration for EGM-Net training.
    
    Attributes:
        learning_rate: Initial learning rate for optimizer.
        weight_decay: L2 regularization weight.
        num_epochs: Total training epochs.
        batch_size: Training batch size.
        num_points: Number of coordinate samples per image.
        boundary_ratio: Ratio of boundary vs uniform samples.
        spatial_weight: Loss weight for coarse branch.
        point_weight: Loss weight for fine branch.
        consistency_weight: Loss weight for coarse-fine consistency.
        energy_weight: Extra weight for high-energy boundary regions.
        save_interval: Checkpoint saving frequency (epochs).
        checkpoint_dir: Directory for saving checkpoints.
        device: Training device ('cuda' or 'cpu').
    """
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
    """
    Coordinate sampler for implicit neural network training.
    
    Instead of training on full images, we sample points and their labels.
    This enables resolution-free learning.
    """
    
    def __init__(self, num_points: int = 4096, 
                 boundary_ratio: float = 0.5,
                 jitter_scale: float = 0.01):
        """
        Initialize sampler.
        
        Args:
            num_points: Number of points to sample per image
            boundary_ratio: Ratio of boundary points (vs uniform)
            jitter_scale: Scale of random jitter for boundary points
        """
        self.num_points = num_points
        self.boundary_ratio = boundary_ratio
        self.jitter_scale = jitter_scale
    
    def sample(self, mask: torch.Tensor, 
               energy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample coordinates and labels from mask.
        
        Args:
            mask: Segmentation mask (B, H, W) with class labels
            energy: Optional energy map (B, 1, H, W) for importance sampling
            
        Returns:
            Tuple of (coords, labels):
            - coords: (B, num_points, 2) normalized to [-1, 1]
            - labels: (B, num_points) class labels
        """
        B, H, W = mask.shape
        device = mask.device
        
        num_boundary = int(self.num_points * self.boundary_ratio)
        num_uniform = self.num_points - num_boundary
        
        all_coords = []
        all_labels = []
        
        for b in range(B):
            coords_list = []
            
            # 1. Uniform random sampling
            if num_uniform > 0:
                y_uniform = torch.rand(num_uniform, device=device) * 2 - 1
                x_uniform = torch.rand(num_uniform, device=device) * 2 - 1
                uniform_coords = torch.stack([x_uniform, y_uniform], dim=-1)
                coords_list.append(uniform_coords)
            
            # 2. Boundary-focused sampling (using energy map or gradient)
            if num_boundary > 0:
                if energy is not None:
                    # Sample proportional to energy
                    energy_b = energy[b, 0]  # (H, W)
                    energy_flat = energy_b.view(-1)
                    probs = energy_flat / (energy_flat.sum() + 1e-8)
                    
                    indices = torch.multinomial(probs, num_boundary, replacement=True)
                    y_idx = indices // W
                    x_idx = indices % W
                else:
                    # Use gradient magnitude for boundary detection
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
                
                # Convert to normalized coordinates [-1, 1]
                y_norm = (y_idx.float() / (H - 1)) * 2 - 1
                x_norm = (x_idx.float() / (W - 1)) * 2 - 1
                
                # Add jitter
                y_norm = y_norm + torch.randn_like(y_norm) * self.jitter_scale
                x_norm = x_norm + torch.randn_like(x_norm) * self.jitter_scale
                
                # Clamp to valid range
                y_norm = torch.clamp(y_norm, -1, 1)
                x_norm = torch.clamp(x_norm, -1, 1)
                
                boundary_coords = torch.stack([x_norm, y_norm], dim=-1)
                coords_list.append(boundary_coords)
            
            # Combine
            coords = torch.cat(coords_list, dim=0)  # (num_points, 2)
            
            # Sample labels at coordinates
            # Convert normalized coords to pixel indices
            x_px = ((coords[:, 0] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
            y_px = ((coords[:, 1] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
            labels = mask[b, y_px, x_px]
            
            all_coords.append(coords)
            all_labels.append(labels)
        
        coords = torch.stack(all_coords, dim=0)  # (B, num_points, 2)
        labels = torch.stack(all_labels, dim=0)  # (B, num_points)
        
        return coords, labels


class EGMNetLoss(nn.Module):
    """
    Combined loss for EGM-Net training.
    
    Components:
    1. Spatial loss: Dice + CE for coarse branch
    2. Point loss: CE for fine branch (implicit points)
    3. Consistency loss: Encourage coarse and fine to agree
    4. Energy-weighted loss: Focus on boundary regions
    """
    
    def __init__(self, spatial_weight: float = 1.0,
                 point_weight: float = 1.0,
                 consistency_weight: float = 0.1,
                 energy_weight: float = 0.5):
        """
        Initialize loss.
        
        Args:
            spatial_weight: Weight for spatial (coarse) loss
            point_weight: Weight for point (fine) loss
            consistency_weight: Weight for consistency loss
            energy_weight: Extra weight for high-energy (boundary) regions
        """
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
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dict with 'output', 'coarse', 'fine', 'energy'
            target: Ground truth mask (B, H, W)
            point_coords: Sampled coordinates (B, N, 2) for point loss
            point_labels: Labels at sampled points (B, N)
            
        Returns:
            Dictionary with loss components
        """
        coarse = outputs['coarse']
        fine = outputs['fine']
        energy = outputs['energy']
        fused = outputs['output']
        
        B, C, H, W = coarse.shape
        
        losses = {}
        
        # 1. Spatial loss for coarse branch
        dice = self.dice_loss(coarse, target)
        
        # Resize energy to match coarse
        energy_resized = F.interpolate(energy, size=(H, W), 
                                       mode='bilinear', align_corners=True)
        
        # Energy-weighted CE
        ce = self.ce_loss(coarse, target.long())  # (B, H, W)
        weight = 1.0 + self.energy_weight * energy_resized.squeeze(1)
        weighted_ce = (ce * weight).mean()
        
        losses['coarse_dice'] = dice
        losses['coarse_ce'] = weighted_ce
        losses['spatial'] = self.spatial_weight * (dice + weighted_ce)
        
        # 2. Point loss for fine branch (if provided)
        if point_coords is not None and point_labels is not None:
            # The fine branch output should already be at point locations
            # But if it's a grid, we need to sample
            if fine.dim() == 4:  # (B, C, H, W) - grid output
                # Sample at point coordinates
                grid = point_coords.unsqueeze(2)  # (B, N, 1, 2)
                fine_points = F.grid_sample(fine, grid, mode='bilinear', 
                                           align_corners=True)
                fine_points = fine_points.squeeze(-1).permute(0, 2, 1)  # (B, N, C)
            else:
                fine_points = fine  # Already (B, N, C)
            
            # Point CE loss
            fine_points_flat = fine_points.reshape(-1, C)  # (B*N, C)
            point_labels_flat = point_labels.reshape(-1)    # (B*N,)
            point_ce = F.cross_entropy(fine_points_flat, point_labels_flat.long())
            
            losses['point_ce'] = point_ce
            losses['point'] = self.point_weight * point_ce
        else:
            # Use grid loss for fine branch
            fine_dice = self.dice_loss(fine, target)
            losses['fine_dice'] = fine_dice
            losses['point'] = self.point_weight * fine_dice
        
        # 3. Consistency loss
        # Encourage coarse and fine to agree (KL divergence)
        coarse_probs = F.softmax(coarse, dim=1)
        fine_probs = F.softmax(fine, dim=1)
        kl_div = F.kl_div(coarse_probs.log(), fine_probs, reduction='batchmean')
        
        losses['consistency'] = self.consistency_weight * kl_div
        
        # 4. Final (fused) output loss
        fused_dice = self.dice_loss(fused, target)
        fused_ce = F.cross_entropy(fused, target.long())
        losses['fused'] = fused_dice + fused_ce
        
        # Total loss
        losses['total'] = (losses['spatial'] + losses['point'] + 
                          losses['consistency'] + losses['fused'])
        
        return losses


class EGMNetTrainer:
    """
    Trainer for EGM-Net with progressive training strategy.
    
    Training phases:
    1. Coarse phase: Train coarse branch only
    2. Fine phase: Train fine branch with point sampling
    3. Joint phase: Train everything end-to-end
    """
    
    def __init__(self, model: EGMNet, config: dict, device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: EGM-Net model
            config: Training configuration
            device: Training device
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Loss
        self.loss_fn = EGMNetLoss(
            spatial_weight=config.get('spatial_weight', 1.0),
            point_weight=config.get('point_weight', 1.0),
            consistency_weight=config.get('consistency_weight', 0.1),
            energy_weight=config.get('energy_weight', 0.5)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100)
        )
        
        # Coordinate sampler
        self.sampler = CoordinateSampler(
            num_points=config.get('num_points', 4096),
            boundary_ratio=config.get('boundary_ratio', 0.5)
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader, 
                    use_point_sampling: bool = True) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Sample points for fine branch
            if use_point_sampling:
                coords, labels = self.sampler.sample(masks, outputs['energy'])
                coords = coords.to(self.device)
                labels = labels.to(self.device)
            else:
                coords, labels = None, None
            
            # Compute loss
            losses = self.loss_fn(outputs, masks, coords, labels)
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': losses['total'].item()})
        
        # Average losses
        for k in total_losses:
            total_losses[k] /= num_batches
        
        return total_losses
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
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
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs (overrides config)
        """
        num_epochs = num_epochs or self.config.get('num_epochs', 100)
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_losses = self.train_epoch(train_loader, use_point_sampling=True)
            print(f"Train - Total: {train_losses['total']:.4f}, "
                  f"Spatial: {train_losses['spatial']:.4f}, "
                  f"Point: {train_losses['point']:.4f}")
            
            # Validation
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                print(f"Val   - Total: {val_losses['total']:.4f}, "
                      f"Fused: {val_losses['fused']:.4f}")
            
            # Learning rate step
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch + 1)
    
    def save_checkpoint(self, epoch: int):
        """Save checkpoint."""
        path = self.checkpoint_dir / f"egm_net_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {path}")
        return checkpoint['epoch']


def create_dummy_dataset(num_samples: int = 100, img_size: int = 256,
                        num_classes: int = 3) -> TensorDataset:
    """Create dummy dataset for testing."""
    images = torch.randn(num_samples, 1, img_size, img_size)
    masks = torch.randint(0, num_classes, (num_samples, img_size, img_size))
    return TensorDataset(images, masks)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EGM-Net Training Pipeline")
    print("=" * 60)
    
    # Configuration
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 3,  # Short test
        'batch_size': 4,
        'num_points': 2048,
        'boundary_ratio': 0.5,
        'spatial_weight': 1.0,
        'point_weight': 1.0,
        'consistency_weight': 0.1,
        'energy_weight': 0.5,
        'save_interval': 10,
        'checkpoint_dir': './checkpoints_egm'
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = EGMNetLite(in_channels=1, num_classes=3, img_size=256)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    print("\nCreating dummy dataset...")
    dataset = create_dummy_dataset(num_samples=32, img_size=256, num_classes=3)
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = EGMNetTrainer(model, config, device=device)
    
    # Train
    print("\nStarting training...")
    trainer.train(train_loader, num_epochs=config['num_epochs'])
    
    print("\n" + "=" * 60)
    print("✓ Training test completed!")
    print("=" * 60)
