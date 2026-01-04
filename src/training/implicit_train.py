"""
Training Script for Implicit Mamba Network.

Implements coordinate sampling training procedure for continuous neural
representations, enabling resolution-free medical image segmentation.

Key Features:
    - Coordinate-based point sampling instead of full image training
    - Boundary-aware sampling for improved edge accuracy
    - Multi-resolution validation capabilities

References:
    [1] Sitzmann et al., "Implicit Neural Representations with Periodic
        Activation Functions," NeurIPS, 2020.
    [2] Liu et al., "VMamba: Visual State Space Model," arXiv, 2024.
    [3] Tancik et al., "Fourier Features Let Networks Learn High Frequency
        Functions in Low Dimensional Domains," NeurIPS, 2020.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import logging
from typing import Tuple, Optional

from implicit_mamba import ImplicitMambaNet, ImplicitLoss

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
class ImplicitTrainingConfig:
    """Configuration for Implicit Mamba Network training.
    
    Attributes:
        num_samples: Number of coordinate samples per image.
        boundary_ratio: Ratio of boundary vs random samples.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        num_epochs: Total training epochs.
        batch_size: Training batch size.
        checkpoint_dir: Directory for saving checkpoints.
        device: Training device ('cuda' or 'cpu').
    """
    num_samples: int = 4096
    boundary_ratio: float = 0.3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    batch_size: int = 8
    checkpoint_dir: str = './checkpoints'
    device: str = 'cuda'


class CoordinateSamplingDataset(Dataset):
    """
    Dataset that provides images and masks for coordinate sampling.
    """
    
    def __init__(self, images: np.ndarray, masks: np.ndarray,
                 img_size: int = 256, normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            images: Array of shape (N, H, W) or (N, C, H, W)
            masks: Array of shape (N, H, W)
            img_size: Target image size
            normalize: Whether to normalize images
        """
        self.images = torch.from_numpy(images).float()
        self.masks = torch.from_numpy(masks).long()
        self.img_size = img_size
        
        # Ensure 4D images
        if self.images.ndim == 3:
            self.images = self.images.unsqueeze(1)
        
        # Normalize
        if normalize:
            for i in range(len(self.images)):
                img = self.images[i]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    self.images[i] = (img - img_min) / (img_max - img_min)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx]


class ImplicitTrainer:
    """
    Trainer for Implicit Mamba Network with coordinate sampling.
    """
    
    def __init__(self, model: ImplicitMambaNet, 
                 num_samples: int = 4096,
                 boundary_ratio: float = 0.3,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = "cuda"):
        """
        Initialize trainer.
        
        Args:
            model: ImplicitMambaNet model
            num_samples: Number of coordinate samples per image
            boundary_ratio: Ratio of boundary samples vs random samples
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Training device
        """
        self.model = model.to(device)
        self.device = device
        self.num_samples = num_samples
        self.boundary_ratio = boundary_ratio
        
        # Loss function
        self.loss_fn = ImplicitLoss(num_samples=num_samples, use_dice=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
    
    def compute_boundary_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary mask using gradient magnitude.
        
        Args:
            mask: Segmentation mask (B, H, W)
            
        Returns:
            Boundary mask (B, H, W)
        """
        mask_float = mask.float().unsqueeze(1)  # (B, 1, H, W)
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=mask.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=mask.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        grad_x = torch.nn.functional.conv2d(mask_float, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(mask_float, sobel_y, padding=1)
        
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        boundary = (grad_mag > 0).float().squeeze(1)  # (B, H, W)
        
        return boundary
    
    def sample_coordinates(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample coordinates with boundary emphasis.
        
        Args:
            mask: Ground truth mask (B, H, W)
            
        Returns:
            coords: Sampled coordinates (B, N, 2) in [-1, 1]
            labels: Labels at sampled points (B, N)
        """
        B, H, W = mask.shape
        device = mask.device
        
        num_boundary = int(self.num_samples * self.boundary_ratio)
        num_random = self.num_samples - num_boundary
        
        coords_list = []
        labels_list = []
        
        for b in range(B):
            # Get boundary pixels
            boundary = self.compute_boundary_mask(mask[b:b+1])[0]  # (H, W)
            boundary_pixels = torch.nonzero(boundary, as_tuple=False)  # (K, 2)
            
            # Sample from boundary
            if len(boundary_pixels) > 0:
                indices = torch.randint(0, len(boundary_pixels), (num_boundary,), device=device)
                boundary_coords = boundary_pixels[indices]  # (num_boundary, 2)
            else:
                # If no boundary, use random
                boundary_coords = torch.stack([
                    torch.randint(0, H, (num_boundary,), device=device),
                    torch.randint(0, W, (num_boundary,), device=device)
                ], dim=1)
            
            # Random samples
            random_coords = torch.stack([
                torch.randint(0, H, (num_random,), device=device),
                torch.randint(0, W, (num_random,), device=device)
            ], dim=1)
            
            # Combine
            all_coords = torch.cat([boundary_coords, random_coords], dim=0)  # (N, 2)
            
            # Convert to normalized coordinates [-1, 1]
            # all_coords[:, 0] is y (row), all_coords[:, 1] is x (col)
            norm_coords = torch.zeros(self.num_samples, 2, device=device)
            norm_coords[:, 0] = (all_coords[:, 1].float() / (W - 1)) * 2 - 1  # x
            norm_coords[:, 1] = (all_coords[:, 0].float() / (H - 1)) * 2 - 1  # y
            
            # Get labels
            labels = mask[b, all_coords[:, 0], all_coords[:, 1]]
            
            coords_list.append(norm_coords)
            labels_list.append(labels)
        
        coords = torch.stack(coords_list, dim=0)  # (B, N, 2)
        labels = torch.stack(labels_list, dim=0)  # (B, N)
        
        return coords, labels
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Sample coordinates
            coords, labels = self.sample_coordinates(masks)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_logits = self.model.query_points(images, coords)
            
            # Compute loss
            loss = self.loss_fn(pred_logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred_classes = pred_logits.argmax(dim=-1)
            total_correct += (pred_classes == labels).sum().item()
            total_samples += labels.numel()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': total_correct / total_samples
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, 
                 full_resolution: bool = True) -> dict:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            full_resolution: Whether to evaluate at full resolution
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        total_dice = 0
        num_batches = 0
        
        progress_bar = tqdm(val_loader, desc="Validation")
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            if full_resolution:
                # Render at full resolution
                H, W = masks.shape[1], masks.shape[2]
                pred_logits = self.model.render(images, output_size=(H, W))
                
                # Compute metrics
                pred_classes = pred_logits.argmax(dim=1)
                
                # Dice score
                for c in range(self.model.num_classes):
                    pred_c = (pred_classes == c).float()
                    mask_c = (masks == c).float()
                    intersection = (pred_c * mask_c).sum()
                    union = pred_c.sum() + mask_c.sum()
                    dice = (2 * intersection + 1e-5) / (union + 1e-5)
                    total_dice += dice.item()
            else:
                # Sample points for validation
                coords, labels = self.sample_coordinates(masks)
                pred_logits = self.model.query_points(images, coords)
                
                loss = self.loss_fn(pred_logits, labels)
                total_loss += loss.item()
            
            num_batches += 1
        
        metrics = {}
        if full_resolution:
            metrics['dice'] = total_dice / (num_batches * self.model.num_classes)
        else:
            metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def train(self, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              save_dir: str = "./checkpoints"):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_dice = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, full_resolution=True)
                print(f"Val Dice: {val_metrics['dice']:.4f}")
                
                # Save best model
                if val_metrics['dice'] > best_dice:
                    best_dice = val_metrics['dice']
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_dice': best_dice
                    }, save_path / "best_model.pt")
                    print(f"  -> New best model saved!")
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, save_path / f"epoch_{epoch + 1}.pt")


def create_dummy_dataset(num_samples: int = 100, img_size: int = 256,
                         num_classes: int = 3):
    """Create dummy dataset for testing."""
    images = np.random.rand(num_samples, img_size, img_size).astype(np.float32)
    masks = np.random.randint(0, num_classes, (num_samples, img_size, img_size))
    return CoordinateSamplingDataset(images, masks.astype(np.int64))


if __name__ == "__main__":
    print("=" * 60)
    print("Implicit Mamba Network - Training Demo")
    print("=" * 60)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    model = ImplicitMambaNet(
        in_channels=1,
        num_classes=3,
        img_size=256,
        base_channels=64,
        num_stages=4,
        depth=2,
        fourier_scale=10.0,
        fourier_size=256,
        hidden_dim=256,
        num_mlp_layers=4,
        use_spectral=True,
        use_siren=True,
        multiscale=False
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy dataset
    print("\nCreating dummy dataset...")
    dataset = create_dummy_dataset(num_samples=50, img_size=256, num_classes=3)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create trainer
    trainer = ImplicitTrainer(
        model=model,
        num_samples=4096,
        boundary_ratio=0.3,
        learning_rate=1e-4,
        device=device
    )
    
    # Train for a few epochs
    print("\nStarting training (3 epochs for demo)...")
    trainer.train(train_loader, num_epochs=3)
    
    # Test rendering at different resolutions
    print("\nTesting multi-resolution rendering...")
    model.eval()
    test_image = torch.randn(1, 1, 256, 256).to(device)
    
    with torch.no_grad():
        for res in [128, 256, 512, 1024]:
            output = model.render(test_image, output_size=(res, res))
            print(f"  Resolution {res}x{res}: {output.shape}")
    
    print("\n" + "=" * 60)
    print("Training demo complete!")
    print("=" * 60)
