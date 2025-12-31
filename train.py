"""
Training Script for Spectral Mamba
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
import os

from spectral_mamba import SpectralVMUNet
from physics_loss import SpectralDualLoss, BoundaryAwareLoss
from config import TrainingConfig, ModelConfig, LossConfig


class Trainer:
    """Training wrapper for SpectralVMUNet."""
    
    def __init__(self, config: TrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Create model
        self.model = SpectralVMUNet(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            img_size=config.model.img_size,
            base_channels=config.model.base_channels,
            num_stages=config.model.num_stages,
            depth=config.model.depth
        ).to(device)
        
        # Create losses
        self.loss_fn = SpectralDualLoss(
            spatial_weight=config.loss.spatial_weight,
            freq_weight=config.loss.freq_weight,
            use_dice=config.loss.use_dice,
            use_focal=config.loss.use_focal
        )
        
        self.boundary_loss_fn = BoundaryAwareLoss(weight=config.loss.boundary_weight)
        
        # Create optimizer
        if config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        
        # Create scheduler
        if config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )
        elif config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss, _ = self.loss_fn(outputs, masks, return_components=True)
            
            # Add boundary loss if enabled
            if self.config.loss.boundary_weight > 0:
                boundary_loss = self.boundary_loss_fn(outputs, masks)
                loss = loss + self.config.loss.boundary_weight * boundary_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % self.config.log_interval == 0:
                progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        progress_bar = tqdm(val_loader, desc="Validation")
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images)
            loss, _ = self.loss_fn(outputs, masks, return_components=True)
            
            if self.config.loss.boundary_weight > 0:
                boundary_loss = self.boundary_loss_fn(outputs, masks)
                loss = loss + self.config.loss.boundary_weight * boundary_loss
            
            total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train for multiple epochs."""
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validation
            if val_loader is not None and (epoch + 1) % self.config.val_interval == 0:
                val_loss = self.validate(val_loader)
                print(f"Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")


def create_dummy_dataset(num_samples: int = 100, img_size: int = 256,
                        num_classes: int = 3):
    """Create dummy dataset for testing."""
    images = torch.randn(num_samples, 1, img_size, img_size)
    masks = torch.randint(0, num_classes, (num_samples, img_size, img_size))
    return TensorDataset(images, masks)


if __name__ == "__main__":
    # Configuration
    config = TrainingConfig(
        model=ModelConfig(
            in_channels=1,
            out_channels=3,
            img_size=256,
            base_channels=64,
            num_stages=4,
            depth=2
        ),
        loss=LossConfig(
            spatial_weight=1.0,
            freq_weight=0.1,
            use_dice=True,
            use_focal=True,
            boundary_weight=0.05
        ),
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=5,  # Short for testing
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Using device: {config.device}")
    
    # Create dummy dataset
    print("Creating dummy dataset...")
    dataset = create_dummy_dataset(num_samples=100, img_size=256, num_classes=3)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(config, device=config.device)
    
    # Train
    print("Starting training...")
    trainer.train(train_loader)
    
    print("Training completed!")
