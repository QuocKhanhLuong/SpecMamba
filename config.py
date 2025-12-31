"""
Training Configuration and Utilities
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for SpectralVMUNet model."""
    in_channels: int = 1
    out_channels: int = 3
    img_size: int = 256
    base_channels: int = 64
    num_stages: int = 4
    depth: int = 2


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    spatial_weight: float = 1.0
    freq_weight: float = 0.1
    use_dice: bool = True
    use_focal: bool = True
    boundary_weight: float = 0.0  # 0 to disable boundary loss


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    model: ModelConfig = ModelConfig()
    
    # Loss
    loss: LossConfig = LossConfig()
    
    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # "adamw" or "sgd"
    
    # Training
    num_epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    
    # Learning rate schedule
    warmup_epochs: int = 10
    scheduler: str = "cosine"  # "cosine", "step", or "exponential"
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 10  # Save every N epochs
    
    # Device
    device: str = "cuda"
    mixed_precision: bool = False
    
    # Logging
    log_interval: int = 100  # Log every N iterations
    val_interval: int = 1  # Validate every N epochs


if __name__ == "__main__":
    config = TrainingConfig()
    print(config)
