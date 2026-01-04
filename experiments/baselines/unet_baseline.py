"""
Baseline Experiments: Standard U-Net and Variants.

Implements baseline models for fair comparison:
1. Standard U-Net
2. U-Net with ResNet encoder
3. Attention U-Net

These serve as reference points for EGM-Net evaluation.
"""

import sys
sys.path.insert(0, '../..')

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict
import json
from pathlib import Path


@dataclass 
class BaselineConfig:
    """Configuration for baseline experiments."""
    experiment_name: str = "baseline_training"
    dataset: str = "ACDC"
    num_classes: int = 4
    img_size: int = 224
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    device: str = "cuda"
    results_dir: str = "../../results/baselines"


# =============================================================================
# Simple U-Net Baseline
# =============================================================================

class DoubleConv(nn.Module):
    """Double convolution block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net baseline."""
    def __init__(self, in_channels=1, num_classes=4, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Output
        self.out = nn.Conv2d(features[0], num_classes, 1)
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        
        return self.out(x)


def train_baseline(config: BaselineConfig, model_name: str) -> Dict:
    """Train a baseline model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # TODO: Implement training loop
    
    results = {
        "model": model_name,
        "epochs": config.num_epochs,
        "dice_score": 0.0,  # Placeholder
    }
    
    return results


def main():
    config = BaselineConfig()
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Train U-Net baseline
    results = train_baseline(config, "UNet")
    
    # Save results
    with open(f"{config.results_dir}/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Baseline training complete!")


if __name__ == "__main__":
    main()
