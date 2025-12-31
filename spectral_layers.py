"""
Spectral Layers Module
Implements frequency domain processing using FFT-based gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralGating(nn.Module):
    """
    Spectral Gating Module using FFT for frequency domain filtering.
    
    Applies learnable frequency filtering in the frequency domain to enhance
    edge sharpness and remove noise by modulating frequency components.
    """
    
    def __init__(self, channels: int, height: int, width: int, 
                 threshold: float = 0.1, complex_init: str = "kaiming"):
        """
        Initialize SpectralGating module.
        
        Args:
            channels: Number of input channels
            height: Input height (should be divisible by some factor)
            width: Input width (should be divisible by some factor)
            threshold: Hard thresholding value for amplitude (0 to disable)
            complex_init: Initialization strategy ("kaiming" or "identity")
        """
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.threshold = threshold
        
        # Create learnable complex weights for frequency domain
        # Shape: (channels, height, width//2 + 1) for rfft2
        self.register_buffer(
            "freq_shape",
            torch.tensor([channels, height, width // 2 + 1], dtype=torch.long)
        )
        
        # Real and Imaginary parts of complex weights
        self.weight_real = nn.Parameter(
            torch.zeros(channels, height, width // 2 + 1)
        )
        self.weight_imag = nn.Parameter(
            torch.zeros(channels, height, width // 2 + 1)
        )
        
        self._init_weights(complex_init)
        
    def _init_weights(self, strategy: str = "kaiming"):
        """Initialize complex weights."""
        if strategy == "identity":
            # Initialize close to identity (magnitude ~1, phase ~0)
            nn.init.ones_(self.weight_real)
            nn.init.zeros_(self.weight_imag)
        elif strategy == "kaiming":
            # Kaiming initialization adapted for complex numbers
            fan_in = self.height * (self.width // 2 + 1)
            std = (2.0 / fan_in) ** 0.5
            nn.init.normal_(self.weight_real, 0, std)
            nn.init.normal_(self.weight_imag, 0, std)
        else:
            raise ValueError(f"Unknown init strategy: {strategy}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral gating to input.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Filtered output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Apply FFT to convert to frequency domain
        # rfft2 returns complex tensor
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        
        # Create complex weight matrix: weight_real + 1j * weight_imag
        # Reshape to (1, C, H, W//2+1) for broadcasting
        complex_weight = (
            self.weight_real.unsqueeze(0) + 
            1j * self.weight_imag.unsqueeze(0)
        )
        
        # Apply channel-wise multiplication in frequency domain
        # Shape: (B, C, H, W//2+1) * (1, C, H, W//2+1) -> (B, C, H, W//2+1)
        x_filtered = x_freq * complex_weight
        
        # Optional: Hard thresholding to remove low-amplitude noise
        if self.threshold > 0:
            magnitude = torch.abs(x_filtered)
            mask = magnitude > self.threshold
            x_filtered = x_filtered * mask.float()
        
        # Apply inverse FFT to return to spatial domain
        output = torch.fft.irfft2(x_filtered, s=(H, W), dim=(-2, -1), norm="ortho")
        
        return output


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss for enforcing edge sharpness.
    Computes L2 distance between FFT of prediction and ground truth.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize FrequencyLoss.
        
        Args:
            weight: Weight factor for frequency loss in combined loss
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency domain loss.
        
        Args:
            pred: Prediction tensor of shape (B, C, H, W)
            target: Ground truth tensor of shape (B, C, H, W)
            
        Returns:
            Scalar loss value
        """
        # Apply FFT
        pred_freq = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        target_freq = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
        
        # Compute L2 distance in frequency domain
        # Using both magnitude and phase information
        loss_real = F.mse_loss(pred_freq.real, target_freq.real)
        loss_imag = F.mse_loss(pred_freq.imag, target_freq.imag)
        
        return loss_real + loss_imag


if __name__ == "__main__":
    # Test SpectralGating
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width)
    
    spec_gate = SpectralGating(channels, height, width)
    output = spec_gate(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Module parameters: {sum(p.numel() for p in spec_gate.parameters())}")
