"""
Gabor Implicit Neural Representation Module

Replaces standard Fourier Features with Gabor Basis for implicit neural representations.

Key insight: 
- Fourier (sin/cos): Oscillates infinitely → causes Gibbs ringing artifacts
- Gabor (Gaussian × sin): Localized oscillation → clean boundaries without ringing

This implements the "Fine Branch" of the EGM-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class GaborBasis(nn.Module):
    """
    Gabor Basis Functions for coordinate encoding.
    
    Gabor function: g(x) = exp(-x²/2σ²) × cos(2πfx + φ)
    
    Unlike Fourier features which oscillate infinitely, Gabor wavelets
    are spatially localized, preventing Gibbs phenomenon (ringing artifacts).
    """
    
    def __init__(self, input_dim: int = 2, num_frequencies: int = 64,
                 sigma_range: Tuple[float, float] = (0.1, 2.0),
                 freq_range: Tuple[float, float] = (1.0, 10.0),
                 learnable: bool = True):
        """
        Initialize Gabor Basis.
        
        Args:
            input_dim: Dimension of input coordinates (2 for images)
            num_frequencies: Number of Gabor basis functions
            sigma_range: Range of Gaussian envelope widths
            freq_range: Range of oscillation frequencies
            learnable: Whether parameters are learnable
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = num_frequencies * 2  # sin and cos components
        
        # Initialize frequencies uniformly in log space
        log_freqs = torch.linspace(
            math.log(freq_range[0]), 
            math.log(freq_range[1]), 
            num_frequencies
        )
        freqs = torch.exp(log_freqs)
        
        # Initialize sigmas (Gaussian envelope widths)
        sigmas = torch.linspace(sigma_range[0], sigma_range[1], num_frequencies)
        
        # Random orientations for 2D
        orientations = torch.rand(num_frequencies) * 2 * math.pi
        
        # Random phases
        phases = torch.rand(num_frequencies) * 2 * math.pi
        
        # Create direction vectors from orientations
        directions = torch.stack([
            torch.cos(orientations),
            torch.sin(orientations)
        ], dim=-1)  # (num_freq, 2)
        
        if learnable:
            self.freqs = nn.Parameter(freqs)
            self.sigmas = nn.Parameter(sigmas)
            self.directions = nn.Parameter(directions)
            self.phases = nn.Parameter(phases)
        else:
            self.register_buffer('freqs', freqs)
            self.register_buffer('sigmas', sigmas)
            self.register_buffer('directions', directions)
            self.register_buffer('phases', phases)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates using Gabor basis.
        
        Args:
            coords: Coordinate tensor of shape (..., input_dim)
            
        Returns:
            Gabor features of shape (..., output_dim)
        """
        # Normalize directions
        directions = F.normalize(self.directions, dim=-1)  # (num_freq, 2)
        
        # Project coordinates onto directions
        # coords: (..., 2), directions: (num_freq, 2)
        proj = torch.matmul(coords, directions.T)  # (..., num_freq)
        
        # Compute Gaussian envelope
        # exp(-proj² / (2σ²))
        sigmas = torch.abs(self.sigmas) + 0.01  # Ensure positive
        gaussian = torch.exp(-proj**2 / (2 * sigmas**2 + 1e-8))
        
        # Compute oscillatory component
        # cos(2πf·proj + φ), sin(2πf·proj + φ)
        freqs = torch.abs(self.freqs) + 0.1  # Ensure positive
        arg = 2 * math.pi * freqs * proj + self.phases
        
        cos_comp = gaussian * torch.cos(arg)
        sin_comp = gaussian * torch.sin(arg)
        
        # Concatenate sin and cos
        gabor_features = torch.cat([cos_comp, sin_comp], dim=-1)
        
        return gabor_features


class FourierFeatures(nn.Module):
    """
    Standard Fourier Features for comparison.
    
    From "Fourier Features Let Networks Learn High Frequency Functions"
    """
    
    def __init__(self, input_dim: int = 2, num_frequencies: int = 64,
                 scale: float = 10.0, learnable: bool = False):
        """
        Initialize Fourier Features.
        
        Args:
            input_dim: Dimension of input coordinates
            num_frequencies: Number of frequency bands
            scale: Standard deviation for frequency sampling
            learnable: Whether B matrix is learnable
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = num_frequencies * 2
        
        # Random frequency matrix
        B = torch.randn(input_dim, num_frequencies) * scale
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates using Fourier features.
        
        Args:
            coords: Coordinate tensor of shape (..., input_dim)
            
        Returns:
            Fourier features of shape (..., output_dim)
        """
        # Project: coords @ B
        proj = 2 * math.pi * torch.matmul(coords, self.B)  # (..., num_freq)
        
        # Sin and cos
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class SIRENLayer(nn.Module):
    """
    SIREN (Sinusoidal Representation Networks) layer.
    
    Uses periodic sine activation instead of ReLU.
    From "Implicit Neural Representations with Periodic Activation Functions"
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 omega_0: float = 30.0, is_first: bool = False):
        """
        Initialize SIREN layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            omega_0: Frequency multiplier (ω₀ in paper)
            is_first: Whether this is the first layer (uses different init)
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        """SIREN-specific weight initialization."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/n, 1/n]
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)
            else:
                # Other layers: uniform in [-sqrt(6/n)/ω₀, sqrt(6/n)/ω₀]
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class GaborNet(nn.Module):
    """
    GaborNet: MLP with Gabor basis input encoding.
    
    Architecture: Gabor Encoding → SIREN Layers → Output
    
    This replaces standard MLP/KAN for implicit neural representations,
    providing better stability and localized high-frequency learning.
    """
    
    def __init__(self, coord_dim: int = 2, feature_dim: int = 256,
                 hidden_dim: int = 256, output_dim: int = 1,
                 num_layers: int = 4, num_frequencies: int = 64,
                 use_gabor: bool = True, omega_0: float = 30.0):
        """
        Initialize GaborNet.
        
        Args:
            coord_dim: Dimension of input coordinates (2 for images)
            feature_dim: Dimension of input features from encoder
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (num_classes for segmentation)
            num_layers: Number of SIREN layers
            num_frequencies: Number of Gabor/Fourier frequencies
            use_gabor: Use Gabor (True) or Fourier (False) features
            omega_0: SIREN frequency parameter
        """
        super().__init__()
        
        # Coordinate encoding
        if use_gabor:
            self.coord_encoder = GaborBasis(
                input_dim=coord_dim,
                num_frequencies=num_frequencies,
                learnable=True
            )
        else:
            self.coord_encoder = FourierFeatures(
                input_dim=coord_dim,
                num_frequencies=num_frequencies,
                learnable=False
            )
        
        coord_encoded_dim = self.coord_encoder.output_dim
        
        # Input dimension: encoded coords + features
        input_dim = coord_encoded_dim + feature_dim
        
        # Build SIREN network
        layers = []
        
        # First layer
        layers.append(SIRENLayer(input_dim, hidden_dim, omega_0, is_first=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0, is_first=False))
        
        # Final layer (linear, no sine activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coords: Normalized coordinates of shape (B, N, 2) in [-1, 1]
            features: Features from encoder of shape (B, N, feature_dim)
            
        Returns:
            Output of shape (B, N, output_dim)
        """
        # Encode coordinates
        coord_encoded = self.coord_encoder(coords)  # (B, N, coord_encoded_dim)
        
        # Concatenate with features
        x = torch.cat([coord_encoded, features], dim=-1)  # (B, N, input_dim)
        
        # Pass through network
        output = self.network(x)  # (B, N, output_dim)
        
        return output


class ImplicitSegmentationHead(nn.Module):
    """
    Implicit Segmentation Head for continuous boundary representation.
    
    Takes feature maps from encoder and outputs continuous segmentation
    at arbitrary resolution using Gabor-based implicit representation.
    """
    
    def __init__(self, feature_channels: int = 64, num_classes: int = 2,
                 hidden_dim: int = 256, num_layers: int = 4,
                 num_frequencies: int = 64, use_gabor: bool = True):
        """
        Initialize Implicit Segmentation Head.
        
        Args:
            feature_channels: Number of channels in input feature map
            num_classes: Number of segmentation classes
            hidden_dim: Hidden dimension of GaborNet
            num_layers: Number of layers in GaborNet
            num_frequencies: Number of Gabor frequencies
            use_gabor: Use Gabor (True) or Fourier (False) basis
        """
        super().__init__()
        
        self.feature_channels = feature_channels
        self.num_classes = num_classes
        
        # Feature projector (reduce channel dimension)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU()
        )
        
        # Implicit decoder
        self.implicit_decoder = GaborNet(
            coord_dim=2,
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            num_frequencies=num_frequencies,
            use_gabor=use_gabor
        )
    
    def sample_features(self, feature_map: torch.Tensor, 
                       coords: torch.Tensor) -> torch.Tensor:
        """
        Sample features at given coordinates using bilinear interpolation.
        
        Args:
            feature_map: Feature tensor of shape (B, C, H, W)
            coords: Coordinates of shape (B, N, 2) in [-1, 1]
            
        Returns:
            Sampled features of shape (B, N, C)
        """
        B, C, H, W = feature_map.shape
        N = coords.shape[1]
        
        # Reshape coords for grid_sample: (B, N, 1, 2) -> (B, 1, N, 2)
        # grid_sample expects (B, H, W, 2) where last dim is (x, y)
        grid = coords.view(B, 1, N, 2)
        
        # Sample using bilinear interpolation
        # feature_map: (B, C, H, W), grid: (B, 1, N, 2)
        # output: (B, C, 1, N)
        sampled = F.grid_sample(
            feature_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Reshape: (B, C, 1, N) -> (B, N, C)
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        
        return sampled
    
    def forward(self, feature_map: torch.Tensor, 
                coords: Optional[torch.Tensor] = None,
                output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feature_map: Feature tensor of shape (B, C, H, W)
            coords: Optional query coordinates of shape (B, N, 2) in [-1, 1]
                   If None, generates grid based on output_size
            output_size: Optional (H, W) for output resolution
                        If None and coords is None, uses feature_map size * 4
            
        Returns:
            Segmentation logits of shape (B, num_classes, H_out, W_out) or (B, N, num_classes)
        """
        B, C, H_feat, W_feat = feature_map.shape
        device = feature_map.device
        
        # Project features
        feature_map = self.feature_proj(feature_map)  # (B, hidden_dim, H, W)
        
        # Generate coordinates if not provided
        if coords is None:
            if output_size is None:
                output_size = (H_feat * 4, W_feat * 4)
            
            H_out, W_out = output_size
            
            # Create normalized coordinate grid [-1, 1]
            y = torch.linspace(-1, 1, H_out, device=device)
            x = torch.linspace(-1, 1, W_out, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1)  # (H_out, W_out, 2)
            coords = coords.view(1, -1, 2).expand(B, -1, -1)  # (B, H*W, 2)
            
            reshape_output = True
        else:
            reshape_output = False
            H_out, W_out = None, None
        
        # Sample features at coordinates
        features = self.sample_features(feature_map, coords)  # (B, N, hidden_dim)
        
        # Implicit decoding
        logits = self.implicit_decoder(coords, features)  # (B, N, num_classes)
        
        # Reshape to image if using grid
        if reshape_output:
            logits = logits.view(B, H_out, W_out, self.num_classes)
            logits = logits.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return logits


if __name__ == "__main__":
    print("Testing Gabor Implicit Modules...")
    
    # Test Gabor Basis
    print("\n[1] Testing GaborBasis...")
    gabor = GaborBasis(input_dim=2, num_frequencies=32)
    coords = torch.randn(4, 100, 2)  # (B, N, 2)
    encoded = gabor(coords)
    print(f"Input coords: {coords.shape}")
    print(f"Gabor encoded: {encoded.shape}")
    
    # Test GaborNet
    print("\n[2] Testing GaborNet...")
    net = GaborNet(coord_dim=2, feature_dim=64, hidden_dim=128, 
                   output_dim=3, num_layers=3, num_frequencies=32)
    features = torch.randn(4, 100, 64)
    output = net(coords, features)
    print(f"GaborNet output: {output.shape}")
    
    # Test ImplicitSegmentationHead
    print("\n[3] Testing ImplicitSegmentationHead...")
    head = ImplicitSegmentationHead(
        feature_channels=64, num_classes=3,
        hidden_dim=128, num_layers=3, num_frequencies=32
    )
    feature_map = torch.randn(2, 64, 32, 32)
    
    # Test with automatic grid
    seg_output = head(feature_map, output_size=(128, 128))
    print(f"Feature map: {feature_map.shape}")
    print(f"Segmentation output (grid): {seg_output.shape}")
    
    # Test with custom coordinates
    custom_coords = torch.rand(2, 500, 2) * 2 - 1  # Random points in [-1, 1]
    seg_points = head(feature_map, coords=custom_coords)
    print(f"Segmentation output (points): {seg_points.shape}")
    
    print("\n✓ All tests passed!")
