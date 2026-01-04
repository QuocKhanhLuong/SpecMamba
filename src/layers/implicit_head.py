"""
Implicit Fourier Head for Continuous Segmentation
Based on: "Fourier Features Let Networks Learn High Frequency Functions"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FourierMapping(nn.Module):
    """
    Fourier Feature Mapping for positional encoding.
    
    Transforms low-dimensional coordinates into high-dimensional Fourier features
    to overcome spectral bias and enable learning of high-frequency functions.
    
    γ(v) = [cos(2πBv), sin(2πBv)]
    where B is a Gaussian random matrix.
    """
    
    def __init__(self, input_dim: int = 2, mapping_size: int = 256, 
                 scale: float = 10.0, learnable: bool = False):
        """
        Initialize Fourier Mapping.
        
        Args:
            input_dim: Dimension of input coordinates (2 for 2D images)
            mapping_size: Number of Fourier basis functions
            scale: Standard deviation of Gaussian for frequency matrix B
            learnable: Whether B matrix should be learnable
        """
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.output_dim = mapping_size * 2  # cos and sin
        
        # Gaussian random matrix B
        B = torch.randn(mapping_size, input_dim) * scale
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier mapping to coordinates.
        
        Args:
            coords: Coordinate tensor of shape (..., input_dim)
                    Values should be normalized to [-1, 1] or [0, 1]
            
        Returns:
            Fourier features of shape (..., mapping_size * 2)
        """
        # coords: (..., input_dim)
        # B: (mapping_size, input_dim)
        # projection: (..., mapping_size)
        projection = coords @ self.B.T  # Linear projection
        projection = 2 * math.pi * projection
        
        # Concatenate cos and sin
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)


class SIRENLayer(nn.Module):
    """
    SIREN (Sinusoidal Representation Networks) layer.
    Uses sin activation instead of ReLU for better gradient flow
    and high-frequency learning.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 is_first: bool = False, omega_0: float = 30.0):
        """
        Initialize SIREN layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            is_first: Whether this is the first layer (affects initialization)
            omega_0: Frequency scaling factor
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        """SIREN-specific weight initialization."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.linear.in_features
            else:
                # Other layers: uniform in [-sqrt(6/in)/omega, sqrt(6/in)/omega]
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class ImplicitDecoder(nn.Module):
    """
    Implicit MLP Decoder for continuous segmentation.
    
    Takes feature vectors and Fourier-encoded coordinates,
    outputs class probabilities at arbitrary resolution.
    """
    
    def __init__(self, feature_dim: int, coord_dim: int = 512,
                 hidden_dim: int = 256, num_layers: int = 4,
                 num_classes: int = 3, use_siren: bool = True):
        """
        Initialize Implicit Decoder.
        
        Args:
            feature_dim: Dimension of encoder features
            coord_dim: Dimension of Fourier-encoded coordinates
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            num_classes: Number of output classes
            use_siren: Whether to use SIREN layers (sin activation)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.use_siren = use_siren
        
        # Input dimension = features + Fourier coordinates
        input_dim = feature_dim + coord_dim
        
        # Build MLP layers
        layers = []
        
        if use_siren:
            # First layer
            layers.append(SIRENLayer(input_dim, hidden_dim, is_first=True))
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(SIRENLayer(hidden_dim, hidden_dim))
            # Output layer (no activation)
            layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            # Standard MLP with ReLU
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, 
                fourier_coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Feature vectors of shape (B, N, feature_dim)
                      where N is number of query points
            fourier_coords: Fourier-encoded coordinates of shape (B, N, coord_dim)
            
        Returns:
            Class logits of shape (B, N, num_classes)
        """
        # Concatenate features and coordinates
        x = torch.cat([features, fourier_coords], dim=-1)
        
        # Pass through MLP
        return self.mlp(x)


class ImplicitSegmentationHead(nn.Module):
    """
    Complete Implicit Segmentation Head.
    
    Combines Fourier Mapping + Feature Interpolation + Implicit Decoder.
    Can query segmentation at any continuous coordinate.
    """
    
    def __init__(self, feature_dim: int, num_classes: int = 3,
                 fourier_scale: float = 10.0, fourier_size: int = 256,
                 hidden_dim: int = 256, num_layers: int = 4,
                 use_siren: bool = True, learnable_fourier: bool = False):
        """
        Initialize Implicit Segmentation Head.
        
        Args:
            feature_dim: Dimension of encoder feature maps
            num_classes: Number of segmentation classes
            fourier_scale: Scale of Gaussian for Fourier mapping
            fourier_size: Number of Fourier basis functions
            hidden_dim: Hidden dimension for MLP
            num_layers: Number of MLP layers
            use_siren: Whether to use SIREN activation
            learnable_fourier: Whether Fourier matrix is learnable
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Fourier mapping for coordinates
        self.fourier_mapping = FourierMapping(
            input_dim=2,
            mapping_size=fourier_size,
            scale=fourier_scale,
            learnable=learnable_fourier
        )
        
        # Implicit decoder
        self.decoder = ImplicitDecoder(
            feature_dim=feature_dim,
            coord_dim=self.fourier_mapping.output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            use_siren=use_siren
        )
    
    def sample_features(self, feature_map: torch.Tensor, 
                        coords: torch.Tensor) -> torch.Tensor:
        """
        Sample features at given coordinates using bilinear interpolation.
        
        Args:
            feature_map: Feature map of shape (B, C, H, W)
            coords: Normalized coordinates in [-1, 1], shape (B, N, 2)
                    where coords[..., 0] is x and coords[..., 1] is y
            
        Returns:
            Interpolated features of shape (B, N, C)
        """
        B, C, H, W = feature_map.shape
        N = coords.shape[1]
        
        # Reshape coords for grid_sample: (B, N, 1, 2) -> (B, 1, N, 2)
        # grid_sample expects (B, H_out, W_out, 2)
        grid = coords.view(B, 1, N, 2)
        
        # Sample features using bilinear interpolation
        # Output: (B, C, 1, N)
        sampled = F.grid_sample(
            feature_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Reshape to (B, N, C)
        sampled = sampled.view(B, C, N).permute(0, 2, 1)
        
        return sampled
    
    def forward(self, feature_map: torch.Tensor, 
                coords: Optional[torch.Tensor] = None,
                output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass - query segmentation at coordinates.
        
        Args:
            feature_map: Encoder output of shape (B, C, H_feat, W_feat)
            coords: Query coordinates in [-1, 1], shape (B, N, 2)
                    If None, creates a regular grid based on output_size
            output_size: (H_out, W_out) for regular grid generation
                         Only used if coords is None
            
        Returns:
            Segmentation logits:
            - If coords provided: (B, N, num_classes)
            - If output_size provided: (B, num_classes, H_out, W_out)
        """
        B = feature_map.shape[0]
        device = feature_map.device
        
        # Generate coordinates if not provided
        if coords is None:
            if output_size is None:
                # Default to feature map size * 4
                H_out, W_out = feature_map.shape[2] * 4, feature_map.shape[3] * 4
            else:
                H_out, W_out = output_size
            
            # Create normalized coordinate grid [-1, 1]
            y = torch.linspace(-1, 1, H_out, device=device)
            x = torch.linspace(-1, 1, W_out, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
            coords = coords.view(1, -1, 2).expand(B, -1, -1)  # (B, H*W, 2)
            reshape_output = True
        else:
            reshape_output = False
            H_out, W_out = None, None
        
        # Sample features at query coordinates
        features = self.sample_features(feature_map, coords)  # (B, N, C)
        
        # Apply Fourier mapping to coordinates
        fourier_coords = self.fourier_mapping(coords)  # (B, N, fourier_dim*2)
        
        # Decode to class logits
        logits = self.decoder(features, fourier_coords)  # (B, N, num_classes)
        
        # Reshape to image format if using grid
        if reshape_output:
            logits = logits.view(B, H_out, W_out, -1)
            logits = logits.permute(0, 3, 1, 2)  # (B, num_classes, H, W)
        
        return logits


class MultiScaleImplicitHead(nn.Module):
    """
    Multi-scale Implicit Head that combines features from multiple encoder levels.
    """
    
    def __init__(self, feature_dims: list, num_classes: int = 3,
                 fourier_scale: float = 10.0, fourier_size: int = 256,
                 hidden_dim: int = 256, num_layers: int = 4):
        """
        Initialize Multi-scale Implicit Head.
        
        Args:
            feature_dims: List of feature dimensions from encoder levels
            num_classes: Number of segmentation classes
            fourier_scale: Scale for Fourier mapping
            fourier_size: Size of Fourier features
            hidden_dim: Hidden dimension for MLP
            num_layers: Number of MLP layers
        """
        super().__init__()
        
        self.num_scales = len(feature_dims)
        
        # Fourier mapping (shared)
        self.fourier_mapping = FourierMapping(
            input_dim=2,
            mapping_size=fourier_size,
            scale=fourier_scale
        )
        
        # Feature projection for each scale
        total_feat_dim = sum(feature_dims)
        self.feature_proj = nn.Linear(total_feat_dim, hidden_dim)
        
        # Implicit decoder
        self.decoder = ImplicitDecoder(
            feature_dim=hidden_dim,
            coord_dim=self.fourier_mapping.output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            use_siren=True
        )
    
    def forward(self, feature_maps: list, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale features.
        
        Args:
            feature_maps: List of feature maps from encoder levels
            coords: Query coordinates in [-1, 1], shape (B, N, 2)
            
        Returns:
            Segmentation logits of shape (B, N, num_classes)
        """
        B = coords.shape[0]
        N = coords.shape[1]
        device = coords.device
        
        # Sample and concatenate features from all scales
        all_features = []
        for feat_map in feature_maps:
            grid = coords.view(B, 1, N, 2)
            sampled = F.grid_sample(
                feat_map, grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            sampled = sampled.view(B, -1, N).permute(0, 2, 1)  # (B, N, C)
            all_features.append(sampled)
        
        # Concatenate multi-scale features
        features = torch.cat(all_features, dim=-1)  # (B, N, sum(C))
        
        # Project to hidden dimension
        features = self.feature_proj(features)  # (B, N, hidden_dim)
        
        # Fourier encode coordinates
        fourier_coords = self.fourier_mapping(coords)
        
        # Decode
        return self.decoder(features, fourier_coords)


if __name__ == "__main__":
    # Test Fourier Mapping
    print("Testing FourierMapping...")
    coords = torch.rand(2, 100, 2) * 2 - 1  # (B, N, 2) in [-1, 1]
    fourier = FourierMapping(input_dim=2, mapping_size=256, scale=10.0)
    fourier_feats = fourier(coords)
    print(f"  Coords: {coords.shape} -> Fourier: {fourier_feats.shape}")
    
    # Test Implicit Segmentation Head
    print("\nTesting ImplicitSegmentationHead...")
    feature_map = torch.randn(2, 64, 64, 64)  # (B, C, H, W)
    head = ImplicitSegmentationHead(
        feature_dim=64,
        num_classes=3,
        fourier_scale=10.0,
        fourier_size=256,
        hidden_dim=256,
        num_layers=4
    )
    
    # Query at specific coordinates
    query_coords = torch.rand(2, 1000, 2) * 2 - 1
    logits = head(feature_map, coords=query_coords)
    print(f"  Feature map: {feature_map.shape}")
    print(f"  Query coords: {query_coords.shape}")
    print(f"  Output logits: {logits.shape}")
    
    # Query at regular grid
    logits_grid = head(feature_map, output_size=(256, 256))
    print(f"  Grid output (256x256): {logits_grid.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in head.parameters())
    print(f"\n  Total parameters: {params:,}")
