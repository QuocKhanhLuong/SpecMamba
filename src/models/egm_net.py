"""
Energy-Gated Gabor Mamba Network (EGM-Net).

A hybrid architecture for medical image segmentation combining physics-based
signal processing with deep learning for artifact-free boundary delineation.

Architecture Components:
    1. Monogenic Signal Processing: Physics-based edge detection via Riesz transform
    2. Mamba Encoder: Global context extraction with O(N) complexity
    3. Coarse Branch: Convolutional decoder for smooth body segmentation
    4. Fine Branch: Gabor implicit decoder for sharp boundaries
    5. Energy-Gated Fusion: Selective boundary refinement

Key Innovations:
    - Dual-path decoding: Coarse (Conv) + Fine (Implicit) branches
    - Energy gating: Suppress artifacts in flat/homogeneous regions
    - Gabor basis: Localized oscillations prevent Gibbs ringing artifacts

References:
    [1] Felsberg & Sommer, "The Monogenic Signal," IEEE TSP, 2001.
    [2] Sitzmann et al., "Implicit Neural Representations with Periodic
        Activation Functions," NeurIPS, 2020.
    [3] Liu et al., "VMamba: Visual State Space Model," arXiv, 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from monogenic import EnergyMap, MonogenicSignal
from gabor_implicit import GaborNet, ImplicitSegmentationHead, GaborBasis
from mamba_block import VSSBlock, MambaBlockStack


class PatchEmbedding(nn.Module):
    """Patch embedding for initial feature extraction."""
    
    def __init__(self, in_channels: int = 1, embed_dim: int = 64, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, 
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class DownsampleBlock(nn.Module):
    """Downsample with channel expansion."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(32, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class UpsampleBlock(nn.Module):
    """Upsample with channel reduction."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(self.up(x)))


class MambaEncoderStage(nn.Module):
    """Single encoder stage with Mamba blocks."""
    
    def __init__(self, channels: int, depth: int = 2, spatial_size: int = 64):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(channels, scan_dim=min(64, channels))
            for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CoarseBranch(nn.Module):
    """
    Coarse Branch: Standard convolutional decoder for rough segmentation.
    
    Produces a base mask that is smooth but may have blurry boundaries.
    This handles the "body" of segmented regions.
    """
    
    def __init__(self, in_channels: int, num_classes: int, num_stages: int = 3):
        super().__init__()
        
        self.upsample_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        channels = in_channels
        for i in range(num_stages):
            out_ch = max(channels // 2, 64)
            self.upsample_layers.append(UpsampleBlock(channels, out_ch))
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.GELU()
            ))
            channels = out_ch
        
        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for up, conv in zip(self.upsample_layers, self.conv_layers):
            x = up(x)
            x = conv(x)
        return self.head(x)


class EnergyGatedFusion(nn.Module):
    """
    Energy-Gated Fusion Module.
    
    Uses monogenic energy map to gate the implicit (fine) branch:
    - High energy (edges): Activate fine branch for sharp boundaries
    - Low energy (flat): Suppress fine branch to avoid artifacts
    
    Final output = Coarse + Energy × (Fine - Coarse)
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize Energy-Gated Fusion.
        
        Args:
            temperature: Softness of gating (lower = harder gating)
        """
        super().__init__()
        self.temperature = temperature
        self.gate_scale = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, coarse: torch.Tensor, fine: torch.Tensor, 
                energy: torch.Tensor) -> torch.Tensor:
        """
        Fuse coarse and fine predictions using energy gating.
        
        Args:
            coarse: Coarse prediction (B, C, H, W)
            fine: Fine prediction (B, C, H, W) 
            energy: Energy map (B, 1, H, W) in [0, 1]
            
        Returns:
            Fused prediction (B, C, H, W)
        """
        # Resize energy to match prediction size
        if energy.shape[-2:] != coarse.shape[-2:]:
            energy = F.interpolate(energy, size=coarse.shape[-2:], 
                                   mode='bilinear', align_corners=True)
        
        # Apply learnable scaling and temperature
        gate = torch.sigmoid((energy * self.gate_scale + self.gate_bias) / self.temperature)
        
        # Blend: high energy → use fine, low energy → use coarse
        output = coarse + gate * (fine - coarse)
        
        return output


class FineBranch(nn.Module):
    """
    Fine Branch: Implicit neural representation for sharp boundaries.
    
    Uses Gabor basis encoding for localized high-frequency representation,
    enabling resolution-free rendering without Gibbs artifacts.
    """
    
    def __init__(self, feature_channels: int, num_classes: int,
                 hidden_dim: int = 256, num_layers: int = 4,
                 num_frequencies: int = 64):
        super().__init__()
        
        self.implicit_head = ImplicitSegmentationHead(
            feature_channels=feature_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_frequencies=num_frequencies,
            use_gabor=True  # Use Gabor instead of Fourier
        )
    
    def forward(self, features: torch.Tensor, 
                coords: Optional[torch.Tensor] = None,
                output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Feature map from encoder (B, C, H, W)
            coords: Query coordinates (B, N, 2) or None for grid
            output_size: Output resolution if coords is None
            
        Returns:
            Fine segmentation (B, num_classes, H, W) or (B, N, num_classes)
        """
        return self.implicit_head(features, coords, output_size)


class EGMNet(nn.Module):
    """
    Energy-Gated Gabor Mamba Network (EGM-Net).
    
    Full architecture combining:
    1. Monogenic energy extraction (physics-based edge detection)
    2. Mamba encoder (global context learning)
    3. Coarse branch (smooth body segmentation)
    4. Fine branch (sharp boundary via Gabor implicit)
    5. Energy-gated fusion (artifact-free blending)
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 img_size: int = 256, base_channels: int = 64,
                 num_stages: int = 4, encoder_depth: int = 2,
                 implicit_hidden: int = 256, implicit_layers: int = 4,
                 num_frequencies: int = 64):
        """
        Initialize EGM-Net.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            num_classes: Number of segmentation classes
            img_size: Input image size
            base_channels: Base channel count
            num_stages: Number of encoder stages
            encoder_depth: Depth of each encoder stage
            implicit_hidden: Hidden dim for implicit decoder
            implicit_layers: Number of implicit decoder layers
            num_frequencies: Number of Gabor frequencies
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_stages = num_stages
        
        # 1. Monogenic Energy Extractor (fixed, non-trainable)
        self.energy_extractor = EnergyMap(normalize=True, smoothing_sigma=1.0)
        
        # 2. Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels, base_channels, patch_size=4)
        feat_size = img_size // 4
        
        # 3. Mamba Encoder
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        channels = base_channels
        for i in range(num_stages):
            self.encoder_stages.append(
                MambaEncoderStage(channels, depth=encoder_depth, spatial_size=feat_size)
            )
            if i < num_stages - 1:
                self.downsample_layers.append(
                    DownsampleBlock(channels, channels * 2)
                )
                channels *= 2
                feat_size //= 2
        
        # Store final encoder channels
        self.encoder_channels = channels
        
        # 4. Bottleneck
        self.bottleneck = MambaEncoderStage(
            channels, depth=encoder_depth + 1, spatial_size=feat_size
        )
        
        # 5. Coarse Branch (standard decoder)
        self.coarse_branch = CoarseBranch(
            in_channels=channels,
            num_classes=num_classes,
            num_stages=num_stages - 1
        )
        
        # 6. Fine Branch (Gabor implicit decoder)
        self.fine_branch = FineBranch(
            feature_channels=channels,
            num_classes=num_classes,
            hidden_dim=implicit_hidden,
            num_layers=implicit_layers,
            num_frequencies=num_frequencies
        )
        
        # 7. Energy-Gated Fusion
        self.fusion = EnergyGatedFusion(temperature=1.0)
    
    def forward(self, x: torch.Tensor, 
                output_size: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image (B, C, H, W)
            output_size: Optional output resolution for fine branch
            
        Returns:
            Dictionary containing:
            - 'output': Final fused segmentation
            - 'coarse': Coarse branch output
            - 'fine': Fine branch output
            - 'energy': Energy map
        """
        B, C, H, W = x.shape
        
        if output_size is None:
            output_size = (H, W)
        
        # 1. Extract energy map (detached, no gradients for physics module)
        with torch.no_grad():
            # Convert to grayscale if needed
            x_gray = x.mean(dim=1, keepdim=True) if C > 1 else x
            energy, mono_out = self.energy_extractor(x_gray)
        
        # 2. Patch embedding
        features = self.patch_embed(x)
        
        # 3. Encoder (Mamba stages)
        encoder_features = []
        for i, stage in enumerate(self.encoder_stages):
            features = stage(features)
            encoder_features.append(features)
            if i < len(self.downsample_layers):
                features = self.downsample_layers[i](features)
        
        # 4. Bottleneck
        features = self.bottleneck(features)
        
        # 5. Coarse branch
        coarse = self.coarse_branch(features)
        coarse = F.interpolate(coarse, size=output_size, 
                               mode='bilinear', align_corners=True)
        
        # 6. Fine branch (implicit decoder)
        fine = self.fine_branch(features, output_size=output_size)
        
        # 7. Energy-gated fusion
        output = self.fusion(coarse, fine, energy)
        
        return {
            'output': output,
            'coarse': coarse,
            'fine': fine,
            'energy': energy
        }
    
    def inference(self, x: torch.Tensor, 
                  output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Simplified inference returning only final output.
        
        Args:
            x: Input image (B, C, H, W)
            output_size: Optional output resolution
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        return self.forward(x, output_size)['output']
    
    def query_points(self, x: torch.Tensor, 
                     coords: torch.Tensor) -> torch.Tensor:
        """
        Query segmentation at arbitrary coordinates (implicit representation).
        
        This enables resolution-free inference - you can zoom into boundaries
        at any resolution without pixelation.
        
        Args:
            x: Input image (B, C, H, W)
            coords: Query coordinates (B, N, 2) normalized to [-1, 1]
            
        Returns:
            Segmentation logits at query points (B, N, num_classes)
        """
        B, C, H, W = x.shape
        
        # Encode image
        features = self.patch_embed(x)
        for i, stage in enumerate(self.encoder_stages):
            features = stage(features)
            if i < len(self.downsample_layers):
                features = self.downsample_layers[i](features)
        features = self.bottleneck(features)
        
        # Query fine branch at coordinates
        fine_points = self.fine_branch.implicit_head(features, coords=coords)
        
        return fine_points


class EGMNetLite(nn.Module):
    """
    Lightweight version of EGM-Net for faster training/inference.
    
    Reduced channels and stages for lower memory/compute.
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 img_size: int = 256):
        super().__init__()
        
        self.model = EGMNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            base_channels=32,  # Reduced from 64
            num_stages=3,      # Reduced from 4
            encoder_depth=1,   # Reduced from 2
            implicit_hidden=128,  # Reduced from 256
            implicit_layers=3,    # Reduced from 4
            num_frequencies=32    # Reduced from 64
        )
    
    def forward(self, x, output_size=None):
        return self.model(x, output_size)
    
    def inference(self, x, output_size=None):
        return self.model.inference(x, output_size)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EGM-Net (Energy-Gated Gabor Mamba Network)")
    print("=" * 60)
    
    # Test full model
    print("\n[1] Testing EGM-Net Full...")
    model = EGMNet(
        in_channels=1,
        num_classes=3,
        img_size=256,
        base_channels=64,
        num_stages=4,
        encoder_depth=2
    )
    
    x = torch.randn(2, 1, 256, 256)
    outputs = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {outputs['output'].shape}")
    print(f"Coarse: {outputs['coarse'].shape}")
    print(f"Fine: {outputs['fine'].shape}")
    print(f"Energy: {outputs['energy'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test point query (resolution-free inference)
    print("\n[2] Testing Point Query (Resolution-Free)...")
    coords = torch.rand(2, 1000, 2) * 2 - 1  # Random points in [-1, 1]
    point_output = model.query_points(x, coords)
    print(f"Query coords: {coords.shape}")
    print(f"Point output: {point_output.shape}")
    
    # Test lite model
    print("\n[3] Testing EGM-Net Lite...")
    lite_model = EGMNetLite(in_channels=1, num_classes=3, img_size=256)
    lite_outputs = lite_model(x)
    
    lite_params = sum(p.numel() for p in lite_model.parameters())
    print(f"Lite model parameters: {lite_params:,}")
    print(f"Lite output: {lite_outputs['output'].shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
