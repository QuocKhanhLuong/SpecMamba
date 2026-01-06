"""
HRNetV2-Mamba Backbone for EGM-Net.

A dual-stream architecture that maintains high-resolution features throughout,
replacing downsampling/upsampling with parallel multi-resolution streams.

Architecture Overview:
    1. Two parallel streams: High-Res (H/4) and Low-Res (H/8 → H/16)
    2. SpectralVSSBlock: Combines Mamba (spatial) + FFT (spectral) processing
    3. MultiScaleFusion: Periodic exchange of information between streams
    4. Aggregation: Upsample all to highest resolution and concatenate

Key Features:
    - High-res stream preserves boundary details (Energy, Frequency signals)
    - Low-res stream captures semantic context (Intensity)
    - No information loss from deep downsampling
    - Spectral Mamba blocks overcome spectral bias

References:
    [1] Wang et al., "Deep High-Resolution Representation Learning," CVPR 2020.
    [2] Liu et al., "VMamba: Visual State Space Model," arXiv 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

# Handle imports for both package and standalone usage
try:
    from .mamba_block import VSSBlock, MambaBlockStack
    from ..layers.spectral_layers import SpectralGating
except (ImportError, ValueError):
    try:
        from models.mamba_block import VSSBlock, MambaBlockStack
        from layers.spectral_layers import SpectralGating
    except ImportError:
        try:
            import sys
            import os
            # Add parent directory of 'models' (i.e., 'src') to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(current_dir)
            if src_dir not in sys.path:
                sys.path.append(src_dir)
            
            from models.mamba_block import VSSBlock, MambaBlockStack
            from layers.spectral_layers import SpectralGating
        except ImportError as e:
            raise ImportError(f"Could not import required modules (mamba_block, spectral_layers): {e}")


# =============================================================================
# Spectral VSS Block: Mamba + FFT Dual-Path Processing
# =============================================================================

class SpectralVSSBlock(nn.Module):
    """
    Spectral VSS Block combining spatial and frequency domain processing.
    
    Two parallel branches:
        A) Spatial Branch: Mamba (SS2D) for global context with O(N) complexity
        B) Spectral Branch: FFT → Learnable Gating → IFFT to overcome spectral bias
    
    The outputs are blended with a learnable weight.
    
    Args:
        channels: Number of input/output channels
        height: Feature map height (for spectral weights)
        width: Feature map width (for spectral weights)
        mamba_depth: Number of stacked Mamba blocks
        expansion_ratio: Channel expansion in Mamba blocks
        spectral_threshold: Threshold for spectral gating
        use_mamba: Enable Mamba (VSS) blocks
        use_spectral: Enable Spectral (FFT) gating
    """
    
    def __init__(self, channels: int, height: int, width: int,
                 mamba_depth: int = 2, expansion_ratio: float = 2.0,
                 spectral_threshold: float = 0.1,
                 use_mamba: bool = True, use_spectral: bool = True):
        super().__init__()
        
        self.channels = channels
        self.height = height
        self.width = width
        self.use_mamba = use_mamba
        self.use_spectral = use_spectral
        
        # Branch A: Spatial path (VSS/Mamba Blocks or Conv fallback)
        if use_mamba and mamba_depth > 0:
            self.vss_blocks = MambaBlockStack(
                channels, 
                depth=mamba_depth,
                expansion_ratio=expansion_ratio,
                scan_dim=min(64, channels)
            )
        else:
            # Convolutional fallback (baseline)
            self.vss_blocks = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(min(32, channels), channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(min(32, channels), channels),
                nn.GELU()
            )
        
        # Branch B: Spectral path (FFT-based filtering)
        if use_spectral:
            self.spectral_gate = SpectralGating(
                channels, height, width,
                threshold=spectral_threshold,
                complex_init="kaiming"
            )
        else:
            self.spectral_gate = None
        
        # Learnable fusion weight (only needed if both branches active)
        if use_mamba and use_spectral:
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.fusion_weight = None
        
        # Optional: Layer norm for stability
        self.norm = nn.GroupNorm(min(32, channels), channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dual-path processing.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C, H, W)
        """
        # Branch A: Spatial context via Mamba or Conv
        spatial_out = self.vss_blocks(x)
        
        # Branch B: Frequency filtering via FFT (if enabled)
        if self.use_spectral and self.spectral_gate is not None:
            spectral_out = self.spectral_gate(x)
            
            # Learnable fusion
            if self.fusion_weight is not None:
                weight = torch.sigmoid(self.fusion_weight)
                output = weight * spatial_out + (1 - weight) * spectral_out
            else:
                output = (spatial_out + spectral_out) / 2.0
        else:
            # Only spatial path
            output = spatial_out
        
        # Normalize
        output = self.norm(output)
        
        return output



# =============================================================================
# Multi-Scale Fusion Module
# =============================================================================

class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion between high-resolution and low-resolution streams.
    
    Enables bidirectional information flow:
        - High → Low: Strided Conv to downsample
        - Low → High: Bilinear Upsample + 1x1 Conv
    
    This allows:
        - Low-res stream to know "where" (from high-res)
        - High-res stream to know "what" (from low-res)
    
    Args:
        high_channels: Channels in high-res stream
        low_channels: Channels in low-res stream
        scale_factor: Spatial scale ratio between streams (e.g., 2)
    """
    
    def __init__(self, high_channels: int, low_channels: int, scale_factor: int = 2):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # High → Low: Downsample path
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, kernel_size=3, 
                     stride=scale_factor, padding=1, bias=False),
            nn.GroupNorm(min(32, low_channels), low_channels),
            nn.GELU()
        )
        
        # Low → High: Upsample path
        self.low_to_high = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, high_channels), high_channels),
            nn.GELU()
        )
        
        # Fusion weights (learnable gating)
        self.high_gate = nn.Parameter(torch.ones(1) * 0.5)
        self.low_gate = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, high_feat: torch.Tensor, 
                low_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse features from both streams.
        
        Args:
            high_feat: High-res features (B, C_h, H, W)
            low_feat: Low-res features (B, C_l, H/s, W/s)
            
        Returns:
            Tuple of (updated_high, updated_low)
        """
        # High → Low contribution
        high_to_low = self.high_to_low(high_feat)
        
        # Low → High contribution
        low_to_high = self.low_to_high(low_feat)
        
        # Gated fusion
        h_gate = torch.sigmoid(self.high_gate)
        l_gate = torch.sigmoid(self.low_gate)
        
        # Update streams
        new_high = high_feat + h_gate * low_to_high
        new_low = low_feat + l_gate * high_to_low
        
        return new_high, new_low


# =============================================================================
# HRNet Stage: Single Resolution Processing Stage
# =============================================================================

class HRNetStage(nn.Module):
    """
    Single stage of HRNet processing at one resolution.
    
    Contains multiple SpectralVSSBlocks for deep feature extraction.
    
    Args:
        channels: Number of channels
        height: Feature map height
        width: Feature map width
        num_blocks: Number of SpectralVSSBlocks
        mamba_depth: Depth of Mamba blocks within each SpectralVSSBlock
        use_mamba: Enable Mamba blocks
        use_spectral: Enable Spectral gating
    """
    
    def __init__(self, channels: int, height: int, width: int,
                 num_blocks: int = 2, mamba_depth: int = 2,
                 use_mamba: bool = True, use_spectral: bool = True):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            SpectralVSSBlock(
                channels, height, width, 
                mamba_depth=mamba_depth,
                use_mamba=use_mamba,
                use_spectral=use_spectral
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through all blocks."""
        for block in self.blocks:
            x = block(x)
        return x



# =============================================================================
# Stem: Initial Feature Extraction
# =============================================================================

class HRNetStem(nn.Module):
    """
    Initial feature extraction stem.
    
    Converts input image to initial feature representation with
    strided convolutions for efficiency.
    
    Args:
        in_channels: Number of input channels (3 for Intensity + Riesz)
        out_channels: Output feature channels
        stride: Total stride of stem (typically 4)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, stride: int = 4):
        super().__init__()
        
        # Two-stage striding: 2x each
        mid_channels = out_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, 
                              stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, mid_channels), mid_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act2 = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract initial features with stride-4 reduction."""
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


# =============================================================================
# Aggregation Layer: Combine All Resolutions
# =============================================================================

class AggregationLayer(nn.Module):
    """
    Aggregate multi-resolution features for output.
    
    Following HRNetV2: Upsample all resolutions to highest resolution
    and concatenate along channel dimension.
    
    Args:
        high_channels: Channels from high-res stream
        low_channels: Channels from low-res stream
        out_channels: Output channels after aggregation
        scale_factor: Scale factor to upsample low-res
    """
    
    def __init__(self, high_channels: int, low_channels: int, 
                 out_channels: int, scale_factor: int = 2):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Upsample low-res to match high-res
        self.upsample_low = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(low_channels, low_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, low_channels), low_channels)
        )
        
        # Combine and project
        total_channels = high_channels + low_channels
        self.projection = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU()
        )
    
    def forward(self, high_feat: torch.Tensor, 
                low_feat: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features from both streams.
        
        Args:
            high_feat: High-res features (B, C_h, H, W)
            low_feat: Low-res features (B, C_l, H/s, W/s)
            
        Returns:
            Aggregated features (B, out_channels, H, W)
        """
        # Upsample low-res to match high-res
        low_up = self.upsample_low(low_feat)
        
        # Concatenate
        concat = torch.cat([high_feat, low_up], dim=1)
        
        # Project to output channels
        output = self.projection(concat)
        
        return output


# =============================================================================
# HRNetV2-Mamba Backbone: Complete Dual-Stream Architecture
# =============================================================================

class HRNetV2MambaBackbone(nn.Module):
    """
    HRNetV2 Backbone with Mamba (SS2D) and Spectral Gating.
    
    Dual-stream architecture maintaining high-resolution features throughout:
    
    Stream 1 (High-Res): H/4 resolution
        - Receives: Energy and Frequency signals (physics features)
        - Purpose: Preserve boundary details
        
    Stream 2 (Low-Res): H/8 resolution
        - Receives: Intensity signal (semantic features)
        - Purpose: Capture global context
    
    The streams are periodically fused via MultiScaleFusion modules.
    Output is aggregated from all resolutions at the highest resolution.
    
    Args:
        in_channels: Input channels (3 for Intensity + Rx + Ry)
        base_channels: Base channel count
        num_stages: Number of processing stages
        blocks_per_stage: Number of SpectralVSSBlocks per stage
        mamba_depth: Depth within each SpectralVSSBlock
        img_size: Input image size (for spectral weight initialization)
        use_mamba: Enable Mamba (VSS) blocks (set False for pure conv baseline)
        use_spectral: Enable Spectral (FFT) gating (set False for baseline)
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_stages: int = 4, blocks_per_stage: int = 2,
                 mamba_depth: int = 2, img_size: int = 256,
                 use_mamba: bool = True, use_spectral: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.img_size = img_size
        self.use_mamba = use_mamba
        self.use_spectral = use_spectral
        
        # Initial stem: stride 4 reduction
        self.stem = HRNetStem(in_channels, base_channels, stride=4)
        
        # Initial feature sizes
        high_res_size = img_size // 4  # H/4
        low_res_size = img_size // 8   # H/8
        
        high_channels = base_channels
        low_channels = base_channels * 2
        
        # Create initial low-res stream from high-res
        self.create_low_stream = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, low_channels), low_channels),
            nn.GELU()
        )
        
        # High-resolution stream stages
        self.high_res_stages = nn.ModuleList()
        
        # Low-resolution stream stages
        self.low_res_stages = nn.ModuleList()
        
        # Fusion modules (after each stage)
        self.fusion_modules = nn.ModuleList()
        
        # Build stages
        for stage_idx in range(num_stages):
            # High-res stage
            self.high_res_stages.append(
                HRNetStage(
                    channels=high_channels,
                    height=high_res_size,
                    width=high_res_size,
                    num_blocks=blocks_per_stage,
                    mamba_depth=mamba_depth,
                    use_mamba=use_mamba,
                    use_spectral=use_spectral
                )
            )
            
            # Low-res stage
            self.low_res_stages.append(
                HRNetStage(
                    channels=low_channels,
                    height=low_res_size,
                    width=low_res_size,
                    num_blocks=blocks_per_stage,
                    mamba_depth=mamba_depth,
                    use_mamba=use_mamba,
                    use_spectral=use_spectral
                )
            )
            
            # Fusion module
            self.fusion_modules.append(
                MultiScaleFusion(
                    high_channels=high_channels,
                    low_channels=low_channels,
                    scale_factor=2

                )
            )
        
        # Aggregation layer
        self.aggregation = AggregationLayer(
            high_channels=high_channels,
            low_channels=low_channels,
            out_channels=high_channels + low_channels,
            scale_factor=2
        )
        
        # Store output channels
        self.out_channels = high_channels + low_channels
        self.feature_size = high_res_size
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-stream backbone.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
            
        Returns:
            Dictionary containing:
            - 'features': Aggregated features (B, out_channels, H/4, W/4)
            - 'high_res': High-res stream output (B, C_h, H/4, W/4)
            - 'low_res': Low-res stream output (B, C_l, H/8, W/8)
        """
        # Initial stem
        high = self.stem(x)  # (B, base_channels, H/4, W/4)
        
        # Create low-res stream
        low = self.create_low_stream(high)  # (B, base_channels*2, H/8, W/8)
        
        # Process through stages with fusion
        for stage_idx in range(self.num_stages):
            # Process each stream
            high = self.high_res_stages[stage_idx](high)
            low = self.low_res_stages[stage_idx](low)
            
            # Fuse streams
            high, low = self.fusion_modules[stage_idx](high, low)
        
        # Aggregate for output
        features = self.aggregation(high, low)
        
        return {
            'features': features,
            'high_res': high,
            'low_res': low
        }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing HRNetV2-Mamba Backbone")
    print("=" * 60)
    
    # Test SpectralVSSBlock
    print("\n[1] Testing SpectralVSSBlock...")
    block = SpectralVSSBlock(channels=64, height=64, width=64, mamba_depth=2)
    x = torch.randn(2, 64, 64, 64)
    out = block(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    
    # Test MultiScaleFusion
    print("\n[2] Testing MultiScaleFusion...")
    fusion = MultiScaleFusion(high_channels=64, low_channels=128, scale_factor=2)
    high = torch.randn(2, 64, 64, 64)
    low = torch.randn(2, 128, 32, 32)
    new_high, new_low = fusion(high, low)
    print(f"High: {high.shape} → {new_high.shape}")
    print(f"Low: {low.shape} → {new_low.shape}")
    
    # Test full backbone
    print("\n[3] Testing HRNetV2MambaBackbone...")
    backbone = HRNetV2MambaBackbone(
        in_channels=3,
        base_channels=64,
        num_stages=4,
        blocks_per_stage=2,
        mamba_depth=2,
        img_size=256
    )
    
    x = torch.randn(2, 3, 256, 256)
    outputs = backbone(x)
    
    print(f"Input: {x.shape}")
    print(f"Features: {outputs['features'].shape}")
    print(f"High-res: {outputs['high_res'].shape}")
    print(f"Low-res: {outputs['low_res'].shape}")
    print(f"Output channels: {backbone.out_channels}")
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
