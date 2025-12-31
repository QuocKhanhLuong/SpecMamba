"""
Spectral Mamba Architecture (Spec-VMamba)
Combines Spectral Analysis with Visual State Space Models for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from spectral_layers import SpectralGating
from mamba_block import VSSBlock, MambaBlockStack


class SpectralVSSBlock(nn.Module):
    """
    Spectral Visual State Space Block - Main Building Unit.
    
    Implements parallel dual-path architecture:
    - Path A (Spatial): VSS Block for long-range context
    - Path B (Spectral): Spectral Gating for boundary sharpness
    """
    
    def __init__(self, channels: int, height: int, width: int,
                 depth: int = 2, expansion_ratio: float = 2.0,
                 threshold: float = 0.1):
        """
        Initialize SpectralVSSBlock.
        
        Args:
            channels: Number of input channels
            height: Input height
            width: Input width
            depth: Number of VSS blocks to stack
            expansion_ratio: Channel expansion for VSS processing
            threshold: Hard thresholding value for spectral filtering
        """
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # Branch A: Spatial path (VSS Blocks)
        self.vss_blocks = MambaBlockStack(
            channels, depth=depth, 
            expansion_ratio=expansion_ratio, 
            scan_dim=min(64, channels)
        )
        
        # Branch B: Spectral path (FFT-based filtering)
        self.spectral_gate = SpectralGating(
            channels, height, width, 
            threshold=threshold, 
            complex_init="kaiming"
        )
        
        # Fusion layer (learnable weighting)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dual-path processing and fusion.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Fused output tensor of shape (B, C, H, W)
        """
        # Branch A: Spatial context (VSS)
        spatial_out = self.vss_blocks(x)
        
        # Branch B: Frequency filtering (Spectral)
        spectral_out = self.spectral_gate(x)
        
        # Learnable fusion with sigmoid weight
        weight = torch.sigmoid(self.fusion_weight)
        output = weight * spatial_out + (1 - weight) * spectral_out
        
        return output


class PatchEmbedding(nn.Module):
    """Patch embedding layer for converting images to patch sequences."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, 
                 patch_size: int = 4):
        """
        Initialize patch embedding.
        
        Args:
            in_channels: Input image channels (1 for medical images)
            out_channels: Output embedding dimension
            patch_size: Size of patches (4x4)
        """
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Embedded tensor (B, C, H', W') where H' = H / patch_size
        """
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class PatchMerging(nn.Module):
    """Downsampling layer that reduces spatial dimensions."""
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        """
        Initialize patch merging.
        
        Args:
            channels: Input channels
            out_channels: Output channels (default: channels * 2)
        """
        super().__init__()
        out_channels = out_channels or channels * 2
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=2, 
                              stride=2, bias=True)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class PatchExpanding(nn.Module):
    """Upsampling layer that increases spatial dimensions."""
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        """
        Initialize patch expanding.
        
        Args:
            channels: Input channels
            out_channels: Output channels (default: channels // 2)
        """
        super().__init__()
        out_channels = out_channels or channels // 2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=True)
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class SpectralVMUNet(nn.Module):
    """
    Spectral Mamba U-Net - Full segmentation architecture.
    
    Combines Spectral Analysis with Visual State Space Models in a U-shaped
    architecture for medical image segmentation.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 img_size: int = 256, base_channels: int = 64,
                 num_stages: int = 4, depth: int = 2):
        """
        Initialize SpectralVMUNet.
        
        Args:
            in_channels: Number of input channels (1 for medical images)
            out_channels: Number of output classes
            img_size: Input image size (assumed square)
            base_channels: Base number of channels
            num_stages: Number of encoder/decoder stages
            depth: Depth of VSS blocks in each stage
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.base_channels = base_channels
        self.num_stages = num_stages
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, base_channels, patch_size=4)
        initial_size = img_size // 4
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(num_stages):
            in_ch = base_channels * (2 ** i)
            out_ch = in_ch
            h = w = initial_size // (2 ** i)
            
            # SpectralVSSBlock
            block = SpectralVSSBlock(
                in_ch, h, w, depth=depth, expansion_ratio=2.0, threshold=0.1
            )
            self.encoder_blocks.append(block)
            
            # Downsampling (except after last encoder block)
            if i < num_stages - 1:
                down = PatchMerging(in_ch, in_ch * 2)
                self.downsample_layers.append(down)
        
        # Bottleneck - uses the last encoder's output channels
        # After num_stages-1 downsamplings, channels = base_channels * 2^(num_stages-1)
        bottleneck_ch = base_channels * (2 ** (num_stages - 1))
        bottleneck_h = bottleneck_w = initial_size // (2 ** (num_stages - 1))
        self.bottleneck = SpectralVSSBlock(
            bottleneck_ch, bottleneck_h, bottleneck_w,
            depth=depth + 1, expansion_ratio=2.0, threshold=0.1
        )
        
        # Decoder
        # We have num_stages - 1 decoder stages (matching skip connections)
        # Each decoder stage: upsample -> concat with skip -> fusion -> SpectralVSSBlock
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        num_decoder_stages = num_stages - 1
        
        for i in range(num_decoder_stages):
            # Going from deepest to shallowest
            # i=0: from bottleneck (8x8, 512ch) -> upsample to (16x16, 256ch)
            # i=1: from 16x16, 256ch -> upsample to (32x32, 128ch)
            # i=2: from 32x32, 128ch -> upsample to (64x64, 64ch)
            
            # Input channels: for i=0, it's bottleneck_ch; else from previous decoder output
            if i == 0:
                in_ch = bottleneck_ch  # 512 for default
            else:
                in_ch = base_channels * (2 ** (num_stages - 1 - i))
            
            # Output channels after upsampling
            out_ch = base_channels * (2 ** (num_stages - 2 - i))
            
            # Upsampling layer
            up = PatchExpanding(in_ch, out_ch)
            self.upsample_layers.append(up)
            
            # Skip connection comes from encoder at level (num_decoder_stages - 1 - i)
            # which has same spatial size after upsampling
            skip_ch = out_ch  # Skip has same channels as upsampled output
            
            # Spatial size at this level
            h = w = initial_size // (2 ** (num_stages - 2 - i))
            
            # Fusion: concatenate upsampled + skip, then reduce channels
            fused_ch = out_ch + skip_ch  # After concatenation
            fusion = nn.Sequential(
                nn.Conv2d(fused_ch, out_ch, kernel_size=1, bias=True),
                nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch, eps=1e-6)
            )
            self.decoder_blocks.append(fusion)
            
            # SpectralVSSBlock after fusion
            vss = SpectralVSSBlock(
                out_ch, h, w, depth=depth, expansion_ratio=2.0, threshold=0.1
            )
            self.decoder_blocks.append(vss)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, 
                      padding=1, bias=True),
            nn.GroupNorm(num_groups=32, num_channels=base_channels // 2, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=1, bias=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full network.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Encoder path with skip connections storage
        # Skip connections are saved BEFORE downsampling
        skips = []
        for i in range(self.num_stages):
            x = self.encoder_blocks[i](x)
            # Save skip connection before downsampling
            if i < self.num_stages - 1:
                skips.append(x)
                x = self.downsample_layers[i](x)
        
        # The last encoder output goes to bottleneck (no skip for this level)
        # skips now contains: [stage0_out, stage1_out, stage2_out] for 4 stages
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        # Decoder stages: num_stages - 1 (since last encoder has no skip)
        num_decoder_stages = self.num_stages - 1
        
        for i in range(num_decoder_stages):
            # Upsample
            x = self.upsample_layers[i](x)
            
            # Concatenate skip connection (in reverse order)
            # For i=0: skip from encoder stage num_stages-2 (last skip)
            # For i=1: skip from encoder stage num_stages-3
            skip_idx = num_decoder_stages - 1 - i
            skip = skips[skip_idx]
            x = torch.cat([x, skip], dim=1)
            
            # Fusion and processing
            x = self.decoder_blocks[2 * i](x)  # Fusion conv
            x = self.decoder_blocks[2 * i + 1](x)  # SpectralVSSBlock
        
        # Segmentation head
        output = self.seg_head(x)
        
        # Upsample to original resolution (since patch embedding uses stride 4)
        output = F.interpolate(output, size=(self.img_size, self.img_size),
                               mode='bilinear', align_corners=True)
        
        return output


if __name__ == "__main__":
    # Test the full architecture
    batch_size = 2
    in_channels = 1
    out_channels = 3  # Binary segmentation + background
    img_size = 256
    
    model = SpectralVMUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        base_channels=64,
        num_stages=4,
        depth=2
    )
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
