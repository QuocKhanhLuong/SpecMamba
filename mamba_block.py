"""
Mamba-like Block Module (Lightweight VSS Block Implementation)
Pure PyTorch implementation of Visual State Space scanning without external SSM libraries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DirectionalScanner(nn.Module):
    """
    Scans 2D feature maps in multiple directions to simulate 2D-SS2D.
    Implements sequential scanning in 4 directions: right, down, left, up.
    """
    
    def __init__(self, channels: int, scan_dim: int = 64):
        """
        Initialize DirectionalScanner.
        
        Args:
            channels: Number of input channels
            scan_dim: Hidden dimension for sequential processing
        """
        super().__init__()
        self.channels = channels
        self.scan_dim = scan_dim
        
        # Learnable projection to scan_dim for each direction
        self.proj_in = nn.Linear(channels, scan_dim)
        
        # GRU cell for sequential state processing (simulates SSM)
        self.gru_cell = nn.GRUCell(scan_dim, scan_dim)
        
        # Project back to original channels
        self.proj_out = nn.Linear(scan_dim, channels)
        
    def _scan_direction(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        """
        Scan feature map in a specific direction.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            direction: One of ['right', 'down', 'left', 'up']
            
        Returns:
            Scanned tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Prepare sequence based on direction
        if direction == "right":
            # Scan left-to-right: (B, H*W, C) after reshape
            x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        elif direction == "down":
            # Scan top-to-bottom
            x = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        elif direction == "left":
            # Scan right-to-left (reverse)
            x = x.permute(0, 2, 3, 1).flip(1).reshape(B * H, W, C)  # (B*H, W, C)
        elif direction == "up":
            # Scan bottom-to-top (reverse)
            x = x.permute(0, 3, 2, 1).flip(1).reshape(B * W, H, C)  # (B*W, H, C)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Project to scan dimension
        x = self.proj_in(x)  # (*, W/H, scan_dim)
        
        # Apply GRU cell sequentially (simulates SSM forward pass)
        outputs = []
        h = torch.zeros(x.shape[0], self.scan_dim, device=x.device, dtype=x.dtype)
        
        for t in range(x.shape[1]):
            h = self.gru_cell(x[:, t], h)  # GRU step
            outputs.append(h)
        
        x = torch.stack(outputs, dim=1)  # (*, W/H, scan_dim)
        
        # Project back to original channels
        x = self.proj_out(x)  # (*, W/H, C)
        
        # Reshape back to (B, C, H, W)
        if direction == "right":
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif direction == "down":
            x = x.reshape(B, W, H, C).permute(0, 3, 2, 1)
        elif direction == "left":
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).flip(-1)
        elif direction == "up":
            x = x.reshape(B, W, H, C).permute(0, 3, 2, 1).flip(-2)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply directional scanning in 4 directions and aggregate.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Aggregated output tensor of shape (B, C, H, W)
        """
        # Scan in all 4 directions
        scan_right = self._scan_direction(x, "right")
        scan_down = self._scan_direction(x, "down")
        scan_left = self._scan_direction(x, "left")
        scan_up = self._scan_direction(x, "up")
        
        # Aggregate by averaging
        output = (scan_right + scan_down + scan_left + scan_up) / 4.0
        
        return output


class VSSBlock(nn.Module):
    """
    Visual State Space Block - Core Mamba-like component.
    Combines convolutional preprocessing with multi-directional scanning.
    """
    
    def __init__(self, channels: int, hidden_dim: Optional[int] = None, 
                 scan_dim: int = 64, expansion_ratio: float = 2.0):
        """
        Initialize VSSBlock.
        
        Args:
            channels: Number of input channels
            hidden_dim: Hidden dimension (default: channels * expansion_ratio)
            scan_dim: Hidden dimension for directional scanner
            expansion_ratio: Channel expansion ratio for internal processing
        """
        super().__init__()
        self.channels = channels
        hidden_dim = hidden_dim or int(channels * expansion_ratio)
        
        # Preprocessing: expand channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv_expand = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=True)
        
        # Directional scanning
        self.scanner = DirectionalScanner(hidden_dim, scan_dim=scan_dim)
        
        # Postprocessing: contract channels back
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6)
        self.conv_contract = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W) with residual
        """
        residual = x
        
        # Preprocessing
        x = self.norm1(x)
        x = self.conv_expand(x)
        x = F.gelu(x)
        
        # Directional scanning (core SSM-like operation)
        x = self.scanner(x)
        
        # Postprocessing
        x = self.norm2(x)
        x = self.conv_contract(x)
        
        # Residual connection
        output = x + residual
        
        return output


class MambaBlockStack(nn.Module):
    """Stack of multiple VSS blocks for hierarchical processing."""
    
    def __init__(self, channels: int, depth: int = 2, **kwargs):
        """
        Initialize stack of VSS blocks.
        
        Args:
            channels: Number of channels
            depth: Number of VSS blocks to stack
            **kwargs: Additional arguments passed to VSSBlock
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(channels, **kwargs) for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    # Test VSSBlock
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width)
    
    vss_block = VSSBlock(channels, scan_dim=32)
    output = vss_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Module parameters: {sum(p.numel() for p in vss_block.parameters())}")
