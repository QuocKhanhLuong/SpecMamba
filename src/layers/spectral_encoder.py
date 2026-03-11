"""
Lightweight Spectral Encoder

Encodes concatenated Phase Congruency map + Shearlet Energy maps
into feature representations matching the spatial branch dimension.

Architecture: 3× (Conv 3×3 + GroupNorm + GELU) with residual connections
Target: ~2M params — kept lightweight to justify efficiency claim.
"""

import torch
import torch.nn as nn


def get_norm(num_channels, num_groups=8):
    """GroupNorm for stability with small batch sizes."""
    if num_channels < num_groups:
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class SpectralEncoder(nn.Module):
    """
    Lightweight encoder for spectral boundary features.

    Takes concatenated PC map [1 ch] + Shearlet maps [9 ch] = 10 channels
    and produces feature maps matching the spatial branch dimension.

    Args:
        in_channels: Number of input channels (default: 10 = 1 PC + 9 Shearlet)
        out_channels: Output feature dimension (must match spatial branch)
        mid_channels: Hidden layer channels (default: 64)

    Input:  [B, in_channels, H, W]
    Output: [B, out_channels, H, W]
    """

    def __init__(self, in_channels=10, out_channels=64, mid_channels=64):
        super().__init__()

        # Block 1: in_channels → mid_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            get_norm(mid_channels),
            nn.GELU(),
        )

        # Block 2: mid → mid (with residual)
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            get_norm(mid_channels),
            nn.GELU(),
        )

        # Block 3: mid → out_channels (with residual if dims match)
        self.block3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            get_norm(out_channels),
            nn.GELU(),
        )

        # Projection for residual if mid_channels != out_channels
        if mid_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                get_norm(out_channels),
            )
        else:
            self.proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W] concatenation of PC map + Shearlet maps

        Returns:
            features: [B, out_channels, H, W]
        """
        # Block 1 (no residual — channel change)
        x = self.block1(x)

        # Block 2 (with residual)
        identity = x
        x = self.block2(x) + identity

        # Block 3 (with projected residual)
        identity = self.proj(x)
        x = self.block3(x) + identity

        return x


if __name__ == "__main__":
    B, H, W = 2, 64, 64
    x = torch.randn(B, 10, H, W)  # 1 PC + 9 Shearlet channels
    encoder = SpectralEncoder(in_channels=10, out_channels=64)
    out = encoder(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in encoder.parameters())
    print(f"Params: {params:,}")
