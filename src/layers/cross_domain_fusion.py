"""
Cross-Domain Attention Fusion

Fuses spatial features (from HRNet encoder) with spectral boundary features
(from PC + Shearlet encoder) using cross-attention.

Spatial features "ask" (Query) spectral features (Key, Value) about
boundary locations, allowing the network to attend to noise-robust
boundary cues from the frequency domain.

Includes a lightweight SE-block fallback for memory-constrained settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_norm(num_channels, num_groups=8):
    """GroupNorm for stability with small batch sizes."""
    if num_channels < num_groups:
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class CrossDomainAttentionFusion(nn.Module):
    """
    Cross-attention fusion between spatial and spectral features.

    Q = W_q × F_spatial
    K = W_k × F_spectral
    V = W_v × F_spectral
    F_fused = F_spatial + softmax(QK^T / √d) × V

    Uses efficient windowed cross-attention to avoid O(N²) on full resolution.

    Args:
        dim: Feature dimension (must be same for both branches)
        num_heads: Number of attention heads (default: 8)
        window_size: Window size for windowed attention (default: 8)
        qkv_bias: Whether to use bias in Q/K/V projections (default: True)

    Input:
        f_spatial:  [B, C, H, W] — spatial features from HRNet encoder
        f_spectral: [B, C, H, W] — spectral features from SpectralEncoder

    Output:
        f_fused: [B, C, H, W] — fused features
    """

    def __init__(self, dim, num_heads=8, window_size=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query from spatial, Key/Value from spectral
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Layer norms
        self.norm_spatial = nn.LayerNorm(dim)
        self.norm_spectral = nn.LayerNorm(dim)

        # Gate for residual
        self.gate = nn.Parameter(torch.zeros(1))

    def _window_partition(self, x, window_size):
        """Partition feature map into non-overlapping windows.

        Args:
            x: [B, H, W, C]
            window_size: int

        Returns:
            windows: [B * num_windows, window_size, window_size, C]
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size,
                    W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def _window_reverse(self, windows, window_size, H, W):
        """Reverse window partition.

        Args:
            windows: [B * num_windows, window_size, window_size, C]

        Returns:
            x: [B, H, W, C]
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size,
                         window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, f_spatial, f_spectral):
        """
        Args:
            f_spatial:  [B, C, H, W]
            f_spectral: [B, C, H, W]

        Returns:
            f_fused: [B, C, H, W]
        """
        B, C, H, W = f_spatial.shape
        ws = self.window_size

        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            f_spatial = F.pad(f_spatial, (0, pad_w, 0, pad_h))
            f_spectral = F.pad(f_spectral, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = f_spatial.shape

        # Reshape to (B, H, W, C) for attention
        spatial_hwc = f_spatial.permute(0, 2, 3, 1)   # [B, Hp, Wp, C]
        spectral_hwc = f_spectral.permute(0, 2, 3, 1)  # [B, Hp, Wp, C]

        # Window partition
        spatial_win = self._window_partition(spatial_hwc, ws)   # [nW*B, ws, ws, C]
        spectral_win = self._window_partition(spectral_hwc, ws)  # [nW*B, ws, ws, C]

        nWB = spatial_win.shape[0]
        N = ws * ws

        # Flatten windows
        spatial_flat = spatial_win.view(nWB, N, C)    # [nW*B, N, C]
        spectral_flat = spectral_win.view(nWB, N, C)  # [nW*B, N, C]

        # Layer norm
        spatial_normed = self.norm_spatial(spatial_flat)
        spectral_normed = self.norm_spectral(spectral_flat)

        # Q from spatial, K/V from spectral
        q = self.q_proj(spatial_normed).view(nWB, N, self.num_heads, self.head_dim)
        k = self.k_proj(spectral_normed).view(nWB, N, self.num_heads, self.head_dim)
        v = self.v_proj(spectral_normed).view(nWB, N, self.num_heads, self.head_dim)

        # Transpose to [nWB, heads, N, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [nWB, heads, N, N]
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = (attn @ v)  # [nWB, heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(nWB, N, C)
        out = self.out_proj(out)

        # Reshape back to windows
        out = out.view(nWB, ws, ws, C)
        # Reverse window partition
        out = self._window_reverse(out, ws, Hp, Wp)  # [B, Hp, Wp, C]
        out = out.permute(0, 3, 1, 2)  # [B, C, Hp, Wp]

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
            f_spatial = f_spatial[:, :, :H, :W]

        # Gated residual
        f_fused = f_spatial + torch.sigmoid(self.gate) * out

        return f_fused


class SEFusion(nn.Module):
    """
    Lightweight SE-block fusion (fallback for memory-constrained settings).

    Uses channel-wise attention to weight spectral features before
    adding them to spatial features.

    Args:
        dim: Feature dimension
        reduction: Channel reduction ratio for SE bottleneck (default: 16)
    """

    def __init__(self, dim, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 2, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, f_spatial, f_spectral):
        """
        Args:
            f_spatial:  [B, C, H, W]
            f_spectral: [B, C, H, W]

        Returns:
            f_fused: [B, C, H, W]
        """
        # Channel descriptor from both branches
        combined = torch.cat([
            F.adaptive_avg_pool2d(f_spatial, 1).flatten(1),
            F.adaptive_avg_pool2d(f_spectral, 1).flatten(1)
        ], dim=1)  # [B, C*2]

        # SE weights (recompute from concat)
        weights = self.se(
            torch.cat([f_spatial, f_spectral], dim=1)
        )  # [B, C]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Weighted spectral + residual
        f_fused = f_spatial + torch.sigmoid(self.gate) * (f_spectral * weights)
        return f_fused


if __name__ == "__main__":
    B, C, H, W = 2, 64, 16, 16

    f_spatial = torch.randn(B, C, H, W)
    f_spectral = torch.randn(B, C, H, W)

    # Test cross-attention fusion
    fusion = CrossDomainAttentionFusion(dim=C, num_heads=8, window_size=8)
    out = fusion(f_spatial, f_spectral)
    print(f"CrossDomainAttention: {f_spatial.shape} + {f_spectral.shape} -> {out.shape}")
    print(f"  Params: {sum(p.numel() for p in fusion.parameters()):,}")

    # Test SE fusion (lightweight fallback)
    se_fusion = SEFusion(dim=C)
    out_se = se_fusion(f_spatial, f_spectral)
    print(f"SEFusion: {f_spatial.shape} + {f_spectral.shape} -> {out_se.shape}")
    print(f"  Params: {sum(p.numel() for p in se_fusion.parameters()):,}")
