"""
Modular Blocks for HRNet Backbone (Standardized & Corrected)
- ConvNeXt Block (SOTA CNN)
- Deformable Conv Block (DCNv2)
- Inverted Residual Block (MobileNetV2)
- Swin Transformer Block (Fixed Masking)
- Fourier Neural Operator (FNO) Block
- Wavelet Block (Handled Padding)
- RWKV Block (Corrected Spatial Mixing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
import math
from typing import Optional, Tuple, List


class DropPath(nn.Module):
    """Stochastic Depth / Drop Path"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for Channel First images (B, C, H, W)"""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

# =============================================================================
# NHÓM 1: CONVOLUTIONAL MODERN
# =============================================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block - SOTA CNN (Facebook Research 2022)"""
    def __init__(self, dim, drop_path=0., layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = input + self.drop_path(x)
        return x


class DeformableConvBlock(nn.Module):
    """
    Deformable Conv v3 (Pyramid Ready)
    - Supports dilation for multi-scale context (ASPP-style)
    - Supports channel change (in_dim != out_dim) with projection shortcut
    """
    def __init__(self, dim, out_dim=None, kernel_size=3, dilation=1):
        super().__init__()
        self.in_dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Dynamic padding to maintain spatial size with dilation
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        
        # Offset & Modulator (must use same dilation)
        self.offset_conv = nn.Conv2d(
            self.in_dim, 2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, padding=self.padding, dilation=dilation
        )
        self.modulator_conv = nn.Conv2d(
            self.in_dim, kernel_size * kernel_size,
            kernel_size=kernel_size, padding=self.padding, dilation=dilation
        )
        
        # Main conv (in -> out)
        self.regular_conv = nn.Conv2d(
            self.in_dim, self.out_dim, 
            kernel_size=kernel_size, padding=self.padding, 
            dilation=dilation, bias=True
        )
        
        self.norm = nn.BatchNorm2d(self.out_dim)
        self.act = nn.GELU()
        
        # Shortcut projection for channel mismatch
        if self.in_dim != self.out_dim:
            self.shortcut_proj = nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_dim)
            )
        else:
            self.shortcut_proj = nn.Identity()
        
        # Init weights (hardened)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.modulator_conv.weight)
        nn.init.constant_(self.modulator_conv.bias, -2.0)

    def forward(self, x):
        # Offset bounded by tanh
        offset = torch.tanh(self.offset_conv(x)) * self.kernel_size 
        modulator = torch.sigmoid(self.modulator_conv(x))
        
        out = torchvision.ops.deform_conv2d(
            input=x, 
            offset=offset, 
            weight=self.regular_conv.weight, 
            bias=self.regular_conv.bias,     
            padding=self.padding,
            mask=modulator,
            dilation=self.dilation  # Enable multi-scale receptive field
        )
        
        out = self.norm(out)
        out = self.act(out)
        
        # Residual with projection
        shortcut = self.shortcut_proj(x)
        return shortcut + out


class InvertedResidualBlock(nn.Module):
    """MobileNetV2 Inverted Residual Block"""
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv(x)


# =============================================================================
# NHÓM 2: ATTENTION MECHANISMS
# =============================================================================

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with Correct Masked Shifted Window Attention"""
    def __init__(self, dim, num_heads=8, window_size=7, shift_size=0, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def calculate_mask(self, H, W, device):
        # Calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)

        # Pad checks
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.calculate_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = shortcut + self.drop_path(x.permute(0, 3, 1, 2))
        
        # FFN
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            # Mask should be (nW, N, N)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =============================================================================
# NHÓM 3: SIGNAL PROCESSING
# =============================================================================

class FourierNeuralOperatorBlock(nn.Module):
    """Fourier Neural Operator (FNO) Block (Better Init)"""
    def __init__(self, dim, modes=16, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.modes = modes
        
        # Complex weights: (in, out, modes1, modes2)
        # Shape thực tế trong pytorch view_as_complex là (dim, dim, modes, modes, 2)
        self.weights = nn.Parameter(torch.empty(dim, dim, modes, modes, 2))
        
        # Kaiming/Xavier Initialization cho số phức
        # Scale = 1 / sqrt(in_channels * modes * modes)
        scale = (1 / (dim * modes * modes))
        nn.init.normal_(self.weights, std=scale)
        
        self.norm = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, dim, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros_like(x_ft)
        
        # Dynamic shape handling
        m1 = min(self.modes, H // 2 + 1)
        m2 = min(self.modes, W // 2 + 1)
        
        # Complex multiplication
        weights = torch.view_as_complex(self.weights)
        # Slice weights đúng kích thước (nếu ảnh nhỏ hơn modes)
        w_curr = weights[:, :, :m1, :m2]
        
        # Einstein summation: batch, in, x, y -> batch, out, x, y
        out_ft[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy', x_ft[:, :, :m1, :m2], w_curr)
        
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        x = shortcut + x
        x = x + self.mlp(x)
        return x


class WaveletBlock(nn.Module):
    """Wavelet Transform Block with Padding Support"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer('low_filter', torch.tensor([[1, 1], [1, 1]]) / 2)
        self.register_buffer('high_filter', torch.tensor([[-1, 1], [-1, 1]]) / 2)
        
        self.ll_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.lh_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.hl_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.hh_conv = nn.Conv2d(dim, dim, 3, padding=1)
        
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.act = nn.GELU()

    def dwt2d(self, x):
        # Handle odd dimensions
        if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
            x = F.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2))
            
        ll = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2
        lh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 2
        hl = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 2
        hh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2
        return ll, lh, hl, hh

    def idwt2d(self, ll, lh, hl, hh):
        x = torch.zeros(ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2, device=ll.device)
        x[:, :, 0::2, 0::2] = (ll + lh + hl + hh) / 2
        x[:, :, 0::2, 1::2] = (ll - lh + hl - hh) / 2
        x[:, :, 1::2, 0::2] = (ll + lh - hl - hh) / 2
        x[:, :, 1::2, 1::2] = (ll - lh - hl + hh) / 2
        return x

    def forward(self, x):
        shortcut = x
        ll, lh, hl, hh = self.dwt2d(x)
        
        ll = self.ll_conv(ll)
        lh = self.lh_conv(lh)
        hl = self.hl_conv(hl)
        hh = self.hh_conv(hh)
        
        out = self.idwt2d(ll, lh, hl, hh)
        
        # Crop back if padded
        if out.shape != x.shape:
             out = out[:, :, :x.shape[2], :x.shape[3]]
             
        out = self.norm(out)
        out = self.act(out)
        return shortcut + out


# =============================================================================
# NHÓM 4: NEXT-GEN SEQUENCE MODELS
# =============================================================================

class RWKVBlock(nn.Module):
    """
    RWKV Block (Simplified for Vision / AFT Style)
    Replaced incorrect softmax logic with Global Linear Attention / AFT
    to ensure proper spatial mixing (O(N) complexity).
    """
    def __init__(self, dim, layer_id=0):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        
        # Time mixing params
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        mlp_hidden = dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C) # Flatten to sequence
        
        # 1. Token Shift (Spatial Mixing Lite)
        x_prev = F.pad(x, (0, 0, 1, -1)) # Shift sequence by 1
        
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        
        k = self.key(self.ln1(xk))
        v = self.value(self.ln1(xv))
        r = torch.sigmoid(self.receptance(self.ln1(xr)))
        
        # 2. WKV / Global Linear Attention (Corrected)
        # Original code did softmax(k, dim=1) which is just normalization, no mixing.
        # We use AFT-Simple style global context: 
        # Context = sum(exp(K) * V) / sum(exp(K))
        
        k_exp = torch.exp(k - torch.max(k, dim=1, keepdim=True)[0]) # Stable exp
        wkv = (k_exp * v).sum(dim=1, keepdim=True) / (k_exp.sum(dim=1, keepdim=True) + 1e-6)
        
        # Output is Receptance * Global Context
        # Note: True RWKV uses causal scan, but for images, Global/Bidirectional is better.
        out = r * self.output(wkv) # Broadcast global context
        
        x = x + out
        x = x + self.mlp(self.ln2(x))
        
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


# =============================================================================
# BLOCK FACTORY
# =============================================================================

def get_block(block_type: str, dim: int, **kwargs):
    """Factory function to get block by name"""
    
    # BasicBlock wrapper (simple conv residual)
    class BasicBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(dim)
            self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(dim)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + residual)
    
    blocks = {
        'basic': BasicBlock,
        'convnext': ConvNeXtBlock,
        'dcn': DeformableConvBlock,
        'inverted_residual': InvertedResidualBlock,
        'swin': SwinTransformerBlock,
        'fno': FourierNeuralOperatorBlock,
        'wavelet': WaveletBlock,
        'rwkv': RWKVBlock,
    }
    
    if block_type not in blocks:
        raise ValueError(f"Unknown block type: {block_type}. Available: {list(blocks.keys())}")
    
    return blocks[block_type](dim, **kwargs)


class BlockStack(nn.Module):
    """Stack of blocks for HRNet stages"""
    def __init__(self, block_type: str, dim: int, depth: int = 2, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            get_block(block_type, dim, **kwargs) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
