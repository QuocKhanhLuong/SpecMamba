"""
HRNet with ResNet-34 Encoder - Asymmetric Encoder-Decoder
- Encoder: ResNet-34 style BasicBlocks (heavier)
- Decoder: Lightweight upsampling (lighter)
- Multi-resolution fusion like HRNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# NORMALIZATION HELPER (GroupNorm for small batch stability)
# =============================================================================

def get_norm(num_channels, num_groups=8):
    """GroupNorm for stability with small batch sizes."""
    if num_channels < num_groups:
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


# =============================================================================
# RESNET-34 BASIC BLOCK
# =============================================================================

class BasicBlock(nn.Module):
    """ResNet-34 BasicBlock: 2 x (3x3 conv)"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = get_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = get_norm(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


# =============================================================================
# DECOUPLED SPATIAL-CHANNEL BLOCKS (Spectral & Mamba)
# =============================================================================

class FourierSpatialMixer(nn.Module):
    """Global Spatial Mixing via FFT (Con đường 1)."""
    def __init__(self, dim, modes=16):
        super().__init__()
        self.modes = modes
        # Complex weights for spatial frequencies
        self.weights = nn.Parameter(torch.empty(dim, modes, modes, 2))
        scale = 1 / (dim * modes * modes)
        nn.init.normal_(self.weights, std=scale)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros_like(x_ft)
        
        m1 = min(self.modes, H // 2 + 1)
        m2 = min(self.modes, W // 2 + 1)
        
        weights = torch.view_as_complex(self.weights)
        w_curr = weights[:, :m1, :m2]
        
        # Multiply only the lower frequencies
        out_ft[:, :, :m1, :m2] = x_ft[:, :, :m1, :m2] * w_curr
        
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x


class SpectralDecoupledBlock(nn.Module):
    """
    Decouples Spatial and Channel Mixing using FFT.
    - 1x1 Conv (Channel Mix 1)
    - Fourier Spatial Mixer (Spatial Mix)
    - 1x1 Conv (Channel Mix 2)
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.stride = stride
        
        # Channel Mix 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm1 = get_norm(out_channels)
        self.act1 = nn.GELU()
        
        # Spatial Mix (FFT)
        self.spatial_mixer = FourierSpatialMixer(out_channels, modes=16)
        
        # Channel Mix 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.norm2 = get_norm(out_channels)
        
        # Handle stride if needed (FFT requires same dims, so we handle stride via downsample path and pooling before FFT)
        self.pool = nn.AvgPool2d(2, 2) if stride > 1 else nn.Identity()

    def forward(self, x):
        identity = x
        
        # Channel mix 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        # Pool if stride > 1
        out = self.pool(out)
        
        # Spatial Mix
        shortcut = out
        out = self.spatial_mixer(out) + shortcut
        
        # Channel mix 2
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act1(out)
        return out


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    def selective_scan_fn(u, delta, A, B_mat, C_mat, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        """Pure PyTorch fallback mock for shape testing when mamba_ssm is not installed."""
        return u.clone()

class SimpleMambaBlock(nn.Module):
    """
    Lite-Mamba Block: Spatial Mixing via Vanilla SSM with Expansion=1 and no Conv1D.
    Designed specifically for Medical Image Decoupling, avoiding NLP overhead.
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # 1. Linear projection to create x, delta, B, C completely without Expansion
        # Total added dims = dim*2 + d_state*2 (x, delta, B, C)
        self.proj = nn.Linear(dim, dim * 2 + 2 * d_state)
        
        # A matrix: (dim, d_state) - HiPPO or random init. Usually kept negative for stability.
        self.A_log = nn.Parameter(torch.log(torch.rand(dim, d_state) + 1e-4))
        
        # D matrix mapping (skip connection inside Mamba)
        self.D = nn.Parameter(torch.ones(dim))
        
        # DT Bias (delta bias)
        self.dt_proj_bias = nn.Parameter(torch.zeros(dim))

        # 2. Output Linear
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        
        # Sequence format: (B, L, C)
        seq = x.view(B, C, L).transpose(1, 2)
        
        # 1. Linear projection
        proj_out = self.proj(seq) # (B, L, dim*2 + 2*d_state)
        
        # Split into x_in, delta, B_mat, C_mat
        x_in, delta, B_mat, C_mat = torch.split(
            proj_out, 
            [self.dim, self.dim, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Mamba's core function expects inputs in shape (B, D, L) or (B, N, L)
        x_in = x_in.transpose(1, 2).contiguous() # (B, dim, L)
        delta = delta.transpose(1, 2).contiguous() # (B, dim, L)
        B_mat = B_mat.transpose(1, 2).contiguous() # (B, d_state, L)
        C_mat = C_mat.transpose(1, 2).contiguous() # (B, d_state, L)
        
        A = -torch.exp(self.A_log) # Keep A negative, shape (dim, d_state)
        
        # 2. Selective Scan
        if HAS_MAMBA:
            y = selective_scan_fn(
                x_in, delta, A, B_mat, C_mat,
                D=self.D, z=None, delta_bias=self.dt_proj_bias, delta_softplus=True
            ) # y: (B, dim, L)
        else:
            # Fallback mock for testing dimensions without Mamba installed
            import torch.nn.functional as F
            y = x_in * F.softplus(delta) # Just a dimension-preserving mock operation
        
        # 3. Output projection
        y = y.transpose(1, 2) # (B, L, dim)
        out = self.out_proj(y)
        
        # Back to image shape
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


class MambaDecoupledBlock(nn.Module):
    """
    Decouples Spatial and Channel Mixing using SSM.
    - 1x1 Conv (Channel Mix 1)
    - SimpleSSM (Spatial Mix)
    - 1x1 Conv (Channel Mix 2)
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.stride = stride
        
        # Channel Mix 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm1 = get_norm(out_channels)
        self.act1 = nn.GELU()
        
        # Spatial Mix (Lite-Mamba SSM)
        self.spatial_mixer = SimpleMambaBlock(out_channels)
        
        # Channel Mix 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.norm2 = get_norm(out_channels)
        
        self.pool = nn.AvgPool2d(2, 2) if stride > 1 else nn.Identity()

    def forward(self, x):
        identity = x
        
        # Channel mix 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        # Pool if stride > 1
        out = self.pool(out)
        
        # Spatial Mix
        shortcut = out
        out = self.spatial_mixer(out) + shortcut
        
        # Channel mix 2
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act1(out)
        return out


# =============================================================================
# LIGHTWEIGHT DECODER BLOCK
# =============================================================================

class LightweightDecoderBlock(nn.Module):
    """Lightweight decoder block: 1x1 conv + upsample"""
    
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            get_norm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.scale_factor = scale_factor
    
    def forward(self, x, target_size=None):
        x = self.conv(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        elif self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return x


# =============================================================================
# MULTI-SCALE FUSION MODULE
# =============================================================================

class MultiScaleFusion(nn.Module):
    """Fuse features from multiple scales."""
    
    def __init__(self, channels_list):
        super().__init__()
        self.num_scales = len(channels_list)
        
        # Cross-scale fusion
        self.fusions = nn.ModuleList()
        for i, ch in enumerate(channels_list):
            fusion = nn.ModuleList()
            for j, ch_j in enumerate(channels_list):
                if i == j:
                    fusion.append(nn.Identity())
                elif j < i:  # Upsample from lower resolution
                    fusion.append(nn.Sequential(
                        nn.Conv2d(ch_j, ch, 1, bias=False),
                        get_norm(ch),
                    ))
                else:  # Downsample from higher resolution
                    fusion.append(nn.Sequential(
                        nn.Conv2d(ch_j, ch, 3, stride=2**(j-i), padding=1, bias=False),
                        get_norm(ch),
                    ))
            self.fusions.append(fusion)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x_list):
        out = []
        for i in range(self.num_scales):
            y = None
            for j in range(self.num_scales):
                if j < i:  # Need to downsample
                    feat = self.fusions[i][j](x_list[j])
                    feat = F.interpolate(feat, size=x_list[i].shape[2:], mode='bilinear', align_corners=True)
                elif j > i:  # Need to upsample
                    feat = self.fusions[i][j](x_list[j])
                    feat = F.interpolate(feat, size=x_list[i].shape[2:], mode='bilinear', align_corners=True)
                else:
                    feat = x_list[j]
                
                if y is None:
                    y = feat
                else:
                    y = y + feat
            out.append(self.relu(y))
        return out


# =============================================================================
# HRNET-RESNET34 MAIN MODEL
# =============================================================================

class HRNetResNet34(nn.Module):
    """
    HRNet with ResNet-34 Encoder and Asymmetric Decoder
    
    Encoder (Heavy):
    - Stage 1: [3, 3, 3, 3] BasicBlocks like ResNet-34
    - Multi-resolution branches
    
    Decoder (Light):
    - Simple 1x1 conv + bilinear upsample
    - Feature aggregation
    """
    
    def __init__(self, in_channels=3, num_classes=4, base_channels=64, 
                 encoder_depths=[3, 4, 6, 3], use_deep_supervision=False,
                 full_resolution_mode=False, block_type='basic'):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_deep_supervision = use_deep_supervision
        self.full_resolution_mode = full_resolution_mode
        self.block_type = block_type
        
        C = base_channels  # 64 default (like ResNet-34)
        
        # =====================================================================
        # STEM 
        # =====================================================================
        if full_resolution_mode:
            # Full Resolution Stem (stride=1) - keeps 224x224 in stream 1
            print(">>> FULL RESOLUTION MODE: Stream 1 keeps 224x224. High VRAM!")
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, C//2, 3, stride=1, padding=1, bias=False),
                get_norm(C//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C//2, C, 3, stride=1, padding=1, bias=False),
                get_norm(C),
                nn.ReLU(inplace=True),
            )
        else:
            # Standard Stem (stride 4: 224 -> 56)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, C, 7, stride=2, padding=3, bias=False),
                get_norm(C),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
        
        # =====================================================================
        # ENCODER - ResNet-34 style (Asymmetric: Heavier)
        # =====================================================================
        
        # Stage 1: Full resolution branch (56x56)
        self.encoder1 = self._make_layer(C, C, encoder_depths[0])
        
        # Stage 2: Add 1/2 resolution branch (28x28)
        self.transition1 = nn.Sequential(
            nn.Conv2d(C, C*2, 3, stride=2, padding=1, bias=False),
            get_norm(C*2),
            nn.ReLU(inplace=True)
        )
        self.encoder2_high = self._make_layer(C, C, encoder_depths[1])
        self.encoder2_low = self._make_layer(C*2, C*2, encoder_depths[1])
        self.fusion2 = MultiScaleFusion([C, C*2])
        
        # Stage 3: Add 1/4 resolution branch (14x14)
        self.transition2 = nn.Sequential(
            nn.Conv2d(C*2, C*4, 3, stride=2, padding=1, bias=False),
            get_norm(C*4),
            nn.ReLU(inplace=True)
        )
        self.encoder3_high = self._make_layer(C, C, encoder_depths[2])
        self.encoder3_mid = self._make_layer(C*2, C*2, encoder_depths[2])
        self.encoder3_low = self._make_layer(C*4, C*4, encoder_depths[2])
        self.fusion3 = MultiScaleFusion([C, C*2, C*4])
        
        # Stage 4: Add 1/8 resolution branch (7x7)
        self.transition3 = nn.Sequential(
            nn.Conv2d(C*4, C*8, 3, stride=2, padding=1, bias=False),
            get_norm(C*8),
            nn.ReLU(inplace=True)
        )
        self.encoder4_high = self._make_layer(C, C, encoder_depths[3])
        self.encoder4_mid1 = self._make_layer(C*2, C*2, encoder_depths[3])
        self.encoder4_mid2 = self._make_layer(C*4, C*4, encoder_depths[3])
        self.encoder4_low = self._make_layer(C*8, C*8, encoder_depths[3])
        self.fusion4 = MultiScaleFusion([C, C*2, C*4, C*8])
        
        # =====================================================================
        # DECODER - Lightweight (Asymmetric: Lighter)
        # =====================================================================
        
        # Aggregate all scales to high resolution
        total_channels = C + C*2 + C*4 + C*8  # 15C
        
        self.decoder_low = LightweightDecoderBlock(C*8, C*2)
        self.decoder_mid2 = LightweightDecoderBlock(C*4, C*2)
        self.decoder_mid1 = LightweightDecoderBlock(C*2, C*2)
        self.decoder_high = LightweightDecoderBlock(C, C*2)
        
        # Final fusion (4 branches of C*2 each = 8C)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(C*2 * 4, C*2, 3, padding=1, bias=False),
            get_norm(C*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C*2, C, 3, padding=1, bias=False),
            get_norm(C),
            nn.ReLU(inplace=True),
        )
        
        # Segmentation head
        self.seg_head = nn.Conv2d(C, num_classes, 1)
        
        # Deep supervision heads
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(C*2, num_classes, 1),
                nn.Conv2d(C*4, num_classes, 1),
                nn.Conv2d(C*8, num_classes, 1),
            ])
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Create a stage with multiple Blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                get_norm(out_channels)
            )
        
        if self.block_type == 'basic':
            block_cls = BasicBlock
        elif self.block_type == 'spectral':
            block_cls = SpectralDecoupledBlock
        elif self.block_type == 'mamba':
            block_cls = MambaDecoupledBlock
        else:
            raise ValueError(f"Unknown block_type: {self.block_type}")

        layers = [block_cls(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(block_cls(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        target_size = x.shape[2:]
        
        # Stem
        x = self.stem(x)  # -> C, H/4, W/4
        
        # =====================================================================
        # ENCODER
        # =====================================================================
        
        # Stage 1
        x1 = self.encoder1(x)
        
        # Stage 2
        x2_low = self.transition1(x1)
        x1 = self.encoder2_high(x1)
        x2_low = self.encoder2_low(x2_low)
        x1, x2_low = self.fusion2([x1, x2_low])
        
        # Stage 3
        x3_low = self.transition2(x2_low)
        x1 = self.encoder3_high(x1)
        x2_low = self.encoder3_mid(x2_low)
        x3_low = self.encoder3_low(x3_low)
        x1, x2_low, x3_low = self.fusion3([x1, x2_low, x3_low])
        
        # Stage 4
        x4_low = self.transition3(x3_low)
        x1 = self.encoder4_high(x1)
        x2_low = self.encoder4_mid1(x2_low)
        x3_low = self.encoder4_mid2(x3_low)
        x4_low = self.encoder4_low(x4_low)
        x1, x2_low, x3_low, x4_low = self.fusion4([x1, x2_low, x3_low, x4_low])
        
        # =====================================================================
        # DECODER - Aggregate to high resolution
        # =====================================================================
        
        # Use x1 (highest resolution encoder output) as target size
        high_size = x1.shape[2:]
        
        # All decoders output to the same high resolution
        d1 = self.decoder_high(x1, target_size=high_size)  # Already at high_size, but ensure consistency
        d2 = self.decoder_mid1(x2_low, target_size=high_size)
        d3 = self.decoder_mid2(x3_low, target_size=high_size)
        d4 = self.decoder_low(x4_low, target_size=high_size)
        
        # Concatenate and fuse
        features = torch.cat([d1, d2, d3, d4], dim=1)
        features = self.final_fusion(features)
        
        # Segmentation
        logits = self.seg_head(features)
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=True)
        
        result = {'output': logits}
        
        # Deep supervision
        if self.use_deep_supervision and hasattr(self, 'aux_heads'):
            aux_outputs = []
            for i, (feat, head) in enumerate(zip([x2_low, x3_low, x4_low], self.aux_heads)):
                aux = head(feat)
                aux = F.interpolate(aux, size=target_size, mode='bilinear', align_corners=True)
                aux_outputs.append(aux)
            result['aux_outputs'] = aux_outputs
        
        return result


# =============================================================================
# MODEL FACTORY FUNCTIONS
# =============================================================================

def hrnet_resnet34_small(num_classes=4, in_channels=3):
    """Small: base_channels=32, ~8M params"""
    return HRNetResNet34(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=32,
        encoder_depths=[2, 3, 4, 2]
    )


def hrnet_resnet34_base(num_classes=4, in_channels=3):
    """Base: base_channels=64, ~30M params (like ResNet-34)"""
    return HRNetResNet34(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        encoder_depths=[3, 4, 6, 3]
    )


def hrnet_resnet34_large(num_classes=4, in_channels=3):
    """Large: base_channels=64, deeper, ~50M params"""
    return HRNetResNet34(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        encoder_depths=[3, 4, 23, 3]  # Like ResNet-101 depth
    )


def hrnet_spectral_decoupled(num_classes=4, in_channels=3):
    """Spectral Decoupled (Fast Fourier Transform + Linear)"""
    return HRNetResNet34(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        encoder_depths=[3, 4, 6, 3],
        block_type='spectral'
    )


def hrnet_mamba_decoupled(num_classes=4, in_channels=3):
    """Mamba Decoupled (SSM Scan + Linear)"""
    return HRNetResNet34(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        encoder_depths=[3, 4, 6, 3],
        block_type='mamba'
    )


if __name__ == '__main__':
    # Test
    model = hrnet_resnet34_base(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out['output'].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
