"""
PCShear-HRNet: Noise-Robust Frequency-Aware Boundary Representation
for HD95-Optimized 3D Medical Image Segmentation

Dual-branch architecture:
  - Spatial Branch:  HRNet multi-scale encoder (ResNet-34 / Spectral / Mamba blocks)
  - Spectral Branch: Phase Congruency + Shearlet Energy → Spectral Encoder
  - Fusion: Cross-Domain Attention (Q=spatial, K/V=spectral)
  - Heads:  Segmentation head + Boundary weight head

The spectral branch provides noise-robust boundary features (Phase Congruency)
and curvature-aware directional energy (Shearlet transform). These are fused
with spatial features via cross-attention so the network can attend to
clean boundary cues from the frequency domain.

Key properties:
  - Contrast-invariant edge detection (Phase Congruency)
  - Curvature-aware boundary supervision (Shearlet entropy)
  - Direct HD95 optimization via curvature-weighted boundary loss

Reference:
  PCShear_HRNet_Blueprint.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.phase_congruency import PhaseCongruencyModule
from layers.shearlet_energy import ShearletEnergyModule
from layers.spectral_encoder import SpectralEncoder
from layers.cross_domain_fusion import CrossDomainAttentionFusion, SEFusion


# =============================================================================
# NORMALIZATION HELPER
# =============================================================================

def get_norm(num_channels, num_groups=8):
    """GroupNorm for stability with small batch sizes."""
    if num_channels < num_groups:
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


# =============================================================================
# ENCODER BLOCKS
# =============================================================================

class BasicBlock(nn.Module):
    """ResNet-34 BasicBlock: 2 × (3×3 conv)"""
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class FourierSpatialMixer(nn.Module):
    """Global Spatial Mixing via FFT — Channel/Spatial decoupling."""

    def __init__(self, dim, modes=16):
        super().__init__()
        self.modes = modes
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
        out_ft[:, :, :m1, :m2] = x_ft[:, :, :m1, :m2] * w_curr
        return torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')


class SpectralDecoupledBlock(nn.Module):
    """Decoupled Spatial/Channel Mixing using FFT."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm1 = get_norm(out_channels)
        self.act1 = nn.GELU()
        self.spatial_mixer = FourierSpatialMixer(out_channels, modes=16)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.norm2 = get_norm(out_channels)
        self.pool = nn.AvgPool2d(2, 2) if stride > 1 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.pool(out)
        out = self.spatial_mixer(out) + out
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act1(out + identity)


# =============================================================================
# LIGHTWEIGHT DECODER BLOCK
# =============================================================================

class LightweightDecoderBlock(nn.Module):
    """Lightweight decoder block: 1×1 conv + upsample"""

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
            x = F.interpolate(x, size=target_size, mode='bilinear',
                              align_corners=True)
        elif self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor,
                              mode='bilinear', align_corners=True)
        return x


# =============================================================================
# MULTI-SCALE FUSION
# =============================================================================

class MultiScaleFusion(nn.Module):
    """Fuse features from multiple scales (HRNet-style)."""

    def __init__(self, channels_list):
        super().__init__()
        self.num_scales = len(channels_list)
        self.fusions = nn.ModuleList()
        for i, ch in enumerate(channels_list):
            fusion = nn.ModuleList()
            for j, ch_j in enumerate(channels_list):
                if i == j:
                    fusion.append(nn.Identity())
                elif j < i:
                    fusion.append(nn.Sequential(
                        nn.Conv2d(ch_j, ch, 1, bias=False),
                        get_norm(ch),
                    ))
                else:
                    fusion.append(nn.Sequential(
                        nn.Conv2d(ch_j, ch, 3, stride=2 ** (j - i),
                                  padding=1, bias=False),
                        get_norm(ch),
                    ))
            self.fusions.append(fusion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_list):
        out = []
        for i in range(self.num_scales):
            y = None
            for j in range(self.num_scales):
                feat = self.fusions[i][j](x_list[j])
                if j != i:
                    feat = F.interpolate(feat, size=x_list[i].shape[2:],
                                         mode='bilinear', align_corners=True)
                y = feat if y is None else y + feat
            out.append(self.relu(y))
        return out


# =============================================================================
# PCSHEAR-HRNET MAIN MODEL
# =============================================================================

class PCShearHRNet(nn.Module):
    """
    PCShear-HRNet: Noise-Robust Frequency-Aware Boundary Representation

    Architecture:
      - Spatial Branch:  HRNet encoder with multi-resolution streams
      - Spectral Branch: Phase Congruency + Shearlet Energy → lightweight encoder
      - Fusion:          Cross-domain attention (spatial queries spectral)
      - Seg Head:        Conv 1×1 → upsample → N_class logits
      - Boundary Head:   Produces curvature weight map w(x) for loss

    Args:
        in_channels: Number of input image channels (default: 3)
        num_classes: Number of segmentation classes (default: 4)
        base_channels: Base channel width C (default: 64)
        encoder_depths: Number of blocks per stage [s1, s2, s3, s4]
        block_type: 'basic' | 'spectral' (FFT) (default: 'basic')
        use_deep_supervision: Enable auxiliary losses (default: False)
        fusion_type: 'attention' | 'se' (default: 'attention')
        pc_learnable: Whether PC filter bank is learnable (default: False)
        n_pc_scales: Number of Phase Congruency scales (default: 5)
        n_pc_orientations: Number of PC orientations (default: 6)
        n_shearlet_scales: Number of Shearlet scales (default: 4)
        n_shearlet_orientations: Number of Shearlet orientations (default: 8)
    """

    def __init__(self, in_channels=3, num_classes=4, base_channels=64,
                 encoder_depths=(3, 4, 6, 3), block_type='basic',
                 use_deep_supervision=False, fusion_type='attention',
                 pc_learnable=False,
                 n_pc_scales=5, n_pc_orientations=6,
                 n_shearlet_scales=4, n_shearlet_orientations=8):
        super().__init__()

        self.num_classes = num_classes
        self.use_deep_supervision = use_deep_supervision
        self.block_type = block_type
        C = base_channels

        # =====================================================================
        # SPECTRAL BRANCH (Noise-Robust Boundary Features)
        # =====================================================================

        # Phase Congruency Module
        self.pc_module = PhaseCongruencyModule(
            n_scales=n_pc_scales,
            n_orientations=n_pc_orientations,
            learnable=pc_learnable
        )

        # Shearlet Directional Energy Module
        self.shearlet_module = ShearletEnergyModule(
            n_scales=n_shearlet_scales,
            n_orientations=n_shearlet_orientations
        )

        # Lightweight Spectral Encoder
        # Input: PC map (1 ch) + Shearlet (8 dir + 1 entropy = 9 ch) = 10 channels
        spectral_in_ch = 1 + n_shearlet_orientations + 1
        self.spectral_encoder = SpectralEncoder(
            in_channels=spectral_in_ch,
            out_channels=C,
            mid_channels=min(C, 64)
        )

        # =====================================================================
        # SPATIAL BRANCH (HRNet Multi-Scale Encoder)
        # =====================================================================

        # Stem (stride 4: 224 → 56)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 7, stride=2, padding=3, bias=False),
            get_norm(C),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Stage 1: Full resolution branch
        self.encoder1 = self._make_layer(C, C, encoder_depths[0])

        # Stage 2: Add 1/2 resolution branch
        self.transition1 = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1, bias=False),
            get_norm(C * 2), nn.ReLU(inplace=True)
        )
        self.encoder2_high = self._make_layer(C, C, encoder_depths[1])
        self.encoder2_low = self._make_layer(C * 2, C * 2, encoder_depths[1])
        self.fusion2 = MultiScaleFusion([C, C * 2])

        # Stage 3: Add 1/4 resolution branch
        self.transition2 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1, bias=False),
            get_norm(C * 4), nn.ReLU(inplace=True)
        )
        self.encoder3_high = self._make_layer(C, C, encoder_depths[2])
        self.encoder3_mid = self._make_layer(C * 2, C * 2, encoder_depths[2])
        self.encoder3_low = self._make_layer(C * 4, C * 4, encoder_depths[2])
        self.fusion3 = MultiScaleFusion([C, C * 2, C * 4])

        # Stage 4: Add 1/8 resolution branch
        self.transition3 = nn.Sequential(
            nn.Conv2d(C * 4, C * 8, 3, stride=2, padding=1, bias=False),
            get_norm(C * 8), nn.ReLU(inplace=True)
        )
        self.encoder4_high = self._make_layer(C, C, encoder_depths[3])
        self.encoder4_mid1 = self._make_layer(C * 2, C * 2, encoder_depths[3])
        self.encoder4_mid2 = self._make_layer(C * 4, C * 4, encoder_depths[3])
        self.encoder4_low = self._make_layer(C * 8, C * 8, encoder_depths[3])
        self.fusion4 = MultiScaleFusion([C, C * 2, C * 4, C * 8])

        # =====================================================================
        # DECODER (Lightweight — Asymmetric)
        # =====================================================================

        self.decoder_high = LightweightDecoderBlock(C, C * 2)
        self.decoder_mid1 = LightweightDecoderBlock(C * 2, C * 2)
        self.decoder_mid2 = LightweightDecoderBlock(C * 4, C * 2)
        self.decoder_low = LightweightDecoderBlock(C * 8, C * 2)

        # Final fusion (4 branches × C*2 = 8C → C)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(C * 2 * 4, C * 2, 3, padding=1, bias=False),
            get_norm(C * 2), nn.ReLU(inplace=True),
            nn.Conv2d(C * 2, C, 3, padding=1, bias=False),
            get_norm(C), nn.ReLU(inplace=True),
        )

        # =====================================================================
        # CROSS-DOMAIN FUSION
        # =====================================================================

        if fusion_type == 'attention':
            self.cross_fusion = CrossDomainAttentionFusion(
                dim=C, num_heads=8, window_size=8
            )
        else:
            self.cross_fusion = SEFusion(dim=C)

        # =====================================================================
        # HEADS
        # =====================================================================

        # Segmentation head
        self.seg_head = nn.Conv2d(C, num_classes, 1)

        # Boundary weight head: produces w(x) = 1 + α·E_curv + β·PC
        self.boundary_head = nn.Sequential(
            nn.Conv2d(C, C // 4, 3, padding=1, bias=False),
            get_norm(C // 4), nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, 1, 1),
            nn.Softplus(),  # Ensure positive weights
        )

        # Deep supervision heads
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(C * 2, num_classes, 1),
                nn.Conv2d(C * 4, num_classes, 1),
                nn.Conv2d(C * 8, num_classes, 1),
            ])

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Create stage with multiple blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                get_norm(out_channels)
            )

        if self.block_type == 'spectral':
            block_cls = SpectralDecoupledBlock
        else:
            block_cls = BasicBlock

        layers = [block_cls(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(block_cls(out_channels, out_channels))
        return nn.Sequential(*layers)

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
            x: Input image [B, in_channels, H, W]

        Returns:
            dict with keys:
              - 'output': Segmentation logits [B, num_classes, H, W]
              - 'pc_map': Phase Congruency map [B, 1, H, W]
              - 'boundary_weight': Curvature weight map [B, 1, H, W]
              - 'aux_outputs' (if deep supervision): List of aux logits
        """
        target_size = x.shape[2:]

        # =================================================================
        # SPECTRAL BRANCH
        # =================================================================
        # Phase Congruency (noise-robust edge detection)
        pc_map = self.pc_module(x)  # [B, 1, H, W]

        # Shearlet Energy (directional energy + curvature entropy)
        shearlet_out = self.shearlet_module(x)  # [B, 9, H, W]

        # Concatenate spectral features
        spectral_input = torch.cat([pc_map, shearlet_out], dim=1)  # [B, 10, H, W]

        # Encode spectral features
        f_spectral = self.spectral_encoder(spectral_input)  # [B, C, H, W]

        # =================================================================
        # SPATIAL BRANCH (HRNet Encoder)
        # =================================================================
        x_enc = self.stem(x)  # → [B, C, H/4, W/4]

        # Stage 1
        x1 = self.encoder1(x_enc)

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
        x1, x2_low, x3_low, x4_low = self.fusion4(
            [x1, x2_low, x3_low, x4_low]
        )

        # =================================================================
        # DECODER
        # =================================================================
        high_size = x1.shape[2:]

        d1 = self.decoder_high(x1, target_size=high_size)
        d2 = self.decoder_mid1(x2_low, target_size=high_size)
        d3 = self.decoder_mid2(x3_low, target_size=high_size)
        d4 = self.decoder_low(x4_low, target_size=high_size)

        features = torch.cat([d1, d2, d3, d4], dim=1)
        f_spatial = self.final_fusion(features)  # [B, C, H/4, W/4]

        # =================================================================
        # CROSS-DOMAIN FUSION
        # =================================================================
        # Downsample spectral features to match spatial resolution
        f_spectral_ds = F.interpolate(
            f_spectral, size=f_spatial.shape[2:],
            mode='bilinear', align_corners=True
        )

        # Attention fusion: spatial queries spectral about boundaries
        f_fused = self.cross_fusion(f_spatial, f_spectral_ds)  # [B, C, H/4, W/4]

        # =================================================================
        # HEADS
        # =================================================================
        # Segmentation
        logits = self.seg_head(f_fused)
        logits = F.interpolate(logits, size=target_size, mode='bilinear',
                               align_corners=True)

        # Boundary weight map
        boundary_weight = self.boundary_head(f_fused)
        boundary_weight = F.interpolate(boundary_weight, size=target_size,
                                        mode='bilinear', align_corners=True)

        result = {
            'output': logits,
            'pc_map': pc_map,
            'boundary_weight': boundary_weight,
        }

        # Deep supervision
        if self.use_deep_supervision and hasattr(self, 'aux_heads'):
            aux_outputs = []
            for feat, head in zip([x2_low, x3_low, x4_low], self.aux_heads):
                aux = head(feat)
                aux = F.interpolate(aux, size=target_size, mode='bilinear',
                                    align_corners=True)
                aux_outputs.append(aux)
            result['aux_outputs'] = aux_outputs

        return result


# =============================================================================
# MODEL FACTORY FUNCTIONS
# =============================================================================

def pcshear_hrnet_small(num_classes=4, in_channels=3, **kwargs):
    """Small: base_channels=32, ~12M params"""
    return PCShearHRNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=32,
        encoder_depths=(2, 3, 4, 2),
        block_type='basic',
        **kwargs
    )


def pcshear_hrnet_base(num_classes=4, in_channels=3, **kwargs):
    """Base: base_channels=64, ~35M params"""
    return PCShearHRNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        encoder_depths=(3, 4, 6, 3),
        block_type='basic',
        **kwargs
    )


def pcshear_hrnet_spectral(num_classes=4, in_channels=3, **kwargs):
    """Spectral Decoupled (FFT spatial mixing)"""
    return PCShearHRNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        encoder_depths=(3, 4, 6, 3),
        block_type='spectral',
        **kwargs
    )


if __name__ == '__main__':
    # Quick validation
    model = pcshear_hrnet_small(num_classes=4, in_channels=3)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out['output'].shape}")
    print(f"PC map: {out['pc_map'].shape}")
    print(f"Boundary weight: {out['boundary_weight'].shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
