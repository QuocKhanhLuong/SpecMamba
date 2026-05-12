"""
SpecMamba model family — legacy tri-stream baseline and current 2.5D model.

`SpecMambaNet` is the older 3-stream baseline. The current paper-facing model is
`AsymSpecMambaDCN`, an asymmetric 2.5D dual-branch architecture with full-res
precision features, low-res Fourier context, ABX exchange, and SDF-gated
frequency-split fusion.

The scan blocks are Mamba-inspired approximations built from linear layers and
depthwise Conv1d, not true selective SSM/Mamba kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PRIOR KNOWLEDGE INPUT
# =============================================================================

class PriorKnowledgeConstructor(nn.Module):
    """3-channel: Raw + Sobel Edge + Local Variance."""
    def __init__(self, pool_size=5):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sx.view(1,1,3,3))
        self.register_buffer('sobel_y', sy.view(1,1,3,3))
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2)

    def forward(self, x):
        gray = x[:, :1]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx**2 + gy**2 + 1e-8)
        edge = edge / (edge.amax(dim=(-2,-1), keepdim=True) + 1e-8)
        var = (self.avg_pool(gray**2) - self.avg_pool(gray)**2).clamp(min=0)
        var = var / (var.amax(dim=(-2,-1), keepdim=True) + 1e-8)
        return torch.cat([gray, edge, var], dim=1)


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class DCNv3Block(nn.Module):
    """Deformable-convolution block used by the legacy tri-stream baseline.
    
    Inverted Bottleneck: Expand -> deformable conv with offsets/mask -> shrink.
    Supports HDC dilation for multi-scale receptive fields.
    """
    def __init__(self, dim, expansion=4, kernel_size=3, num_groups=4, dilation=1):
        super().__init__()
        from torchvision.ops import deform_conv2d
        self.deform_conv2d = deform_conv2d
        
        mid = dim * expansion
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.dilation = dilation
        pad = dilation * (kernel_size // 2)  # same padding with dilation
        
        # Inverted bottleneck: expand → deform → shrink
        self.pw_expand = nn.Conv2d(dim, mid, 1, bias=False)
        
        # DCNv3: offset (2*K*K per group) + mask (K*K per group)
        # offset_mask conv also uses same dilation for consistent RF
        kk = kernel_size * kernel_size
        om_pad = dilation * (kernel_size // 2)
        self.offset_mask = nn.Conv2d(mid, num_groups * 3 * kk, kernel_size,
                                     padding=om_pad, dilation=dilation, bias=True)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        
        # Grouped deformable conv weight
        self.dcn_weight = nn.Parameter(torch.randn(mid, mid // num_groups, kernel_size, kernel_size) * 0.02)
        self.dcn_pad = pad
        
        self.pw_shrink = nn.Conv2d(mid, dim, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        h = self.act(self.pw_expand(x))
        
        # Predict offsets + modulation masks
        kk = self.kernel_size * self.kernel_size
        om = self.offset_mask(h)
        offset = om[:, :self.num_groups * 2 * kk]
        mask = torch.sigmoid(om[:, self.num_groups * 2 * kk:])
        
        # Deformable conv with dilation
        h = self.deform_conv2d(h, offset, self.dcn_weight,
                               padding=self.dcn_pad, dilation=self.dilation, mask=mask)
        
        return self.act(self.norm(self.pw_shrink(self.act(h))))


class AdaptiveFourierMixer(nn.Module):
    """FFT → learned mode weights → Linear channel mix → IFFT."""
    def __init__(self, dim, num_modes=32):
        super().__init__()
        self.num_modes = num_modes
        self.mode_weight = nn.Parameter(torch.stack([
            torch.ones(dim, num_modes, num_modes),
            torch.zeros(dim, num_modes, num_modes),
        ], dim=-1))
        self.channel_mix = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.amp.autocast('cuda', enabled=False):
            xf = x.float()
            xq = torch.fft.rfft2(xf, norm='ortho')
            mh, mw = min(self.num_modes, H), min(self.num_modes, W//2+1)
            w = torch.view_as_complex(self.mode_weight[:, :mh, :mw].contiguous())
            xw = xq.clone()
            xw[:, :, :mh, :mw] = xq[:, :, :mh, :mw] * w.unsqueeze(0)
            r, i = self.channel_mix(xw.real), self.channel_mix(xw.imag)
            out = torch.fft.irfft2(torch.complex(r, i), s=(H, W), norm='ortho')
        return self.act(self.norm(out))


class CrossScanGatedMixer(nn.Module):
    """Cross-scan with configurable scan passes + gating.
    
    num_passes=1: forward-only (H scan)
    num_passes=2: bidirectional (H↕)
    num_passes=4: full cross-scan (H↕ + W↔, bidirectional)
    """
    def __init__(self, dim, kernel_size=3, num_passes=4):
        super().__init__()
        self.num_passes = num_passes
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_gate = nn.Linear(dim, dim, bias=False)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size-1,
                                 groups=dim, bias=False)
        self.linear_out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()

    def _scan_fwd(self, seq):
        """Single direction scan."""
        L = seq.shape[1]
        h = self.act(self.linear_in(seq)).transpose(1, 2)
        return self.dw_conv(h)[..., :L].transpose(1, 2)

    def _scan_bidi(self, seq):
        """Bidirectional scan."""
        L = seq.shape[1]
        h = self.act(self.linear_in(seq)).transpose(1, 2)
        return ((self.dw_conv(h)[..., :L] +
                 self.dw_conv(h.flip(-1))[..., :L].flip(-1)) * 0.5).transpose(1, 2)

    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.num_passes >= 4:
            # Full cross-scan: H↕ + W↔ bidirectional
            hh = self._scan_bidi(x.permute(0,3,2,1).reshape(B*W,H,C)).reshape(B,W,H,C).permute(0,3,2,1)
            hw = self._scan_bidi(x.permute(0,2,3,1).reshape(B*H,W,C)).reshape(B,H,W,C).permute(0,3,1,2)
            h = (hh + hw) * 0.5
        elif self.num_passes >= 2:
            # Bidirectional H-scan only
            hh = self._scan_bidi(x.permute(0,3,2,1).reshape(B*W,H,C)).reshape(B,W,H,C).permute(0,3,2,1)
            h = hh
        else:
            # Forward H-scan only
            hh = self._scan_fwd(x.permute(0,3,2,1).reshape(B*W,H,C)).reshape(B,W,H,C).permute(0,3,2,1)
            h = hh
        
        g = torch.sigmoid(self.linear_gate(x.permute(0,2,3,1))).permute(0,3,1,2)
        return self.act(self.norm(self.linear_out((h*g).permute(0,2,3,1)).permute(0,3,1,2)))


class ResidualBlock(nn.Module):
    """Pre-norm residual: x + Block(Norm(x))"""
    def __init__(self, block, dim):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.block = block
    def forward(self, x):
        return x + self.block(self.norm(x))


# =============================================================================
# ASYMMETRIC SKIP ATTENTION (stream-specific denoising)
# =============================================================================

class FRSkipAttention(nn.Module):
    """FR: Spatial + Channel attention (CBAM-lite) for MRI artifact removal."""
    def __init__(self, dim):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1, groups=dim // 4, bias=False),
            nn.Conv2d(dim // 4, 1, 1), nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1), nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.spatial(x) * self.channel(x)


class HRSkipAttention(nn.Module):
    """HR: Frequency energy weighting + spatial gate for FFT ringing suppression."""
    def __init__(self, dim):
        super().__init__()
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, 1, 7, padding=3), nn.Sigmoid(),
        )
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            xf = torch.fft.rfft2(x.float(), norm='ortho')
        energy = xf.abs().mean(dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
        energy = energy / (energy.max(dim=1, keepdim=True)[0] + 1e-6)
        return x * (self.freq_gate(x) * energy) * self.spatial_gate(x)


class LRSkipAttention(nn.Module):
    """LR: Uncertainty-guided attention for scan-mixer ambiguity suppression."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim // 2, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 4, 3, padding=1, bias=False),
            nn.GELU(), nn.Conv2d(dim // 4, 1, 1), nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1), nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        h = self.proj(x)
        conf = torch.exp(-x.std(dim=1, keepdim=True))  # low std = confident
        return x * (self.gate(h) * conf) * self.channel(x)


# =============================================================================
# TRI-FUSE LAYER (All-to-All Cross-Fuse)
# =============================================================================

class TriFuseLayer(nn.Module):
    """Asymmetric 3-stream cross-fuse: 5 paths (LR→FR dropped).

    FR→HR: Conv3×3↓2          HR→FR: Conv1×1 + ↑2
    FR→LR: Conv3×3↓2 ×2       LR→HR: Conv1×1 + ↑2
    HR→LR: Conv3×3↓2          (LR→FR: dropped — FR does local only)
    """
    def __init__(self, fr_ch, hr_ch, lr_ch):
        super().__init__()
        def down(ci, co, s=2):
            return nn.Sequential(nn.Conv2d(ci,co,3,stride=s,padding=1,bias=False),
                                 nn.GroupNorm(min(8,co),co))
        def proj(ci, co):
            return nn.Sequential(nn.Conv2d(ci,co,1,bias=False),
                                 nn.GroupNorm(min(8,co),co))
        # Downsample paths
        self.fr_to_hr = down(fr_ch, hr_ch)
        self.fr_to_lr = nn.Sequential(down(fr_ch, hr_ch), nn.ReLU(True),
                                       down(hr_ch, lr_ch))
        self.hr_to_lr = down(hr_ch, lr_ch)
        # Upsample paths (project-first)
        self.hr_to_fr = proj(hr_ch, fr_ch)
        self.lr_to_hr = proj(lr_ch, hr_ch)
        # LR→FR: DROPPED (FR does local edge only, no semantic injection)
        self.act = nn.ReLU(inplace=True)

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)

    def forward(self, fr, hr, lr):
        fr_new = self.act(fr + self._up(self.hr_to_fr(hr), fr))  # only HR→FR, no LR→FR
        hr_new = self.act(hr + self.fr_to_hr(fr) + self._up(self.lr_to_hr(lr), hr))
        lr_new = self.act(lr + self.fr_to_lr(fr) + self.hr_to_lr(hr))
        return fr_new, hr_new, lr_new


# =============================================================================
# TRI-STREAM FUSION (Final)
# =============================================================================

class TriStreamFusion(nn.Module):
    """FR edges gate HR+LR with SEPARATE gates (frequency-specific gating)."""
    def __init__(self, fr_ch, hr_ch, lr_ch, out_ch):
        super().__init__()
        self.hr_proj = nn.Conv2d(hr_ch, fr_ch, 1, bias=False)
        self.lr_proj = nn.Conv2d(lr_ch, fr_ch, 1, bias=False)
        self.gate_hr = nn.Conv2d(fr_ch, fr_ch, 1, bias=False)  # separate gate for HR
        self.gate_lr = nn.Conv2d(fr_ch, fr_ch, 1, bias=False)  # separate gate for LR
        self.fuse = nn.Sequential(
            nn.Conv2d(fr_ch * 3, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
        )

    def forward(self, fr, hr, lr):
        hr_up = F.interpolate(self.hr_proj(hr), size=fr.shape[2:],
                               mode='bilinear', align_corners=True)
        lr_up = F.interpolate(self.lr_proj(lr), size=fr.shape[2:],
                               mode='bilinear', align_corners=True)
        g_hr = torch.sigmoid(self.gate_hr(fr))  # edge gate for spectral stream
        g_lr = torch.sigmoid(self.gate_lr(fr))  # edge gate for sequential stream
        return self.fuse(torch.cat([fr, hr_up * g_hr, lr_up * g_lr], dim=1))


# =============================================================================
# 3-STREAM ASYMMETRIC SPEC-HRNET
# =============================================================================

class SpecMambaNet(nn.Module):
    """3-Stream Frequency-Guided HRNet.

    FR (224², C):   Deformable-conv block — adaptive edge refinement
    HR (112², C):   AdaptiveFourierMixer — global spectral mixing
    LR (56², 2C):   CrossScanGatedMixer — sequential context
    TriFuseLayer per stage, TriStreamFusion at end.
    """

    def __init__(self, in_channels=3, num_classes=4, base_channels=48,
                 img_size=224, deep_supervision=False, num_modes=32,
                 blocks_per_stage=2, num_stages=3):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_stages = num_stages
        C = base_channels

        # Asymmetric depth: [2, 4, 6] blocks per stage
        stage_depths = [blocks_per_stage * (i + 1) for i in range(num_stages)]
        
        # HDC dilation pyramid per FR stage
        # Stage 1: [1, 2],  Stage 2: [1, 2, 4, 8],  Stage 3: [1, 2, 4, 8, 16, 32]
        hdc_dilations = []
        for d in stage_depths:
            dils = [2**i for i in range(d)]  # [1, 2, 4, 8, ...]
            hdc_dilations.append(dils)
        
        # Mode pyramid per HR stage
        # Stage 1: modes=H/8, Stage 2: modes=H/4, Stage 3: modes=H/2
        hr_size = img_size // 2  # 112
        mode_pyramid = [max(4, hr_size // (2 ** (num_stages - i))) for i in range(num_stages)]
        
        # Scan depth pyramid per LR stage
        # Stage 1: 1-pass, Stage 2: 2-pass, Stage 3: 4-pass
        scan_pyramid = [min(4, 2 ** i) for i in range(num_stages)]

        self.prior = PriorKnowledgeConstructor()

        # Stem: full resolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, C), C), nn.GELU(),
        )

        # Stream init: split right after stem
        self.fr_init = nn.Identity()
        self.hr_init = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, C), C), nn.GELU(),
        )
        self.lr_init = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, C), C), nn.GELU(),
            nn.Conv2d(C, C*2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, C*2), C*2), nn.GELU(),
        )

        # Per-stage blocks + TriFuseLayer (asymmetric depth)
        self.fr_stages = nn.ModuleList()
        self.hr_stages = nn.ModuleList()
        self.lr_stages = nn.ModuleList()
        self.tri_fuse = nn.ModuleList()

        for s in range(num_stages):
            depth = stage_depths[s]
            dils = hdc_dilations[s]
            modes = mode_pyramid[s]
            passes = scan_pyramid[s]
            
            # FR: deformable conv with dilation pyramid
            self.fr_stages.append(nn.Sequential(*[
                ResidualBlock(DCNv3Block(C, dilation=dils[i]), C)
                for i in range(depth)]))
            
            # HR: SpectralBlock with mode pyramid
            self.hr_stages.append(nn.Sequential(*[
                ResidualBlock(AdaptiveFourierMixer(C, modes), C)
                for _ in range(depth)]))
            
            # LR: Mamba-inspired scan mixer with scan-depth pyramid
            self.lr_stages.append(nn.Sequential(*[
                ResidualBlock(CrossScanGatedMixer(C*2, num_passes=passes), C*2)
                for _ in range(depth)]))
            
            self.tri_fuse.append(TriFuseLayer(C, C, C*2))

        # Asymmetric Skip Attention (stream-specific denoising)
        self.skip_fr = FRSkipAttention(C)
        self.skip_hr = HRSkipAttention(C)
        self.skip_lr = LRSkipAttention(C*2)

        # Final Fusion
        self.final_fusion = TriStreamFusion(C, C, C*2, C)

        # Heads
        self.seg_head = nn.Conv2d(C, num_classes, 1)
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(C, num_classes, 1) for _ in range(num_stages - 1)])

    def forward(self, x):
        target = x.shape[2:]
        x = self.prior(x)
        x = self.stem(x)

        fr = self.fr_init(x)
        hr = self.hr_init(x)
        lr = self.lr_init(x)

        aux_feats = []
        for s in range(self.num_stages):
            fr = self.fr_stages[s](fr)
            hr = self.hr_stages[s](hr)
            lr = self.lr_stages[s](lr)
            fr, hr, lr = self.tri_fuse[s](fr, hr, lr)
            if self.deep_supervision and s < self.num_stages - 1:
                aux_feats.append(hr)  # Issue 2: aux from HR_fused (has context), not FR

        fused = self.final_fusion(self.skip_fr(fr), self.skip_hr(hr), self.skip_lr(lr))

        logits = self.seg_head(fused)
        if logits.shape[2:] != target:
            logits = F.interpolate(logits, target, mode='bilinear', align_corners=True)

        result = {'output': logits}
        if self.deep_supervision and self.training and aux_feats:
            result['aux_outputs'] = [
                F.interpolate(h(f), target, mode='bilinear', align_corners=True)
                if f.shape[2:] != target else h(f)
                for h, f in zip(self.aux_heads, aux_feats)]
        return result


# Backward compat
SpectralBlock = AdaptiveFourierMixer
PseudoMambaBlock = CrossScanGatedMixer
SpecMambaBlock = None

def specmamba_small(num_classes=4, in_channels=3, deep_supervision=False):
    return SpecMambaNet(in_channels, num_classes, 32, deep_supervision=deep_supervision)
def specmamba_base(num_classes=4, in_channels=3, deep_supervision=False):
    return SpecMambaNet(in_channels, num_classes, 48, deep_supervision=deep_supervision)
def specmamba_large(num_classes=4, in_channels=3, deep_supervision=False):
    return SpecMambaNet(in_channels, num_classes, 64, deep_supervision=deep_supervision)


# ═════════════════════════════════════════════════════════════════════════════
#
#  ASYMMETRIC SPEC-MAMBA DCN  (Asymmetric 2.5D Architecture)
#  ──────────────────────────────────────────────────────────
#  Input:  (B, 5, H, W) — 5 consecutive slices; predict center-slice mask.
#
#  Branch A — Precision Stem:
#       x[:, 1:4] (k-1, k, k+1)  →  12× SG-DeformConv @ 224²  →  feat_center
#
#  Branch B — Global Context:
#       x (all 5 slices)  →  progressive downsampling → 3× FFT mixer @ 56²
#
#  Branch C — Frequency-Split Fusion:
#       feat_z_ctx → PixelShuffle(×4) → ctx_up
#       feat_center → FFT → [Low-Freq + gate·ctx_up] + [High-Freq]  →  feat_fused
#
#  Branch D — Dual Heads:
#       SDF Head:  feat_center → (B, 3, H, W) per-class SDF
#       Seg Head:  feat_fused  → (B, 4, H, W) logits
#
# ═════════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
#  Core Block: Spectral Guidance (for DCN offset generation)
# ---------------------------------------------------------------------------

class SpectralGuidanceBlock(nn.Module):
    """rfft2 → learned Conv1×1 on real/imag → irfft2(s=(H,W)) → GN + GELU.

    Pure Conv2d (no nn.Linear) in frequency domain.  Always passes s=(H,W)
    to irfft2 to prevent 1-pixel spatial mismatches.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv_real = nn.Conv2d(dim, dim, 1)
        self.conv_imag = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.amp.autocast('cuda', enabled=False):
            xf = torch.fft.rfft2(x.float(), norm='ortho')
            real = self.conv_real(xf.real)
            imag = self.conv_imag(xf.imag)
            out = torch.fft.irfft2(torch.complex(real, imag), s=(H, W), norm='ortho')
        return self.act(self.norm(out))


# ---------------------------------------------------------------------------
#  Core Block: Spectral-Guided Deformable Convolution
# ---------------------------------------------------------------------------

class SGDCNv4Block(nn.Module):
    """Spectral-guided deformable conv + post-DCN FFN.

    1) SpectralGuidanceBlock produces spectral features from input.
    2) concat(x_norm, x_spec) → Linear → offsets + modulation masks.
    3) deform_conv2d at `dim` channels (memory-safe at full resolution).
    4) Post-DCN FFN (inverted bottleneck: dim → dim*expansion → dim).
    Two residual connections per block.

    The class name is retained for checkpoint/code compatibility; paper text
    should describe this as spectral-guided deformable convolution rather than
    a verified DCNv4 operator.
    """

    def __init__(self, dim, dilation=1, kernel_size=3, num_groups=4, expansion=4):
        super().__init__()
        self.dim = dim
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        kk = kernel_size * kernel_size
        pad = dilation * (kernel_size // 2)

        self.spec_norm = nn.GroupNorm(min(8, dim), dim)
        self.spectral = SpectralGuidanceBlock(dim)
        self.norm = nn.GroupNorm(min(8, dim), dim)

        offset_dim = num_groups * 3 * kk
        self.offset_proj = nn.Linear(2 * dim, offset_dim)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)

        self.dcn_weight = nn.Parameter(
            torch.randn(dim, dim // num_groups, kernel_size, kernel_size) * 0.02
        )
        self.dcn_pad = pad
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.dcn_norm = nn.GroupNorm(min(8, dim), dim)

        mid = dim * expansion
        self.ffn = nn.Sequential(
            nn.GroupNorm(min(8, dim), dim),
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, dim, 1, bias=False),
        )

    def forward(self, x):
        from torchvision.ops import deform_conv2d

        B, C, H, W = x.shape
        kk = self.kernel_size * self.kernel_size

        x_norm = self.norm(x)
        x_spec = self.spectral(self.spec_norm(x))

        # Offset generation: spatial (x_norm) + spectral (x_spec) guidance
        x_flat = x_norm.permute(0, 2, 3, 1).reshape(B * H * W, C)
        s_flat = x_spec.permute(0, 2, 3, 1).reshape(B * H * W, C)
        om = self.offset_proj(torch.cat([x_flat, s_flat], dim=-1))
        om = om.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        offset = om[:, : self.num_groups * 2 * kk]
        mask = torch.sigmoid(om[:, self.num_groups * 2 * kk :])

        x_dcn = deform_conv2d(
            x_norm, offset, self.dcn_weight,
            padding=self.dcn_pad, dilation=self.dilation, mask=mask,
        )

        # Residual 1: DCN
        x = x + self.dcn_norm(self.proj(x_dcn))
        # Residual 2: FFN
        x = x + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
#  Branch A: Precision Stem  (12 spectral-guided deformable blocks)
# ---------------------------------------------------------------------------

class PrecisionStem(nn.Module):
    """Isotropic full-resolution stem for boundary-precise features.

    Input:  x_local (B, 3, H, W) — center 3 slices [k-1, k, k+1].
    Stages: [2, 4, 6] spectral-guided deformable blocks with dilation pyramids,
            each followed by a CrossScanGatedMixer.
    Output: feat_center (B, C, H, W) at full resolution.
    """

    def __init__(self, in_channels=3, base_channels=48):
        super().__init__()
        C = base_channels
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, C), C),
            nn.GELU(),
        )
        # Stage 1: 2 deformable blocks [d=1,2] + scan mixer
        self.stage1_dcn = nn.ModuleList([
            SGDCNv4Block(C, dilation=1), SGDCNv4Block(C, dilation=2),
        ])
        self.stage1_mamba = CrossScanGatedMixer(C, num_passes=1)

        # Stage 2: 4 deformable blocks [d=1,2,4,8] + scan mixer
        self.stage2_dcn = nn.ModuleList([
            SGDCNv4Block(C, dilation=d) for d in [1, 2, 4, 8]
        ])
        self.stage2_mamba = CrossScanGatedMixer(C, num_passes=2)

        # Stage 3: 6 deformable blocks [d=1,2,4,8,16,32] + cross-scan mixer
        self.stage3_dcn = nn.ModuleList([
            SGDCNv4Block(C, dilation=d) for d in [1, 2, 4, 8, 16, 32]
        ])
        self.stage3_mamba = CrossScanGatedMixer(C, num_passes=4)

    def stem_init(self, x):
        """Initial projection only — used by the main model for staged execution."""
        return self.stem_conv(x)

    def forward_stage(self, x, stage_idx):
        """Run a single deformable-conv stage plus its local scan mixer."""
        dcn_blocks = [self.stage1_dcn, self.stage2_dcn, self.stage3_dcn][stage_idx]
        mamba_blk = [self.stage1_mamba, self.stage2_mamba, self.stage3_mamba][stage_idx]
        for blk in dcn_blocks:
            x = blk(x)
        x = x + mamba_blk(x)
        return x

    def forward(self, x, return_intermediates=False):
        x = self.stem_init(x)

        x = self.forward_stage(x, 0)
        s1 = x
        x = self.forward_stage(x, 1)
        s2 = x
        x = self.forward_stage(x, 2)

        if return_intermediates:
            return x, [s1, s2]
        return x


# ---------------------------------------------------------------------------
#  Branch B: Global 2D FFT Mixer  (replaces Conv1d scan approximation)
# ---------------------------------------------------------------------------

class Global2DFFTMixer(nn.Module):
    """Learned 2D spectral filter with full spatial receptive field.

    Every channel gets a complex-valued weight map covering the entire
    frequency plane.  Gating prevents the filter from hallucinating in
    featureless regions.  Pure PyTorch, no custom CUDA.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_real = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_imag = nn.Conv2d(dim, dim, 1, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.amp.autocast('cuda', enabled=False):
            xf = torch.fft.rfft2(x.float(), norm='ortho')
            real = self.conv_real(xf.real)
            imag = self.conv_imag(xf.imag)
            y = torch.fft.irfft2(torch.complex(real, imag),
                                 s=(H, W), norm='ortho')
        y = y.to(x.dtype)
        g = torch.sigmoid(
            self.gate_proj(x.permute(0, 2, 3, 1))
        ).permute(0, 3, 1, 2)
        out = self.out_proj((y * g).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(self.norm(out))


# ---------------------------------------------------------------------------
#  Asymmetric Bidirectional Exchange  (ABX)
# ---------------------------------------------------------------------------

class AsymBidirectionalExchange(nn.Module):
    """Per-stage cross-stream information exchange.

    Direction 1 — Context → Precision  (channel-wise reweighting):
        GAP(ctx) → MLP → scale   ;   prec = prec * (1 + scale)
        No spatial info injected → zero boundary blur.

    Direction 2 — Precision → Context  (spatial residual injection):
        prec_down = AvgPool(prec)  ;  delta = prec_down - ctx
        ctx = ctx + gate · Conv(delta)
        Difference signal forces context to learn *complementary* info.
    """

    def __init__(self, dim, ctx_hw=56, prec_hw=224):
        super().__init__()
        self.pool_factor = prec_hw // ctx_hw

        # Context → Precision: channel-wise scale (like SE from context)
        self.ctx_to_prec = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dim, dim // 4, bias=False),
            nn.GELU(),
            nn.Linear(dim // 4, dim, bias=False),
        )

        # Precision → Context: spatial residual injection
        self.delta_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dim), dim),
            nn.GELU(),
        )
        self.delta_gate = nn.Sequential(
            nn.Conv2d(dim, 1, 1), nn.Sigmoid(),
        )

    def forward(self, prec_feat, ctx_feat):
        """
        Args:
            prec_feat: (B, C, H_p, W_p)  — precision branch feature (224²)
            ctx_feat:  (B, C, H_c, W_c)  — context branch feature   (56²)
        Returns:
            prec_out, ctx_out  — updated features for both branches
        """
        # Direction 1: Context → Precision (channel reweight)
        scale = self.ctx_to_prec(ctx_feat).unsqueeze(-1).unsqueeze(-1)
        prec_out = prec_feat * (1.0 + scale.tanh())

        # Direction 2: Precision → Context (spatial residual)
        pf = F.avg_pool2d(prec_feat, self.pool_factor)
        delta = pf - ctx_feat
        g = self.delta_gate(ctx_feat)
        ctx_out = ctx_feat + g * self.delta_conv(delta)

        return prec_out, ctx_out


# ---------------------------------------------------------------------------
#  Branch B: Global Context Encoder  v2
# ---------------------------------------------------------------------------

class GlobalContextEncoder(nn.Module):
    """All-5-slice context branch at ¼ resolution (56²) with global RF.

    v2 changes over v1:
      - Progressive 2-step downsampling (224→112→56) instead of 1-step stride-4
      - Global2DFFTMixer blocks (full spatial RF) instead of Conv1d scan approximation
      - 3 stages matching PrecisionStem for per-stage ABX exchange

    Input:   x (B, 5, H, W) — full 2.5D stack.
    Output:  list of 3 stage features, each (B, C, H/4, W/4).
    """

    def __init__(self, dim, in_slices=5, num_stages=3):
        super().__init__()
        mid = max(dim // 2, 16)
        self.down = nn.Sequential(
            nn.Conv2d(in_slices, mid, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, mid), mid),
            nn.GELU(),
            nn.Conv2d(mid, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, dim), dim),
            nn.GELU(),
        )
        self.stages = nn.ModuleList([
            Global2DFFTMixer(dim) for _ in range(num_stages)
        ])

    def forward_stage(self, h, stage_idx):
        """Run a single stage with residual. Used by the main model."""
        return h + self.stages[stage_idx](h)

    def forward(self, x):
        """Full forward (for standalone use / testing)."""
        h = self.down(x)
        for i in range(len(self.stages)):
            h = self.forward_stage(h, i)
        return h


# ---------------------------------------------------------------------------
#  Branch C-1: Cascaded PixelShuffle  (56² → 224²)
# ---------------------------------------------------------------------------

class CascadedPixelShuffle(nn.Module):
    """Two-step learned upsampling with anti-checkerboard DW-Conv.

    Step 1:  Conv1×1(C→4C) → PixelShuffle(2) → DWConv3×3 → GELU  (×2 spatial)
    Step 2:  Conv1×1(C→4C) → PixelShuffle(2) → DWConv3×3 → GELU  (×2 spatial)
    Total:   ×4 spatial upsampling.
    """

    def __init__(self, dim):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(min(8, dim), dim),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(min(8, dim), dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.up2(self.up1(x))


# ---------------------------------------------------------------------------
#  Branch C-2: SDF Gate
# ---------------------------------------------------------------------------

class SDFGate(nn.Module):
    """Per-class SDF spatial gate with explicit stop-gradient.

    Input:  sdf_pred (B, num_fg, H, W) — detached inside forward.
    Output: gate (B, 1, H, W) in [0, 1].
    """

    def __init__(self, sdf_ch=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(sdf_ch, sdf_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(4, sdf_ch), sdf_ch),
            nn.GELU(),
            nn.Conv2d(sdf_ch, 1, 1),
        )

    def forward(self, sdf):
        with torch.no_grad():
            s = sdf.detach()
        return torch.sigmoid(self.conv(s))


# ---------------------------------------------------------------------------
#  Branch C-3: Frequency-Split Fusion
# ---------------------------------------------------------------------------

class FrequencySplitFusion(nn.Module):
    """FFT-based low/high frequency decoupling for artifact-free fusion.

    1. rfft2(feat_center) → apply Gaussian radial LP mask → irfft2 → feat_low
    2. feat_high = feat_center − feat_low   (preserved intact)
    3. fused_low = feat_low + gate · ctx_up
    4. output = fused_low + feat_high

    Uses a Gaussian kernel (not sigmoid) to avoid Gibbs ringing.
    """

    def __init__(self, dim, cutoff_ratio=0.25):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        # Learnable channel-wise scale for the injected context
        self.ctx_scale = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GroupNorm(min(8, dim), dim),
            nn.Sigmoid(),
        )

    def _gaussian_lp_mask(self, H, W_freq, device):
        """Gaussian radial low-pass filter in frequency domain."""
        cy = H // 2
        fy = torch.arange(H, device=device, dtype=torch.float32) - cy
        fy = torch.fft.fftshift(fy)
        fx = torch.arange(W_freq, device=device, dtype=torch.float32)
        r2 = fy[:, None] ** 2 + fx[None, :] ** 2
        sigma = self.cutoff_ratio * min(H, W_freq * 2)
        return torch.exp(-r2 / (2.0 * sigma ** 2))       # (H, W_freq)

    def forward(self, feat_center, ctx_up, gate):
        B, C, H, W = feat_center.shape

        # Split feat_center into low-freq and high-freq via FFT
        with torch.amp.autocast('cuda', enabled=False):
            x = feat_center.float()
            xf = torch.fft.rfft2(x, norm='ortho')
            lp = self._gaussian_lp_mask(H, xf.shape[-1], x.device)  # (H, W/2+1)
            feat_low = torch.fft.irfft2(xf * lp, s=(H, W), norm='ortho')

        feat_high = feat_center - feat_low.to(feat_center.dtype)
        feat_low = feat_low.to(feat_center.dtype)

        # Adaptive channel-wise scaling for context injection
        alpha = self.ctx_scale(torch.cat([feat_low, ctx_up], dim=1))
        fused_low = feat_low + gate * alpha * ctx_up

        return fused_low + feat_high


# ---------------------------------------------------------------------------
#  Main Model: AsymSpecMambaDCN
# ---------------------------------------------------------------------------

class AsymSpecMambaDCN(nn.Module):
    """Asymmetric 2.5D Cardiac MRI Segmentation  (v3.1).

    Decouples boundary precision (center-slice DCN @ 224²) from global context
    (all-5-slice FFT mixer @ 56²), with per-stage Asymmetric Bidirectional
    Exchange (ABX) and final frequency-split fusion to preserve edges.

    v3.1 changes over v3:
      - GlobalContextEncoder uses Global2DFFTMixer (full spatial RF)
      - ABX exchange after every stage (3 exchanges total)
      - Auxiliary context segmentation head for gradient health

    Forward returns ``{'output': logits, 'sdf': sdf_pred}``.
    With ``deep_supervision=True`` and ``model.training``, also returns
    ``'aux_outputs'`` and ``'ctx_logits'``.
    """

    NUM_STAGES = 3

    def __init__(self, in_ch=5, num_classes=4, base_ch=48,
                 deep_supervision=False, cutoff_ratio=0.25):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_fg = num_classes - 1
        C = base_ch

        # Branch A: Precision Stem — center 3 slices through full-res DCN
        self.precision_stem = PrecisionStem(in_channels=3, base_channels=C)

        # Branch B: Global Context — all 5 slices through FFT mixer @ 56²
        self.context_encoder = GlobalContextEncoder(
            dim=C, in_slices=in_ch, num_stages=self.NUM_STAGES)

        # ABX: per-stage cross-stream exchange
        self.abx = nn.ModuleList([
            AsymBidirectionalExchange(C) for _ in range(self.NUM_STAGES)
        ])

        # Branch C: Upsampler + Fusion
        self.upsampler = CascadedPixelShuffle(C)
        self.sdf_gate = SDFGate(sdf_ch=self.num_fg)
        self.freq_fusion = FrequencySplitFusion(C, cutoff_ratio=cutoff_ratio)

        # Branch D: Dual heads
        self.sdf_head = nn.Sequential(
            nn.Conv2d(C, self.num_fg, 1),
            nn.Tanh(),
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, C), C),
            nn.GELU(),
            nn.Conv2d(C, num_classes, 1),
        )

        # Auxiliary context head — ensures context branch gets direct grad
        self.ctx_seg_head = nn.Conv2d(C, num_classes, 1)

        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(C, num_classes, 1),
                nn.Conv2d(C, num_classes, 1),
            ])

    def forward(self, x):
        x_local = x[:, 1:4, :, :]

        # -- Init both branches --
        prec = self.precision_stem.stem_init(x_local)
        ctx = self.context_encoder.down(x)

        # -- Interleaved stages with ABX exchange --
        intermediates = []
        for s in range(self.NUM_STAGES):
            prec = self.precision_stem.forward_stage(prec, s)
            ctx = self.context_encoder.forward_stage(ctx, s)
            prec, ctx = self.abx[s](prec, ctx)
            if s < self.NUM_STAGES - 1:
                intermediates.append(prec)

        feat_center = prec
        feat_z_ctx = ctx

        # -- Branch C: upsample context + frequency-split fusion --
        ctx_up = self.upsampler(feat_z_ctx)

        sdf_pred = self.sdf_head(feat_center)
        gate = self.sdf_gate(sdf_pred)

        feat_fused = self.freq_fusion(feat_center, ctx_up, gate)

        # -- Branch D: dual heads --
        logits = self.seg_head(feat_fused)

        result = {'output': logits, 'sdf': sdf_pred}

        if self.training:
            ctx_logits = F.interpolate(
                self.ctx_seg_head(feat_z_ctx),
                size=logits.shape[2:], mode='bilinear', align_corners=False,
            )
            result['ctx_logits'] = ctx_logits

        need_aux = self.training and self.deep_supervision
        if need_aux:
            result['aux_outputs'] = [
                head(f) for head, f in zip(self.aux_heads, intermediates)
            ]
        return result


# ---------------------------------------------------------------------------
#  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = AsymSpecMambaDCN(
        in_ch=5, num_classes=4, base_ch=48, deep_supervision=True,
    ).to(device)
    model.train()

    x = torch.randn(2, 5, 224, 224, device=device)
    out = model(x)

    assert out['output'].shape == (2, 4, 224, 224), \
        f"Bad logits shape: {out['output'].shape}"
    assert out['sdf'].shape == (2, 3, 224, 224), \
        f"Bad SDF shape: {out['sdf'].shape}"
    assert 'ctx_logits' in out, "Missing ctx_logits in training mode"
    assert out['ctx_logits'].shape == (2, 4, 224, 224), \
        f"Bad ctx_logits shape: {out['ctx_logits'].shape}"
    assert len(out.get('aux_outputs', [])) == 2, \
        f"Expected 2 aux outputs, got {len(out.get('aux_outputs', []))}"

    loss = out['output'].sum() + out['sdf'].sum() + 0.3 * out['ctx_logits'].sum()
    for aux in out['aux_outputs']:
        loss = loss + 0.1 * aux.sum()
    loss.backward()

    n = sum(p.numel() for p in model.parameters())
    print(f"Smoke test passed! Forward + backward OK.")
    print(f"  output:      {out['output'].shape}")
    print(f"  sdf:         {out['sdf'].shape}")
    print(f"  ctx_logits:  {out['ctx_logits'].shape}")
    print(f"  aux:         {[a.shape for a in out['aux_outputs']]}")
    print(f"  params:      {n:,}")
