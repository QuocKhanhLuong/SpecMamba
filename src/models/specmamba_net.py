"""
3-Stream Asymmetric Spec-HRNet — Frequency-Guided Architecture

FR (224², DWConv):  Local edge refinement at full resolution
HR (112², FFT):     Global spectral mixing at half resolution
LR (56²,  Mamba):   Long-range sequential context at quarter resolution

Per-stage TriFuseLayer: all-to-all asymmetric cross-fuse (6 paths).
Final TriStreamFusion: FR edges gate HR+LR to suppress false positives.
Pure PyTorch — no external dependencies.
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

class DWConvBlock(nn.Module):
    """Inverted Bottleneck DWConv — local edge refinement at full-res.
    Expand → DWConv3×3 → Shrink (MobileNetV2-style).
    """
    def __init__(self, dim, expansion=4):
        super().__init__()
        mid = dim * expansion
        self.pw_expand = nn.Conv2d(dim, mid, 1, bias=False)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.pw_shrink = nn.Conv2d(mid, dim, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.pw_shrink(self.act(self.dw(self.act(self.pw_expand(x)))))))


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
    """4-way cross-scan (H↕ + W↔ × bidirectional) + gating."""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_gate = nn.Linear(dim, dim, bias=False)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size-1,
                                 groups=dim, bias=False)
        self.linear_out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()

    def _bidi(self, seq):
        L = seq.shape[1]
        h = self.act(self.linear_in(seq)).transpose(1, 2)
        return ((self.dw_conv(h)[..., :L] +
                 self.dw_conv(h.flip(-1))[..., :L].flip(-1)) * 0.5).transpose(1, 2)

    def forward(self, x):
        B, C, H, W = x.shape
        hh = self._bidi(x.permute(0,3,2,1).reshape(B*W,H,C)).reshape(B,W,H,C).permute(0,3,2,1)
        hw = self._bidi(x.permute(0,2,3,1).reshape(B*H,W,C)).reshape(B,H,W,C).permute(0,3,1,2)
        h = (hh + hw) * 0.5
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
    """LR: Uncertainty-guided attention for Mamba hallucination suppression."""
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

    FR (224², C):   DWConvBlock — local edge refinement
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

        # Per-stage blocks + TriFuseLayer
        self.fr_stages = nn.ModuleList()
        self.hr_stages = nn.ModuleList()
        self.lr_stages = nn.ModuleList()
        self.tri_fuse = nn.ModuleList()

        for _ in range(num_stages):
            self.fr_stages.append(nn.Sequential(*[
                ResidualBlock(DWConvBlock(C), C) for _ in range(blocks_per_stage)]))
            self.hr_stages.append(nn.Sequential(*[
                ResidualBlock(AdaptiveFourierMixer(C, num_modes), C)
                for _ in range(blocks_per_stage)]))
            self.lr_stages.append(nn.Sequential(*[
                ResidualBlock(CrossScanGatedMixer(C*2), C*2)
                for _ in range(blocks_per_stage)]))
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
