"""
HRNet with Asymmetric DCN (Dilation Pyramid)
Production-ready model file

Features:
- Asymmetric depth per stage (2,4,6 blocks)
- DCN with Dilation Pyramid (HDC: 1,2,4,8,16,32)
- Optional PointRend for boundary refinement
- Full resolution mode (no downsampling in stem)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import get_block




class HRNetStem(nn.Module):
    """Standard HRNet stem with stride 4 (224 -> 56)."""
    
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, out_ch, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Bottleneck(nn.Module):
    """Bottleneck block for Layer1."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)


class FuseLayer(nn.Module):
    """Multi-scale feature fusion layer."""
    
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.num_in = len(in_channels_list)
        self.num_out = len(out_channels_list)
        self.fuse = nn.ModuleList()

        for j in range(self.num_out):
            fuse_j = nn.ModuleList()
            for i in range(self.num_in):
                if i == j:
                    fuse_j.append(nn.Identity())
                elif i < j:
                    fuse_j.append(nn.Sequential(
                        nn.Conv2d(in_channels_list[i], out_channels_list[j], 3, 2**(j-i), 1, bias=False),
                        nn.BatchNorm2d(out_channels_list[j])
                    ))
                else:
                    fuse_j.append(nn.Sequential(
                        nn.Conv2d(in_channels_list[i], out_channels_list[j], 1, bias=False),
                        nn.BatchNorm2d(out_channels_list[j]),
                        nn.Upsample(scale_factor=2**(i-j), mode='bilinear', align_corners=True)
                    ))
            self.fuse.append(fuse_j)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_list):
        out = []
        for j in range(self.num_out):
            y = None
            for i in range(self.num_in):
                if y is None:
                    y = self.fuse[j][i](x_list[i])
                else:
                    y = y + self.fuse[j][i](x_list[i])
            out.append(self.relu(y))
        return out


# =============================================================================
# FULL RESOLUTION STEM
# =============================================================================

class HRNetStemFullRes(nn.Module):
    """Full Resolution Stem (stride=1) - keeps 224x224 throughout. High VRAM!"""
    
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch // 2)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class HRNetDCN(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=4, base_channels=64, img_size=224,
                 stage_configs=None, use_pointrend=False, full_resolution_mode=True,
                 deep_supervision=False, use_shearlet=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.full_resolution_mode = full_resolution_mode
        self.deep_supervision = deep_supervision
        
        # Default: Asymmetric DCN (2-4-6)
        if stage_configs is None:
            stage_configs = [
                {'blocks': ['dcn'] * 2},
                {'blocks': ['dcn'] * 4},
                {'blocks': ['dcn'] * 6},
            ]
        
        self.stage_configs = stage_configs
        
        # Stem
        if full_resolution_mode:
            print(">>> WARNING: FULL RESOLUTION MODE (224x224). High VRAM!")
            self.stem = HRNetStemFullRes(in_channels, 64)
            s = img_size  # No downsample
        else:
            self.stem = HRNetStem(in_channels, 64)
            s = img_size // 4  # Normal: 224 -> 56
        
        # Layer 1 (Bottleneck)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=nn.Sequential(
                nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256)
            )),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        
        C = base_channels
        
        # Transitions and Stages
        self.transition1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, C, 3, 1, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(256, C*2, 3, 2, 1, bias=False), nn.BatchNorm2d(C*2), nn.ReLU(True))
        ])
        
        self.stage2 = self._make_stage([C, C*2], stage_configs[0]['blocks'], [(s, s), (s//2, s//2)])
        
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(nn.Conv2d(C*2, C*4, 3, 2, 1, bias=False), nn.BatchNorm2d(C*4), nn.ReLU(True))
        ])
        
        self.stage3 = self._make_stage([C, C*2, C*4], stage_configs[1]['blocks'], 
                                        [(s, s), (s//2, s//2), (s//4, s//4)])
        
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(nn.Conv2d(C*4, C*8, 3, 2, 1, bias=False), nn.BatchNorm2d(C*8), nn.ReLU(True))
        ])
        
        self.stage4 = self._make_stage([C, C*2, C*4, C*8], stage_configs[2]['blocks'],
                                        [(s, s), (s//2, s//2), (s//4, s//4), (s//8, s//8)])
        
        # Output channels
        self.out_channels = C + C*2 + C*4 + C*8
        
        # Segmentation head
        self.seg_head = nn.Conv2d(self.out_channels, num_classes, 1)
        
        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(C, num_classes, 1),      # Stage2 stream1
                nn.Conv2d(C*2, num_classes, 1),    # Stage2 stream2
                nn.Conv2d(C*4, num_classes, 1),    # Stage3 stream3
                nn.Conv2d(C*8, num_classes, 1),    # Stage4 stream4
            ])
        
        # Optional PointRend
        self.use_pointrend = use_pointrend
        if use_pointrend:
            from layers.pointrend import PointRend
            self.pointrend = PointRend(
                in_channels=self.out_channels,
                num_classes=num_classes,
                num_points=1024,
                hidden_dim=128
            )
        
        # Optional Shearlet Head for fine-grained boundary refinement
        self.use_shearlet = use_shearlet
        if use_shearlet:
            from layers.shearlet_implicit import ShearletImplicitHead
            self.shearlet_head = ShearletImplicitHead(
                feature_dim=self.out_channels,
                num_classes=num_classes,
                hidden_dim=256,
                num_orientations=8,
                num_frequencies=4
            )
            # Fusion weight for shearlet output
            self.shearlet_fusion = nn.Parameter(torch.tensor(0.5))
    
    def _make_stage(self, channels_list, block_types, sizes):
        """Create stage with DCN Dilation Pyramid (HDC strategy)."""
        branches = nn.ModuleList()
        num_blocks = len(block_types)
        
        for idx, ch in enumerate(channels_list):
            blocks = []
            for i, bt in enumerate(block_types):
                # Dilation Pyramid for DCN blocks
                if bt == 'dcn' and num_blocks >= 3:
                    current_dilation = 2 ** min(i, 5)  # Cap at 32
                    blocks.append(get_block(bt, ch, dilation=current_dilation))
                else:
                    blocks.append(get_block(bt, ch))
            branches.append(nn.Sequential(*blocks))
        
        fuse = FuseLayer(channels_list, channels_list)
        
        return nn.ModuleDict({
            'branches': branches,
            'fuse': fuse
        })
    
    def _forward_stage(self, stage, x_list):
        out = [stage['branches'][i](x_list[i]) for i in range(len(x_list))]
        return stage['fuse'](out)
    
    def forward(self, x):
        target_size = x.shape[2:]
        
        x = self.stem(x)
        x = self.layer1(x)
        
        x_list = [self.transition1[0](x), self.transition1[1](x)]
        x_list = self._forward_stage(self.stage2, x_list)
        
        x_list = [
            self.transition2[0](x_list[0]),
            self.transition2[1](x_list[1]),
            self.transition2[2](x_list[1])
        ]
        x_list = self._forward_stage(self.stage3, x_list)
        
        x_list = [
            self.transition3[0](x_list[0]),
            self.transition3[1](x_list[1]),
            self.transition3[2](x_list[2]),
            self.transition3[3](x_list[2])
        ]
        x_list = self._forward_stage(self.stage4, x_list)
        
        # Aggregate multi-scale features
        x0_h, x0_w = x_list[0].shape[2:]
        feats = [
            x_list[0],
            F.interpolate(x_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
            F.interpolate(x_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
            F.interpolate(x_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
        ]
        features = torch.cat(feats, dim=1)
        
        # Segmentation
        logits = self.seg_head(features)
        
        if self.use_pointrend:
            logits = self.pointrend(logits, features, target_size)
        else:
            logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=True)
        
        # Shearlet refinement head - fuse with main logits
        if self.use_shearlet and hasattr(self, 'shearlet_head'):
            features_up = F.interpolate(features, size=target_size, mode='bilinear', align_corners=True)
            shearlet_logits = self.shearlet_head(features_up, output_size=target_size)
            w = torch.sigmoid(self.shearlet_fusion)
            logits = (1 - w) * logits + w * shearlet_logits
        
        result = {'output': logits}
        
        # Deep supervision auxiliary outputs
        if self.deep_supervision and hasattr(self, 'aux_heads'):
            aux_outputs = []
            for i, head in enumerate(self.aux_heads):
                aux_logits = head(x_list[i])
                aux_logits = F.interpolate(aux_logits, size=target_size, mode='bilinear', align_corners=True)
                aux_outputs.append(aux_logits)
            result['aux_outputs'] = aux_outputs
        
        return result


def hrnet_dcn_small(num_classes=4, in_channels=3, use_pointrend=False):
    """Small config: base_channels=32, ~10M params"""
    return HRNetDCN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=32,
        use_pointrend=use_pointrend
    )


def hrnet_dcn_base(num_classes=4, in_channels=3, use_pointrend=True):
    """Base config: base_channels=48, ~25M params"""
    return HRNetDCN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=48,
        use_pointrend=use_pointrend
    )


def hrnet_dcn_large(num_classes=4, in_channels=3, use_pointrend=True):
    """Large config: base_channels=64, ~40M params"""
    return HRNetDCN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        use_pointrend=use_pointrend
    )
