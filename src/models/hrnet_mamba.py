import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import math

try:
    from .mamba_block import VSSBlock, MambaBlockStack
    from ..layers.spectral_layers import SpectralGating
except (ImportError, ValueError):
    try:
        from models.mamba_block import VSSBlock, MambaBlockStack
        from layers.spectral_layers import SpectralGating
    except ImportError:
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.mamba_block import VSSBlock, MambaBlockStack
        from layers.spectral_layers import SpectralGating


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
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


class MambaBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 use_mamba=True, use_spectral=True, height=64, width=64):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_spectral = use_spectral

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = downsample

        if use_mamba:
            self.mamba = MambaBlockStack(planes, depth=2, expansion_ratio=2.0, scan_dim=min(64, planes))
        else:
            self.mamba = nn.Sequential(
                nn.Conv2d(planes, planes, 3, 1, 1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, 3, 1, 1, bias=False),
                nn.BatchNorm2d(planes)
            )

        if use_spectral:
            self.spectral = SpectralGating(planes, height, width, threshold=0.1, complex_init="kaiming")
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.spectral = None
            self.fusion_weight = None

        self.proj = nn.Conv2d(inplanes, planes, 1, stride, bias=False) if (stride != 1 or inplanes != planes) else nn.Identity()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.proj(x)

        out = self.mamba(residual)

        if self.spectral is not None:
            spec_out = self.spectral(residual)
            w = torch.sigmoid(self.fusion_weight)
            out = w * out + (1 - w) * spec_out

        out = self.bn(out)
        return self.relu(out + residual)


class HRNetStem(nn.Module):
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


class FuseLayer(nn.Module):
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


class HRNetStage(nn.Module):
    def __init__(self, channels_list, num_blocks=4, block=BasicBlock,
                 use_mamba=False, use_spectral=False, sizes=None,
                 block_type="basic"):
        """
        Args:
            block_type: "basic", "mamba", "convnext", "dcn", "swin", "fno", "wavelet", "rwkv"
        """
        super().__init__()
        self.branches = nn.ModuleList()

        # Import modular blocks if needed
        if block_type not in ["basic", "mamba"]:
            try:
                from models.blocks import get_block
            except ImportError:
                from .blocks import get_block

        for idx, ch in enumerate(channels_list):
            h, w = sizes[idx] if sizes else (64, 64)
            
            if block_type == "mamba" or (block_type == "basic" and use_mamba):
                # Original Mamba blocks
                blocks = [MambaBlock(ch, ch, use_mamba=use_mamba, use_spectral=use_spectral, height=h, width=w)
                          for _ in range(num_blocks)]
            elif block_type == "basic":
                # Original BasicBlock
                blocks = [block(ch, ch) for _ in range(num_blocks)]
            else:
                # Modular blocks (ConvNeXt, DCN, Swin, FNO, Wavelet, RWKV)
                blocks = [get_block(block_type, ch) for _ in range(num_blocks)]
            
            self.branches.append(nn.Sequential(*blocks))

        self.fuse = FuseLayer(channels_list, channels_list)

    def forward(self, x_list):
        out = [self.branches[i](x_list[i]) for i in range(len(x_list))]
        return self.fuse(out)


class HRNetV2MambaBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, num_stages=4,
                 blocks_per_stage=4, use_mamba=True, use_spectral=True,
                 img_size=224, mamba_depth=2, block_type="basic"):
        """
        Args:
            block_type: "basic", "mamba", "convnext", "dcn", "swin", "fno", "wavelet", "rwkv"
        """
        super().__init__()
        
        self.block_type = block_type

        self.stem = HRNetStem(in_channels, 64)

        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=nn.Sequential(
                nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256)
            )),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        C = base_channels
        self.transition1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, C, 3, 1, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(256, C*2, 3, 2, 1, bias=False), nn.BatchNorm2d(C*2), nn.ReLU(True))
        ])

        s = img_size // 4
        self.stage2 = HRNetStage([C, C*2], blocks_per_stage, BasicBlock,
                                  use_mamba, use_spectral, [(s, s), (s//2, s//2)],
                                  block_type=block_type)

        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(nn.Conv2d(C*2, C*4, 3, 2, 1, bias=False), nn.BatchNorm2d(C*4), nn.ReLU(True))
        ])

        self.stage3 = HRNetStage([C, C*2, C*4], blocks_per_stage, BasicBlock,
                                  use_mamba, use_spectral, [(s, s), (s//2, s//2), (s//4, s//4)],
                                  block_type=block_type)

        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(nn.Conv2d(C*4, C*8, 3, 2, 1, bias=False), nn.BatchNorm2d(C*8), nn.ReLU(True))
        ])

        self.stage4 = HRNetStage([C, C*2, C*4, C*8], blocks_per_stage, BasicBlock,
                                  use_mamba, use_spectral, [(s, s), (s//2, s//2), (s//4, s//4), (s//8, s//8)],
                                  block_type=block_type)

        self.out_channels = C + C*2 + C*4 + C*8
        self.feature_size = s

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)

        x_list = [self.transition1[0](x), self.transition1[1](x)]
        x_list = self.stage2(x_list)

        x_list = [
            self.transition2[0](x_list[0]),
            self.transition2[1](x_list[1]),
            self.transition2[2](x_list[1])
        ]
        x_list = self.stage3(x_list)

        x_list = [
            self.transition3[0](x_list[0]),
            self.transition3[1](x_list[1]),
            self.transition3[2](x_list[2]),
            self.transition3[3](x_list[2])
        ]
        x_list = self.stage4(x_list)

        h0 = x_list[0]
        h1 = F.interpolate(x_list[1], size=h0.shape[2:], mode='bilinear', align_corners=True)
        h2 = F.interpolate(x_list[2], size=h0.shape[2:], mode='bilinear', align_corners=True)
        h3 = F.interpolate(x_list[3], size=h0.shape[2:], mode='bilinear', align_corners=True)

        features = torch.cat([h0, h1, h2, h3], dim=1)

        return {
            'features': features,
            'high_res': h0,
            'low_res': h3
        }


if __name__ == "__main__":
    model = HRNetV2MambaBackbone(in_channels=3, base_channels=32, use_mamba=True, use_spectral=True, img_size=256)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Features: {out['features'].shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
