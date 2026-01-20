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
                 encoder_depths=[3, 4, 6, 3], use_deep_supervision=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_deep_supervision = use_deep_supervision
        
        C = base_channels  # 64 default (like ResNet-34)
        
        # =====================================================================
        # STEM (stride 4: 224 -> 56)
        # =====================================================================
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
        """Create a stage with multiple BasicBlocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                get_norm(out_channels)
            )
        
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
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


if __name__ == '__main__':
    # Test
    model = hrnet_resnet34_base(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out['output'].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
