
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from layers.monogenic import EnergyMap, MonogenicSignal
    from models.hrnet_mamba import HRNetV2MambaBackbone
    from layers.constellation_head import RBFConstellationHead
    from layers.gabor_implicit import EnergyGatedImplicitHead, GaborNet, ImplicitSegmentationHead
    from models.mamba_block import VSSBlock, MambaBlockStack
except ImportError:
    from ..layers.monogenic import EnergyMap, MonogenicSignal
    from .hrnet_mamba import HRNetV2MambaBackbone
    from ..layers.constellation_head import RBFConstellationHead
    from ..layers.gabor_implicit import EnergyGatedImplicitHead, GaborNet, ImplicitSegmentationHead
    from .mamba_block import VSSBlock, MambaBlockStack


class EnergyGatedFusion(nn.Module):

    def __init__(self, temperature: float = 1.0):

        super().__init__()
        self.temperature = temperature
        self.gate_scale = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self, coarse: torch.Tensor, fine: torch.Tensor,
                energy: torch.Tensor) -> torch.Tensor:

        if energy.shape[-2:] != coarse.shape[-2:]:
            energy = F.interpolate(energy, size=coarse.shape[-2:],
                                   mode='bilinear', align_corners=True)

        if fine.shape[-2:] != coarse.shape[-2:]:
            fine = F.interpolate(fine, size=coarse.shape[-2:],
                                mode='bilinear', align_corners=True)

        gate = torch.sigmoid((energy * self.gate_scale + self.gate_bias) / self.temperature)

        output = coarse + gate * (fine - coarse)

        return output

class EGMNet(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 4,
                 img_size: int = 256, base_channels: int = 64,
                 num_stages: int = 4, use_hrnet: bool = True,
                 use_mamba: bool = True, use_spectral: bool = True,
                 use_fine_head: bool = True,
                 coarse_head_type: str = "constellation",
                 fusion_type: str = "energy_gated",
                 implicit_hidden: int = 256, implicit_layers: int = 4,
                 num_frequencies: int = 64):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_hrnet = use_hrnet
        self.use_fine_head = use_fine_head
        self.coarse_head_type = coarse_head_type
        self.fusion_type = fusion_type

        self.energy_extractor = EnergyMap(normalize=True, smoothing_sigma=1.0)

        if use_hrnet:
            self.backbone = HRNetV2MambaBackbone(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=num_stages,
                blocks_per_stage=2,
                mamba_depth=2 if use_mamba else 0,
                img_size=img_size,
                use_mamba=use_mamba,
                use_spectral=use_spectral
            )
            backbone_channels = self.backbone.out_channels
        else:

            self.backbone = SimpleEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=num_stages
            )
            backbone_channels = base_channels * (2 ** (num_stages - 1))

        self.backbone_channels = backbone_channels

        if coarse_head_type == "constellation":
            self.coarse_head = RBFConstellationHead(
                in_channels=backbone_channels,
                num_classes=num_classes,
                embedding_dim=2,
                init_gamma=1.0
            )
        else:

            self.coarse_head = nn.Sequential(
                nn.Conv2d(backbone_channels, backbone_channels // 2, 3, padding=1),
                nn.GroupNorm(min(32, backbone_channels // 2), backbone_channels // 2),
                nn.GELU(),
                nn.Conv2d(backbone_channels // 2, num_classes, 1)
            )

        if use_fine_head:
            self.fine_head = EnergyGatedImplicitHead(
                feature_channels=backbone_channels,
                num_classes=num_classes,
                hidden_dim=implicit_hidden,
                num_layers=implicit_layers,
                num_frequencies=num_frequencies
            )
        else:
            self.fine_head = None

        if use_fine_head:
            if fusion_type == "energy_gated":
                self.fusion = EnergyGatedFusion(temperature=1.0)
            else:
                self.fusion = None
        else:
            self.fusion = None

        if use_hrnet:
            self.feature_size = img_size // 4
        else:
            self.feature_size = img_size // (2 ** num_stages)

    def _compute_energy(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:

        with torch.no_grad():

            x_gray = x[:, 0:1] if x.shape[1] > 1 else x
            energy, mono_out = self.energy_extractor(x_gray)
        return energy, mono_out

    def forward(self, x: torch.Tensor,
                output_size: Optional[Tuple[int, int]] = None,
                return_intermediates: bool = True) -> Dict[str, torch.Tensor]:

        B, C, H, W = x.shape

        if output_size is None:
            output_size = (H, W)

        if C >= 3:

            intensity = x[:, 0:1]
            riesz_x = x[:, 1:2]
            riesz_y = x[:, 2:3]

            energy = torch.sqrt(intensity**2 + riesz_x**2 + riesz_y**2 + 1e-8)

            energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        else:

            energy, _ = self._compute_energy(x)

        if self.use_hrnet:
            backbone_out = self.backbone(x)
            features = backbone_out['features']
        else:
            features = self.backbone(x)

        if self.coarse_head_type == "constellation":
            coarse_logits, embeddings = self.coarse_head(features)
        else:

            coarse_logits = self.coarse_head(features)
            embeddings = None

        coarse = F.interpolate(coarse_logits, size=output_size,
                              mode='bilinear', align_corners=True)

        if self.use_fine_head and self.fine_head is not None:

            energy_for_fine = F.interpolate(energy, size=features.shape[-2:],
                                            mode='bilinear', align_corners=True)

            fine_logits = self.fine_head(features, energy_for_fine, output_size=output_size)

            if fine_logits.shape[-2:] != output_size:
                fine_logits = F.interpolate(fine_logits, size=output_size,
                                           mode='bilinear', align_corners=True)

            if self.fusion is not None:
                output = self.fusion(coarse, fine_logits, energy)
            else:

                output = (coarse + fine_logits) / 2.0
        else:

            fine_logits = None
            output = coarse

        if return_intermediates:
            return {
                'output': output,
                'coarse': coarse,
                'fine': fine_logits if fine_logits is not None else coarse,
                'energy': energy,
                'embeddings': embeddings
            }
        else:
            return {'output': output}

    def inference(self, x: torch.Tensor,
                  output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:

        return self.forward(x, output_size, return_intermediates=False)['output']

    def sample_points(self, coarse_logits: torch.Tensor,
                      energy_map: torch.Tensor,
                      num_samples: int = 4096) -> torch.Tensor:

        B, C, H, W = coarse_logits.shape
        device = coarse_logits.device

        probs = F.softmax(coarse_logits, dim=1)

        max_prob, _ = probs.max(dim=1, keepdim=True)

        uncertainty = 1.0 - max_prob

        if energy_map.shape[-2:] != (H, W):
            energy_map = F.interpolate(
                energy_map, size=(H, W), mode='bilinear', align_corners=True
            )

        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)

        sample_weights = uncertainty + 0.5 * energy_map

        sample_weights_flat = sample_weights.view(B, -1)

        indices = torch.multinomial(sample_weights_flat, num_samples, replacement=True)

        x_idx = indices % W
        y_idx = indices // W

        x_norm = (x_idx.float() + 0.5) / W * 2.0 - 1.0
        y_norm = (y_idx.float() + 0.5) / H * 2.0 - 1.0

        coords = torch.stack([x_norm, y_norm], dim=-1)

        return coords

    def query_points(self, x: torch.Tensor,
                     coords: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        if C >= 3:
            intensity = x[:, 0:1]
            riesz_x = x[:, 1:2]
            riesz_y = x[:, 2:3]
            energy = torch.sqrt(intensity**2 + riesz_x**2 + riesz_y**2 + 1e-8)
            energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        else:
            energy, _ = self._compute_energy(x)

        if self.use_hrnet:
            backbone_out = self.backbone(x)
            features = backbone_out['features']
        else:
            features = self.backbone(x)

        N = coords.shape[1]
        grid = coords.view(B, 1, N, 2)

        features_proj = self.fine_head.feature_proj(features)

        feat_sampled = F.grid_sample(
            features_proj, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        ).squeeze(2).permute(0, 2, 1)

        energy_sampled = F.grid_sample(
            energy, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        ).squeeze(2).permute(0, 2, 1)

        point_logits = self.fine_head.implicit_decoder(coords, feat_sampled, energy_sampled)

        return point_logits

class SimpleEncoder(nn.Module):

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_stages: int = 4):
        super().__init__()

        layers = []
        channels = in_channels
        out_channels = base_channels

        for i in range(num_stages):
            layers.append(nn.Conv2d(channels, out_channels, 3, stride=2, padding=1))
            layers.append(nn.GroupNorm(min(32, out_channels), out_channels))
            layers.append(nn.GELU())
            channels = out_channels
            out_channels = min(out_channels * 2, 512)

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class EGMNetLite(nn.Module):

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 img_size: int = 256):
        super().__init__()

        self.model = EGMNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            base_channels=32,
            num_stages=3,
            use_hrnet=False,
            implicit_hidden=128,
            implicit_layers=3,
            num_frequencies=32
        )

    def forward(self, x, output_size=None):
        return self.model(x, output_size)

    def inference(self, x, output_size=None):
        return self.model.inference(x, output_size)

if __name__ == "__main__":
    print("=" * 60)
    print("Testing EGM-Net (Energy-Gated Gabor Mamba Network)")
    print("=" * 60)

    print("\n[1] Testing Full EGM-Net (HRNetV2-Mamba Backbone)...")
    model = EGMNet(
        in_channels=3,
        num_classes=4,
        img_size=256,
        base_channels=64,
        num_stages=4,
        use_hrnet=True
    )

    x = torch.randn(2, 3, 256, 256)
    outputs = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {outputs['output'].shape}")
    print(f"Coarse: {outputs['coarse'].shape}")
    print(f"Fine: {outputs['fine'].shape}")
    print(f"Energy: {outputs['energy'].shape}")
    print(f"Embeddings: {outputs['embeddings'].shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n[2] Testing Point Query (Resolution-Free)...")
    coords = torch.rand(2, 1000, 2) * 2 - 1
    point_output = model.query_points(x, coords)
    print(f"Query coords: {coords.shape}")
    print(f"Point output: {point_output.shape}")

    print("\n[3] Testing EGM-Net Lite...")
    lite_model = EGMNetLite(in_channels=1, num_classes=3, img_size=256)
    x_lite = torch.randn(2, 1, 256, 256)
    lite_outputs = lite_model(x_lite)

    lite_params = sum(p.numel() for p in lite_model.parameters())
    print(f"Lite model parameters: {lite_params:,}")
    print(f"Lite output: {lite_outputs['output'].shape}")

    print("\n[4] Testing Training Sampling (Uncertainty + Energy)...")
    coarse_logits = outputs['coarse']
    energy_map = outputs['energy']
    sampled_coords = model.sample_points(coarse_logits, energy_map, num_samples=1024)
    print(f"Sampled coords: {sampled_coords.shape}")
    print(f"Range: [{sampled_coords.min():.3f}, {sampled_coords.max():.3f}]")

    print("\n[5] Testing Single-Channel Input (Auto Energy)...")
    x_single = torch.randn(2, 1, 256, 256)
    model_single = EGMNet(in_channels=1, num_classes=4, img_size=256, use_hrnet=False)
    outputs_single = model_single(x_single)
    print(f"Single-channel output: {outputs_single['output'].shape}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
