"""Small segmentation model for SSRBlockV3 debugging."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ssr_blocks import SSRBlockV3


class ConvBlock(nn.Module):
    """Two-convolution block that preserves spatial shape."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MiniSSRSegNetV3(nn.Module):
    """Minimal SSR segmentation network for controlled ACDC experiments.

    Input shape: `[B, in_channels, H, W]`.
    Output dictionary:
        `seg_logits`: `[B, num_classes, H, W]`
        `boundary_logits`: `[B, 1, H, W]`
        `gate_reg`: scalar regularization tensor
        `logs`: optional SSR diagnostics when `return_logs=True`
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_classes: int = 4,
        num_bands: int = 4,
        ssr: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        ssr = dict(ssr or {})
        ssr.setdefault("num_bands", num_bands)

        self.stem = ConvBlock(in_channels, base_channels)
        self.ssr1 = SSRBlockV3(base_channels, **ssr)
        self.mid = ConvBlock(base_channels, base_channels)
        self.ssr2 = SSRBlockV3(base_channels, **ssr)
        self.shared_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(base_channels), base_channels),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        boundary_mask: Tensor | None = None,
        return_logs: bool = False,
    ) -> dict[str, Tensor | dict[str, Any]]:
        # x: [B, C, H, W]
        feat = self.stem(x)

        if return_logs:
            feat, ssr1_logs, gate_reg1 = self.ssr1(
                feat,
                boundary_mask=boundary_mask,
                return_logs=True,
            )
        else:
            feat, gate_reg1 = self.ssr1(feat, boundary_mask=boundary_mask)
            ssr1_logs = {}

        feat = self.mid(feat)

        if return_logs:
            feat, ssr2_logs, gate_reg2 = self.ssr2(
                feat,
                boundary_mask=boundary_mask,
                return_logs=True,
            )
        else:
            feat, gate_reg2 = self.ssr2(feat, boundary_mask=boundary_mask)
            ssr2_logs = {}

        feat = self.shared_head(feat)
        outputs: dict[str, Tensor | dict[str, Any]] = {
            "seg_logits": self.seg_head(feat),
            "boundary_logits": self.boundary_head(feat),
            "gate_reg": gate_reg1 + gate_reg2,
        }
        if return_logs:
            outputs["logs"] = {
                "ssr1": ssr1_logs,
                "ssr2": ssr2_logs,
            }
        return outputs


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
