"""Selective Spectral Retention blocks for isolated ACDC experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def build_radial_frequency_masks(
    H: int,
    W: int,
    num_bands: int,
    device: torch.device | str,
) -> Tensor:
    """Return radial masks for an `rfft2` spectrum.

    The output shape is `[num_bands, H, W // 2 + 1]` and follows the unshifted
    frequency layout produced by `torch.fft.rfft2`.
    """
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")

    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.rfftfreq(W, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    radius = radius / radius.max().clamp_min(1e-8)

    edges = torch.linspace(0.0, 1.0, num_bands + 1, device=device)
    masks = []
    for band in range(num_bands):
        left = edges[band]
        right = edges[band + 1]
        if band == num_bands - 1:
            mask = (radius >= left) & (radius <= right)
        else:
            mask = (radius >= left) & (radius < right)
        masks.append(mask.float())
    return torch.stack(masks, dim=0)


def boundary_map_from_mask(mask: Tensor) -> Tensor:
    """Create a 1-pixel-thick multiclass boundary map from labels."""
    if mask.ndim == 4:
        mask_2d = mask[:, 0]
    elif mask.ndim == 3:
        mask_2d = mask
    else:
        raise ValueError(f"Expected mask rank 3 or 4, got shape {tuple(mask.shape)}")

    mask_2d = mask_2d.long()
    boundary = torch.zeros_like(mask_2d, dtype=torch.bool)
    boundary[:, :, 1:] |= mask_2d[:, :, 1:] != mask_2d[:, :, :-1]
    boundary[:, :, :-1] |= mask_2d[:, :, 1:] != mask_2d[:, :, :-1]
    boundary[:, 1:, :] |= mask_2d[:, 1:, :] != mask_2d[:, :-1, :]
    boundary[:, :-1, :] |= mask_2d[:, 1:, :] != mask_2d[:, :-1, :]
    boundary &= mask_2d > 0
    return boundary.unsqueeze(1).float()


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise residual update gating."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // max(int(reduction), 1), 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def gate(self, x: Tensor) -> Tensor:
        return self.mlp(self.pool(x))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)


class ResidualChannelGate(nn.Module):
    """Channel gate that inspects identity features and spectral update."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // max(int(reduction), 1), 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def gate(self, x: Tensor, delta: Tensor) -> Tensor:
        return self.mlp(self.pool(torch.cat([x, delta], dim=1)))

    def forward(self, x: Tensor, delta: Tensor) -> Tensor:
        return delta * self.gate(x, delta)


class LargeKernelRefine(nn.Module):
    """Lightweight spatial geometry refinement after SSR spectral update."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("large_kernel_size must be odd to preserve shape")
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeformableRefine(nn.Module):
    """Optional local geometry refinement using torchvision deform_conv2d.

    This is a modulated deformable-convolution path exposed by torchvision, not
    a verified DCNv4 operator.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        try:
            from torchvision.ops import deform_conv2d
        except Exception as exc:  # pragma: no cover - depends on local install
            raise ImportError(
                "geometry_refine='deformable' requires torchvision.ops.deform_conv2d. "
                "Install a torchvision build compatible with the active PyTorch."
            ) from exc

        if kernel_size % 2 == 0:
            raise ValueError("deformable kernel_size must be odd to preserve shape")
        self.deform_conv2d = deform_conv2d
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        k2 = self.kernel_size * self.kernel_size
        self.offset_conv = nn.Conv2d(channels, 2 * k2, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(channels, k2, kernel_size=3, padding=1)
        self.weight = nn.Parameter(torch.empty(channels, channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        y = self.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=(1, 1),
            padding=(self.padding, self.padding),
            dilation=(1, 1),
            mask=mask,
        )
        return self.act(self.norm(y))


class DCNv4Refine(nn.Module):
    """Pure-PyTorch DCNv4-style local geometry refinement.

    The official DCNv4 Python module projects values, predicts grouped
    offset/mask tensors, and then calls a fused CUDA operator for unnormalized
    deformable aggregation. This experimental block keeps the same module-level
    structure but implements the sampling path with `grid_sample`, so it runs
    without compiling the external DCNv4 extension.

    It is intended for controlled ablations, not as a speed-equivalent drop-in
    replacement for the official CUDA operator.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        group: int = 2,
        offset_scale: float = 1.0,
        dw_kernel_size: int | None = 3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("dcnv4 kernel_size must be odd to preserve shape")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.kernel_points = self.kernel_size * self.kernel_size
        self.group = _valid_group(self.channels, int(group))
        self.group_channels = self.channels // self.group
        self.offset_scale = float(offset_scale)

        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1)
        if dw_kernel_size is None:
            self.offset_mask_dw = None
        else:
            self.offset_mask_dw = nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                padding=dw_kernel_size // 2,
                groups=channels,
            )
        self.offset_mask = nn.Conv2d(channels, self.group * self.kernel_points * 3, kernel_size=1)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.GELU()

        center_prior = torch.zeros(1, self.group, self.kernel_points, 1, 1)
        center_prior[:, :, self.kernel_points // 2] = 1.0
        self.register_buffer("center_prior", center_prior, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {C}")

        value = self.value_proj(x)
        offset_source = self.offset_mask_dw(x) if self.offset_mask_dw is not None else x
        offset_mask = self.offset_mask(offset_source).view(B, self.group, self.kernel_points, 3, H, W)
        raw_offset = torch.tanh(offset_mask[:, :, :, :2]) * self.offset_scale
        weight = self.center_prior.to(dtype=x.dtype, device=x.device) + offset_mask[:, :, :, 2]

        sampled = self._deformable_aggregate(value, raw_offset, weight)
        y = self.output_proj(sampled)
        return self.act(self.norm(y))

    def _deformable_aggregate(self, value: Tensor, offset: Tensor, weight: Tensor) -> Tensor:
        """Grouped unnormalized deformable aggregation.

        Shapes:
            value:  `[B,C,H,W]`
            offset: `[B,G,K,2,H,W]` in pixel units, ordered as dy/dx
            weight: `[B,G,K,H,W]`, intentionally not softmax-normalized
        """
        B, C, H, W = value.shape
        base_y, base_x = _base_coordinate_grid(H, W, value.device, value.dtype)
        kernel_offsets = _kernel_offsets(self.kernel_size, value.device, value.dtype)
        output_groups: list[Tensor] = []

        for group_idx in range(self.group):
            c0 = group_idx * self.group_channels
            c1 = c0 + self.group_channels
            group_value = value[:, c0:c1]
            group_out = torch.zeros_like(group_value)
            for kernel_idx in range(self.kernel_points):
                dy = offset[:, group_idx, kernel_idx, 0]
                dx = offset[:, group_idx, kernel_idx, 1]
                ky, kx = kernel_offsets[kernel_idx]
                sample_y = base_y.unsqueeze(0) + ky + dy
                sample_x = base_x.unsqueeze(0) + kx + dx
                grid_y = _normalize_grid_coordinate(sample_y, H)
                grid_x = _normalize_grid_coordinate(sample_x, W)
                grid = torch.stack((grid_x, grid_y), dim=-1)
                sampled = F.grid_sample(
                    group_value,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                group_out = group_out + sampled * weight[:, group_idx, kernel_idx].unsqueeze(1)
            output_groups.append(group_out)
        return torch.cat(output_groups, dim=1)


class _BandMLP(nn.Module):
    """Small shared MLP applied independently to each radial band."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        # features: [B, K, 3]
        B, K, D = features.shape
        raw = self.net(features.reshape(B * K, D))
        return raw.reshape(B, K)


class SSRBlockV3(nn.Module):
    """Selective Spectral Retention Block v3 with phase-2 stabilization."""

    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
        update_budget: float = 1.5,
        min_update: float = 0.08,
        noise_strength: float = 0.04,
        retain_floor: tuple[float, ...] | list[float] = (0.15, 0.18, 0.22, 0.28),
        suppress_min: tuple[float, ...] | list[float] = (0.00, 0.02, 0.03, 0.03),
        suppress_max: tuple[float, ...] | list[float] = (0.05, 0.15, 0.25, 0.25),
        update_target: tuple[float, ...] | list[float] = (0.30, 0.30, 0.25, 0.15),
        noise_aware_suppress: bool = True,
        use_bounded_gamma: bool = True,
        gamma_max: float = 0.25,
        gamma_init: float = -2.0,
        residual_gate_type: str = "se_update",
        se_reduction: int = 4,
        geometry_refine: str = "none",
        large_kernel_size: int = 7,
        dcnv4_group: int = 2,
        use_hf_ratio_penalty: bool = True,
        hf_ratio_threshold: float = 4.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.update_budget = float(update_budget)
        self.min_update = float(min_update)
        self.noise_strength = float(noise_strength)
        self.noise_aware_suppress = bool(noise_aware_suppress)
        self.use_bounded_gamma = bool(use_bounded_gamma)
        self.gamma_max = float(gamma_max)
        self.residual_gate_type = str(residual_gate_type)
        self.geometry_refine_type = str(geometry_refine)
        self.use_hf_ratio_penalty = bool(use_hf_ratio_penalty)
        self.hf_ratio_threshold = float(hf_ratio_threshold)

        for name, values in (
            ("retain_floor", retain_floor),
            ("suppress_min", suppress_min),
            ("suppress_max", suppress_max),
            ("update_target", update_target),
        ):
            if len(values) != self.num_bands:
                raise ValueError(f"{name} length must match num_bands")

        self.retain_mlp = _BandMLP()
        self.suppress_mlp = _BandMLP()
        self.update_mlp = _BandMLP()

        self.delta_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.local_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        if self.use_bounded_gamma:
            self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))
            self.gamma = None
        else:
            self.gamma = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
            self.gamma_raw = None

        if self.residual_gate_type == "none":
            self.se_update = None
            self.residual_channel_gate = None
        elif self.residual_gate_type == "se_update":
            self.se_update = SEBlock(channels, reduction=se_reduction)
            self.residual_channel_gate = None
        elif self.residual_gate_type == "residual_channel_gate":
            self.se_update = None
            self.residual_channel_gate = ResidualChannelGate(channels, reduction=se_reduction)
        else:
            raise ValueError(
                "residual_gate_type must be one of: none, se_update, residual_channel_gate"
            )

        if self.geometry_refine_type == "none":
            self.geometry_refine = nn.Identity()
        elif self.geometry_refine_type == "large_kernel":
            self.geometry_refine = LargeKernelRefine(channels, kernel_size=large_kernel_size)
        elif self.geometry_refine_type == "deformable":
            self.geometry_refine = DeformableRefine(channels, kernel_size=3)
        elif self.geometry_refine_type == "dcnv4":
            self.geometry_refine = DCNv4Refine(channels, kernel_size=3, group=dcnv4_group)
        else:
            raise ValueError("geometry_refine must be one of: none, large_kernel, deformable, dcnv4")

        self.register_buffer(
            "retain_floor",
            torch.tensor(retain_floor, dtype=torch.float32).view(1, self.num_bands),
            persistent=False,
        )
        self.register_buffer(
            "suppress_min",
            torch.tensor(suppress_min, dtype=torch.float32).view(1, self.num_bands),
            persistent=False,
        )
        self.register_buffer(
            "suppress_max",
            torch.tensor(suppress_max, dtype=torch.float32).view(1, self.num_bands),
            persistent=False,
        )
        target = torch.tensor(update_target, dtype=torch.float32)
        target = target / target.sum().clamp_min(1e-8)
        self.register_buffer("update_target", target.view(1, self.num_bands), persistent=False)

    def forward(
        self,
        x: Tensor,
        boundary_mask: Tensor | None = None,
        return_logs: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, dict[str, Any], Tensor, Tensor]:
        """Run SSRBlockV3.

        Args:
            x: Input tensor `[B, C, H, W]`.
            boundary_mask: Optional boundary mask `[B, 1, H, W]`.
            return_logs: Return detached spectral/gate diagnostics.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {C}")

        eps = 1e-8
        x_float = x.float()
        X = torch.fft.rfft2(x_float, norm="ortho")
        Wf = X.shape[-1]
        masks = build_radial_frequency_masks(H, W, self.num_bands, x.device).to(X.real.dtype)
        if masks.shape[-1] != Wf:
            raise RuntimeError("Internal radial mask shape mismatch for rfft2 spectrum.")

        mask_bc = masks.view(1, self.num_bands, 1, H, Wf)
        abs_x = X.abs()
        power = abs_x.square()
        denom = masks.sum(dim=(1, 2)).clamp_min(1.0).view(1, self.num_bands) * C

        energy = (power.unsqueeze(1) * mask_bc).sum(dim=(2, 3, 4)) / denom
        mean_mag = (abs_x.unsqueeze(1) * mask_bc).sum(dim=(2, 3, 4)) / denom
        variance = (
            (abs_x.unsqueeze(1) - mean_mag.view(B, self.num_bands, 1, 1, 1)).square()
            * mask_bc
        ).sum(dim=(2, 3, 4)) / denom

        phase = torch.angle(X)
        cos_mean = (torch.cos(phase).unsqueeze(1) * mask_bc).sum(dim=(2, 3, 4)) / denom
        sin_mean = (torch.sin(phase).unsqueeze(1) * mask_bc).sum(dim=(2, 3, 4)) / denom
        phase_coherence = torch.sqrt(cos_mean.square() + sin_mean.square()).clamp(0.0, 1.0)

        features = torch.stack(
            [torch.log(energy + eps), torch.log(variance + eps), phase_coherence],
            dim=-1,
        )
        raw_retain = self.retain_mlp(features)
        raw_suppress = self.suppress_mlp(features)
        raw_update = self.update_mlp(features)

        retain_gate = self.retain_floor + (1.0 - self.retain_floor) * torch.sigmoid(raw_retain)
        remaining_budget = max(self.update_budget - self.num_bands * self.min_update, 0.0)
        update_gate = self.min_update + remaining_budget * torch.softmax(raw_update, dim=1)

        base_suppress = torch.sigmoid(raw_suppress)
        if self.noise_aware_suppress:
            energy_norm = energy / energy.mean(dim=1, keepdim=True).clamp_min(eps)
            noise_score = (energy_norm * (1.0 - phase_coherence)).clamp(0.0, 2.0) / 2.0
            noise_score = 0.25 + 0.75 * noise_score
            suppress_gate = self.suppress_min + (
                self.suppress_max - self.suppress_min
            ) * base_suppress * noise_score
        else:
            suppress_gate = self.suppress_min + (
                self.suppress_max - self.suppress_min
            ) * base_suppress

        gate_reg = F.mse_loss(
            update_gate.mean(dim=0),
            (self.update_target.squeeze(0) * self.update_budget).to(update_gate.device),
        )

        band_outputs = []
        input_energy = []
        output_energy = []
        retain_contrib = []
        update_contrib = []
        suppress_contrib = []
        high_band_energy_map = None
        in_high = None
        out_high = None

        for band in range(self.num_bands):
            mask = masks[band].view(1, 1, H, Wf)
            xk = torch.fft.irfft2(X * mask, s=(H, W), norm="ortho")
            delta = self.delta_net(xk)
            noise_like = xk - F.avg_pool2d(xk, kernel_size=3, stride=1, padding=1)

            retain = retain_gate[:, band].view(B, 1, 1, 1) * xk
            update = update_gate[:, band].view(B, 1, 1, 1) * delta
            suppress = self.noise_strength * suppress_gate[:, band].view(B, 1, 1, 1) * noise_like
            out_k = retain + update - suppress
            band_outputs.append(out_k)

            if band == self.num_bands - 1:
                in_high = xk.square().mean(dim=(1, 2, 3))
                out_high = out_k.square().mean(dim=(1, 2, 3))
                high_band_energy_map = xk.square().mean(dim=1, keepdim=True)

            if return_logs:
                input_energy.append(xk.square().mean(dim=(1, 2, 3)))
                output_energy.append(out_k.square().mean(dim=(1, 2, 3)))
                retain_contrib.append(retain.abs().mean(dim=(1, 2, 3)))
                update_contrib.append(update.abs().mean(dim=(1, 2, 3)))
                suppress_contrib.append(suppress.abs().mean(dim=(1, 2, 3)))

        if in_high is None or out_high is None:
            raise RuntimeError("High-frequency band was not computed.")
        hf_ratio = out_high / in_high.clamp_min(eps)
        if self.use_hf_ratio_penalty:
            hf_ratio_penalty = F.relu(hf_ratio - self.hf_ratio_threshold).square().mean()
        else:
            hf_ratio_penalty = torch.zeros((), device=x.device, dtype=x_float.dtype)

        spectral_out = torch.stack(band_outputs, dim=0).sum(dim=0)
        spectral_out = self.local_refine(spectral_out)
        gated_update, residual_gate = self._apply_residual_gate(x_float, spectral_out)
        gated_update = self.geometry_refine(gated_update)
        gamma = self._gamma_value().to(x_float.dtype)
        y = x_float + gamma * gated_update
        y = y.to(x.dtype)

        if not return_logs:
            return y, gate_reg, hf_ratio_penalty

        logs = self._build_logs(
            retain_gate=retain_gate,
            suppress_gate=suppress_gate,
            update_gate=update_gate,
            variance=variance,
            phase_coherence=phase_coherence,
            input_energy=torch.stack(input_energy, dim=1),
            output_energy=torch.stack(output_energy, dim=1),
            retain_contrib=torch.stack(retain_contrib, dim=1),
            update_contrib=torch.stack(update_contrib, dim=1),
            suppress_contrib=torch.stack(suppress_contrib, dim=1),
            high_band_energy_map=high_band_energy_map,
            boundary_mask=boundary_mask,
            gate_reg=gate_reg,
            hf_ratio=hf_ratio,
            hf_ratio_penalty=hf_ratio_penalty,
            gamma=gamma,
            residual_gate=residual_gate,
        )
        return y, logs, gate_reg, hf_ratio_penalty

    def _gamma_value(self) -> Tensor:
        if self.use_bounded_gamma:
            assert self.gamma_raw is not None
            return self.gamma_max * torch.sigmoid(self.gamma_raw)
        assert self.gamma is not None
        return self.gamma

    def _apply_residual_gate(self, x: Tensor, delta: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.residual_gate_type == "none":
            return delta, None
        if self.residual_gate_type == "se_update":
            assert self.se_update is not None
            gate = self.se_update.gate(delta)
            return delta * gate, gate
        assert self.residual_channel_gate is not None
        gate = self.residual_channel_gate.gate(x, delta)
        return delta * gate, gate

    def _build_logs(
        self,
        *,
        retain_gate: Tensor,
        suppress_gate: Tensor,
        update_gate: Tensor,
        variance: Tensor,
        phase_coherence: Tensor,
        input_energy: Tensor,
        output_energy: Tensor,
        retain_contrib: Tensor,
        update_contrib: Tensor,
        suppress_contrib: Tensor,
        high_band_energy_map: Tensor | None,
        boundary_mask: Tensor | None,
        gate_reg: Tensor,
        hf_ratio: Tensor,
        hf_ratio_penalty: Tensor,
        gamma: Tensor,
        residual_gate: Tensor | None,
    ) -> dict[str, Any]:
        """Build detached scalar/list logs for CSV serialization."""
        eps = 1e-8

        def per_band_mean(t: Tensor) -> list[float]:
            return [float(v) for v in t.detach().mean(dim=0).cpu()]

        def per_band_std(t: Tensor) -> list[float]:
            return [float(v) for v in t.detach().std(dim=0, unbiased=False).cpu()]

        logs: dict[str, Any] = {
            "retain_gate_mean": per_band_mean(retain_gate),
            "retain_gate_std": per_band_std(retain_gate),
            "suppress_gate_mean": per_band_mean(suppress_gate),
            "suppress_gate_std": per_band_std(suppress_gate),
            "update_gate_mean": per_band_mean(update_gate),
            "update_gate_std": per_band_std(update_gate),
            "input_energy": per_band_mean(input_energy),
            "output_energy": per_band_mean(output_energy),
            "phase_coherence": per_band_mean(phase_coherence),
            "variance": per_band_mean(variance),
            "retain_contribution": per_band_mean(retain_contrib),
            "update_contribution": per_band_mean(update_contrib),
            "suppress_contribution": per_band_mean(suppress_contrib),
            "high_freq_ratio": float(hf_ratio.detach().mean().cpu()),
            "high_freq_penalty": float(hf_ratio_penalty.detach().cpu()),
            "update_budget_sum": float(update_gate.detach().sum(dim=1).mean().cpu()),
            "gate_reg": float(gate_reg.detach().cpu()),
            "gamma": float(gamma.detach().cpu()),
        }
        if residual_gate is not None:
            gate_flat = residual_gate.detach().flatten(1)
            logs["residual_gate_mean"] = float(gate_flat.mean().cpu())
            logs["residual_gate_std"] = float(gate_flat.std(unbiased=False).cpu())

        boundary_density = torch.tensor(0.0, device=input_energy.device)
        nonboundary_density = torch.tensor(0.0, device=input_energy.device)
        ratio = torch.tensor(0.0, device=input_energy.device)
        if boundary_mask is not None and high_band_energy_map is not None:
            bmask = boundary_mask.float()
            if bmask.ndim == 3:
                bmask = bmask.unsqueeze(1)
            if bmask.shape[-2:] != high_band_energy_map.shape[-2:]:
                bmask = F.interpolate(bmask, size=high_band_energy_map.shape[-2:], mode="nearest")
            bmask = (bmask > 0).float()
            non = 1.0 - bmask
            boundary_area = bmask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            nonboundary_area = non.sum(dim=(1, 2, 3)).clamp_min(1.0)
            boundary_density = (high_band_energy_map * bmask).sum(dim=(1, 2, 3)) / boundary_area
            nonboundary_density = (high_band_energy_map * non).sum(dim=(1, 2, 3)) / nonboundary_area
            ratio = boundary_density / nonboundary_density.clamp_min(eps)

        logs["boundary_high_density"] = float(boundary_density.detach().mean().cpu())
        logs["nonboundary_high_density"] = float(nonboundary_density.detach().mean().cpu())
        logs["boundary_to_nonboundary_high_ratio"] = float(ratio.detach().mean().cpu())
        return logs


def _num_groups(channels: int) -> int:
    """Choose a stable GroupNorm group count for small debug models."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _valid_group(channels: int, requested: int) -> int:
    for group in (requested, 4, 2, 1):
        if group > 0 and channels % group == 0 and (channels // group) % 16 == 0:
            return group
    raise ValueError(
        "DCNv4-style refine follows the official channels/group divisibility. "
        f"Got channels={channels}, requested group={requested}."
    )


def _base_coordinate_grid(
    H: int,
    W: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy, xx


def _kernel_offsets(kernel_size: int, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(kernel_size * kernel_size, 2)


def _normalize_grid_coordinate(coord: Tensor, size: int) -> Tensor:
    if size <= 1:
        return torch.zeros_like(coord)
    return coord * (2.0 / float(size - 1)) - 1.0
