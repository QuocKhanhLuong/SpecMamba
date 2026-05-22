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
    max_radius = radius.max().clamp_min(1e-8)
    radius = radius / max_radius

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
    """Create a 1-pixel-thick multiclass boundary map from labels.

    Args:
        mask: Tensor shaped `[B, H, W]` or `[B, 1, H, W]`.

    Returns:
        Float tensor shaped `[B, 1, H, W]`.
    """
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
    """Selective Spectral Retention Block v3.

    The block splits an `rfft2` spectrum into radial bands, computes band-wise
    spectral state features, and applies separate retain, suppress, and update
    gates before a small local refinement and residual update.
    """

    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
        update_budget: float = 1.5,
        min_update: float = 0.08,
        noise_strength: float = 0.04,
        retain_floor: tuple[float, ...] | list[float] = (0.15, 0.18, 0.22, 0.28),
        suppress_max: tuple[float, ...] | list[float] = (0.10, 0.30, 0.50, 0.50),
        update_target: tuple[float, ...] | list[float] = (0.30, 0.30, 0.25, 0.15),
        noise_aware_suppress: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.update_budget = float(update_budget)
        self.min_update = float(min_update)
        self.noise_strength = float(noise_strength)
        self.noise_aware_suppress = bool(noise_aware_suppress)

        if len(retain_floor) != self.num_bands:
            raise ValueError("retain_floor length must match num_bands")
        if len(suppress_max) != self.num_bands:
            raise ValueError("suppress_max length must match num_bands")
        if len(update_target) != self.num_bands:
            raise ValueError("update_target length must match num_bands")

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
        self.gamma = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        self.register_buffer(
            "retain_floor",
            torch.tensor(retain_floor, dtype=torch.float32).view(1, self.num_bands),
            persistent=False,
        )
        self.register_buffer(
            "suppress_max",
            torch.tensor(suppress_max, dtype=torch.float32).view(1, self.num_bands),
            persistent=False,
        )
        target = torch.tensor(update_target, dtype=torch.float32)
        target = target / target.sum().clamp_min(1e-8)
        self.register_buffer(
            "update_target",
            target.view(1, self.num_bands),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        boundary_mask: Tensor | None = None,
        return_logs: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Any], Tensor]:
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
        phase_denom = denom
        cos_mean = (torch.cos(phase).unsqueeze(1) * mask_bc).sum(dim=(2, 3, 4)) / phase_denom
        sin_mean = (torch.sin(phase).unsqueeze(1) * mask_bc).sum(dim=(2, 3, 4)) / phase_denom
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

        energy_norm = energy / energy.mean(dim=1, keepdim=True).clamp_min(eps)
        noise_score = (energy_norm * (1.0 - phase_coherence)).clamp(0.0, 2.0) / 2.0
        if not self.noise_aware_suppress:
            noise_score = torch.ones_like(noise_score)
        suppress_gate = self.suppress_max * torch.sigmoid(raw_suppress) * noise_score

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

        for band in range(self.num_bands):
            mask = masks[band].view(1, 1, H, Wf)
            xk = torch.fft.irfft2(X * mask, s=(H, W), norm="ortho")
            delta = self.delta_net(xk)
            noise_like = xk - F.avg_pool2d(xk, kernel_size=3, stride=1, padding=1)

            retain = retain_gate[:, band].view(B, 1, 1, 1) * xk
            update = update_gate[:, band].view(B, 1, 1, 1) * delta
            suppress = (
                self.noise_strength
                * suppress_gate[:, band].view(B, 1, 1, 1)
                * noise_like
            )
            out_k = retain + update - suppress
            band_outputs.append(out_k)

            if return_logs:
                input_energy.append(xk.square().mean(dim=(1, 2, 3)))
                output_energy.append(out_k.square().mean(dim=(1, 2, 3)))
                retain_contrib.append(retain.abs().mean(dim=(1, 2, 3)))
                update_contrib.append(update.abs().mean(dim=(1, 2, 3)))
                suppress_contrib.append(suppress.abs().mean(dim=(1, 2, 3)))
                if band == self.num_bands - 1:
                    high_band_energy_map = xk.square().mean(dim=1, keepdim=True)

        spectral_out = torch.stack(band_outputs, dim=0).sum(dim=0)
        spectral_out = self.local_refine(spectral_out)
        y = x + self.gamma.to(x.dtype) * spectral_out.to(x.dtype)

        if not return_logs:
            return y, gate_reg

        logs = self._build_logs(
            retain_gate=retain_gate,
            suppress_gate=suppress_gate,
            update_gate=update_gate,
            energy=energy,
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
        )
        return y, logs, gate_reg

    def _build_logs(
        self,
        *,
        retain_gate: Tensor,
        suppress_gate: Tensor,
        update_gate: Tensor,
        energy: Tensor,
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
            "update_budget_sum": float(update_gate.detach().sum(dim=1).mean().cpu()),
            "gate_reg": float(gate_reg.detach().cpu()),
            "gamma": float(self.gamma.detach().cpu()),
        }

        in_high = input_energy[:, -1].mean()
        out_high = output_energy[:, -1].mean()
        logs["high_freq_ratio"] = float((out_high / in_high.clamp_min(eps)).detach().cpu())

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
