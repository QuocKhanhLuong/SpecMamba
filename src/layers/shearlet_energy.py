"""
Shearlet Directional Energy Module

Implements a digital Shearlet transform to analyze images across multiple
directions and scales. Produces:
  - 8 directional energy maps (energy in each shearlet orientation)
  - 1 curvature entropy map (Shannon entropy of directional distribution)

The curvature entropy map serves as a proxy for local boundary curvature:
  - Low entropy = straight edge (energy concentrated in one direction)
  - High entropy = curved edge (energy spread across directions)
  → high-entropy regions are where HD95 fails most

Reference:
  Labate, D. et al. (2005). "Sparse Multidimensional Representation
  using Shearlets." SPIE Wavelets XI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ShearletFilterBank(nn.Module):
    """
    Generates a bank of digital Shearlet filters in frequency domain.

    Shearlet filters are constructed as the product of:
      - Radial (scale) component: band-pass filter selecting a frequency band
      - Angular (shear) component: directional filter selecting an orientation

    Args:
        n_scales: Number of decomposition scales (default: 4)
        n_orientations: Number of orientations per scale (default: 8)
    """

    def __init__(self, n_scales=4, n_orientations=8):
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations

        # Precompute orientation angles
        angles = torch.linspace(0, math.pi * (1 - 1 / n_orientations),
                                n_orientations)
        self.register_buffer('angles', angles)

    def _meyer_window(self, x):
        """Smooth bump function for frequency band selection."""
        # Polynomial auxiliary function for Meyer wavelet construction
        y = torch.zeros_like(x)
        mask = (x > 0) & (x < 1)
        t = x[mask]
        y[mask] = t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)
        y[x >= 1] = 1.0
        return y

    def _build_filters(self, H, W, device):
        """
        Build Shearlet filter bank for given spatial dimensions.

        Returns: [n_scales, n_orientations, H, W] real-valued filters
        """
        # Frequency grid (centered, normalized to [-0.5, 0.5])
        u = torch.linspace(-0.5, 0.5, W, device=device)
        v = torch.linspace(-0.5, 0.5, H, device=device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')

        # Radius and angle in frequency domain
        radius = torch.sqrt(grid_u ** 2 + grid_v ** 2).clamp(min=1e-8)
        theta = torch.atan2(grid_v, grid_u)

        filters = []

        for s in range(self.n_scales):
            # Band-pass radial component: frequency band for scale s
            # Scale factor: higher scale → lower frequency
            scale_factor = 2.0 ** (self.n_scales - 1 - s)
            r_scaled = radius * scale_factor * 2  # Normalize to [0, 2] range

            # Radial window: smooth band-pass using Meyer-like construction
            radial = self._meyer_window(r_scaled) * (1 - self._meyer_window(r_scaled / 2))

            scale_filters = []
            for o in range(self.n_orientations):
                angle = self.angles[o]
                # Angular difference (wrapped to [-pi/2, pi/2])
                d_theta = theta - angle
                d_theta = torch.atan2(torch.sin(d_theta), torch.cos(d_theta))

                # Angular window: Gaussian-like windowing
                angular_sigma = math.pi / (self.n_orientations * 1.2)
                angular = torch.exp(-0.5 * (d_theta / angular_sigma) ** 2)

                # Also include the opposite direction (π rotation)
                d_theta_opp = theta - (angle + math.pi)
                d_theta_opp = torch.atan2(torch.sin(d_theta_opp),
                                          torch.cos(d_theta_opp))
                angular_opp = torch.exp(-0.5 * (d_theta_opp / angular_sigma) ** 2)

                angular_combined = angular + angular_opp

                # Combined filter
                filt = radial * angular_combined
                # Normalize
                filt = filt / (filt.max() + 1e-8)
                scale_filters.append(filt)

            filters.append(torch.stack(scale_filters, dim=0))

        return torch.stack(filters, dim=0)  # [n_scales, n_orient, H, W]


class ShearletEnergyModule(nn.Module):
    """
    Computes directional energy maps and curvature entropy from Shearlet transform.

    Args:
        n_scales: Number of scales (default: 4)
        n_orientations: Number of orientations (default: 8)
        epsilon: Numerical stability constant (default: 1e-8)

    Input:  [B, C, H, W]  (if C > 1, converts to grayscale)
    Output: [B, 9, H, W]  = 8 directional energy maps + 1 curvature entropy map
    """

    def __init__(self, n_scales=4, n_orientations=8, epsilon=1e-8):
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.epsilon = epsilon

        self.filter_bank = ShearletFilterBank(
            n_scales=n_scales,
            n_orientations=n_orientations
        )

        # Cache
        self._cached_filters = None
        self._cached_size = None

    def _to_grayscale(self, x):
        """Convert multi-channel input to single channel."""
        if x.shape[1] == 1:
            return x
        if x.shape[1] == 3:
            weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                   device=x.device, dtype=x.dtype)
            return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        return x.mean(dim=1, keepdim=True)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            output: [B, 9, H, W] where:
                    channels 0-7 = directional energy (one per orientation)
                    channel 8 = curvature entropy map
        """
        B, C, H, W = x.shape

        # Convert to grayscale
        gray = self._to_grayscale(x)  # [B, 1, H, W]

        # Build or reuse filter bank
        if self._cached_size != (H, W):
            self._cached_filters = self.filter_bank._build_filters(H, W, x.device)
            self._cached_size = (H, W)
        filters = self._cached_filters  # [n_scales, n_orient, H, W]

        # FFT of input (centered)
        img_fft = torch.fft.fftshift(
            torch.fft.fft2(gray.squeeze(1)),  # [B, H, W]
            dim=(-2, -1)
        )

        # Compute energy per orientation (summed across scales)
        dir_energy = torch.zeros(B, self.n_orientations, H, W,
                                 device=x.device, dtype=x.dtype)

        for s in range(self.n_scales):
            for o in range(self.n_orientations):
                filt = filters[s, o]  # [H, W]
                # Apply filter in frequency domain
                filtered = img_fft * filt.unsqueeze(0)  # [B, H, W]
                # Back to spatial domain — take magnitude squared (energy)
                response = torch.fft.ifft2(
                    torch.fft.ifftshift(filtered, dim=(-2, -1))
                )
                energy = response.real ** 2 + response.imag ** 2
                dir_energy[:, o] += energy

        # Normalize to probability distribution over directions (per pixel)
        energy_sum = dir_energy.sum(dim=1, keepdim=True) + self.epsilon
        p = dir_energy / energy_sum  # [B, n_orient, H, W]

        # Shannon entropy of directional distribution (curvature proxy)
        # High entropy = energy spread across directions = high curvature
        log_p = torch.log(p + self.epsilon)
        entropy = -(p * log_p).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Normalize entropy to [0, 1] range
        max_entropy = math.log(self.n_orientations)  # Maximum possible entropy
        entropy = entropy / (max_entropy + self.epsilon)
        entropy = entropy.clamp(0, 1)

        # Normalize directional energy maps to [0, 1] per batch
        dir_max = dir_energy.amax(dim=(-2, -1), keepdim=True) + self.epsilon
        dir_energy_norm = dir_energy / dir_max

        # Output: [B, 9, H, W] = 8 directional + 1 curvature
        return torch.cat([dir_energy_norm, entropy], dim=1)


if __name__ == "__main__":
    # Quick test
    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W)
    module = ShearletEnergyModule()
    out = module(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Directional energy range: [{out[:, :8].min():.4f}, {out[:, :8].max():.4f}]")
    print(f"Curvature entropy range:  [{out[:, 8:].min():.4f}, {out[:, 8:].max():.4f}]")
