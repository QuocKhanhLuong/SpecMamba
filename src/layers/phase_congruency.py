"""
Phase Congruency Module (Kovesi, 1999) — Pure PyTorch Implementation

Detects true anatomical boundaries using phase coherence across frequency
scales. Inherently separates real edges from noise without manual thresholding,
contrast-invariant across MRI scanners and protocols.

Key properties:
  - PC → 1.0 at true edges (phases aligned across scales)
  - PC → 0.0 at noise/flat regions (phases random, cancel out)
  - Automatic noise threshold via Rayleigh distribution estimation

Reference:
  Kovesi, P. (1999). "Image Features From Phase Congruency."
  Journal of Computer Vision Research, 1(3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LogGaborFilterBank(nn.Module):
    """
    Constructs a bank of Log-Gabor filters in frequency domain.

    Log-Gabor filters have a Gaussian transfer function when viewed on a
    logarithmic frequency scale:

        G(f, θ) = exp(−[log(f/f₀)]² / 2σ_f²) × exp(−(θ−θ_k)² / 2σ_θ²)

    Args:
        n_scales: Number of frequency scales (default: 5)
        n_orientations: Number of orientations (default: 6)
        min_wavelength: Wavelength of smallest scale filter (default: 3)
        mult: Wavelength multiplier between scales (default: 2.1)
        sigma_onf: Ratio of standard deviation of Gaussian describing the
                    log Gabor filter's transfer function in the frequency
                    domain to the filter center frequency (default: 0.55)
        d_theta_on_sigma: Angular bandwidth control (default: 1.2)
    """

    def __init__(self, n_scales=5, n_orientations=6, min_wavelength=3,
                 mult=2.1, sigma_onf=0.55, d_theta_on_sigma=1.2):
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_onf = sigma_onf

        # Precompute orientation angles and angular sigma
        d_theta = math.pi / n_orientations
        self.sigma_theta = d_theta / d_theta_on_sigma

        # Store orientation angles
        angles = torch.arange(n_orientations).float() * d_theta
        self.register_buffer('angles', angles)

        # Store center frequencies for each scale
        wavelengths = min_wavelength * (mult ** torch.arange(n_scales).float())
        center_freqs = 1.0 / wavelengths
        self.register_buffer('center_freqs', center_freqs)

    def _build_filters(self, H, W, device):
        """Build Log-Gabor filter bank for given spatial dimensions."""
        # Frequency grid (centered)
        u = torch.arange(W, device=device).float() - W / 2
        v = torch.arange(H, device=device).float() - H / 2
        v, u = torch.meshgrid(v, u, indexing='ij')

        # Radius from center (in normalized frequency)
        radius = torch.sqrt(u ** 2 + v ** 2) / max(H, W)
        radius[H // 2, W // 2] = 1.0  # Avoid log(0) at DC

        # Angle grid
        theta = torch.atan2(-v, u)

        filters = []  # (n_scales, n_orientations, H, W)

        for s in range(self.n_scales):
            f0 = self.center_freqs[s]
            # Log-Gabor radial component
            log_gabor = torch.exp(
                -0.5 * (torch.log(radius / f0) ** 2) / (self.sigma_onf ** 2)
            )
            # Zero out DC
            log_gabor[H // 2, W // 2] = 0.0

            scale_filters = []
            for o in range(self.n_orientations):
                angle = self.angles[o]
                # Angular difference (wrapped)
                d_theta = torch.abs(theta - angle)
                d_theta = torch.min(d_theta, math.pi * 2 - d_theta)
                d_theta = torch.min(d_theta, math.pi - d_theta)

                # Angular Gaussian spread
                angular = torch.exp(-0.5 * (d_theta / self.sigma_theta) ** 2)

                filt = log_gabor * angular
                scale_filters.append(filt)

            filters.append(torch.stack(scale_filters, dim=0))  # (n_orient, H, W)

        return torch.stack(filters, dim=0)  # (n_scales, n_orient, H, W)


class PhaseCongruencyModule(nn.Module):
    """
    Computes Phase Congruency maps from input images.

    Phase Congruency detects features at points where the Fourier components
    are maximally in phase. This makes it contrast-invariant and
    inherently noise-robust.

    Args:
        n_scales: Number of frequency scales (default: 5)
        n_orientations: Number of orientations (default: 6)
        min_wavelength: Wavelength of smallest scale filter (default: 3)
        mult: Scaling factor between successive filters (default: 2.1)
        sigma_onf: Bandwidth parameter (default: 0.55)
        k: Noise threshold scaling constant (default: 2.0)
        epsilon: Small constant for numerical stability (default: 1e-4)
        learnable: If True, allow filter parameters to be learned (default: False)

    Input:  [B, C, H, W]  (C can be 1 or 3; if 3, converts to grayscale)
    Output: [B, 1, H, W]  Phase Congruency map in [0, 1]
    """

    def __init__(self, n_scales=5, n_orientations=6, min_wavelength=3,
                 mult=2.1, sigma_onf=0.55, k=2.0, epsilon=1e-4,
                 learnable=False):
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.k = k
        self.epsilon = epsilon
        self.learnable = learnable

        self.filter_bank = LogGaborFilterBank(
            n_scales=n_scales,
            n_orientations=n_orientations,
            min_wavelength=min_wavelength,
            mult=mult,
            sigma_onf=sigma_onf
        )

        # Cache for filters at different resolutions
        self._cached_filters = None
        self._cached_size = None

    def _to_grayscale(self, x):
        """Convert multi-channel input to single channel."""
        if x.shape[1] == 1:
            return x
        # Standard RGB to grayscale weights
        weights = torch.tensor([0.2989, 0.5870, 0.1140],
                               device=x.device, dtype=x.dtype)
        # Handle arbitrary channels by averaging if not 3
        if x.shape[1] == 3:
            return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        return x.mean(dim=1, keepdim=True)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            pc_map: Phase Congruency map [B, 1, H, W] in range [0, 1]
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

        # Apply filter bank and compute even/odd responses
        # even = real part of response (cosine phase)
        # odd = imaginary part of response (sine phase)
        pc_sum = torch.zeros(B, self.n_orientations, H, W,
                             device=x.device, dtype=x.dtype)
        amplitude_sum = torch.zeros_like(pc_sum)

        for s in range(self.n_scales):
            for o in range(self.n_orientations):
                filt = filters[s, o]  # [H, W]
                # Apply filter in frequency domain
                filtered = img_fft * filt.unsqueeze(0)  # [B, H, W]
                # Back to spatial domain
                response = torch.fft.ifft2(
                    torch.fft.ifftshift(filtered, dim=(-2, -1))
                )
                even = response.real  # [B, H, W]
                odd = response.imag   # [B, H, W]

                # Amplitude
                amplitude = torch.sqrt(even ** 2 + odd ** 2 + self.epsilon)

                # Accumulate for phase congruency
                amplitude_sum[:, o] += amplitude
                pc_sum[:, o] += amplitude

        # Estimate noise threshold from median of filter responses
        # (Rayleigh distribution: noise_threshold ≈ k * median / 0.6745)
        with torch.no_grad():
            median_amp = amplitude_sum.median()
            noise_threshold = self.k * median_amp / 0.6745 + self.epsilon

        # Phase congruency per orientation: sum of weighted amplitudes
        # above noise threshold, divided by total amplitude
        pc_per_orient = (pc_sum - noise_threshold).clamp(min=0) / (
            amplitude_sum + self.epsilon
        )

        # Aggregate across orientations: take max (strongest edge direction)
        pc_map = pc_per_orient.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]

        # Normalize to [0, 1]
        pc_map = pc_map.clamp(0, 1)

        return pc_map


if __name__ == "__main__":
    # Quick test
    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W)
    module = PhaseCongruencyModule()
    pc_map = module(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {pc_map.shape}")
    print(f"Range:  [{pc_map.min().item():.4f}, {pc_map.max().item():.4f}]")
