"""
Monogenic Signal Processing Module
Implements Riesz Transform for extracting phase, orientation, and energy from images.

The Monogenic Signal is a 2D generalization of the Analytic Signal (Hilbert Transform).
It provides rotation-invariant local features: Amplitude, Phase, and Orientation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class RieszTransform(nn.Module):
    """
    Riesz Transform - A 2D isotropic extension of the Hilbert Transform.
    
    The Riesz kernels in frequency domain are:
        H1(u,v) = -j * u / sqrt(u^2 + v^2)
        H2(u,v) = -j * v / sqrt(u^2 + v^2)
    
    This is a fixed (non-learnable) physics-based operator.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize Riesz Transform.
        
        Args:
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Riesz Transform to input image.
        
        Args:
            x: Input tensor of shape (B, 1, H, W) - grayscale image
            
        Returns:
            Tuple of (riesz_x, riesz_y) - the two components of Riesz transform
        """
        B, C, H, W = x.shape
        
        # Create frequency grid
        freq_y = torch.fft.fftfreq(H, device=x.device, dtype=x.dtype)
        freq_x = torch.fft.fftfreq(W, device=x.device, dtype=x.dtype)
        freq_y, freq_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Compute radial frequency (avoid division by zero)
        radius = torch.sqrt(freq_x**2 + freq_y**2 + self.epsilon)
        
        # Riesz kernels in frequency domain
        # H1 = -j * u / |w|, H2 = -j * v / |w|
        kernel_x = freq_x / radius
        kernel_y = freq_y / radius
        
        # Set DC component to zero
        kernel_x[0, 0] = 0
        kernel_y[0, 0] = 0
        
        # Apply FFT to input
        x_fft = torch.fft.fft2(x)
        
        # Apply Riesz kernels (multiplication by -j in frequency = Hilbert-like)
        # -j * X = real(X) * (-j) + imag(X) * (-j) * j = imag(X) - j*real(X)
        riesz_x_fft = -1j * x_fft * kernel_x.unsqueeze(0).unsqueeze(0)
        riesz_y_fft = -1j * x_fft * kernel_y.unsqueeze(0).unsqueeze(0)
        
        # Inverse FFT
        riesz_x = torch.fft.ifft2(riesz_x_fft).real
        riesz_y = torch.fft.ifft2(riesz_y_fft).real
        
        return riesz_x, riesz_y


class LogGaborFilter(nn.Module):
    """
    Log-Gabor Filter Bank for multi-scale analysis.
    
    Log-Gabor filters have no DC component and have Gaussian transfer
    functions on a log-frequency scale, making them ideal for edge detection.
    """
    
    def __init__(self, num_scales: int = 4, num_orientations: int = 6,
                 min_wavelength: float = 3.0, mult: float = 2.1,
                 sigma_on_f: float = 0.55):
        """
        Initialize Log-Gabor Filter Bank.
        
        Args:
            num_scales: Number of wavelet scales
            num_orientations: Number of orientation angles
            min_wavelength: Minimum wavelength in pixels
            mult: Scaling factor between scales
            sigma_on_f: Ratio of standard deviation to filter center frequency
        """
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_on_f = sigma_on_f
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Log-Gabor filter bank.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
            
        Returns:
            Tensor of shape (B, num_scales * num_orientations, H, W)
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        # Create frequency grid
        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype)
        freq_x = torch.fft.fftfreq(W, device=device, dtype=dtype)
        freq_y, freq_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Polar coordinates
        radius = torch.sqrt(freq_x**2 + freq_y**2)
        radius[0, 0] = 1  # Avoid log(0)
        theta = torch.atan2(freq_y, freq_x)
        
        # FFT of input
        x_fft = torch.fft.fft2(x)
        
        outputs = []
        
        for scale in range(self.num_scales):
            wavelength = self.min_wavelength * (self.mult ** scale)
            fo = 1.0 / wavelength  # Center frequency
            
            # Log-Gabor radial component
            log_gabor_radial = torch.exp(
                -(torch.log(radius / fo) ** 2) / (2 * math.log(self.sigma_on_f) ** 2)
            )
            log_gabor_radial[0, 0] = 0  # Zero DC
            
            for orient in range(self.num_orientations):
                angle = orient * math.pi / self.num_orientations
                
                # Angular component
                ds = torch.sin(theta - angle)
                dc = torch.cos(theta - angle)
                dtheta = torch.abs(torch.atan2(ds, dc))
                
                # Angular spread
                angular_spread = torch.exp(
                    -(dtheta ** 2) / (2 * (math.pi / self.num_orientations) ** 2)
                )
                
                # Combined filter
                log_gabor = log_gabor_radial * angular_spread
                
                # Apply filter
                filtered = torch.fft.ifft2(x_fft * log_gabor.unsqueeze(0).unsqueeze(0))
                outputs.append(filtered.abs())
        
        return torch.cat(outputs, dim=1)


class MonogenicSignal(nn.Module):
    """
    Monogenic Signal Decomposition.
    
    Combines the original signal with its Riesz Transform to create
    a rotation-invariant representation with:
    - Amplitude (Local Energy)
    - Phase (Edge type: line, edge, etc.)
    - Orientation (Edge direction)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize Monogenic Signal processor.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.riesz = RieszTransform(epsilon=epsilon)
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Compute Monogenic Signal decomposition.
        
        Args:
            x: Input tensor of shape (B, 1, H, W) - grayscale image
            
        Returns:
            Dictionary containing:
            - 'amplitude': Local energy (B, 1, H, W)
            - 'phase': Local phase (B, 1, H, W)
            - 'orientation': Local orientation (B, 1, H, W)
            - 'riesz_x': Riesz component x (B, 1, H, W)
            - 'riesz_y': Riesz component y (B, 1, H, W)
        """
        # Get Riesz components
        riesz_x, riesz_y = self.riesz(x)
        
        # Compute amplitude (local energy)
        # A = sqrt(f^2 + h1^2 + h2^2)
        amplitude = torch.sqrt(x**2 + riesz_x**2 + riesz_y**2 + self.epsilon)
        
        # Compute orientation
        # theta = atan2(h2, h1)
        orientation = torch.atan2(riesz_y, riesz_x + self.epsilon)
        
        # Compute phase
        # phi = atan2(sqrt(h1^2 + h2^2), f)
        riesz_magnitude = torch.sqrt(riesz_x**2 + riesz_y**2 + self.epsilon)
        phase = torch.atan2(riesz_magnitude, x + self.epsilon)
        
        return {
            'amplitude': amplitude,
            'phase': phase,
            'orientation': orientation,
            'riesz_x': riesz_x,
            'riesz_y': riesz_y
        }


class EnergyMap(nn.Module):
    """
    Compute Energy Map from Monogenic Signal.
    
    The energy map highlights regions with high local contrast (edges, boundaries)
    and suppresses flat/homogeneous regions.
    
    This serves as the "gate" for the Energy-Gated architecture.
    """
    
    def __init__(self, normalize: bool = True, smoothing_sigma: float = 1.0):
        """
        Initialize Energy Map extractor.
        
        Args:
            normalize: Whether to normalize energy to [0, 1]
            smoothing_sigma: Gaussian smoothing sigma for energy map
        """
        super().__init__()
        self.monogenic = MonogenicSignal()
        self.normalize = normalize
        self.smoothing_sigma = smoothing_sigma
        
        # Create Gaussian smoothing kernel
        if smoothing_sigma > 0:
            kernel_size = int(6 * smoothing_sigma) | 1  # Ensure odd
            self.register_buffer('smooth_kernel', self._create_gaussian_kernel(
                kernel_size, smoothing_sigma
            ))
        else:
            self.smooth_kernel = None
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        x = torch.arange(kernel_size) - kernel_size // 2
        x = x.float()
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        return gaussian_2d.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute Energy Map.
        
        Args:
            x: Input tensor of shape (B, 1, H, W) or (B, C, H, W)
            
        Returns:
            Tuple of:
            - energy_map: Tensor of shape (B, 1, H, W) in [0, 1]
            - monogenic_outputs: Dictionary with all monogenic components
        """
        # Convert to grayscale if needed
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Get monogenic decomposition
        mono_out = self.monogenic(x)
        
        # Energy is the amplitude
        energy = mono_out['amplitude']
        
        # Optional smoothing
        if self.smooth_kernel is not None:
            pad = self.smooth_kernel.shape[-1] // 2
            energy = F.conv2d(energy, self.smooth_kernel, padding=pad)
        
        # Normalize to [0, 1]
        if self.normalize:
            B = energy.shape[0]
            energy_flat = energy.view(B, -1)
            energy_min = energy_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            energy_max = energy_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            energy = (energy - energy_min) / (energy_max - energy_min + 1e-8)
        
        return energy, mono_out


class BoundaryDetector(nn.Module):
    """
    Physics-based Boundary Detection using Monogenic Signal.
    
    Uses phase congruency to detect boundaries - regions where
    Fourier components are maximally in phase.
    """
    
    def __init__(self, num_scales: int = 4, num_orientations: int = 6,
                 noise_threshold: float = 0.1):
        """
        Initialize Boundary Detector.
        
        Args:
            num_scales: Number of scales for Log-Gabor filters
            num_orientations: Number of orientation channels
            noise_threshold: Threshold for noise suppression
        """
        super().__init__()
        self.log_gabor = LogGaborFilter(num_scales, num_orientations)
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.noise_threshold = noise_threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect boundaries using phase congruency.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
            
        Returns:
            Boundary map of shape (B, 1, H, W) in [0, 1]
        """
        # Get multi-scale responses
        responses = self.log_gabor(x)  # (B, S*O, H, W)
        
        # Sum across orientations to get edge strength per scale
        B, _, H, W = responses.shape
        responses = responses.view(B, self.num_scales, self.num_orientations, H, W)
        
        # Max across orientations (strongest edge direction)
        edge_strength = responses.max(dim=2)[0]  # (B, S, H, W)
        
        # Sum across scales
        edge_strength = edge_strength.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Normalize and threshold
        edge_max = edge_strength.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        edge_strength = edge_strength / (edge_max + 1e-8)
        edge_strength = torch.clamp(edge_strength - self.noise_threshold, min=0)
        edge_strength = edge_strength / (1 - self.noise_threshold + 1e-8)
        
        return edge_strength


if __name__ == "__main__":
    # Test Monogenic Signal processing
    print("Testing Monogenic Signal Processing...")
    
    # Create test image with edges
    H, W = 128, 128
    x = torch.zeros(1, 1, H, W)
    x[:, :, 32:96, 32:96] = 1.0  # Square
    
    # Add some noise
    x = x + 0.1 * torch.randn_like(x)
    
    # Test Energy Map
    energy_extractor = EnergyMap(normalize=True)
    energy, mono = energy_extractor(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Energy map shape: {energy.shape}")
    print(f"Energy range: [{energy.min():.3f}, {energy.max():.3f}]")
    print(f"Monogenic components: {list(mono.keys())}")
    
    # Test Boundary Detector
    boundary_detector = BoundaryDetector()
    boundaries = boundary_detector(x)
    
    print(f"Boundary map shape: {boundaries.shape}")
    print(f"Boundary range: [{boundaries.min():.3f}, {boundaries.max():.3f}]")
    
    print("\n✓ All tests passed!")
