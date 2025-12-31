"""
Implicit Mamba Network for Continuous Segmentation
Combines Mamba Encoder with Implicit Fourier Decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from mamba_block import VSSBlock, MambaBlockStack
from spectral_layers import SpectralGating
from implicit_head import ImplicitSegmentationHead, FourierMapping, MultiScaleImplicitHead


class MambaEncoder(nn.Module):
    """
    Mamba-based Encoder for extracting hierarchical features.
    Uses SpectralVSS blocks for dual spatial-frequency processing.
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 num_stages: int = 4, depth: int = 2):
        """
        Initialize Mamba Encoder.
        
        Args:
            in_channels: Input image channels
            base_channels: Base number of channels
            num_stages: Number of encoder stages
            depth: Depth of VSS blocks per stage
        """
        super().__init__()
        
        self.num_stages = num_stages
        self.base_channels = base_channels
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=32, num_channels=base_channels),
            nn.GELU()
        )
        
        # Encoder stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(num_stages):
            channels = base_channels * (2 ** i)
            
            # VSS Block stack
            stage = MambaBlockStack(
                channels, 
                depth=depth,
                scan_dim=min(64, channels)
            )
            self.stages.append(stage)
            
            # Downsampling (except last stage)
            if i < num_stages - 1:
                down = nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2),
                    nn.GroupNorm(num_groups=32, num_channels=channels * 2)
                )
                self.downsamples.append(down)
    
    def forward(self, x: torch.Tensor, 
                return_multiscale: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image of shape (B, C, H, W)
            return_multiscale: Whether to return features from all stages
            
        Returns:
            If return_multiscale=False: Final feature map (B, C, H', W')
            If return_multiscale=True: List of feature maps from all stages
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        features = []
        
        for i in range(self.num_stages):
            x = self.stages[i](x)
            features.append(x)
            
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)
        
        if return_multiscale:
            return features
        else:
            return x


class SpectralMambaEncoder(nn.Module):
    """
    Enhanced Mamba Encoder with Spectral Gating.
    Combines spatial (Mamba) and frequency (FFT) processing.
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 num_stages: int = 4, depth: int = 2, img_size: int = 256):
        """
        Initialize Spectral Mamba Encoder.
        
        Args:
            in_channels: Input image channels
            base_channels: Base number of channels
            num_stages: Number of encoder stages
            depth: Depth of VSS blocks per stage
            img_size: Input image size
        """
        super().__init__()
        
        self.num_stages = num_stages
        self.base_channels = base_channels
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=32, num_channels=base_channels),
            nn.GELU()
        )
        
        initial_size = img_size // 4
        
        # Encoder stages with spectral gating
        self.stages = nn.ModuleList()
        self.spectral_gates = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.fusion_weights = nn.ParameterList()
        
        for i in range(num_stages):
            channels = base_channels * (2 ** i)
            h = w = initial_size // (2 ** i)
            
            # VSS Block stack
            stage = MambaBlockStack(
                channels, 
                depth=depth,
                scan_dim=min(64, channels)
            )
            self.stages.append(stage)
            
            # Spectral gating
            spectral = SpectralGating(channels, h, w, threshold=0.1)
            self.spectral_gates.append(spectral)
            
            # Learnable fusion weight
            self.fusion_weights.append(nn.Parameter(torch.tensor(0.5)))
            
            # Downsampling (except last stage)
            if i < num_stages - 1:
                down = nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2),
                    nn.GroupNorm(num_groups=32, num_channels=channels * 2)
                )
                self.downsamples.append(down)
    
    def forward(self, x: torch.Tensor, 
                return_multiscale: bool = False) -> torch.Tensor:
        """
        Forward pass with dual-path processing.
        
        Args:
            x: Input image of shape (B, C, H, W)
            return_multiscale: Whether to return features from all stages
            
        Returns:
            Feature map(s)
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        features = []
        
        for i in range(self.num_stages):
            # Dual path: Mamba + Spectral
            spatial_out = self.stages[i](x)
            spectral_out = self.spectral_gates[i](x)
            
            # Learnable fusion
            w = torch.sigmoid(self.fusion_weights[i])
            x = w * spatial_out + (1 - w) * spectral_out
            
            features.append(x)
            
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)
        
        if return_multiscale:
            return features
        else:
            return x


class ImplicitMambaNet(nn.Module):
    """
    Implicit Continuous Segmentation Network.
    
    Combines:
    - Mamba Encoder for global context extraction
    - Implicit Fourier Decoder for resolution-free segmentation
    
    Key Features:
    - Resolution-independent: Can output at any resolution
    - Smooth boundaries: No pixelation artifacts
    - Global context: Mamba captures long-range dependencies
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 3,
                 img_size: int = 256, base_channels: int = 64,
                 num_stages: int = 4, depth: int = 2,
                 fourier_scale: float = 10.0, fourier_size: int = 256,
                 hidden_dim: int = 256, num_mlp_layers: int = 4,
                 use_spectral: bool = True, use_siren: bool = True,
                 multiscale: bool = False):
        """
        Initialize Implicit Mamba Network.
        
        Args:
            in_channels: Input image channels
            num_classes: Number of segmentation classes
            img_size: Input image size
            base_channels: Base encoder channels
            num_stages: Number of encoder stages
            depth: Depth per stage
            fourier_scale: Scale for Fourier mapping
            fourier_size: Fourier feature dimension
            hidden_dim: MLP hidden dimension
            num_mlp_layers: Number of MLP layers
            use_spectral: Use spectral gating in encoder
            use_siren: Use SIREN activation in decoder
            multiscale: Use multi-scale features
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.multiscale = multiscale
        
        # Encoder
        if use_spectral:
            self.encoder = SpectralMambaEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=num_stages,
                depth=depth,
                img_size=img_size
            )
        else:
            self.encoder = MambaEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=num_stages,
                depth=depth
            )
        
        # Feature dimension at final stage
        final_channels = base_channels * (2 ** (num_stages - 1))
        
        # Implicit decoder
        if multiscale:
            # Multi-scale: use features from all stages
            feature_dims = [base_channels * (2 ** i) for i in range(num_stages)]
            self.implicit_head = MultiScaleImplicitHead(
                feature_dims=feature_dims,
                num_classes=num_classes,
                fourier_scale=fourier_scale,
                fourier_size=fourier_size,
                hidden_dim=hidden_dim,
                num_layers=num_mlp_layers
            )
        else:
            # Single-scale: use only final features
            self.implicit_head = ImplicitSegmentationHead(
                feature_dim=final_channels,
                num_classes=num_classes,
                fourier_scale=fourier_scale,
                fourier_size=fourier_size,
                hidden_dim=hidden_dim,
                num_layers=num_mlp_layers,
                use_siren=use_siren
            )
    
    def forward(self, x: torch.Tensor, 
                coords: Optional[torch.Tensor] = None,
                output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image of shape (B, C, H, W)
            coords: Query coordinates in [-1, 1], shape (B, N, 2)
                    If None, uses regular grid
            output_size: Output resolution (H, W) for grid generation
                         If None, uses input resolution
            
        Returns:
            Segmentation logits:
            - If coords: (B, N, num_classes)
            - If grid: (B, num_classes, H_out, W_out)
        """
        # Default output size
        if output_size is None and coords is None:
            output_size = (self.img_size, self.img_size)
        
        # Encode
        if self.multiscale:
            features = self.encoder(x, return_multiscale=True)
            
            # Generate coords if needed
            if coords is None:
                B = x.shape[0]
                device = x.device
                H_out, W_out = output_size
                
                y = torch.linspace(-1, 1, H_out, device=device)
                xc = torch.linspace(-1, 1, W_out, device=device)
                grid_y, grid_x = torch.meshgrid(y, xc, indexing='ij')
                coords = torch.stack([grid_x, grid_y], dim=-1)
                coords = coords.view(1, -1, 2).expand(B, -1, -1)
            
            # Decode
            logits = self.implicit_head(features, coords)
            
            # Reshape if using grid
            if output_size is not None:
                B = x.shape[0]
                H_out, W_out = output_size
                logits = logits.view(B, H_out, W_out, -1)
                logits = logits.permute(0, 3, 1, 2)
        else:
            features = self.encoder(x, return_multiscale=False)
            logits = self.implicit_head(features, coords=coords, 
                                        output_size=output_size)
        
        return logits
    
    def query_points(self, x: torch.Tensor, 
                     coords: torch.Tensor) -> torch.Tensor:
        """
        Query segmentation at specific coordinates.
        Useful for training with point sampling.
        
        Args:
            x: Input image (B, C, H, W)
            coords: Normalized coordinates (B, N, 2) in [-1, 1]
            
        Returns:
            Class logits (B, N, num_classes)
        """
        return self.forward(x, coords=coords)
    
    def render(self, x: torch.Tensor, 
               output_size: Tuple[int, int]) -> torch.Tensor:
        """
        Render segmentation at arbitrary resolution.
        
        Args:
            x: Input image (B, C, H, W)
            output_size: Desired output resolution (H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        return self.forward(x, output_size=output_size)


class ImplicitLoss(nn.Module):
    """
    Loss function for implicit segmentation with point sampling.
    """
    
    def __init__(self, num_samples: int = 4096, 
                 boundary_weight: float = 2.0,
                 use_dice: bool = True):
        """
        Initialize Implicit Loss.
        
        Args:
            num_samples: Number of points to sample per image
            boundary_weight: Weight for boundary points
            use_dice: Include Dice loss
        """
        super().__init__()
        self.num_samples = num_samples
        self.boundary_weight = boundary_weight
        self.use_dice = use_dice
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def sample_points(self, mask: torch.Tensor, 
                      num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random points with boundary emphasis.
        
        Args:
            mask: Ground truth mask (B, H, W)
            num_samples: Number of points to sample
            
        Returns:
            coords: Sampled coordinates (B, N, 2) in [-1, 1]
            labels: Labels at sampled points (B, N)
        """
        B, H, W = mask.shape
        device = mask.device
        
        # Generate random coordinates
        coords = torch.rand(B, num_samples, 2, device=device) * 2 - 1  # [-1, 1]
        
        # Convert to pixel indices
        pixel_x = ((coords[..., 0] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
        pixel_y = ((coords[..., 1] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
        
        # Get labels at sampled points
        labels = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        for b in range(B):
            labels[b] = mask[b, pixel_y[b], pixel_x[b]]
        
        return coords, labels
    
    def forward(self, pred_logits: torch.Tensor, 
                target_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            pred_logits: Predicted logits (B, N, num_classes)
            target_labels: Target labels (B, N)
            
        Returns:
            Scalar loss
        """
        # Cross entropy
        B, N, C = pred_logits.shape
        pred_flat = pred_logits.view(B * N, C)
        target_flat = target_labels.view(B * N)
        
        ce = self.ce_loss(pred_flat, target_flat).view(B, N)
        loss = ce.mean()
        
        # Optional Dice loss
        if self.use_dice:
            pred_probs = F.softmax(pred_logits, dim=-1)  # (B, N, C)
            target_onehot = F.one_hot(target_labels, C).float()  # (B, N, C)
            
            intersection = (pred_probs * target_onehot).sum(dim=1)  # (B, C)
            union = pred_probs.sum(dim=1) + target_onehot.sum(dim=1)  # (B, C)
            dice = (2 * intersection + 1e-5) / (union + 1e-5)
            dice_loss = 1 - dice.mean()
            
            loss = loss + dice_loss
        
        return loss


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Implicit Mamba Network")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    in_channels = 1
    num_classes = 3
    img_size = 256
    
    # Create model
    model = ImplicitMambaNet(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        base_channels=64,
        num_stages=4,
        depth=2,
        fourier_scale=10.0,
        fourier_size=256,
        hidden_dim=256,
        num_mlp_layers=4,
        use_spectral=True,
        use_siren=True,
        multiscale=False
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # Test grid rendering at different resolutions
    print("\nGrid Rendering:")
    for res in [128, 256, 512]:
        output = model.render(x, output_size=(res, res))
        print(f"  Resolution {res}x{res}: {output.shape}")
    
    # Test point querying
    print("\nPoint Querying:")
    coords = torch.rand(batch_size, 1000, 2) * 2 - 1
    output = model.query_points(x, coords)
    print(f"  Query 1000 points: {output.shape}")
    
    # Test training with point sampling
    print("\nTraining Simulation:")
    mask = torch.randint(0, num_classes, (batch_size, img_size, img_size))
    loss_fn = ImplicitLoss(num_samples=4096)
    
    coords, labels = loss_fn.sample_points(mask, 4096)
    print(f"  Sampled coords: {coords.shape}")
    print(f"  Sampled labels: {labels.shape}")
    
    pred = model.query_points(x, coords)
    loss = loss_fn(pred, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
