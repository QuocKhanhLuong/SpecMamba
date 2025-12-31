"""
Physics-Inspired Dual Loss Function
Combines spatial (Dice + CE) and frequency domain losses for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """Dice Loss for binary/multi-class segmentation."""
    
    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing constant to avoid division by zero
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            pred: Prediction logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W) or (B, C, H, W)
            
        Returns:
            Scalar Dice loss
        """
        # Convert logits to probabilities
        pred = torch.softmax(pred, dim=1)
        
        # Ensure target has same shape as pred for multi-class
        if target.ndim == 3:  # (B, H, W) -> convert to one-hot
            target = F.one_hot(target.long(), num_classes=pred.shape[1])
            target = target.permute(0, 3, 1, 2).float()
        
        # Flatten spatial dimensions
        pred = pred.view(pred.shape[0], pred.shape[1], -1)
        target = target.view(target.shape[0], target.shape[1], -1)
        
        # Compute Dice score
        intersection = torch.sum(pred * target, dim=2)
        union = torch.sum(pred, dim=2) + torch.sum(target, dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - Dice)
        loss = 1.0 - dice
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class 1
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            pred: Prediction logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
            
        Returns:
            Scalar Focal loss
        """
        # Get class probabilities
        p = torch.softmax(pred, dim=1)
        
        # Get class log probabilities
        ce = F.cross_entropy(pred, target.long(), reduction='none')
        
        # Get probability of true class
        p_t = torch.gather(p, 1, target.long().unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FrequencyLoss(nn.Module):
    """
    Frequency Domain Loss for enforcing edge sharpness and detail preservation.
    Computes L2 distance between FFT of prediction and ground truth.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize FrequencyLoss.
        
        Args:
            weight: Weight factor for this loss component in combined loss
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency domain loss.
        
        Uses FFT to compare frequency components, emphasizing edge preservation.
        
        Args:
            pred: Prediction tensor of shape (B, C, H, W) or (B, H, W)
            target: Ground truth tensor of same shape as pred
            
        Returns:
            Scalar frequency loss
        """
        # Ensure both have batch and channel dimensions
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        
        # Flatten to single channel for FFT comparison
        if pred.shape[1] > 1:
            # For multi-channel, convert to grayscale by averaging
            pred = pred.mean(dim=1, keepdim=True)
        if target.shape[1] > 1:
            target = target.mean(dim=1, keepdim=True)
        
        # Apply FFT to convert to frequency domain
        pred_freq = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        target_freq = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
        
        # Compute L2 distance in frequency domain
        # Consider both magnitude and phase information
        loss_real = F.mse_loss(pred_freq.real, target_freq.real, reduction='mean')
        loss_imag = F.mse_loss(pred_freq.imag, target_freq.imag, reduction='mean')
        
        return loss_real + loss_imag


class SpectralDualLoss(nn.Module):
    """
    Combined Spectral Dual Loss.
    
    Combines:
    1. Spatial losses (Dice + Focal CE) - for overall shape and class balance
    2. Frequency loss - for edge sharpness and boundary preservation
    """
    
    def __init__(self, spatial_weight: float = 1.0, freq_weight: float = 0.1,
                 use_dice: bool = True, use_focal: bool = True):
        """
        Initialize SpectralDualLoss.
        
        Args:
            spatial_weight: Weight for spatial losses
            freq_weight: Weight for frequency loss
            use_dice: Whether to include Dice loss
            use_focal: Whether to include Focal loss (else CrossEntropy)
        """
        super().__init__()
        self.spatial_weight = spatial_weight
        self.freq_weight = freq_weight
        self.use_dice = use_dice
        self.use_focal = use_focal
        
        # Spatial losses
        if use_dice:
            self.dice_loss = DiceLoss(smooth=1e-5)
        
        if use_focal:
            self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Frequency loss
        self.freq_loss = FrequencyLoss(weight=freq_weight)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Prediction logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
            return_components: If True, return dict with individual loss components
            
        Returns:
            Scalar combined loss, or dict if return_components=True
        """
        # Ensure target is on same device as pred
        target = target.to(pred.device)
        
        # Spatial losses
        spatial_loss = 0.0
        losses_dict = {}
        
        if self.use_dice:
            dice = self.dice_loss(pred, target)
            spatial_loss = spatial_loss + dice
            losses_dict['dice'] = dice.item()
        
        if self.use_focal:
            focal = self.focal_loss(pred, target)
            spatial_loss = spatial_loss + focal
            losses_dict['focal'] = focal.item()
        else:
            ce = self.ce_loss(pred, target)
            spatial_loss = spatial_loss + ce
            losses_dict['ce'] = ce.item()
        
        # Frequency loss
        # For frequency loss, we need to extract the predicted class (argmax) and compare
        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)  # (B, H, W)
        
        freq = self.freq_loss(pred_class.float(), target.float())
        losses_dict['freq'] = freq.item()
        
        # Weighted combination
        total_loss = (self.spatial_weight * spatial_loss + 
                     self.freq_weight * freq)
        losses_dict['total'] = total_loss.item()
        
        if return_components:
            return total_loss, losses_dict
        else:
            return total_loss


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes pixels near segmentation boundaries.
    Useful for improving edge precision.
    """
    
    def __init__(self, kernel_size: int = 3, weight: float = 1.0):
        """
        Initialize BoundaryAwareLoss.
        
        Args:
            kernel_size: Size of kernel for computing boundary gradients
            weight: Weight for boundary loss
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = weight
    
    def _compute_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary mask using gradient.
        
        Args:
            mask: Binary mask of shape (B, H, W)
            
        Returns:
            Boundary map of shape (B, H, W)
        """
        # Convert to float
        mask = mask.float().unsqueeze(1)  # (B, 1, H, W)
        
        # Compute gradients using Sobel-like operation
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=mask.dtype, device=mask.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=mask.dtype, device=mask.device)
        
        kernel_x = kernel_x.view(1, 1, 3, 3)
        kernel_y = kernel_y.view(1, 1, 3, 3)
        
        grad_x = F.conv2d(mask, kernel_x, padding=1)
        grad_y = F.conv2d(mask, kernel_y, padding=1)
        
        # Compute magnitude of gradient
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Threshold to get boundary pixels
        boundary = (grad_magnitude > 0).float().squeeze(1)
        
        return boundary
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-aware loss.
        
        Args:
            pred: Prediction logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
            
        Returns:
            Scalar loss emphasizing boundaries
        """
        # Get predicted class
        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)  # (B, H, W)
        
        # Compute boundary maps
        pred_boundary = self._compute_boundaries(pred_class)
        target_boundary = self._compute_boundaries(target)
        
        # Compute cross-entropy loss weighted by boundary
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        
        # Apply boundary weight (higher loss for boundary pixels)
        boundary_weight = (pred_boundary + target_boundary).clamp(0, 1)
        boundary_weight = 1.0 + boundary_weight  # Weight between 1 and 2
        
        weighted_loss = ce_loss * boundary_weight
        
        return weighted_loss.mean()


if __name__ == "__main__":
    # Test losses
    batch_size, num_classes, height, width = 2, 3, 64, 64
    
    # Create dummy predictions and targets
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test SpectralDualLoss
    loss_fn = SpectralDualLoss(spatial_weight=1.0, freq_weight=0.1)
    loss, components = loss_fn(pred, target, return_components=True)
    
    print(f"Total Loss: {loss.item():.4f}")
    for name, value in components.items():
        print(f"  {name}: {value:.4f}")
    
    # Test BoundaryAwareLoss
    boundary_loss_fn = BoundaryAwareLoss()
    boundary_loss = boundary_loss_fn(pred, target)
    print(f"\nBoundary Loss: {boundary_loss.item():.4f}")
