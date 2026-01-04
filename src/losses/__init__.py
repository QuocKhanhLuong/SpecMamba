"""Loss functions for medical image segmentation."""

from .physics_loss import (
    DiceLoss, FocalLoss, FrequencyLoss,
    SpectralDualLoss, BoundaryAwareLoss
)

__all__ = [
    'DiceLoss', 'FocalLoss', 'FrequencyLoss',
    'SpectralDualLoss', 'BoundaryAwareLoss'
]
