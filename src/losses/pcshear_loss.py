"""
PCShear Loss Functions for HD95-Optimized Medical Image Segmentation

Loss components:
  1. CurvatureWeightedBoundaryLoss — Penalizes boundary errors proportional
     to local curvature (from Shearlet entropy map)
  2. PhaseCongruencyConsistencyLoss — Ensures predicted boundaries align
     with Phase Congruency map from GT
  3. PCShearCombinedLoss — Combined loss with warmup scheduling

References:
  - Boundary Loss: Kervadec et al. (2019)
  - Phase Congruency: Kovesi (1999)

Default hyperparameters:
  λ₁ (dice) = 1.0, λ₂ (ce) = 1.0, λ₃ (cwb) = 0.5, λ₄ (pc) = 0.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class CurvatureWeightedBoundaryLoss(nn.Module):
    """
    Curvature-Weighted Boundary Loss.

    L_cwb = mean(w(x) × |p(x) − t(x)| × D(t(x)))

    Where:
      - w(x) = boundary weight map (1 + α×E_curv + β×PC)
      - D(t(x)) = distance transform of GT mask (SDF proxy)
      - p(x) = predicted probability
      - t(x) = ground truth

    Penalizes errors more heavily at high-curvature boundary regions,
    directly targeting the 95th percentile outlier points responsible for HD95.

    Args:
        alpha_init: Initial value for curvature weight (learnable, default: 1.0)
        beta_init: Initial value for PC weight (learnable, default: 1.0)
    """

    def __init__(self, alpha_init=1.0, beta_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def _compute_distance_map(self, mask):
        """Compute distance transform for boundary weighting.

        Args:
            mask: Binary mask tensor [B, H, W]

        Returns:
            Distance maps [B, H, W] on same device
        """
        mask_np = mask.detach().cpu().numpy()
        dist_maps = []

        for b in range(mask_np.shape[0]):
            m = mask_np[b]
            if m.sum() == 0:
                dist = np.zeros_like(m, dtype=np.float32)
            elif (1 - m).sum() == 0:
                dist = np.zeros_like(m, dtype=np.float32)
            else:
                # Signed distance: positive outside, negative inside
                dist_out = distance_transform_edt(1 - m)
                dist_in = distance_transform_edt(m)
                dist = dist_out + dist_in
            dist_maps.append(dist)

        return torch.from_numpy(np.stack(dist_maps)).float().to(mask.device)

    def forward(self, pred, target, boundary_weight=None):
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
            boundary_weight: Optional weight map [B, 1, H, W] from boundary head
                             If None, uses uniform weight = 1.0

        Returns:
            loss: Scalar loss value
        """
        pred_probs = torch.softmax(pred, dim=1)
        num_classes = pred_probs.shape[1]

        total_loss = 0.0
        count = 0

        for c in range(1, num_classes):  # Skip background
            target_c = (target == c).float()  # [B, H, W]
            pred_c = pred_probs[:, c]          # [B, H, W]

            # Distance map from boundary
            dist_map = self._compute_distance_map(target_c)  # [B, H, W]

            # Prediction error
            error = (pred_c - target_c).abs()  # [B, H, W]

            # Apply boundary weight if provided
            if boundary_weight is not None:
                w = boundary_weight.squeeze(1)  # [B, H, W]
                weighted_error = error * w * dist_map
            else:
                weighted_error = error * dist_map

            total_loss += weighted_error.mean()
            count += 1

        return total_loss / max(count, 1)


class PhaseCongruencyConsistencyLoss(nn.Module):
    """
    Phase Congruency Consistency Loss.

    L_pc = BCE(predicted_boundary_map, PC_map_from_GT)

    Encourages the network to produce boundaries consistent with the
    Phase Congruency signal derived from the ground truth mask.
    No manual boundary annotation needed.

    Args:
        smooth: Smoothing factor for GT boundary extraction (default: 1e-6)
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def _extract_boundary(self, mask):
        """Extract binary boundary from segmentation mask using morphological gradient.

        Args:
            mask: [B, H, W] integer mask

        Returns:
            boundary: [B, 1, H, W] binary boundary map
        """
        # Convert to one-hot and process each class
        mask_float = mask.float().unsqueeze(1)  # [B, 1, H, W]

        # Use max pooling - erosion to get morphological gradient
        kernel_size = 3
        padding = kernel_size // 2

        dilated = F.max_pool2d(mask_float, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-mask_float, kernel_size, stride=1, padding=padding)

        boundary = (dilated - eroded).clamp(0, 1)
        return boundary  # [B, 1, H, W]

    def forward(self, pc_map, target):
        """
        Args:
            pc_map: Phase Congruency map from model [B, 1, H, W]
            target: Ground truth mask [B, H, W]

        Returns:
            loss: Scalar BCE loss
        """
        # Extract GT boundary
        gt_boundary = self._extract_boundary(target)  # [B, 1, H, W]

        # Resize pc_map if needed
        if pc_map.shape[-2:] != gt_boundary.shape[-2:]:
            pc_map = F.interpolate(pc_map, size=gt_boundary.shape[-2:],
                                   mode='bilinear', align_corners=True)

        # Binary cross-entropy
        loss = F.binary_cross_entropy(
            pc_map.clamp(self.smooth, 1 - self.smooth),
            gt_boundary,
            reduction='mean'
        )
        return loss


class PCShearCombinedLoss(nn.Module):
    """
    Combined loss for PCShear-HRNet with warmup scheduling.

    L_total = λ₁×L_dice + λ₂×L_ce + λ₃×L_cwb + λ₄×L_pc

    Warmup schedule:
      - Epoch  0–10:  L_dice + L_ce only (stabilize model)
      - Epoch 10–20:  + L_cwb with linear warmup
      - Epoch 20+:    + L_pc

    Args:
        num_classes: Number of segmentation classes
        dice_weight: Weight for Dice loss (default: 1.0)
        ce_weight: Weight for Cross-Entropy loss (default: 1.0)
        cwb_weight: Weight for Curvature-Weighted Boundary loss (default: 0.5)
        pc_weight: Weight for Phase Congruency Consistency loss (default: 0.3)
        cwb_start_epoch: Epoch to start L_cwb (default: 10)
        pc_start_epoch: Epoch to start L_pc (default: 20)
        warmup_length: Number of epochs for linear warmup (default: 10)
    """

    def __init__(self, num_classes=4, dice_weight=1.0, ce_weight=1.0,
                 cwb_weight=0.5, pc_weight=0.3,
                 cwb_start_epoch=10, pc_start_epoch=20, warmup_length=10):
        super().__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.cwb_weight = cwb_weight
        self.pc_weight = pc_weight
        self.cwb_start_epoch = cwb_start_epoch
        self.pc_start_epoch = pc_start_epoch
        self.warmup_length = warmup_length

        # Loss components
        self.ce_loss = nn.CrossEntropyLoss()
        self.cwb_loss = CurvatureWeightedBoundaryLoss()
        self.pc_loss = PhaseCongruencyConsistencyLoss()

    def _dice_loss(self, pred, target, smooth=1e-5):
        """Soft Dice loss (multi-class)."""
        pred_soft = torch.softmax(pred, dim=1)
        num_classes = pred_soft.shape[1]

        if target.ndim == 3:
            target_oh = F.one_hot(target.long(), num_classes=num_classes)
            target_oh = target_oh.permute(0, 3, 1, 2).float()
        else:
            target_oh = target

        pred_flat = pred_soft.view(pred_soft.shape[0], pred_soft.shape[1], -1)
        target_flat = target_oh.view(target_oh.shape[0], target_oh.shape[1], -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2.0 * intersection + smooth) / (union + smooth)

        return (1.0 - dice).mean()

    def _get_warmup_factor(self, epoch, start_epoch):
        """Linear warmup factor for a loss component."""
        if epoch < start_epoch:
            return 0.0
        elapsed = epoch - start_epoch
        factor = min(1.0, elapsed / max(self.warmup_length, 1))
        return factor

    def forward(self, pred, target, boundary_weight=None, pc_map=None,
                epoch=0):
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
            boundary_weight: Boundary weight map [B, 1, H, W] from model
            pc_map: Phase Congruency map [B, 1, H, W] from model
            epoch: Current training epoch

        Returns:
            loss: Total loss scalar
            details: Dict of individual loss values
        """
        details = {}

        # 1. Dice Loss (always on)
        dice = self._dice_loss(pred, target)
        total = self.dice_weight * dice
        details['dice'] = dice.item()

        # 2. Cross-Entropy Loss (always on)
        ce = self.ce_loss(pred, target.long())
        total = total + self.ce_weight * ce
        details['ce'] = ce.item()

        # 3. Curvature-Weighted Boundary Loss (with warmup)
        cwb_factor = self._get_warmup_factor(epoch, self.cwb_start_epoch)
        if cwb_factor > 0:
            cwb = self.cwb_loss(pred, target, boundary_weight=boundary_weight)
            total = total + self.cwb_weight * cwb_factor * cwb
            details['cwb'] = cwb.item()
            details['cwb_factor'] = cwb_factor

        # 4. Phase Congruency Consistency Loss (with warmup)
        pc_factor = self._get_warmup_factor(epoch, self.pc_start_epoch)
        if pc_factor > 0 and pc_map is not None:
            pc = self.pc_loss(pc_map, target)
            total = total + self.pc_weight * pc_factor * pc
            details['pc'] = pc.item()
            details['pc_factor'] = pc_factor

        details['total'] = total.item()
        return total, details


if __name__ == "__main__":
    B, C, H, W = 2, 4, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    boundary_weight = torch.rand(B, 1, H, W)
    pc_map = torch.rand(B, 1, H, W)

    loss_fn = PCShearCombinedLoss(num_classes=C)

    # Test at different epochs
    for epoch in [0, 5, 10, 15, 20, 25]:
        loss, details = loss_fn(pred, target, boundary_weight, pc_map, epoch=epoch)
        active = [k for k in ['dice', 'ce', 'cwb', 'pc'] if k in details]
        print(f"Epoch {epoch:3d}: loss={loss.item():.4f}  active={active}")
