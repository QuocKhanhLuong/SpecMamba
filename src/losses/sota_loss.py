"""
SOTA Loss Functions for Medical Image Segmentation
- Boundary Loss (Hausdorff-like)
- Deep Supervision Loss
- Combined Loss với warmup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights  # Tensor of shape (num_classes,)

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        
        if target.ndim == 3:
            target = F.one_hot(target.long(), num_classes=num_classes)
            target = target.permute(0, 3, 1, 2).float()
        
        pred = pred.view(pred.shape[0], pred.shape[1], -1)
        target = target.view(target.shape[0], target.shape[1], -1)
        
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            # Weighted mean across classes
            dice_loss = 1.0 - dice_per_class
            weighted_loss = (dice_loss * weights.view(1, -1)).sum(dim=1) / weights.sum()
            return weighted_loss.mean()
        else:
            return (1.0 - dice_per_class).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target.long(), reduction='none')
        p_t = torch.exp(-ce)
        focal = (1 - p_t) ** self.gamma * ce
        return focal.mean()


class BoundaryLoss(nn.Module):
    """Distance-based boundary loss (Hausdorff-like)"""
    def __init__(self):
        super().__init__()

    def _compute_distance_map(self, mask):
        """Compute distance transform for boundary weighting"""
        mask_np = mask.cpu().numpy()
        dist_maps = []
        
        for b in range(mask_np.shape[0]):
            m = mask_np[b]
            if m.sum() == 0:
                dist = np.zeros_like(m, dtype=np.float32)
            else:
                dist = distance_transform_edt(1 - m) + distance_transform_edt(m)
            dist_maps.append(dist)
        
        return torch.from_numpy(np.stack(dist_maps)).float().to(mask.device)

    def forward(self, pred, target):
        pred_probs = torch.softmax(pred, dim=1)
        num_classes = pred_probs.shape[1]
        
        total_loss = 0
        for c in range(1, num_classes):  # Skip background
            target_c = (target == c).float()
            pred_c = pred_probs[:, c]
            
            # Distance map từ boundary
            dist_map = self._compute_distance_map(target_c)
            
            # Weighted loss
            error = (pred_c - target_c).abs()
            weighted_error = error * dist_map
            total_loss += weighted_error.mean()
        
        return total_loss / max(num_classes - 1, 1)


class BoundaryAwareLoss(nn.Module):
    """Sobel-based boundary enhancement"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def _compute_boundaries(self, mask):
        mask = mask.float().unsqueeze(1)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return (grad_mag > 0).float().squeeze(1)

    def forward(self, pred, target):
        pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        pred_boundary = self._compute_boundaries(pred_class)
        target_boundary = self._compute_boundaries(target)
        
        ce = F.cross_entropy(pred, target.long(), reduction='none')
        boundary_weight = 1.0 + (pred_boundary + target_boundary).clamp(0, 1)
        
        return (ce * boundary_weight).mean()


class DeepSupervisionLoss(nn.Module):
    """Deep supervision với multi-scale outputs"""
    def __init__(self, num_classes, weights=[1.0, 0.5, 0.25, 0.125]):
        super().__init__()
        self.weights = weights
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs_list, target):
        """
        Args:
            outputs_list: List of [main, scale1, scale2, ...] logits
            target: Ground truth mask
        """
        total_loss = 0
        
        for i, (output, w) in enumerate(zip(outputs_list, self.weights)):
            if output.shape[-2:] != target.shape[-2:]:
                target_resized = F.interpolate(
                    target.float().unsqueeze(1),
                    size=output.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                target_resized = target
            
            loss = self.dice(output, target_resized) + self.ce(output, target_resized)
            total_loss += w * loss
        
        return total_loss


class CombinedSOTALoss(nn.Module):
    """Combined loss với warmup cho boundary và class weights (re-weighting)"""
    def __init__(self, num_classes=4, 
                 ce_weight=1.0, 
                 dice_weight=1.0, 
                 boundary_weight=0.5,
                 focal_weight=0.0,
                 warmup_epochs=10,
                 class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.focal_weight = focal_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.num_classes = num_classes
        
        # Class weights for re-weighting (e.g., [0.1, 1.5, 1.5, 1.0] for [BG, RV, MYO, LV])
        # Store as list, convert to tensor on correct device during forward
        self._class_weights_list = class_weights
        
        # Initialize losses without weights (weights applied in forward)
        self.ce = nn.CrossEntropyLoss(reduction='none')  # Apply weight manually
        self.dice = DiceLoss(class_weights=None)  # Will pass weights in forward
        self.boundary = BoundaryAwareLoss()
        self.focal = FocalLoss() if focal_weight > 0 else None

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred, target):
        device = pred.device
        loss = 0
        losses_dict = {}
        
        # Get class weights on correct device
        if self._class_weights_list is not None:
            class_weights = torch.tensor(self._class_weights_list, dtype=torch.float32, device=device)
        else:
            class_weights = None
        
        # CE Loss with manual weighting
        ce_per_pixel = F.cross_entropy(pred, target.long(), reduction='none')
        if class_weights is not None:
            # Apply class-specific weights
            weight_map = class_weights[target.long()]
            ce = (ce_per_pixel * weight_map).mean()
        else:
            ce = ce_per_pixel.mean()
        loss += self.ce_weight * ce
        losses_dict['ce'] = ce.item()
        
        # Dice Loss with class weights
        self.dice.class_weights = class_weights
        dice = self.dice(pred, target)
        loss += self.dice_weight * dice
        losses_dict['dice'] = dice.item()
        
        # Boundary Loss (với warmup)
        if self.current_epoch >= self.warmup_epochs // 2:
            # Tăng dần boundary weight sau warmup
            warmup_factor = min(1.0, (self.current_epoch - self.warmup_epochs // 2) / self.warmup_epochs)
            boundary = self.boundary(pred, target)
            loss += self.boundary_weight * warmup_factor * boundary
            losses_dict['boundary'] = boundary.item()
        
        # Focal Loss (optional)
        if self.focal is not None and self.focal_weight > 0:
            focal = self.focal(pred, target)
            loss += self.focal_weight * focal
            losses_dict['focal'] = focal.item()
        
        losses_dict['total'] = loss.item()
        return loss, losses_dict


class TTAInference:
    """Test Time Augmentation"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, x):
        """TTA: Average predictions từ 4 augmentations"""
        self.model.eval()
        
        preds = []
        
        # Original
        out = self.model(x)['output']
        preds.append(torch.softmax(out, dim=1))
        
        # Horizontal flip
        out_hflip = self.model(torch.flip(x, dims=[-1]))['output']
        preds.append(torch.flip(torch.softmax(out_hflip, dim=1), dims=[-1]))
        
        # Vertical flip
        out_vflip = self.model(torch.flip(x, dims=[-2]))['output']
        preds.append(torch.flip(torch.softmax(out_vflip, dim=1), dims=[-2]))
        
        # 90 degree rotation
        x_rot = torch.rot90(x, 1, dims=[-2, -1])
        out_rot = self.model(x_rot)['output']
        preds.append(torch.rot90(torch.softmax(out_rot, dim=1), -1, dims=[-2, -1]))
        
        # Average
        avg_pred = torch.stack(preds, dim=0).mean(dim=0)
        return avg_pred.argmax(dim=1)

    def predict_8x(self, x):
        """8x TTA: More augmentations"""
        self.model.eval()
        
        preds = []
        
        # 4 rotations
        for k in range(4):
            x_rot = torch.rot90(x, k, dims=[-2, -1])
            out = self.model(x_rot)['output']
            pred = torch.rot90(torch.softmax(out, dim=1), -k, dims=[-2, -1])
            preds.append(pred)
            
            # + horizontal flip
            x_rot_flip = torch.flip(x_rot, dims=[-1])
            out_flip = self.model(x_rot_flip)['output']
            pred_flip = torch.flip(torch.softmax(out_flip, dim=1), dims=[-1])
            pred_flip = torch.rot90(pred_flip, -k, dims=[-2, -1])
            preds.append(pred_flip)
        
        avg_pred = torch.stack(preds, dim=0).mean(dim=0)
        return avg_pred.argmax(dim=1)
