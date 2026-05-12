
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):

        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = torch.softmax(pred, dim=1)

        if target.ndim == 3:
            target = F.one_hot(target.long(), num_classes=pred.shape[1])
            target = target.permute(0, 3, 1, 2).float()

        pred = pred.view(pred.shape[0], pred.shape[1], -1)
        target = target.view(target.shape[0], target.shape[1], -1)

        intersection = torch.sum(pred * target, dim=2)
        union = torch.sum(pred, dim=2) + torch.sum(target, dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        p = torch.softmax(pred, dim=1)

        ce = F.cross_entropy(pred, target.long(), reduction='none')

        p_t = torch.gather(p, 1, target.long().unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class FrequencyLoss(nn.Module):

    def __init__(self, weight: float = 0.1):

        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        if pred.shape[1] > 1:

            pred = pred.mean(dim=1, keepdim=True)
        if target.shape[1] > 1:
            target = target.mean(dim=1, keepdim=True)

        pred_freq = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        target_freq = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")

        loss_real = F.mse_loss(pred_freq.real, target_freq.real, reduction='mean')
        loss_imag = F.mse_loss(pred_freq.imag, target_freq.imag, reduction='mean')

        return loss_real + loss_imag

class SpectralDualLoss(nn.Module):

    def __init__(self, spatial_weight: float = 1.0, freq_weight: float = 0.1,
                 use_dice: bool = True, use_focal: bool = True):

        super().__init__()
        self.spatial_weight = spatial_weight
        self.freq_weight = freq_weight
        self.use_dice = use_dice
        self.use_focal = use_focal

        if use_dice:
            self.dice_loss = DiceLoss(smooth=1e-5)

        if use_focal:
            self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.freq_loss = FrequencyLoss(weight=freq_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:

        target = target.to(pred.device)

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

        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)

        freq = self.freq_loss(pred_class.float(), target.float())
        losses_dict['freq'] = freq.item()

        total_loss = (self.spatial_weight * spatial_loss +
                     self.freq_weight * freq)
        losses_dict['total'] = total_loss.item()

        if return_components:
            return total_loss, losses_dict
        else:
            return total_loss

class BoundaryAwareLoss(nn.Module):

    def __init__(self, kernel_size: int = 3, weight: float = 1.0):

        super().__init__()
        self.kernel_size = kernel_size
        self.weight = weight

    def _compute_boundaries(self, mask: torch.Tensor) -> torch.Tensor:

        mask = mask.float().unsqueeze(1)

        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=mask.dtype, device=mask.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=mask.dtype, device=mask.device)

        kernel_x = kernel_x.view(1, 1, 3, 3)
        kernel_y = kernel_y.view(1, 1, 3, 3)

        grad_x = F.conv2d(mask, kernel_x, padding=1)
        grad_y = F.conv2d(mask, kernel_y, padding=1)

        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        boundary = (grad_magnitude > 0).float().squeeze(1)

        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)

        pred_boundary = self._compute_boundaries(pred_class)
        target_boundary = self._compute_boundaries(target)

        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')

        boundary_weight = (pred_boundary + target_boundary).clamp(0, 1)
        boundary_weight = 1.0 + boundary_weight

        weighted_loss = ce_loss * boundary_weight

        return weighted_loss.mean()

class EyeOpeningLoss(nn.Module):

    def __init__(self, warmup_epochs: int = 5, max_weight: float = 0.1,
                 anneal_rate: float = 0.02, smooth: float = 1e-7):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
        self.anneal_rate = anneal_rate
        self.smooth = smooth

    def get_weight(self, epoch: int) -> float:

        if epoch < self.warmup_epochs:
            return 0.0
        return min(self.max_weight, self.anneal_rate * (epoch - self.warmup_epochs))

    def forward(self, logits: torch.Tensor,
                energy_map: Optional[torch.Tensor] = None,
                epoch: int = 0) -> torch.Tensor:

        weight = self.get_weight(epoch)

        if weight <= 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=False)

        if logits.dim() == 4:

            probs = torch.softmax(logits, dim=1)

            max_probs = probs.max(dim=1, keepdim=True)[0]
        else:

            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1, keepdim=True)[0]

        eye_loss = 4 * max_probs * (1 - max_probs)

        if energy_map is not None:
            if energy_map.shape[-2:] != eye_loss.shape[-2:]:
                energy_map = F.interpolate(
                    energy_map, size=eye_loss.shape[-2:],
                    mode='bilinear', align_corners=True
                )
            eye_loss = eye_loss * energy_map

        return weight * eye_loss.mean()

class EGMCombinedLoss(nn.Module):

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0,
                 fine_weight: float = 1.0, eye_weight: float = 0.1,
                 consistency_weight: float = 0.1, eye_warmup: int = 5):
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.fine_weight = fine_weight
        self.eye_weight = eye_weight
        self.consistency_weight = consistency_weight

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.eye_loss = EyeOpeningLoss(
            warmup_epochs=eye_warmup,
            max_weight=eye_weight,
            anneal_rate=0.02
        )

    def forward(self, outputs: dict, target: torch.Tensor,
                point_logits: Optional[torch.Tensor] = None,
                point_labels: Optional[torch.Tensor] = None,
                epoch: int = 0) -> dict:

        losses = {}

        coarse = outputs.get('coarse', outputs.get('output'))

        if coarse.shape[-2:] != target.shape[-2:]:
            target_resized = F.interpolate(
                target.unsqueeze(1).float(),
                size=coarse.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()
        else:
            target_resized = target.long()

        dice = self.dice_loss(coarse, target_resized)
        ce = self.ce_loss(coarse, target_resized)

        losses['dice'] = dice
        losses['ce'] = ce

        coarse_loss = self.dice_weight * dice + self.ce_weight * ce

        fine_loss = torch.tensor(0.0, device=coarse.device)
        if point_logits is not None and point_labels is not None:

            B, N, C = point_logits.shape
            point_logits_flat = point_logits.view(B * N, C)
            point_labels_flat = point_labels.view(B * N)
            fine_loss = F.cross_entropy(point_logits_flat, point_labels_flat)
            losses['fine'] = fine_loss
        elif 'fine' in outputs:

            fine = outputs['fine']
            if fine.shape[-2:] != target_resized.shape[-2:]:
                target_for_fine = F.interpolate(
                    target_resized.unsqueeze(1).float(),
                    size=fine.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                target_for_fine = target_resized
            fine_loss = self.ce_loss(fine, target_for_fine)
            losses['fine'] = fine_loss

        energy_map = outputs.get('energy', None)
        if 'fine' in outputs:
            eye = self.eye_loss(outputs['fine'], energy_map, epoch)
        else:
            eye = self.eye_loss(coarse, energy_map, epoch)
        losses['eye'] = eye

        consistency_loss = torch.tensor(0.0, device=coarse.device)
        if 'fine' in outputs and self.consistency_weight > 0:
            fine = outputs['fine']
            if fine.shape[-2:] != coarse.shape[-2:]:
                fine_resized = F.interpolate(
                    fine, size=coarse.shape[-2:],
                    mode='bilinear', align_corners=True
                )
            else:
                fine_resized = fine

            coarse_probs = F.log_softmax(coarse, dim=1)
            fine_probs = F.softmax(fine_resized, dim=1)
            consistency_loss = F.kl_div(coarse_probs, fine_probs, reduction='batchmean')
            losses['consistency'] = consistency_loss

        total = (coarse_loss +
                 self.fine_weight * fine_loss +
                 eye +
                 self.consistency_weight * consistency_loss)

        losses['total'] = total

        return losses

def _demo():
    batch_size, num_classes, height, width = 2, 3, 64, 64

    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))

    loss_fn = SpectralDualLoss(spatial_weight=1.0, freq_weight=0.1)
    loss, components = loss_fn(pred, target, return_components=True)

    print(f"Total Loss: {loss.item():.4f}")
    for name, value in components.items():
        print(f"  {name}: {value:.4f}")

    boundary_loss_fn = BoundaryAwareLoss()
    boundary_loss = boundary_loss_fn(pred, target)
    print(f"\nBoundary Loss: {boundary_loss.item():.4f}")


if __name__ == "__main__":
    _demo()
