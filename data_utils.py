"""
Data utilities for medical image segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional


class MedicalImageSegmentationDataset(Dataset):
    """
    Base dataset class for medical image segmentation.
    Can be extended for specific data formats (NIfTI, DICOM, PNG, etc.)
    """
    
    def __init__(self, images: np.ndarray, masks: np.ndarray,
                 img_size: int = 256, normalize: bool = True,
                 augment: bool = False):
        """
        Initialize dataset.
        
        Args:
            images: Array of shape (N, H, W) or (N, C, H, W)
            masks: Array of shape (N, H, W) with class labels
            img_size: Target image size for resizing
            normalize: Whether to normalize images (0-1 or standardize)
            augment: Whether to apply data augmentation
        """
        self.images = images
        self.masks = masks
        self.img_size = img_size
        self.normalize = normalize
        self.augment = augment
        
        assert len(images) == len(masks), "Images and masks must have same length"
        
        # Ensure 4D shape (N, C, H, W)
        if self.images.ndim == 3:
            self.images = np.expand_dims(self.images, axis=1)
        
        # Preprocess
        self.images = torch.from_numpy(self.images).float()
        self.masks = torch.from_numpy(self.masks).long()
        
        if self.normalize:
            self._normalize_images()
    
    def _normalize_images(self):
        """Normalize images to 0-1 range or standardize."""
        # Normalize per image to 0-1
        for i in range(len(self.images)):
            img = self.images[i]
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                self.images[i] = (img - img_min) / (img_max - img_min)
    
    def _resize_if_needed(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple:
        """Resize image and mask if needed."""
        if image.shape[-1] != self.img_size or image.shape[-2] != self.img_size:
            image = F.interpolate(
                image.unsqueeze(0), size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=True
            ).squeeze(0)
            mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
        return image, mask
    
    def _augment_data(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple:
        """Apply basic data augmentation."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        
        # Random rotation (0, 90, 180, 270)
        k = torch.randint(0, 4, (1,)).item()
        image = torch.rot90(image, k=k, dims=[-2, -1])
        mask = torch.rot90(mask, k=k, dims=[-2, -1])
        
        return image, mask
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]
        
        # Resize if needed
        image, mask = self._resize_if_needed(image, mask)
        
        # Augmentation
        if self.augment:
            image, mask = self._augment_data(image, mask)
        
        return image, mask


class MetricsCalculator:
    """Calculate segmentation metrics."""
    
    @staticmethod
    def dice_score(pred: torch.Tensor, target: torch.Tensor, 
                   smooth: float = 1e-5) -> float:
        """
        Calculate Dice coefficient.
        
        Args:
            pred: Predictions of shape (B, C, H, W) or (B, H, W)
            target: Ground truth of shape (B, H, W)
            smooth: Smoothing constant
            
        Returns:
            Dice score (0-1)
        """
        if pred.ndim == 4:
            pred = torch.argmax(pred, dim=1)
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred == target).sum().float()
        union = pred.numel()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
    
    @staticmethod
    def iou_score(pred: torch.Tensor, target: torch.Tensor,
                  num_classes: int = 3, smooth: float = 1e-5) -> dict:
        """
        Calculate Intersection over Union (IoU) for each class.
        
        Args:
            pred: Predictions of shape (B, C, H, W) or (B, H, W)
            target: Ground truth of shape (B, H, W)
            num_classes: Number of classes
            smooth: Smoothing constant
            
        Returns:
            Dictionary with per-class and mean IoU
        """
        if pred.ndim == 4:
            pred = torch.argmax(pred, dim=1)
        
        iou_scores = {}
        mean_iou = 0.0
        
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            iou = (intersection + smooth) / (union + smooth)
            iou_scores[f"class_{cls}"] = iou.item()
            mean_iou += iou.item()
        
        iou_scores["mean"] = mean_iou / num_classes
        return iou_scores
    
    @staticmethod
    def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Hausdorff distance (boundary metric).
        
        Args:
            pred: Predictions of shape (B, H, W)
            target: Ground truth of shape (B, H, W)
            
        Returns:
            Hausdorff distance
        """
        # Simple implementation using max of minimum distances
        pred = pred.float().view(-1, 1)
        target = target.float().view(-1, 1)
        
        # Distance from pred to target
        dist_pt = torch.cdist(pred, target).min(dim=1)[0]
        max_dist_pt = dist_pt.max()
        
        # Distance from target to pred
        dist_tp = torch.cdist(target, pred).min(dim=1)[0]
        max_dist_tp = dist_tp.max()
        
        hd = max(max_dist_pt.item(), max_dist_tp.item())
        return hd
    
    @staticmethod
    def sensitivity_specificity(pred: torch.Tensor, target: torch.Tensor,
                               num_classes: int = 2) -> dict:
        """
        Calculate sensitivity and specificity for binary/multi-class.
        
        Args:
            pred: Predictions of shape (B, C, H, W) or (B, H, W)
            target: Ground truth of shape (B, H, W)
            num_classes: Number of classes
            
        Returns:
            Dictionary with sensitivity and specificity per class
        """
        if pred.ndim == 4:
            pred = torch.argmax(pred, dim=1)
        
        metrics = {}
        
        for cls in range(1, num_classes):  # Skip background class 0
            pred_pos = (pred == cls)
            pred_neg = (pred != cls)
            target_pos = (target == cls)
            target_neg = (target != cls)
            
            tp = (pred_pos & target_pos).sum().float().item()
            tn = (pred_neg & target_neg).sum().float().item()
            fp = (pred_pos & target_neg).sum().float().item()
            fn = (pred_neg & target_pos).sum().float().item()
            
            sensitivity = tp / (tp + fn + 1e-5)
            specificity = tn / (tn + fp + 1e-5)
            
            metrics[f"class_{cls}"] = {
                "sensitivity": sensitivity,
                "specificity": specificity
            }
        
        return metrics


if __name__ == "__main__":
    # Test dataset
    images = np.random.rand(10, 256, 256)
    masks = np.random.randint(0, 3, (10, 256, 256))
    
    dataset = MedicalImageSegmentationDataset(
        images, masks, img_size=256, normalize=True, augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    
    # Test metrics
    pred = torch.randint(0, 3, (2, 256, 256))
    target = torch.randint(0, 3, (2, 256, 256))
    
    calc = MetricsCalculator()
    dice = calc.dice_score(pred, target)
    iou = calc.iou_score(pred, target)
    
    print(f"\nDice Score: {dice:.4f}")
    print(f"IoU Scores: {iou}")
