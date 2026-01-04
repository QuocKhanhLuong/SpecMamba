import torch
import torch.nn.functional as F
import numpy as np
from monai.metrics import compute_hausdorff_distance

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SegmentationMetrics:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.reset()
        
    def reset(self):
        self.batches = 0
        self.total_correct_pixels = 0
        self.total_pixels = 0
        
        # Aggregated stats for Precision/Recall/F1 (Global)
        self.tp = torch.zeros(self.num_classes, device=self.device)
        self.fp = torch.zeros(self.num_classes, device=self.device)
        self.fn = torch.zeros(self.num_classes, device=self.device)
        
        # Accumulators for averaging Batch-wise metrics
        self.dice_sum = torch.zeros(self.num_classes, device=self.device)
        self.iou_sum = torch.zeros(self.num_classes, device=self.device)
        self.hd95_sum = torch.zeros(self.num_classes, device=self.device)
        
        # Track valid batches for HD95 (it can be NaN if class is missing)
        self.hd95_counts = torch.zeros(self.num_classes, device=self.device)

    def update(self, preds, targets):
        """
        Update metrics with a new batch.
        Args:
            preds: (B, H, W) Tensor of class indices.
            targets: (B, H, W) Tensor of class indices.
        """
        self.batches += 1
        
        # Accuracy
        self.total_correct_pixels += (preds == targets).sum().item()
        self.total_pixels += targets.numel()
        
        # Create one-hot for HD95 and Dice
        # preds_oh: (B, C, H, W)
        preds_oh = F.one_hot(preds, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        targets_oh = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Helper for Dice/IoU/TP/FP/FN
        for c in range(self.num_classes):
            p_flat = preds_oh[:, c].reshape(-1)
            t_flat = targets_oh[:, c].reshape(-1)
            
            intersection = (p_flat * t_flat).sum()
            union = p_flat.sum() + t_flat.sum()
            
            # Global TP/FP/FN accumulation
            self.tp[c] += intersection
            self.fp[c] += (p_flat.sum() - intersection)
            self.fn[c] += (t_flat.sum() - intersection)
            
            # Batch-wise Dice/IoU accumulation
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            iou = (intersection + 1e-6) / (union - intersection + 1e-6)
            
            self.dice_sum[c] += dice
            self.iou_sum[c] += iou
            
        # HD95 Compliance (MONAI)
        # compute_hausdorff_distance expects (B, C, spatial...)
        # include_background=True usually, but we iterate.
        # We can compute all classes at once.
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # percentile=95
                hd95_batch = compute_hausdorff_distance(
                    y_pred=preds_oh, 
                    y=targets_oh, 
                    include_background=True, 
                    percentile=95.0,
                    spacing=None  # Pixel space
                )
            # hd95_batch is (B, C)
            
            for c in range(self.num_classes):
                # Filter NaNs/Infs (happens if class missing in both pred and target, or just one)
                # MONAI returns NaN if one is empty. We mostly care if target exists.
                # Common practice: if target is empty, skip. If target exists but pred empty, HD is high (inf).
                # MONAI behavior: Nan if both empty. Inf if one empty.
                
                valid_vals = hd95_batch[:, c]
                valid_mask = ~torch.isnan(valid_vals) & ~torch.isinf(valid_vals)
                
                if valid_mask.any():
                    self.hd95_sum[c] += valid_vals[valid_mask].sum()
                    self.hd95_counts[c] += valid_mask.sum()
                    
        except Exception as e:
            # Fallback or strict error? 
            # Often happens if shapes are weird or empty batch.
            pass

    def compute(self):
        """
        Compute final aggregated metrics.
        Returns:
            dict containing scalar values and per-class lists.
        """
        metrics = {}
        
        # Global Accuracy
        metrics['accuracy'] = self.total_correct_pixels / max(self.total_pixels, 1)
        
        # Per-class metrics
        dice_scores = []
        iou_scores = []
        hd95_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for c in range(self.num_classes):
            # Batch-averaged Dice/IoU
            dice_scores.append((self.dice_sum[c] / max(self.batches, 1)).item())
            iou_scores.append((self.iou_sum[c] / max(self.batches, 1)).item())
            
            # Batch-averaged HD95
            if self.hd95_counts[c] > 0:
                hd95_scores.append((self.hd95_sum[c] / self.hd95_counts[c]).item())
            else:
                hd95_scores.append(float('nan')) # Or 0.0 or inf
            
            # Global-based Precision/Recall/F1
            p = (self.tp[c] / (self.tp[c] + self.fp[c] + 1e-6)).item()
            r = (self.tp[c] / (self.tp[c] + self.fn[c] + 1e-6)).item()
            f1 = 2 * p * r / (p + r + 1e-6) if (p + r) > 0 else 0.0
            
            precision_scores.append(p)
            recall_scores.append(r)
            f1_scores.append(f1)
            
        metrics['dice_scores'] = dice_scores
        metrics['iou'] = iou_scores
        metrics['hd95'] = hd95_scores
        metrics['precision'] = precision_scores
        metrics['recall'] = recall_scores
        metrics['f1_score'] = f1_scores
        
        return metrics
