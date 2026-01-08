import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import math
import random
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, List, Union


class JointVectorRotation:
    def __init__(self, angle_range: Tuple[float, float] = (-180, 180), p: float = 1.0):
        self.angle_range = angle_range
        self.p = p

    def __call__(self, image: torch.Tensor, riesz_vec: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return image, riesz_vec, mask

        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask.unsqueeze(0) if mask.dim() == 2 else mask,
                        angle, interpolation=TF.InterpolationMode.NEAREST)
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        riesz_vec = TF.rotate(riesz_vec, angle, interpolation=TF.InterpolationMode.BILINEAR)

        theta = torch.tensor(-angle * math.pi / 180.0, dtype=riesz_vec.dtype)
        cos_a, sin_a = torch.cos(theta), torch.sin(theta)
        rx, ry = riesz_vec[0], riesz_vec[1]
        new_riesz = torch.stack([rx * cos_a - ry * sin_a, rx * sin_a + ry * cos_a], dim=0)
        return image, new_riesz, mask


class JointRandomFlip:
    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.5):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, image: torch.Tensor, riesz_vec: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.p_horizontal:
            image = TF.hflip(image)
            mask = TF.hflip(mask.unsqueeze(0) if mask.dim() == 2 else mask)
            if mask.dim() == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            riesz_vec = TF.hflip(riesz_vec)
            riesz_vec[0] = -riesz_vec[0]

        if random.random() < self.p_vertical:
            image = TF.vflip(image)
            mask = TF.vflip(mask.unsqueeze(0) if mask.dim() == 2 else mask)
            if mask.dim() == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            riesz_vec = TF.vflip(riesz_vec)
            riesz_vec[1] = -riesz_vec[1]
        return image, riesz_vec, mask


class MonogenicDataset(Dataset):
    def __init__(self, data_dir: Union[str, Path], img_size: int = 256,
                 augment: bool = True, normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize

        self.samples = sorted(list(self.data_dir.glob("*.npy")))
        if len(self.samples) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")

        if augment:
            self.rotation = JointVectorRotation(angle_range=(-30, 30), p=0.5)
            self.flip = JointRandomFlip(p_horizontal=0.5, p_vertical=0.5)
        else:
            self.rotation = None
            self.flip = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, path: Path) -> Dict[str, np.ndarray]:
        return np.load(path, allow_pickle=True).item()

    def _preprocess(self, data: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intensity = torch.from_numpy(data['intensity']).float()
        riesz_x = torch.from_numpy(data['riesz_x']).float()
        riesz_y = torch.from_numpy(data['riesz_y']).float()
        mask = torch.from_numpy(data['mask']).long()

        if intensity.dim() == 2:
            intensity = intensity.unsqueeze(0)
        riesz_vec = torch.stack([riesz_x, riesz_y], dim=0)

        if self.normalize:
            i_min, i_max = intensity.min(), intensity.max()
            if i_max > i_min:
                intensity = (intensity - i_min) / (i_max - i_min)
        return intensity, riesz_vec, mask

    def _resize(self, intensity: torch.Tensor, riesz_vec: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if intensity.shape[-1] != self.img_size or intensity.shape[-2] != self.img_size:
            intensity = F.interpolate(
                intensity.unsqueeze(0), size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=True
            ).squeeze(0)
            riesz_vec = F.interpolate(
                riesz_vec.unsqueeze(0), size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=True
            ).squeeze(0)
            mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=(self.img_size, self.img_size), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        return intensity, riesz_vec, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._load_sample(self.samples[idx])
        intensity, riesz_vec, mask = self._preprocess(data)
        intensity, riesz_vec, mask = self._resize(intensity, riesz_vec, mask)

        if self.augment:
            if self.rotation:
                intensity, riesz_vec, mask = self.rotation(intensity, riesz_vec, mask)
            if self.flip:
                intensity, riesz_vec, mask = self.flip(intensity, riesz_vec, mask)

        energy = torch.sqrt(intensity ** 2 + riesz_vec[0:1] ** 2 + riesz_vec[1:2] ** 2 + 1e-8)
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        combined_input = torch.cat([intensity, riesz_vec], dim=0)

        return {
            'input': combined_input,
            'intensity': intensity,
            'riesz': riesz_vec,
            'mask': mask,
            'energy': energy
        }


class MedicalImageSegmentationDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray,
                 img_size: int = 256, normalize: bool = True, augment: bool = False):
        self.images = images
        self.masks = masks
        self.img_size = img_size
        self.normalize = normalize
        self.augment = augment
        assert len(images) == len(masks)

        if self.images.ndim == 3:
            self.images = np.expand_dims(self.images, axis=1)

        self.images = torch.from_numpy(self.images).float()
        self.masks = torch.from_numpy(self.masks).long()
        if self.normalize:
            self._normalize_images()

    def _normalize_images(self):
        for i in range(len(self.images)):
            img = self.images[i]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                self.images[i] = (img - img_min) / (img_max - img_min)

    def _resize_if_needed(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple:
        if image.shape[-1] != self.img_size or image.shape[-2] != self.img_size:
            image = F.interpolate(
                image.unsqueeze(0), size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=True
            ).squeeze(0)
            mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=(self.img_size, self.img_size), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        return image, mask

    def _augment_data(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple:
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        image = torch.rot90(image, k=k, dims=[-2, -1])
        mask = torch.rot90(mask, k=k, dims=[-2, -1])
        return image, mask

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]
        image, mask = self._resize_if_needed(image, mask)
        if self.augment:
            image, mask = self._augment_data(image, mask)
        return image, mask


class MetricsCalculator:
    @staticmethod
    def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
        if pred.ndim == 4:
            pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred == target).sum().float()
        union = pred.numel()
        return ((2.0 * intersection + smooth) / (union + smooth)).item()

    @staticmethod
    def iou_score(pred: torch.Tensor, target: torch.Tensor,
                  num_classes: int = 3, smooth: float = 1e-5) -> dict:
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


# Re-export ACDC dataset for convenience
try:
    from .acdc_dataset import ACDCDataset2D
except ImportError:
    ACDCDataset2D = None
