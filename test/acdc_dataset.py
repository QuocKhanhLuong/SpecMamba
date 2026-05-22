"""Robust slice-level ACDC dataset loader for isolated SSR experiments."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F


PATIENT_RE = re.compile(r"^(patient\d+)")


def patient_id_from_case(case_id: str) -> str:
    """Return the patient id from an ACDC case id such as `patient001_ED`."""
    match = PATIENT_RE.match(case_id)
    return match.group(1) if match else case_id.split("_")[0]


def discover_acdc_cases(data_root: str | Path) -> list[str]:
    """Discover paired ACDC volume/mask case ids."""
    root = Path(data_root)
    volumes_dir = root / "volumes"
    masks_dir = root / "masks"
    if not volumes_dir.exists():
        raise FileNotFoundError(f"Missing ACDC volumes directory: {volumes_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing ACDC masks directory: {masks_dir}")

    volume_ids = {p.stem for p in volumes_dir.glob("*.npy")}
    mask_ids = {p.stem for p in masks_dir.glob("*.npy")}
    missing_masks = sorted(volume_ids - mask_ids)
    missing_volumes = sorted(mask_ids - volume_ids)
    if missing_masks or missing_volumes:
        raise FileNotFoundError(
            "Volume/mask pairing mismatch. "
            f"Missing masks: {missing_masks[:5]}, missing volumes: {missing_volumes[:5]}"
        )
    if not volume_ids:
        raise FileNotFoundError(f"No .npy volumes found under {volumes_dir}")
    return sorted(volume_ids)


def deterministic_patient_split(
    case_ids: Iterable[str],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str], dict[str, object]]:
    """Create a deterministic patient-level split from case ids."""
    patient_to_cases: dict[str, list[str]] = {}
    for case_id in sorted(case_ids):
        patient_to_cases.setdefault(patient_id_from_case(case_id), []).append(case_id)

    patients = sorted(patient_to_cases)
    if len(patients) < 2:
        raise ValueError("Need at least two ACDC patients for a train/val split.")
    rng = np.random.default_rng(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_fraction)
    n_train = min(max(n_train, 1), len(shuffled) - 1)
    train_patients = sorted(shuffled[:n_train])
    val_patients = sorted(shuffled[n_train:])

    train_cases = sorted(c for p in train_patients for c in patient_to_cases[p])
    val_cases = sorted(c for p in val_patients for c in patient_to_cases[p])
    manifest = {
        "split_level": "patient",
        "seed": seed,
        "train_fraction": train_fraction,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "train_cases": train_cases,
        "val_cases": val_cases,
    }
    return train_cases, val_cases, manifest


def load_or_create_split(
    data_root: str | Path,
    seed: int = 42,
    train_fraction: float = 0.8,
    split_manifest: str | Path = "splits/acdc_patient_split_seed42.json",
) -> tuple[list[str], list[str], dict[str, object]]:
    """Use the repo patient split when present, otherwise create one."""
    all_cases = discover_acdc_cases(data_root)
    split_path = Path(split_manifest)
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            manifest = json.load(f)
        splits = manifest.get("splits", {})
        train_cases = sorted(Path(v).stem for v in splits.get("train", {}).get("volumes", []))
        val_cases = sorted(Path(v).stem for v in splits.get("val", {}).get("volumes", []))
        if train_cases and val_cases:
            _validate_split(train_cases, val_cases, all_cases)
            return train_cases, val_cases, manifest

    train_cases, val_cases, manifest = deterministic_patient_split(
        all_cases,
        train_fraction=train_fraction,
        seed=seed,
    )
    _validate_split(train_cases, val_cases, all_cases)
    return train_cases, val_cases, manifest


class ACDCSSRSliceDataset(Dataset):
    """Slice-level ACDC dataset for SSR debug training.

    Supports:
        `input_mode="2d"`: image `[1, H, W]`.
        `input_mode="25d"`: image `[5, H, W]` from mirrored neighboring slices.
    """

    def __init__(
        self,
        data_root: str | Path = "preprocessed_data/ACDC",
        case_ids: Iterable[str] | None = None,
        input_mode: str = "2d",
        image_size: int = 128,
        foreground_only: bool = True,
        max_slices: int | None = None,
        seed: int = 42,
        use_memmap: bool = True,
        max_cache: int = 8,
    ) -> None:
        self.data_root = Path(data_root)
        self.volumes_dir = self.data_root / "volumes"
        self.masks_dir = self.data_root / "masks"
        self.input_mode = str(input_mode).lower()
        if self.input_mode not in {"2d", "25d"}:
            raise ValueError("input_mode must be '2d' or '25d'")
        self.image_size = int(image_size)
        self.foreground_only = bool(foreground_only)
        self.use_memmap = bool(use_memmap)
        self.max_cache = int(max_cache)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, int, float, float]] = OrderedDict()

        all_cases = discover_acdc_cases(self.data_root)
        selected = sorted(set(case_ids or all_cases))
        missing = sorted(set(selected) - set(all_cases))
        if missing:
            raise FileNotFoundError(f"Requested ACDC cases are missing: {missing[:5]}")

        self.case_ids = selected
        self.vol_paths = [self.volumes_dir / f"{case_id}.npy" for case_id in self.case_ids]
        self.mask_paths = [self.masks_dir / f"{case_id}.npy" for case_id in self.case_ids]
        self.slice_counts: dict[int, int] = {}
        self.depth_axes: dict[int, int] = {}
        self.index_map: list[tuple[int, int]] = []

        volume_info = _load_volume_info(self.data_root)
        for vol_idx, case_id in enumerate(self.case_ids):
            arr = np.load(self.vol_paths[vol_idx], mmap_mode="r")
            mask_arr = np.load(self.mask_paths[vol_idx], mmap_mode="r")
            if arr.ndim != 3 or mask_arr.ndim != 3:
                raise ValueError(f"Expected 3D volume/mask arrays for {case_id}")
            if arr.shape != mask_arr.shape:
                raise ValueError(f"Volume/mask shape mismatch for {case_id}: {arr.shape} vs {mask_arr.shape}")
            expected_slices = volume_info.get(case_id, {}).get("num_slices")
            depth_axis = _infer_depth_axis(arr.shape, expected_slices)
            n_slices = arr.shape[depth_axis]
            self.slice_counts[vol_idx] = n_slices
            self.depth_axes[vol_idx] = depth_axis
            for slice_idx in range(n_slices):
                if self.foreground_only:
                    gt = _take_slice(mask_arr, slice_idx, depth_axis)
                    if not np.any(gt > 0):
                        continue
                self.index_map.append((vol_idx, slice_idx))

        if max_slices is not None and max_slices > 0 and len(self.index_map) > max_slices:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(self.index_map), size=max_slices, replace=False))
            self.index_map = [self.index_map[int(i)] for i in keep]

        if not self.index_map:
            raise ValueError(
                "ACDC dataset produced zero slices. "
                "Try foreground_only=false or check the preprocessed masks."
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str | int]:
        vol_idx, slice_idx = self.index_map[idx]
        vol, mask, depth_axis, mean, std = self._load(vol_idx)
        n_slices = self.slice_counts[vol_idx]

        if self.input_mode == "2d":
            img = _take_slice(vol, slice_idx, depth_axis).astype(np.float32)
            img = ((img - mean) / std)[None, ...]  # [1, H, W]
        else:
            channels = []
            for offset in (-2, -1, 0, 1, 2):
                neighbor = _reflect_index(slice_idx + offset, n_slices)
                img = _take_slice(vol, neighbor, depth_axis).astype(np.float32)
                channels.append((img - mean) / std)
            img = np.stack(channels, axis=0)  # [5, H, W]

        gt = _take_slice(mask, slice_idx, depth_axis).astype(np.int64)
        image_t = torch.from_numpy(np.ascontiguousarray(img)).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(gt)).long()
        image_t, mask_t = _resize_sample(image_t, mask_t, self.image_size)

        return {
            "image": image_t,
            "mask": mask_t,
            "case_id": self.case_ids[vol_idx],
            "slice_idx": int(slice_idx),
        }

    def _load(self, vol_idx: int) -> tuple[np.ndarray, np.ndarray, int, float, float]:
        if vol_idx in self._cache:
            self._cache.move_to_end(vol_idx)
            return self._cache[vol_idx]

        mode = "r" if self.use_memmap else None
        vol = np.load(self.vol_paths[vol_idx], mmap_mode=mode)
        mask = np.load(self.mask_paths[vol_idx], mmap_mode=mode)
        depth_axis = self.depth_axes[vol_idx]
        finite = np.asarray(vol, dtype=np.float32)
        mean = float(np.nanmean(finite))
        std = float(np.nanstd(finite))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        item = (vol, mask, depth_axis, mean, std)
        self._cache[vol_idx] = item
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return item


def _validate_split(train_cases: list[str], val_cases: list[str], all_cases: list[str]) -> None:
    all_set = set(all_cases)
    missing = sorted((set(train_cases) | set(val_cases)) - all_set)
    if missing:
        raise FileNotFoundError(f"Split references missing ACDC cases: {missing[:5]}")
    train_patients = {patient_id_from_case(case_id) for case_id in train_cases}
    val_patients = {patient_id_from_case(case_id) for case_id in val_cases}
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(f"Patient leakage in train/val split: {sorted(overlap)[:10]}")
    if not train_cases or not val_cases:
        raise ValueError("Train/val split must not be empty.")


def _load_volume_info(data_root: Path) -> dict[str, dict[str, object]]:
    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    info = meta.get("volume_info", {})
    return info if isinstance(info, dict) else {}


def _infer_depth_axis(shape: tuple[int, int, int], expected_slices: int | None = None) -> int:
    if expected_slices is not None:
        matches = [axis for axis, size in enumerate(shape) if int(size) == int(expected_slices)]
        if len(matches) == 1:
            return matches[0]
    small_axes = [axis for axis, size in enumerate(shape) if size <= 32]
    if len(small_axes) == 1:
        return small_axes[0]
    return int(np.argmin(shape))


def _take_slice(volume: np.ndarray, slice_idx: int, depth_axis: int) -> np.ndarray:
    if depth_axis == 0:
        return volume[slice_idx, :, :]
    if depth_axis == 1:
        return volume[:, slice_idx, :]
    return volume[:, :, slice_idx]


def _reflect_index(index: int, n: int) -> int:
    if n <= 1:
        return 0
    while index < 0 or index >= n:
        if index < 0:
            index = -index
        if index >= n:
            index = 2 * (n - 1) - index
    return int(index)


def _resize_sample(image: Tensor, mask: Tensor, image_size: int) -> tuple[Tensor, Tensor]:
    if image_size <= 0 or tuple(image.shape[-2:]) == (image_size, image_size):
        return image, mask
    image = F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=(image_size, image_size),
        mode="nearest",
    ).squeeze(0).squeeze(0).long()
    return image, mask
