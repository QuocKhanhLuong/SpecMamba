"""
Preprocess MyoPS2020 dataset: multi-sequence Cardiac MRI -> .npy volumes

MyoPS2020 has 3 MRI modalities per patient (C0/bSSFP, DE/LGE, T2), all
co-registered to the same spatial domain. The preprocessed output keeps these
modalities paired as channels of one patient volume.

Ground truth labels (remapped to contiguous classes for training):
    Original → Class
    0        → 0  Background
    200      → 1  Myocardium (healthy)
    500      → 2  Left Ventricle (LV)
    600      → 3  Right Ventricle (RV)
    1220     → 4  Edema
    2221     → 5  Scar

Output format:
    preprocessed_data/MyoPS2020/training/
        volumes/  <- patient{ID}.npy  (3, H, W, D) float32, modality order C0/DE/T2
        masks/    <- patient{ID}.npy  (H, W, D) uint8
        metadata.json

Usage:
    python scripts/preprocess_myops2020.py \\
        --input data/MyoPS2020 \\
        --output preprocessed_data/MyoPS2020/training \\
        --size 224
"""

import argparse
import json
import os
import re

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from skimage.transform import resize
from tqdm import tqdm


MODALITIES = ["C0", "DE", "T2"]
MODALITY_NAMES = {"C0": "bSSFP", "DE": "LGE", "T2": "T2-weighted"}

ORIGINAL_LABELS = [0, 200, 500, 600, 1220, 2221]
CLASS_NAMES = ["Background", "Myocardium", "LV", "RV", "Edema", "Scar"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_REMAP = {orig: idx for idx, orig in enumerate(ORIGINAL_LABELS)}
ALLOWED_LABELS = set(ORIGINAL_LABELS)


def remap_labels(mask):
    """Remap original MyoPS labels to contiguous classes after validation."""
    mask_int = np.rint(mask).astype(np.int32)
    unknown = sorted(set(np.unique(mask_int).tolist()) - ALLOWED_LABELS)
    if unknown:
        raise ValueError(
            "Unknown MyoPS labels encountered: "
            f"{unknown}. Expected only {sorted(ALLOWED_LABELS)}."
        )

    out = np.zeros_like(mask, dtype=np.uint8)
    for orig, new in LABEL_REMAP.items():
        out[mask_int == orig] = new
    return out


def normalize_zscore(image):
    """Robust per-volume z-score normalization.

    Percentiles and statistics are estimated from nonzero finite voxels when
    available, which is safer for sparse MR backgrounds than using all padded
    image values.
    """
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    region = finite & (image != 0)
    if not np.any(region):
        region = finite
    values = image[region]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32), {
            "clip_percentiles": [0.5, 99.5],
            "clip_values": [0.0, 0.0],
            "mean": 0.0,
            "std": 1.0,
            "stats_region": "empty",
        }

    p05, p995 = np.percentile(values, [0.5, 99.5])
    clipped = np.clip(image, p05, p995)
    clipped_values = clipped[region]
    mean = float(np.mean(clipped_values))
    std = float(np.std(clipped_values))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    normalized = (clipped - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32), {
        "clip_percentiles": [0.5, 99.5],
        "clip_values": [float(p05), float(p995)],
        "mean": mean,
        "std": std,
        "stats_region": "nonzero_finite" if np.any(finite & (image != 0)) else "finite",
    }


def discover_patients(data_dir):
    """Find all patient IDs from the train25 folder."""
    train_dir = os.path.join(data_dir, "train25")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Missing MyoPS train25 directory: {train_dir}. "
            "Expected raw layout with train25/ and train25_myops_gd/."
        )
    ids = set()
    for f in os.listdir(train_dir):
        m = re.match(r"myops_training_(\d+)_", f)
        if m:
            ids.add(int(m.group(1)))
    if not ids:
        raise FileNotFoundError(f"No myops_training_* files found under {train_dir}")
    return sorted(ids)


def _histogram(values):
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(unique, counts)}


def _validate_geometry(pid, reference_nii, candidate_nii, candidate_name):
    """Fail fast if a modality/GT is not aligned to the reference image."""
    ref_shape = reference_nii.shape
    cand_shape = candidate_nii.shape
    if cand_shape != ref_shape:
        raise ValueError(
            f"Patient {pid}: {candidate_name} shape {cand_shape} does not match "
            f"reference C0 shape {ref_shape}."
        )

    ref_affine = reference_nii.affine
    cand_affine = candidate_nii.affine
    if not np.allclose(cand_affine, ref_affine, atol=1e-3):
        raise ValueError(
            f"Patient {pid}: {candidate_name} affine does not match C0 affine. "
            "Do not share GT across modalities until geometry is aligned."
        )

    ref_codes = aff2axcodes(ref_affine)
    cand_codes = aff2axcodes(cand_affine)
    if cand_codes != ref_codes:
        raise ValueError(
            f"Patient {pid}: {candidate_name} orientation {cand_codes} does not "
            f"match C0 orientation {ref_codes}."
        )

    ref_zooms = reference_nii.header.get_zooms()[:3]
    cand_zooms = candidate_nii.header.get_zooms()[:3]
    if not np.allclose(cand_zooms, ref_zooms, atol=1e-3):
        raise ValueError(
            f"Patient {pid}: {candidate_name} spacing {cand_zooms} does not "
            f"match C0 spacing {ref_zooms}."
        )


def preprocess_patient(data_dir, pid, target_size=(224, 224)):
    """Process one MyoPS2020 patient: load all 3 modalities + ground truth.

    The returned image volume is channel-first `[3, H, W, D]`, preserving the
    `C0`, `DE`, `T2` pairing needed for MyoPS multi-sequence experiments.

    Returns:
        list with one `(volume, mask, volume_id, spacing_info)` tuple. An empty
        list is returned only when the patient has no ground truth.
    """
    train_dir = os.path.join(data_dir, "train25")
    gd_dir = os.path.join(data_dir, "train25_myops_gd")

    # Load ground truth
    gd_path = os.path.join(gd_dir, f"myops_training_{pid}_gd.nii.gz")
    if not os.path.exists(gd_path):
        print(f"  Warning: No ground truth for patient {pid}")
        return []

    gd_nii = nib.load(gd_path)
    gd_data = gd_nii.get_fdata()
    gd_remapped = remap_labels(gd_data)

    modality_niis = {}
    for mod in MODALITIES:
        img_path = os.path.join(train_dir, f"myops_training_{pid}_{mod}.nii.gz")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Patient {pid}: missing required modality {mod}: {img_path}")
        modality_niis[mod] = nib.load(img_path)

    ref_nii = modality_niis["C0"]
    _validate_geometry(pid, ref_nii, gd_nii, "ground truth")
    for mod, img_nii in modality_niis.items():
        _validate_geometry(pid, ref_nii, img_nii, mod)

    orig_shape = ref_nii.shape
    if len(orig_shape) != 3:
        raise ValueError(f"Patient {pid}: expected 3D NIfTI volumes, got shape {orig_shape}")
    orig_spacing = ref_nii.header.get_zooms()[:3]
    num_slices = orig_shape[2]

    eff_spacing_x = float(orig_spacing[0]) * orig_shape[0] / target_size[0]
    eff_spacing_y = float(orig_spacing[1]) * orig_shape[1] / target_size[1]
    eff_spacing_z = float(orig_spacing[2])

    resized_modalities = np.zeros(
        (len(MODALITIES), target_size[0], target_size[1], num_slices),
        dtype=np.float32,
    )
    resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
    normalization_stats = {}

    for mod_idx, mod in enumerate(MODALITIES):
        img_data = modality_niis[mod].get_fdata(dtype=np.float32)
        img_norm, stats = normalize_zscore(img_data)
        normalization_stats[mod] = stats
        for i in range(num_slices):
            resized_modalities[mod_idx, :, :, i] = resize(
                img_norm[:, :, i],
                target_size,
                order=1,
                preserve_range=True,
                anti_aliasing=True,
                mode="reflect",
            )

    for i in range(num_slices):
        resized_mask[:, :, i] = resize(
            gd_remapped[:, :, i],
            target_size,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
            mode="reflect",
        ).astype(np.uint8)

    spacing_info = {
        "modalities": MODALITIES,
        "num_modalities": len(MODALITIES),
        "volume_layout": "C,H,W,D",
        "mask_layout": "H,W,D",
        "orig_shape": [int(s) for s in orig_shape],
        "orig_spacing_xyz": [float(s) for s in orig_spacing],
        "effective_spacing_xyz": [eff_spacing_x, eff_spacing_y, eff_spacing_z],
        # Kept for compatibility with existing metadata consumers.
        "effective_spacing": [eff_spacing_z, eff_spacing_y, eff_spacing_x],
        "orientation": "".join(aff2axcodes(ref_nii.affine)),
        "affine": ref_nii.affine.astype(float).tolist(),
        "normalization": normalization_stats,
        "label_histogram_original": _histogram(np.rint(gd_data).astype(np.int32)),
        "label_histogram_remapped": _histogram(gd_remapped),
    }

    return [(resized_modalities, resized_mask, f"patient{pid}", spacing_info)]


def main():
    parser = argparse.ArgumentParser(description='Preprocess MyoPS2020 dataset')
    parser.add_argument('--input', type=str, default='data/MyoPS2020',
                        help='Path to raw MyoPS2020 data root')
    parser.add_argument('--output', type=str, default='preprocessed_data/MyoPS2020/training',
                        help='Output directory for preprocessed data')
    parser.add_argument('--size', type=int, default=224,
                        help='Target spatial size (default: 224)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-process even if output exists')
    args = parser.parse_args()

    target_size = (args.size, args.size)
    os.makedirs(args.output, exist_ok=True)
    volumes_dir = os.path.join(args.output, 'volumes')
    masks_dir = os.path.join(args.output, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    patient_ids = discover_patients(args.input)
    print(f"MyoPS2020 Preprocessing: {len(patient_ids)} paired patients × 3 modalities")
    print(f"  Labels: {dict(zip(CLASS_NAMES, range(NUM_CLASSES)))}")
    print(f"  Target size: {target_size}")
    print()

    volume_info = {}
    processed, skipped = 0, 0

    for pid in tqdm(patient_ids, desc="MyoPS2020"):
        for volume, mask, volume_id, spacing_info in preprocess_patient(args.input, pid, target_size):
            vol_path = os.path.join(volumes_dir, f'{volume_id}.npy')
            mask_path = os.path.join(masks_dir, f'{volume_id}.npy')

            if not args.no_skip and os.path.exists(vol_path):
                skipped += 1
                volume_info[volume_id] = {
                    'num_slices': int(mask.shape[2]),
                    **spacing_info,
                }
                continue

            np.save(vol_path, volume)
            np.save(mask_path, mask)
            volume_info[volume_id] = {
                'num_slices': int(mask.shape[2]),
                **spacing_info,
            }
            processed += 1

    # Save metadata
    metadata = {
        'dataset': 'MyoPS2020',
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'original_labels': ORIGINAL_LABELS,
        'label_remap': {str(k): v for k, v in LABEL_REMAP.items()},
        'modalities': MODALITIES,
        'modality_names': MODALITY_NAMES,
        'input_channels': len(MODALITIES),
        'volume_layout': 'C,H,W,D',
        'mask_layout': 'H,W,D',
        'target_size': list(target_size),
        'total_volumes': len(volume_info),
        'total_patients': len(patient_ids),
        'preprocessing': {
            'pair_modalities_per_patient': True,
            'normalize': 'per-modality robust z-score over nonzero finite voxels',
            'image_resize_order': 1,
            'mask_resize_order': 0,
            'geometry_validation': 'shape, affine, orientation, spacing must match C0',
        },
        'volume_info': volume_info,
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Total volumes: {len(volume_info)} paired patient volumes")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
