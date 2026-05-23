"""
Preprocess MyoPS2020 dataset: Multi-sequence Cardiac MRI → .npy volumes

MyoPS2020 has 3 MRI modalities per patient (C0/bSSFP, DE/LGE, T2), all
co-registered to the same spatial domain. Each modality is stored as a
separate volume but shares the same geometry and ground truth.

Ground truth labels (remapped to contiguous classes for training):
    Original → Class
    0        → 0  Background
    200      → 1  Myocardium (healthy)
    500      → 2  Left Ventricle (LV)
    600      → 3  Right Ventricle (RV)
    1220     → 4  Edema
    2221     → 5  Scar

Output format (matching ACDC preprocessing convention):
    preprocessed_data/MyoPS2020/training/
        volumes/  ← patient{ID}_{modality}.npy  (H, W, D) float32
        masks/    ← patient{ID}_{modality}.npy  (H, W, D) uint8
        metadata.json

Usage:
    python scripts/preprocess_myops2020.py \\
        --input data/MyoPS2020 \\
        --output preprocessed_data/MyoPS2020/training \\
        --size 224
"""

import os
import re
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from skimage.transform import resize


# ─── Label remapping ─────────────────────────────────────────────────────────
ORIGINAL_LABELS = [0, 200, 500, 600, 1220, 2221]
CLASS_NAMES = ["Background", "Myocardium", "LV", "RV", "Edema", "Scar"]
NUM_CLASSES = len(CLASS_NAMES)

LABEL_REMAP = {orig: idx for idx, orig in enumerate(ORIGINAL_LABELS)}


def remap_labels(mask):
    """Remap original MyoPS label values (0,200,500,600,1220,2221) → (0,1,2,3,4,5)."""
    out = np.zeros_like(mask, dtype=np.uint8)
    for orig, new in LABEL_REMAP.items():
        out[mask == orig] = new
    return out


def normalize_zscore(image):
    """Z-score normalization with outlier clipping."""
    p05 = np.percentile(image, 0.5)
    p995 = np.percentile(image, 99.5)
    image = np.clip(image, p05, p995)
    mean, std = np.mean(image), np.std(image)
    return (image - mean) / std if std > 0 else image - mean


def discover_patients(data_dir):
    """Find all patient IDs from the train25 folder."""
    train_dir = os.path.join(data_dir, "train25")
    ids = set()
    for f in os.listdir(train_dir):
        m = re.match(r"myops_training_(\d+)_", f)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def preprocess_patient(data_dir, pid, target_size=(224, 224)):
    """Process one MyoPS2020 patient: load all 3 modalities + ground truth.

    Each modality is saved as a separate volume (same convention as ACDC
    which saves ED and ES as separate volumes). The ground truth is shared
    across modalities since images are co-registered.

    Returns:
        list of (volume, mask, volume_id, spacing_info) tuples
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

    results = []

    for mod in ["C0", "DE", "T2"]:
        img_path = os.path.join(train_dir, f"myops_training_{pid}_{mod}.nii.gz")
        if not os.path.exists(img_path):
            print(f"  Warning: Missing {mod} for patient {pid}")
            continue

        try:
            img_nii = nib.load(img_path)
            img_data = img_nii.get_fdata()
            orig_shape = img_data.shape
            orig_spacing = img_nii.header.get_zooms()  # (sx, sy, sz) in mm
            num_slices = img_data.shape[2]

            # Z-score normalize
            img_norm = normalize_zscore(img_data)

            # Remap ground truth labels to contiguous 0..5
            gd_remapped = remap_labels(gd_data)

            # Compute effective spacing after resize
            eff_spacing_y = float(orig_spacing[0]) * orig_shape[0] / target_size[0]
            eff_spacing_x = float(orig_spacing[1]) * orig_shape[1] / target_size[1]
            eff_spacing_z = float(orig_spacing[2])

            # Resize slice-by-slice
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)

            for i in range(num_slices):
                resized_img[:, :, i] = resize(
                    img_norm[:, :, i], target_size,
                    order=1, preserve_range=True, anti_aliasing=True, mode='reflect'
                )
                resized_mask[:, :, i] = resize(
                    gd_remapped[:, :, i], target_size,
                    order=0, preserve_range=True, anti_aliasing=False, mode='reflect'
                ).astype(np.uint8)

            spacing_info = {
                'orig_shape': [int(s) for s in orig_shape],
                'orig_spacing': [float(s) for s in orig_spacing],
                'effective_spacing': [eff_spacing_z, eff_spacing_y, eff_spacing_x],
            }

            volume_id = f"patient{pid}_{mod}"
            results.append((resized_img, resized_mask, volume_id, spacing_info))

        except Exception as e:
            print(f"  Error: patient {pid} modality {mod}: {e}")

    return results


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
    print(f"MyoPS2020 Preprocessing: {len(patient_ids)} patients × 3 modalities")
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
        'modalities': ['C0', 'DE', 'T2'],
        'modality_names': {'C0': 'bSSFP', 'DE': 'LGE', 'T2': 'T2-weighted'},
        'target_size': list(target_size),
        'total_volumes': len(volume_info),
        'total_patients': len(patient_ids),
        'volume_info': volume_info,
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Total volumes: {len(volume_info)} ({len(patient_ids)} patients × 3 modalities)")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
