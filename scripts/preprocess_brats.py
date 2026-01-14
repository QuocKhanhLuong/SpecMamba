"""
Preprocess BraTS21 dataset: Brain Tumor MRI → .npy volumes
Data structure:
    data/BraTS21/
    ├── BraTS2021_00000/
    │   ├── BraTS2021_00000_flair.nii.gz
    │   ├── BraTS2021_00000_t1.nii.gz
    │   ├── BraTS2021_00000_t1ce.nii.gz
    │   ├── BraTS2021_00000_t2.nii.gz
    │   └── BraTS2021_00000_seg.nii.gz

4 modalities: T1, T1ce, T2, FLAIR (stacked as 4 channels)
Classes: 0=BG, 1=NCR (necrotic), 2=ED (edema), 3=ET (enhancing tumor)
Note: Original label 4 is remapped to 3

Usage:
    python scripts/preprocess_brats.py --input data/BraTS21 --output preprocessed_data/BraTS21/training
"""

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from skimage.transform import resize


def normalize_zscore_masked(image, mask):
    """Z-score normalization within brain mask."""
    brain = image[mask > 0]
    if len(brain) > 0:
        mean, std = np.mean(brain), np.std(brain)
        if std > 0:
            return (image - mean) / std
        return image - mean
    return image


def main():
    parser = argparse.ArgumentParser(description='Preprocess BraTS21 dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to data/BraTS21')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--no-skip', action='store_true')
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    os.makedirs(args.output, exist_ok=True)
    volumes_dir = os.path.join(args.output, 'volumes')
    masks_dir = os.path.join(args.output, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get patient folders (BraTS2021_XXXXX)
    patient_folders = sorted([
        d for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d)) and d.startswith('BraTS')
    ])
    
    print(f"BraTS21 Preprocessing: {len(patient_folders)} patients")
    
    volume_info = {}
    processed, skipped = 0, 0
    
    for patient_id in tqdm(patient_folders, desc="BraTS21"):
        patient_path = os.path.join(args.input, patient_id)
        
        vol_path = os.path.join(volumes_dir, f'{patient_id}.npy')
        mask_path_out = os.path.join(masks_dir, f'{patient_id}.npy')
        
        if not args.no_skip and os.path.exists(vol_path):
            skipped += 1
            continue
        
        # Find modality files
        modalities = ['t1', 't1ce', 't2', 'flair']
        mod_files = {}
        seg_path = None
        
        for f in os.listdir(patient_path):
            f_lower = f.lower()
            if 'seg' in f_lower:
                seg_path = os.path.join(patient_path, f)
            else:
                for mod in modalities:
                    if f_lower.endswith(f'_{mod}.nii.gz') or f_lower.endswith(f'_{mod}.nii'):
                        mod_files[mod] = os.path.join(patient_path, f)
        
        if not seg_path or len(mod_files) < 4:
            print(f"  Warning: Missing files for {patient_id}")
            continue
        
        try:
            # Load all 4 modalities
            mod_data = [nib.load(mod_files[mod]).get_fdata() for mod in modalities]
            img_4d = np.stack(mod_data, axis=-1)  # (H, W, D, 4)
            
            mask_data = nib.load(seg_path).get_fdata().astype(np.uint8)
            mask_data[mask_data == 4] = 3  # Remap label 4 -> 3
            
            num_slices = img_4d.shape[2]
            
            # Normalize each modality within brain mask
            brain_mask = (img_4d[:, :, :, 0] > 0)
            for m in range(4):
                img_4d[:, :, :, m] = normalize_zscore_masked(img_4d[:, :, :, m], brain_mask)
            
            # Resize: output shape (H, W, D, 4)
            resized_img = np.zeros((target_size[0], target_size[1], num_slices, 4), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
            
            for i in range(num_slices):
                for m in range(4):
                    resized_img[:, :, i, m] = resize(img_4d[:, :, i, m], target_size, order=1, preserve_range=True, anti_aliasing=True, mode='reflect')
                resized_mask[:, :, i] = resize(mask_data[:, :, i], target_size, order=0, preserve_range=True, anti_aliasing=False, mode='reflect').astype(np.uint8)
            
            np.save(vol_path, resized_img)
            np.save(mask_path_out, resized_mask)
            volume_info[patient_id] = {'num_slices': int(num_slices)}
            processed += 1
            
        except Exception as e:
            print(f"  Error: {patient_id}: {e}")
    
    metadata = {
        'dataset': 'BraTS21', 'num_classes': 4, 'num_modalities': 4,
        'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
        'class_names': ['Background', 'NCR', 'ED', 'ET'],
        'total_volumes': len(volume_info), 'volume_info': volume_info
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Processed: {processed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
