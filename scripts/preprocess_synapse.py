"""
Preprocess Synapse dataset: Multi-organ CT → .npy volumes
Data structure:
    data/Synapse/
    ├── synapse_train/       (images: DET0000101_avg.nii.gz)
    └── synapse_train_label/ (labels: DET0000101_avg_seg.nii.gz)

Classes: 14 organs

Usage:
    python scripts/preprocess_synapse.py --input data/Synapse --output preprocessed_data/Synapse/training
"""

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from skimage.transform import resize


def normalize_ct(image, window_center=40, window_width=400):
    """CT windowing normalization for abdominal."""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    image = np.clip(image, min_val, max_val)
    return (image - min_val) / (max_val - min_val)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Synapse dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to data/Synapse')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--no-skip', action='store_true')
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    
    # Find train images and labels
    train_dir = os.path.join(args.input, 'synapse_train')
    label_dir = os.path.join(args.input, 'synapse_train_label')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found")
        return
    
    os.makedirs(args.output, exist_ok=True)
    volumes_dir = os.path.join(args.output, 'volumes')
    masks_dir = os.path.join(args.output, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get image files
    img_files = sorted([f for f in os.listdir(train_dir) if f.endswith(('.nii.gz', '.nii'))])
    
    print(f"Synapse Preprocessing: {len(img_files)} cases")
    
    volume_info = {}
    processed, skipped = 0, 0
    
    for img_file in tqdm(img_files, desc="Synapse"):
        img_path = os.path.join(train_dir, img_file)
        
        # Find matching label file
        base_name = img_file.replace('.nii.gz', '').replace('.nii', '')
        label_file = f"{base_name}_seg.nii.gz"
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            # Try without _seg suffix
            label_path = os.path.join(label_dir, img_file)
            if not os.path.exists(label_path):
                print(f"  Warning: No label for {img_file}")
                continue
        
        vol_path = os.path.join(volumes_dir, f'{base_name}.npy')
        mask_path_out = os.path.join(masks_dir, f'{base_name}.npy')
        
        if not args.no_skip and os.path.exists(vol_path):
            skipped += 1
            continue
        
        try:
            img_data = normalize_ct(nib.load(img_path).get_fdata())
            mask_data = nib.load(label_path).get_fdata().astype(np.uint8)
            
            num_slices = img_data.shape[2]
            
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
            
            for i in range(num_slices):
                resized_img[:, :, i] = resize(img_data[:, :, i], target_size, order=1, preserve_range=True, anti_aliasing=True, mode='reflect')
                resized_mask[:, :, i] = resize(mask_data[:, :, i].astype(np.float32), target_size, order=0, preserve_range=True, anti_aliasing=False, mode='reflect').astype(np.uint8)
            
            np.save(vol_path, resized_img)
            np.save(mask_path_out, resized_mask)
            volume_info[base_name] = {'num_slices': int(num_slices)}
            processed += 1
            
        except Exception as e:
            print(f"  Error: {img_file}: {e}")
    
    metadata = {
        'dataset': 'Synapse', 'num_classes': 14,
        'class_names': ['Background', 'Spleen', 'R.Kidney', 'L.Kidney', 'Gallbladder',
                        'Esophagus', 'Liver', 'Stomach', 'Aorta', 'IVC',
                        'Portal Vein', 'Pancreas', 'R.AG', 'L.AG'],
        'total_volumes': len(volume_info), 'volume_info': volume_info
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Processed: {processed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
