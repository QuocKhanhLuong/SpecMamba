import os
import sys
import argparse
import configparser
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
from skimage.transform import resize


def preprocess_single_patient_acdc(patient_path, target_size=(224, 224)):
    patient_folder = os.path.basename(patient_path)
    info_cfg_path = os.path.join(patient_path, 'Info.cfg')
    
    if not os.path.exists(info_cfg_path):
        return []
    
    try:
        parser = configparser.ConfigParser()
        with open(info_cfg_path, 'r') as f:
            config_string = '[DEFAULT]\n' + f.read()
        parser.read_string(config_string)
        ed_frame = int(parser['DEFAULT']['ED'])
        es_frame = int(parser['DEFAULT']['ES'])
    except Exception as e:
        print(f"  Error reading Info.cfg for {patient_folder}: {e}")
        return []
    
    results = []
    
    for frame_num, frame_name in [(ed_frame, 'ED'), (es_frame, 'ES')]:
        img_filename = f'{patient_folder}_frame{frame_num:02d}.nii.gz'
        mask_filename = f'{patient_folder}_frame{frame_num:02d}_gt.nii.gz'
        
        img_path = None
        mask_path = None
        
        for suffix in ['.gz', '']:
            test_img = os.path.join(patient_path, img_filename.replace('.gz', '') if suffix == '' else img_filename)
            test_mask = os.path.join(patient_path, mask_filename.replace('.gz', '') if suffix == '' else mask_filename)
            
            if os.path.exists(test_img):
                img_path = test_img
                mask_path = test_mask
                break
        
        if img_path is None or not os.path.exists(img_path):
            continue
        
        try:
            img_data = nib.load(img_path).get_fdata()
            
            if not os.path.exists(mask_path):
                continue
            mask_data = nib.load(mask_path).get_fdata()
            
            num_slices = img_data.shape[2]
            
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
            
            for i in range(num_slices):
                resized_img[:, :, i] = resize(
                    img_data[:, :, i], target_size, order=1, 
                    preserve_range=True, anti_aliasing=True, mode='reflect'
                )
                resized_mask[:, :, i] = resize(
                    mask_data[:, :, i], target_size, order=0, 
                    preserve_range=True, anti_aliasing=False, mode='reflect'
                )
            
            max_val = resized_img.max()
            if max_val > 0:
                resized_img /= max_val
            
            volume_id = f"{patient_folder}_{frame_name}"
            results.append((resized_img, resized_mask, volume_id))
            
        except Exception as e:
            print(f"  Error processing {patient_folder} frame {frame_num}: {e}")
            continue
    
    return results


def preprocess_acdc_dataset(input_dir, output_dir, target_size=(224, 224), skip_existing=True):
    os.makedirs(output_dir, exist_ok=True)
    
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    patient_folders = sorted([
        os.path.join(input_dir, d) 
        for d in os.listdir(input_dir) 
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('patient')
    ])
    
    print(f"Found {len(patient_folders)} patients in {input_dir}")
    
    volume_info = {}
    processed_count = 0
    skipped_count = 0
    
    for patient_path in tqdm(patient_folders, desc="Preprocessing"):
        patient_results = preprocess_single_patient_acdc(patient_path, target_size)
        
        if not patient_results:
            continue
        
        for volume, mask, volume_id in patient_results:
            volume_save_path = os.path.join(volumes_dir, f'{volume_id}.npy')
            mask_save_path = os.path.join(masks_dir, f'{volume_id}.npy')
            
            if skip_existing and os.path.exists(volume_save_path):
                skipped_count += 1
                continue
            
            np.save(volume_save_path, volume)
            np.save(mask_save_path, mask)
            
            volume_info[volume_id] = {'num_slices': int(mask.shape[2])}
            processed_count += 1
    
    metadata = {
        'dataset': 'ACDC',
        'target_size': list(target_size),
        'total_volumes': len(volume_info),
        'volume_info': volume_info,
        'num_classes': 4,
        'class_names': ['Background', 'RV', 'MYO', 'LV']
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Processed: {processed_count}, Skipped: {skipped_count}")
    return processed_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()
    
    preprocess_acdc_dataset(args.input, args.output, (args.size, args.size))
