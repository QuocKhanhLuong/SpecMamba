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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def normalize_intensity(image):
    """
    Chuẩn hóa Z-score cho MRI (Robust hơn chia cho max).
    1. Cắt bỏ điểm nhiễu (Top/Bottom 0.5%)
    2. Trừ Mean, chia Std
    """
    # Clip outliers
    p05 = np.percentile(image, 0.5)
    p995 = np.percentile(image, 99.5)
    image = np.clip(image, p05, p995)
    
    # Z-score
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    return image

def preprocess_single_patient_acdc(patient_path, output_dir, target_size=(224, 224)):
    """
    Xử lý 1 bệnh nhân ACDC:
    - Đọc ED/ES frame
    - Normalize Z-score
    - Resize slice
    - LƯU TỪNG SLICE RIÊNG BIỆT (cho 2D Network)
    """
    patient_folder = os.path.basename(patient_path)
    info_cfg_path = os.path.join(patient_path, 'Info.cfg')
    
    # Tạo folder output cho slice
    img_save_dir = os.path.join(output_dir, 'images')
    mask_save_dir = os.path.join(output_dir, 'masks')
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    
    # Đọc config để biết frame nào là ED, frame nào là ES
    if not os.path.exists(info_cfg_path):
        return 0
    
    try:
        parser = configparser.ConfigParser()
        with open(info_cfg_path, 'r') as f:
            config_string = '[DEFAULT]\n' + f.read()
        parser.read_string(config_string)
        ed_frame = int(parser['DEFAULT']['ED'])
        es_frame = int(parser['DEFAULT']['ES'])
    except Exception as e:
        print(f"  Error reading Info.cfg for {patient_folder}: {e}")
        return 0
    
    slices_saved = 0
    
    for frame_num, frame_name in [(ed_frame, 'ED'), (es_frame, 'ES')]:
        img_filename = f'{patient_folder}_frame{frame_num:02d}.nii.gz'
        mask_filename = f'{patient_folder}_frame{frame_num:02d}_gt.nii.gz'
        
        # Tìm file (support cả .nii và .nii.gz)
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
            # Load NIfTI
            img_nii = nib.load(img_path)
            img_data = img_nii.get_fdata() # (H, W, D)
            
            mask_data = None
            if os.path.exists(mask_path):
                mask_data = nib.load(mask_path).get_fdata()
            else:
                continue # Bỏ qua nếu không có mask
            
            # 1. Normalize Intensity TRƯỚC khi resize (tính trên toàn volume 3D)
            img_data = normalize_intensity(img_data)
            
            num_slices = img_data.shape[2]
            
            # 2. Xử lý từng slice và lưu ngay lập tức
            for i in range(num_slices):
                slice_img = img_data[:, :, i]
                slice_mask = mask_data[:, :, i]
                
                # Bỏ qua slice đen thui (không có thông tin) để tránh nhiễu training
                if np.sum(slice_img) == 0:
                    continue
                
                # Resize (Lưu ý: resize của skimage range input=output, đã normalize thì vẫn giữ range)
                slice_img_resized = resize(
                    slice_img, target_size, order=1, preserve_range=True, anti_aliasing=True, mode='reflect'
                ).astype(np.float32)
                
                slice_mask_resized = resize(
                    slice_mask, target_size, order=0, preserve_range=True, anti_aliasing=False, mode='reflect'
                ).astype(np.uint8) # Mask phải là int
                
                # Tạo tên file: patient001_ED_slice005.npy
                file_id = f"{patient_folder}_{frame_name}_slice{i:03d}"
                
                np.save(os.path.join(img_save_dir, f"{file_id}.npy"), slice_img_resized)
                np.save(os.path.join(mask_save_dir, f"{file_id}.npy"), slice_mask_resized)
                
                slices_saved += 1
                
        except Exception as e:
            print(f"  Error processing {patient_folder} frame {frame_num}: {e}")
            continue
            
    return slices_saved

def preprocess_acdc_dataset(input_dir, output_dir, target_size=(224, 224)):
    """
    Main function để chạy preprocess toàn bộ dataset
    """
    # Lấy danh sách patient
    patient_folders = sorted([
        os.path.join(input_dir, d) 
        for d in os.listdir(input_dir) 
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('patient')
    ])
    
    print(f"Found {len(patient_folders)} patients. Outputting 2D slices to {output_dir}")
    
    total_slices = 0
    
    for patient_path in tqdm(patient_folders, desc="Processing ACDC"):
        slices = preprocess_single_patient_acdc(patient_path, output_dir, target_size)
        total_slices += slices
        
    print(f"\nCompleted! Saved {total_slices} slices total.")
    print(f"Images: {os.path.join(output_dir, 'images')}")
    print(f"Masks:  {os.path.join(output_dir, 'masks')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to ACDC data folder')
    parser.add_argument('--output', type=str, required=True, help='Path to save .npy slices')
    parser.add_argument('--size', type=int, default=224, help='Target size (e.g., 224)')
    
    args = parser.parse_args()
    
    preprocess_acdc_dataset(args.input, args.output, (args.size, args.size))

if __name__ == '__main__':
    main()