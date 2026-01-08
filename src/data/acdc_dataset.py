import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import json


class ACDCDataset2D(Dataset):
    def __init__(self, npy_dir, use_memmap=True, in_channels=3, max_cache=10):
        self.use_memmap = use_memmap
        self.in_channels = in_channels
        self.max_cache = max_cache
        self._cache = OrderedDict()
        
        volumes_dir = os.path.join(npy_dir, 'volumes')
        masks_dir = os.path.join(npy_dir, 'masks')
        
        self.vol_paths = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
        
        metadata_path = os.path.join(npy_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            volume_info = meta.get('volume_info', {})
        else:
            volume_info = None
        
        self.index_map = []
        for i, vp in enumerate(self.vol_paths):
            vid = os.path.basename(vp).replace('.npy', '')
            if volume_info and vid in volume_info:
                n_slices = volume_info[vid]['num_slices']
            else:
                vol = np.load(vp, mmap_mode='r')
                n_slices = vol.shape[2]
            for s in range(n_slices):
                self.index_map.append((i, s))
        
        print(f"ACDCDataset2D: {len(self.index_map)} slices from {len(self.vol_paths)} volumes")
    
    def _load(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        
        mode = 'r' if self.use_memmap else None
        vol = np.load(self.vol_paths[idx], mmap_mode=mode)
        mask = np.load(self.mask_paths[idx], mmap_mode=mode)
        self._cache[idx] = (vol, mask)
        
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return vol, mask
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx = self.index_map[idx]
        vol, mask = self._load(vol_idx)
        
        img = vol[:, :, slice_idx].copy().astype(np.float32)
        gt = mask[:, :, slice_idx].copy().astype(np.int64)
        
        img = torch.from_numpy(img).unsqueeze(0)
        if self.in_channels == 3:
            img = img.repeat(3, 1, 1)
        
        return img, torch.from_numpy(gt)
