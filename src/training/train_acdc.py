"""
ACDC Training Script — supports SpecMambaNet (3-stream) and AsymSpecMambaDCN (2.5D).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict, OrderedDict
from scipy.ndimage import distance_transform_edt, binary_erosion
from datetime import datetime
import json
import glob
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from data.acdc_dataset import ACDCDataset2D, ACDCDataset2DAugmented
from losses.sota_loss import CombinedSOTALoss

CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


# ── Reproducibility & config helpers ────────────────────────────────────────

CONFIG_SECTIONS = {
    'experiment', 'data', 'model', 'training', 'loss', 'split', 'output',
    'evaluation', 'reproducibility',
}


def load_training_config(config_path):
    """Load a YAML/JSON config and flatten known experiment sections."""
    if config_path is None:
        return {}

    path = Path(config_path)
    with open(path) as f:
        if path.suffix.lower() in {'.yaml', '.yml'}:
            if yaml is None:
                raise ImportError("PyYAML is required for YAML config files.")
            raw = yaml.safe_load(f) or {}
        else:
            raw = json.load(f)

    flat = {}
    for key, value in raw.items():
        if key in CONFIG_SECTIONS and isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if key == 'experiment' and nested_key == 'name':
                    flat['exp_name'] = nested_value
                else:
                    flat[nested_key] = nested_value
        else:
            flat[key] = value
    if isinstance(flat.get('class_weights'), list):
        flat['class_weights'] = ','.join(str(v) for v in flat['class_weights'])
    return flat


def build_arg_parser(config_defaults=None):
    parser = argparse.ArgumentParser(description="ACDC Training")

    parser.add_argument('--config', type=str, default=None,
                        help='YAML/JSON experiment config.')
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='asym_spec_mamba',
                        choices=['specmamba', 'asym_spec_mamba', 'hrnet_dcn', 'hrnet_resnet34'])
    parser.add_argument('--base_channels', type=int, default=48)
    parser.add_argument('--use_pointrend', action='store_true')
    parser.add_argument('--use_shearlet', action='store_true')
    parser.add_argument('--no_full_res', action='store_true')

    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument('--boundary_weight', type=float, default=1.5)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--focal_weight', type=float, default=0.5)
    parser.add_argument('--deep_supervision', action='store_true')
    parser.add_argument('--class_weights', type=str, default='0.1,2.5,1.5,1.0')
    parser.add_argument('--no_class_weights', action='store_true')

    parser.add_argument('--train_fraction', type=float, default=0.8)
    parser.add_argument('--split_manifest', type=str, default='splits/acdc_patient_split_seed42.json',
                        help='Path to save/load patient-level train/val split manifest.')
    parser.add_argument('--overwrite_split', action='store_true',
                        help='Regenerate split_manifest instead of reusing an existing one.')

    parser.add_argument('--hd95_unit', type=str, default='auto',
                        choices=['auto', 'mm', 'pixel'])
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--exp_name', type=str, default=None)

    if config_defaults:
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(config_defaults) - valid_dests)
        if unknown:
            raise ValueError(f"Unknown config keys for train_acdc.py: {unknown}")
        parser.set_defaults(**config_defaults)
    return parser


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_training_config(config_args.config)
    parser = build_arg_parser(config_defaults)
    return parser.parse_args(argv)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def acdc_volume_id(path_or_id):
    return Path(str(path_or_id)).stem


def acdc_patient_id(volume_id):
    return acdc_volume_id(volume_id).split('_')[0]


def create_acdc_patient_split_manifest(volume_ids, seed=42, train_fraction=0.8,
                                       output_path=None):
    """Create a deterministic patient-level split manifest for ACDC.

    ACDC preprocessing emits one ED and one ES volume per patient. Splitting by
    patient keeps both frames in the same fold and prevents patient leakage.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in the open interval (0, 1).")

    unique_volume_ids = sorted({acdc_volume_id(v) for v in volume_ids})
    patient_to_volumes = OrderedDict()
    for volume_id in unique_volume_ids:
        patient_to_volumes.setdefault(acdc_patient_id(volume_id), []).append(volume_id)

    patients = list(patient_to_volumes.keys())
    if len(patients) < 2:
        raise ValueError("Need at least two ACDC patients for train/val splitting.")

    shuffled = patients.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_fraction)
    n_train = min(max(n_train, 1), len(shuffled) - 1)

    train_patients = sorted(shuffled[:n_train])
    val_patients = sorted(shuffled[n_train:])

    def volumes_for(split_patients):
        vols = []
        for patient in split_patients:
            vols.extend(patient_to_volumes[patient])
        return sorted(vols)

    manifest = {
        'dataset': 'ACDC',
        'schema_version': 1,
        'split_level': 'patient',
        'patient_id_rule': 'prefix_before_first_underscore',
        'seed': int(seed),
        'train_fraction': float(train_fraction),
        'num_patients': len(patients),
        'num_volumes': len(unique_volume_ids),
        'splits': {
            'train': {
                'patients': train_patients,
                'volumes': volumes_for(train_patients),
            },
            'val': {
                'patients': val_patients,
                'volumes': volumes_for(val_patients),
            },
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    return manifest


def load_acdc_split_manifest(manifest_path):
    with open(manifest_path) as f:
        manifest = normalize_acdc_split_manifest(json.load(f))
    if manifest.get('split_level') != 'patient':
        raise ValueError("ACDC split manifest must have split_level='patient'.")
    train_patients = set(manifest['splits']['train']['patients'])
    val_patients = set(manifest['splits']['val']['patients'])
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(f"Patient leakage in split manifest: {sorted(overlap)}")
    return manifest


def normalize_acdc_split_manifest(manifest):
    if 'splits' in manifest:
        for split_name in ('train', 'val'):
            split = manifest['splits'][split_name]
            if 'patients' not in split:
                split['patients'] = sorted({acdc_patient_id(v) for v in split['volumes']})
        return manifest
    if manifest.get('split_type') != 'patient':
        return manifest

    required = {'train_patients', 'val_patients', 'train_volumes', 'val_volumes'}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Flat ACDC split manifest missing keys: {sorted(missing)}")

    return {
        'dataset': manifest.get('dataset', 'ACDC'),
        'schema_version': manifest.get('schema_version', 1),
        'split_level': 'patient',
        'patient_id_rule': manifest.get(
            'patient_id_rule', 'prefix_before_first_underscore'),
        'seed': manifest.get('seed'),
        'train_fraction': manifest.get('train_ratio'),
        'num_patients': manifest.get(
            'n_patients',
            len(set(manifest['train_patients']) | set(manifest['val_patients']))),
        'num_volumes': manifest.get(
            'n_volumes',
            len(set(manifest['train_volumes']) | set(manifest['val_volumes']))),
        'splits': {
            'train': {
                'patients': sorted(manifest['train_patients']),
                'volumes': sorted(manifest['train_volumes']),
            },
            'val': {
                'patients': sorted(manifest['val_patients']),
                'volumes': sorted(manifest['val_volumes']),
            },
        },
    }


def indices_from_acdc_split_manifest(dataset, manifest):
    manifest = normalize_acdc_split_manifest(manifest)
    dataset_volume_ids = [acdc_volume_id(p) for p in dataset.vol_paths]
    available = set(dataset_volume_ids)
    train_vols = set(manifest['splits']['train']['volumes'])
    val_vols = set(manifest['splits']['val']['volumes'])
    train_patients = set(manifest['splits']['train']['patients'])
    val_patients = set(manifest['splits']['val']['patients'])

    if train_patients & val_patients:
        raise ValueError("Patient leakage in split manifest.")

    if train_vols & val_vols:
        raise ValueError("Volume leakage in split manifest.")

    missing = sorted((train_vols | val_vols) - available)
    if missing:
        raise ValueError(f"Split manifest references missing ACDC volumes: {missing}")

    train_idx = [
        i for i, (vol_idx, _) in enumerate(dataset.index_map)
        if dataset_volume_ids[vol_idx] in train_vols
    ]
    val_idx = [
        i for i, (vol_idx, _) in enumerate(dataset.index_map)
        if dataset_volume_ids[vol_idx] in val_vols
    ]
    if not train_idx or not val_idx:
        raise ValueError("ACDC split produced an empty train or validation set.")
    return train_idx, val_idx


# ── 2.5D Dataset ────────────────────────────────────────────────────────────

class ACDCDataset25D(Dataset):
    """Stack 5 consecutive slices [k-2..k+2] as input channels.

    Returns (img_5ch, mask) where img_5ch is (5, H, W) and mask is (H, W).
    Boundary slices use mirror padding.
    """

    def __init__(self, npy_dir, use_memmap=True, max_cache=10, augment=False):
        self.use_memmap = use_memmap
        self.max_cache = max_cache
        self.augment = augment
        self._cache: OrderedDict = OrderedDict()

        volumes_dir = os.path.join(npy_dir, 'volumes')
        masks_dir = os.path.join(npy_dir, 'masks')
        self.vol_paths = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))

        metadata_path = os.path.join(npy_dir, 'metadata.json')
        volume_info = None
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                volume_info = json.load(f).get('volume_info', {})

        self.slice_counts: dict[int, int] = {}
        self.index_map: list[tuple[int, int]] = []
        for i, vp in enumerate(self.vol_paths):
            vid = os.path.basename(vp).replace('.npy', '')
            if volume_info and vid in volume_info:
                n_slices = volume_info[vid]['num_slices']
            else:
                n_slices = np.load(vp, mmap_mode='r').shape[2]
            self.slice_counts[i] = n_slices
            for s in range(n_slices):
                self.index_map.append((i, s))

        tag = "aug=ON" if augment else "aug=OFF"
        print(f"ACDCDataset25D: {len(self.index_map)} slices from "
              f"{len(self.vol_paths)} volumes (2.5D, {tag})")

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

    @staticmethod
    def _mirror(s, n):
        if s < 0:
            s = -s
        if s >= n:
            s = 2 * (n - 1) - s
        return max(0, min(s, n - 1))

    def _apply_aug(self, img, mask):
        if np.random.rand() < 0.5:
            img = img[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return img, mask

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.index_map[idx]
        vol, mask = self._load(vol_idx)
        n = self.slice_counts[vol_idx]
        channels = [
            vol[:, :, self._mirror(slice_idx + off, n)].copy().astype(np.float32)
            for off in (-2, -1, 0, 1, 2)
        ]
        img = np.stack(channels, axis=0)
        gt = mask[:, :, slice_idx].copy().astype(np.int64)
        if self.augment:
            img, gt = self._apply_aug(img, gt)
        return torch.from_numpy(img.copy()), torch.from_numpy(gt.copy())


# ── Per-class SDF ground truth ──────────────────────────────────────────────

def compute_sdf_from_mask(mask, num_fg=3):
    """Per-class normalised SDF: (B, num_fg, H, W) in [-1, 1]."""
    mask_np = mask.detach().cpu().numpy()
    B = mask_np.shape[0]
    sdfs = np.zeros((B, num_fg, *mask_np.shape[1:]), dtype=np.float32)
    for b in range(B):
        for c in range(num_fg):
            fg = (mask_np[b] == (c + 1)).astype(np.float64)
            if fg.sum() == 0 or fg.sum() == fg.size:
                continue
            pos = distance_transform_edt(fg)
            neg = distance_transform_edt(1.0 - fg)
            sdf = pos - neg
            mx = max(abs(sdf.max()), abs(sdf.min()), 1e-8)
            sdfs[b, c] = (sdf / mx).astype(np.float32)
    return torch.from_numpy(sdfs).to(mask.device)


# ── Compound HD Loss ────────────────────────────────────────────────────────

class CompoundHDLoss(nn.Module):
    """Dice + 0.5*CE + 0.5*Focal + w_hd*KarimiHD + w_sdf*SDF-MSE."""

    def __init__(self, num_classes=4, class_weights=None,
                 hd_warmup_start=10, sdf_warmup_epochs=20,
                 focal_gamma=2.0, focal_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.hd_warmup_start = hd_warmup_start
        self.sdf_warmup_epochs = sdf_warmup_epochs
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.current_epoch = 0
        self._cw = class_weights if class_weights is not None else [0.1, 2.5, 1.5, 1.0]
        self.mse = nn.MSELoss()

    def _w_hd(self):
        if self.current_epoch < self.hd_warmup_start:
            return 0.0
        return min(1.0, (self.current_epoch - self.hd_warmup_start)
                   / max(1, self.hd_warmup_start))

    def _w_sdf(self):
        if self.sdf_warmup_epochs <= 0:
            return 0.5
        return min(0.5, 0.5 * self.current_epoch / self.sdf_warmup_epochs)

    def _dice_loss(self, pred, target, weights):
        pred_soft = torch.softmax(pred, dim=1)
        C = pred_soft.shape[1]
        tgt_oh = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()
        p, t = pred_soft.flatten(2), tgt_oh.flatten(2)
        inter = (p * t).sum(2)
        union = p.sum(2) + t.sum(2)
        dice = (2.0 * inter + 1e-5) / (union + 1e-5)
        dl = 1.0 - dice
        if weights is not None:
            w = weights.to(pred.device).view(1, -1)
            return (dl * w).sum(1).mean() / w.sum()
        return dl.mean()

    def _ce_loss(self, pred, target):
        w = torch.tensor(self._cw, dtype=torch.float32, device=pred.device)
        return F.cross_entropy(pred, target.long(), weight=w)

    def _focal_loss(self, pred, target):
        w = torch.tensor(self._cw, dtype=torch.float32, device=pred.device)
        ce = F.cross_entropy(pred, target.long(), weight=w, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.focal_gamma * ce).mean()

    def _hd_loss(self, pred, target):
        pred_soft = torch.softmax(pred, dim=1)
        B = pred.shape[0]
        total = torch.tensor(0.0, device=pred.device)
        count = 0
        for c in range(1, self.num_classes):
            tgt_c = (target == c).float()
            pred_c = pred_soft[:, c]
            for b in range(B):
                gt_np = tgt_c[b].detach().cpu().numpy()
                if gt_np.sum() == 0:
                    continue
                dt = distance_transform_edt(1.0 - gt_np) + distance_transform_edt(gt_np)
                dt_t = torch.from_numpy(dt.astype(np.float32)).to(pred.device)
                total = total + ((pred_c[b] - tgt_c[b]) ** 2 * dt_t).mean()
                count += 1
        if count == 0:
            return torch.zeros(1, device=pred.device, requires_grad=True).squeeze()
        return total / count

    def forward(self, pred, target, sdf_pred=None, sdf_gt=None):
        cw_t = torch.tensor(self._cw, dtype=torch.float32, device=pred.device)
        info: dict[str, float] = {}
        L_dice = self._dice_loss(pred, target, cw_t)
        L_ce = self._ce_loss(pred, target)
        L_focal = self._focal_loss(pred, target)
        info['dice'] = L_dice.item()
        info['ce'] = L_ce.item()
        info['focal'] = L_focal.item()
        loss = L_dice + 0.5 * L_ce + self.focal_weight * L_focal

        w_hd = self._w_hd()
        if w_hd > 0:
            L_hd = self._hd_loss(pred, target)
            loss = loss + w_hd * L_hd
            info['hd'] = L_hd.item()

        w_sdf = self._w_sdf()
        if w_sdf > 0 and sdf_pred is not None and sdf_gt is not None:
            L_sdf = self.mse(sdf_pred, sdf_gt.to(pred.device))
            loss = loss + w_sdf * L_sdf
            info['sdf'] = L_sdf.item()

        info['total'] = loss.item()
        return loss, info


# ── 3D Evaluation ───────────────────────────────────────────────────────────

def evaluate_3d(model, dataset, device, num_classes=4, volume_meta=None):
    model.eval()
    vol_preds = defaultdict(list)
    vol_targets = defaultdict(list)

    with torch.no_grad():
        for i in range(len(dataset)):
            vol_idx, slice_idx = dataset.dataset.index_map[dataset.indices[i]]
            img, target = dataset[i]
            img = img.unsqueeze(0).to(device)
            pred = model(img)['output'].argmax(1).squeeze(0).cpu().numpy()
            vol_preds[vol_idx].append((slice_idx, pred))
            vol_targets[vol_idx].append((slice_idx, target.numpy()))

    dice_3d = {c: [] for c in range(1, num_classes)}
    hd95_3d = {c: [] for c in range(1, num_classes)}
    prec_3d = {c: [] for c in range(1, num_classes)}
    recall_3d = {c: [] for c in range(1, num_classes)}
    acc_3d = {c: [] for c in range(1, num_classes)}

    for vol_idx in vol_preds.keys():
        pred_3d = np.stack([p[1] for p in sorted(vol_preds[vol_idx], key=lambda x: x[0])], axis=0)
        target_3d = np.stack([t[1] for t in sorted(vol_targets[vol_idx], key=lambda x: x[0])], axis=0)

        spacing = None
        if volume_meta is not None and vol_idx in volume_meta:
            spacing = volume_meta[vol_idx].get('effective_spacing')

        for c in range(1, num_classes):
            pred_c = (pred_3d == c)
            target_c = (target_3d == c)
            if not target_c.any():
                if not pred_c.any():
                    continue
                dice_3d[c].append(0.0); hd95_3d[c].append(100.0)
                prec_3d[c].append(0.0); recall_3d[c].append(0.0)
                acc_3d[c].append(float((~pred_c).sum() / pred_c.size))
                continue
            tp = (pred_c & target_c).sum()
            fp = (pred_c & ~target_c).sum()
            fn = (~pred_c & target_c).sum()
            tn = (~pred_c & ~target_c).sum()
            dice_3d[c].append((2 * tp) / (2 * tp + fp + fn + 1e-6))
            prec_3d[c].append(tp / (tp + fp + 1e-6))
            recall_3d[c].append(tp / (tp + fn + 1e-6))
            acc_3d[c].append((tp + tn) / (tp + tn + fp + fn + 1e-6))

            if pred_c.any() and target_c.any():
                pb = pred_c ^ binary_erosion(pred_c)
                tb = target_c ^ binary_erosion(target_c)
                if not pb.any(): pb = pred_c
                if not tb.any(): tb = target_c
                d1 = distance_transform_edt(~target_c, sampling=spacing)[pb]
                d2 = distance_transform_edt(~pred_c, sampling=spacing)[tb]
                hd95_3d[c].append(np.percentile(np.concatenate([d1, d2]), 95))
            else:
                hd95_3d[c].append(100.0)

    sm = lambda lst: np.mean(lst) if lst else 0.0
    return {
        'mean_dice': np.mean([sm(dice_3d[c]) for c in range(1, num_classes)]),
        'mean_hd95': np.mean([sm(hd95_3d[c]) for c in range(1, num_classes)]),
        'mean_prec': np.mean([sm(prec_3d[c]) for c in range(1, num_classes)]),
        'mean_recall': np.mean([sm(recall_3d[c]) for c in range(1, num_classes)]),
        'mean_acc': np.mean([sm(acc_3d[c]) for c in range(1, num_classes)]),
        'dice_all': dice_3d, 'hd95_all': hd95_3d,
        'prec_all': prec_3d, 'recall_all': recall_3d, 'acc_all': acc_3d,
    }


# ── Training Loop ───────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch,
                scaler=None, use_amp=False, deep_supervision=False,
                is_hybrid=False):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"E{epoch+1}", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        sdf_gt = compute_sdf_from_mask(masks) if is_hybrid else None

        if use_amp and scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                out = outputs['output']
                if is_hybrid:
                    loss, info = criterion(out, masks,
                                           sdf_pred=outputs.get('sdf'),
                                           sdf_gt=sdf_gt)
                else:
                    loss, info = criterion(out, masks)
                if deep_supervision and 'aux_outputs' in outputs:
                    for i, aux in enumerate(outputs['aux_outputs']):
                        aux_l, _ = criterion(aux, masks) if not is_hybrid \
                            else criterion(aux, masks)
                        loss = loss + [0.4, 0.3][i] * aux_l
                if 'ctx_logits' in outputs:
                    ctx_l = F.cross_entropy(outputs['ctx_logits'], masks.long())
                    loss = loss + 0.3 * ctx_l
                    info['ctx_ce'] = ctx_l.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            out = outputs['output']
            if is_hybrid:
                loss, info = criterion(out, masks,
                                       sdf_pred=outputs.get('sdf'),
                                       sdf_gt=sdf_gt)
            else:
                loss, info = criterion(out, masks)
            if deep_supervision and 'aux_outputs' in outputs:
                for i, aux in enumerate(outputs['aux_outputs']):
                    aux_l, _ = criterion(aux, masks) if not is_hybrid \
                        else criterion(aux, masks)
                    loss = loss + [0.4, 0.3][i] * aux_l
            if 'ctx_logits' in outputs:
                ctx_l = F.cross_entropy(outputs['ctx_logits'], masks.long())
                loss = loss + 0.3 * ctx_l
                info['ctx_ce'] = ctx_l.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return total_loss / len(loader)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed, deterministic=args.deterministic)

    if args.exp_name is None:
        args.exp_name = f"acdc_{args.model}_c{args.base_channels}_{datetime.now().strftime('%m%d_%H%M')}"

    num_classes = 4
    is_hybrid = (args.model == 'asym_spec_mamba')
    in_channels = 5 if is_hybrid else 3

    # ── Model ───────────────────────────────────────────────────────────
    if args.model == 'asym_spec_mamba':
        from models.specmamba_net import AsymSpecMambaDCN
        model = AsymSpecMambaDCN(
            in_ch=in_channels, num_classes=num_classes,
            base_ch=args.base_channels,
            deep_supervision=args.deep_supervision,
        ).to(device)
        model_name = f"AsymSpecMambaDCN-C{args.base_channels}"
    elif args.model == 'specmamba':
        from models.specmamba_net import SpecMambaNet
        model = SpecMambaNet(
            in_channels=in_channels, num_classes=num_classes,
            base_channels=args.base_channels, img_size=224,
            deep_supervision=args.deep_supervision,
        ).to(device)
        model_name = f"SpecMambaNet-C{args.base_channels}"
    elif args.model == 'hrnet_dcn':
        from models.hrnet_dcn import HRNetDCN
        model = HRNetDCN(
            in_channels=in_channels, num_classes=num_classes,
            base_channels=args.base_channels,
            use_pointrend=args.use_pointrend,
            full_resolution_mode=not args.no_full_res,
            deep_supervision=args.deep_supervision,
            use_shearlet=args.use_shearlet,
        ).to(device)
        model_name = f"HRNetDCN-C{args.base_channels}"
    elif args.model == 'hrnet_resnet34':
        from models.hrnet_resnet34 import HRNetResNet34
        model = HRNetResNet34(
            in_channels=in_channels, num_classes=num_classes,
            base_channels=args.base_channels,
            use_deep_supervision=args.deep_supervision,
            full_resolution_mode=not args.no_full_res,
        ).to(device)
        model_name = f"HRNetResNet34-C{args.base_channels}"

    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"ACDC Training — {model_name}")
    print(f"{'='*60}")
    print(f"Model:      {model_name} | Params={params:,}")
    print(f"Training:   BS={args.batch_size} | LR={args.lr} | Epochs={args.epochs}")
    print(f"DeepSup:    {'ON' if args.deep_supervision else 'OFF'} | AMP={'ON' if args.use_amp else 'OFF'}")
    print(f"Seed:       {args.seed} | Deterministic={'ON' if args.deterministic else 'OFF'}")

    # ── HD95 spacing ────────────────────────────────────────────────────
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    volume_meta = {}
    if args.hd95_unit == 'pixel':
        print(f"HD95 Unit:  pixel")
    else:
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            vol_info = meta.get('volume_info', {})
            for idx, vname in enumerate(sorted(vol_info.keys())):
                if 'effective_spacing' in vol_info[vname]:
                    volume_meta[idx] = vol_info[vname]
        if volume_meta:
            sp = next(iter(volume_meta.values()))['effective_spacing']
            print(f"HD95 Unit:  mm (e.g. {[round(s,2) for s in sp]})")
        else:
            print(f"HD95 Unit:  pixel (no spacing)")

    # ── Data ────────────────────────────────────────────────────────────
    if is_hybrid:
        base_dataset = ACDCDataset25D(args.data_dir, augment=False)
        aug_dataset = ACDCDataset25D(args.data_dir, augment=True) if args.augment else None
    else:
        base_dataset = ACDCDataset2D(args.data_dir, in_channels=in_channels)
        aug_dataset = None

    volume_ids = [acdc_volume_id(path) for path in base_dataset.vol_paths]
    if args.split_manifest is None:
        args.split_manifest = os.path.join(
            args.save_dir, f"{args.exp_name}_split_seed{args.seed}.json")

    if os.path.exists(args.split_manifest) and not args.overwrite_split:
        split_manifest = load_acdc_split_manifest(args.split_manifest)
        print(f"Split:      loaded patient manifest {args.split_manifest}")
    else:
        split_manifest = create_acdc_patient_split_manifest(
            volume_ids=volume_ids,
            seed=args.seed,
            train_fraction=args.train_fraction,
            output_path=args.split_manifest,
        )
        print(f"Split:      saved patient manifest {args.split_manifest}")

    train_idx, val_idx = indices_from_acdc_split_manifest(
        base_dataset, split_manifest)
    print(
        f"Patients:   Train={len(split_manifest['splits']['train']['patients'])} | "
        f"Val={len(split_manifest['splits']['val']['patients'])}"
    )

    if is_hybrid and aug_dataset:
        train_ds = Subset(aug_dataset, train_idx)
        print(f"Augmentation: ON (2.5D flip+rot)")
    elif not is_hybrid and args.augment:
        aug_ds = ACDCDataset2DAugmented(args.data_dir, in_channels=in_channels, augment=True)
        aug_ds.index_map = [base_dataset.index_map[i] for i in train_idx]
        train_ds = aug_ds
        print(f"Augmentation: ON")
    else:
        train_ds = Subset(base_dataset, train_idx)
        print(f"Augmentation: OFF")

    val_ds = Subset(base_dataset, val_idx)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == 'cuda'),
                              worker_init_fn=seed_worker,
                              generator=loader_generator)
    print(f"Data:       Train={len(train_ds)} | Val={len(val_ds)} slices")

    resolved_config_path = os.path.join(args.save_dir, f"{args.exp_name}_resolved_config.json")
    with open(resolved_config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config:     resolved args saved to {resolved_config_path}")

    # ── Loss & Optimizer ────────────────────────────────────────────────
    class_weights = None if args.no_class_weights else \
        [float(w) for w in args.class_weights.split(',')]
    print(f"Class Wts:  {class_weights}")

    if is_hybrid:
        criterion = CompoundHDLoss(num_classes=num_classes, class_weights=class_weights)
        print(f"Loss:       CompoundHDLoss (Dice+CE+Focal+HD+SDF)")
    else:
        criterion = CombinedSOTALoss(
            num_classes=num_classes, ce_weight=args.ce_weight,
            dice_weight=args.dice_weight, boundary_weight=args.boundary_weight,
            focal_weight=args.focal_weight, warmup_epochs=args.warmup_epochs,
            class_weights=class_weights,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-7)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None

    best_balanced = float('-inf')
    best_dice = 0.0
    best_hd95 = float('inf')
    epochs_no_improve = 0
    ALPHA_HD95 = 0.7

    history = {
        'epoch': [], 'train_loss': [], 'val_dice': [], 'val_hd95': [],
        'val_prec': [], 'val_recall': [], 'val_acc': [],
        'balanced_score': [], 'lr': [],
        'dice_rv': [], 'dice_myo': [], 'dice_lv': [],
        'hd95_rv': [], 'hd95_myo': [], 'hd95_lv': [],
        'prec_rv': [], 'prec_myo': [], 'prec_lv': [],
        'recall_rv': [], 'recall_myo': [], 'recall_lv': [],
        'acc_rv': [], 'acc_myo': [], 'acc_lv': [],
    }

    print(f"\n{'='*60}")
    print(f"Training Started (Balanced = Dice - {ALPHA_HD95}*HD95)")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * (epoch + 1) / warmup_epochs

        criterion.current_epoch = epoch
        loss = train_epoch(model, train_loader, criterion, optimizer, device,
                           epoch, scaler, args.use_amp, args.deep_supervision,
                           is_hybrid=is_hybrid)

        if epoch >= warmup_epochs:
            cosine_sched.step()

        metrics = evaluate_3d(model, val_ds, device, num_classes,
                              volume_meta=volume_meta)

        dice = metrics['mean_dice']
        hd95 = metrics['mean_hd95']
        prec = metrics['mean_prec']
        rec = metrics['mean_recall']
        acc = metrics['mean_acc']
        penalty = 10.0 if (hd95 > 100 or np.isnan(hd95)) else hd95
        balanced = dice - ALPHA_HD95 * penalty

        if epoch >= warmup_epochs:
            plateau_sched.step(balanced)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(loss)
        history['val_dice'].append(dice)
        history['val_hd95'].append(hd95)
        history['val_prec'].append(prec)
        history['val_recall'].append(rec)
        history['val_acc'].append(acc)
        history['balanced_score'].append(balanced)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        for ci, cn in [(1,'rv'),(2,'myo'),(3,'lv')]:
            history[f'dice_{cn}'].append(np.mean(metrics['dice_all'][ci]) if metrics['dice_all'][ci] else 0)
            history[f'hd95_{cn}'].append(np.mean(metrics['hd95_all'][ci]) if metrics['hd95_all'][ci] else 0)
            history[f'prec_{cn}'].append(np.mean(metrics['prec_all'][ci]) if metrics['prec_all'][ci] else 0)
            history[f'recall_{cn}'].append(np.mean(metrics['recall_all'][ci]) if metrics['recall_all'][ci] else 0)
            history[f'acc_{cn}'].append(np.mean(metrics['acc_all'][ci]) if metrics['acc_all'][ci] else 0)

        print(f"\nE{epoch+1:03d} | Loss: {loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Avg FG: Dice={dice:.4f}  HD95={hd95:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Acc={acc:.4f}")
        print(f"   Balanced: {balanced:.4f}")
        print(f"   {'Class':<5}  {'Dice':>6}  {'HD95':>7}  {'Prec':>6}  {'Rec':>6}  {'Acc':>6}  {'F1':>6}")
        print(f"   {'─'*50}")
        for c in range(1, num_classes):
            cn = CLASS_MAP[c]
            cd = np.mean(metrics['dice_all'][c]) if metrics['dice_all'][c] else 0
            ch = np.mean(metrics['hd95_all'][c]) if metrics['hd95_all'][c] else 0
            cp = np.mean(metrics['prec_all'][c]) if metrics['prec_all'][c] else 0
            cr = np.mean(metrics['recall_all'][c]) if metrics['recall_all'][c] else 0
            ca = np.mean(metrics['acc_all'][c]) if metrics['acc_all'][c] else 0
            cf1 = 2 * cp * cr / (cp + cr + 1e-8)
            print(f"   {cn:<5}  {cd:6.4f}  {ch:7.4f}  {cp:6.4f}  {cr:6.4f}  {ca:6.4f}  {cf1:6.4f}")

        saved = False
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.exp_name}_best_dice.pt"))
            print(f"   * Best Dice: {best_dice:.4f}")
            saved = True
        if hd95 < best_hd95:
            best_hd95 = hd95
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.exp_name}_best_hd95.pt"))
            print(f"   * Best HD95: {best_hd95:.4f}")
            saved = True
        if balanced > best_balanced:
            best_balanced = balanced
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.exp_name}_best_balanced.pt"))
            print(f"   * Best Balanced: {best_balanced:.4f}")
            saved = True
        epochs_no_improve = 0 if saved else epochs_no_improve + 1
        if epochs_no_improve >= args.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Best HD95: {best_hd95:.4f}")
    print(f"Best Balanced: {best_balanced:.4f}")
    print(f"{'='*60}")

    hp = os.path.join(args.save_dir, f"{args.exp_name}_history.json")
    with open(hp, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hp}")


if __name__ == '__main__':
    main()
