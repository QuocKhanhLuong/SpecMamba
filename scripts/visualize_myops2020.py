"""
Visualize raw MyoPS2020 data before preprocessing.

Shows all 3 modalities (C0/bSSFP, DE/LGE, T2) alongside the ground truth
segmentation for each patient. Supports multi-slice navigation.

MyoPS2020 Label Map:
    0    = Background
    200  = Healthy Myocardium (MYO)
    500  = Left Ventricle (LV)
    600  = Right Ventricle (RV)
    1220 = Edema
    2221 = Scar

Usage:
    python scripts/visualize_myops2020.py                          # all patients
    python scripts/visualize_myops2020.py --patient 101            # single patient
    python scripts/visualize_myops2020.py --patient 101 --save     # save to disk
"""

import os
import argparse
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# ─── Label mapping ───────────────────────────────────────────────────────────
LABEL_MAP = {
    0:    ("Background",  [0.0, 0.0, 0.0, 0.0]),   # transparent
    200:  ("Myo",         [0.0, 0.8, 0.0, 0.6]),    # green
    500:  ("LV",          [1.0, 0.2, 0.2, 0.6]),     # red
    600:  ("RV",          [0.2, 0.4, 1.0, 0.6]),     # blue
    1220: ("Edema",       [1.0, 1.0, 0.0, 0.7]),     # yellow
    2221: ("Scar",        [1.0, 0.0, 1.0, 0.7]),     # magenta
}

MODALITY_NAMES = {"C0": "bSSFP (C0)", "DE": "LGE (DE)", "T2": "T2-weighted"}


def discover_patients(data_dir):
    """Find all patient IDs from the train25 folder."""
    train_dir = os.path.join(data_dir, "train25")
    ids = set()
    for f in os.listdir(train_dir):
        m = re.match(r"myops_training_(\d+)_", f)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def load_patient(data_dir, pid):
    """Load all 3 modalities + ground truth for a patient."""
    train_dir = os.path.join(data_dir, "train25")
    gd_dir = os.path.join(data_dir, "train25_myops_gd")

    images = {}
    for mod in ["C0", "DE", "T2"]:
        path = os.path.join(train_dir, f"myops_training_{pid}_{mod}.nii.gz")
        if os.path.exists(path):
            nii = nib.load(path)
            images[mod] = {
                "data": nii.get_fdata(),
                "spacing": nii.header.get_zooms(),
            }

    gd_path = os.path.join(gd_dir, f"myops_training_{pid}_gd.nii.gz")
    gd = None
    if os.path.exists(gd_path):
        gd_nii = nib.load(gd_path)
        gd = gd_nii.get_fdata()

    return images, gd


def make_label_overlay(gd_slice):
    """Convert a ground truth slice to an RGBA overlay image."""
    h, w = gd_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for val, (_, color) in LABEL_MAP.items():
        mask = gd_slice == val
        if mask.any():
            overlay[mask] = color
    return overlay


def make_legend_patches():
    """Create legend patches for the label colormap."""
    patches = []
    for val, (name, color) in LABEL_MAP.items():
        if val == 0:
            continue
        patches.append(mpatches.Patch(color=color, label=f"{name} ({val})"))
    return patches


def visualize_patient(data_dir, pid, save_dir=None):
    """Visualize all slices for a single patient."""
    images, gd = load_patient(data_dir, pid)

    if not images:
        print(f"  [SKIP] Patient {pid}: no image data found")
        return

    ref_mod = list(images.keys())[0]
    num_slices = images[ref_mod]["data"].shape[2]
    spacing = images[ref_mod]["spacing"]

    print(f"  Patient {pid}: shape={images[ref_mod]['data'].shape}, "
          f"spacing={tuple(round(s, 3) for s in spacing)}, slices={num_slices}")

    # Print label distribution
    if gd is not None:
        unique, counts = np.unique(gd, return_counts=True)
        total = gd.size
        print(f"    Labels: " + ", ".join(
            f"{LABEL_MAP.get(int(v), ('?',))[0]}({int(v)})={c/total*100:.1f}%"
            for v, c in zip(unique, counts)
        ))

    n_cols = 4  # C0, DE, T2, GD overlay
    fig, axes = plt.subplots(num_slices, n_cols, figsize=(4 * n_cols, 4 * num_slices))
    if num_slices == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"MyoPS2020 — Patient {pid}  |  Shape: {images[ref_mod]['data'].shape}  |  "
        f"Spacing: {tuple(round(s, 2) for s in spacing)} mm",
        fontsize=14, fontweight="bold", y=1.0
    )

    for s in range(num_slices):
        for col, mod in enumerate(["C0", "DE", "T2"]):
            ax = axes[s, col]
            if mod in images:
                sl = images[mod]["data"][:, :, s]
                ax.imshow(sl, cmap="gray", aspect="equal")
                vmin, vmax = np.percentile(sl[sl > 0], [1, 99]) if sl.max() > 0 else (0, 1)
                ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{MODALITY_NAMES.get(mod, mod)} — Slice {s}", fontsize=10)
            ax.axis("off")

        # Ground truth overlay on C0
        ax = axes[s, 3]
        if "C0" in images and gd is not None:
            c0_sl = images["C0"]["data"][:, :, s]
            vmin, vmax = np.percentile(c0_sl[c0_sl > 0], [1, 99]) if c0_sl.max() > 0 else (0, 1)
            ax.imshow(c0_sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            overlay = make_label_overlay(gd[:, :, s])
            ax.imshow(overlay, aspect="equal")
        ax.set_title(f"GT on C0 — Slice {s}", fontsize=10)
        ax.axis("off")

    # Add legend
    patches = make_legend_patches()
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"myops2020_patient_{pid}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {out_path}")
        plt.close(fig)
    else:
        plt.show()


def print_dataset_summary(data_dir):
    """Print a summary of the MyoPS2020 dataset."""
    patient_ids = discover_patients(data_dir)
    print("=" * 70)
    print("MyoPS2020 Dataset Summary")
    print("=" * 70)
    print(f"  Training patients : {len(patient_ids)}  (IDs: {patient_ids[0]}–{patient_ids[-1]})")
    print(f"  Modalities        : C0 (bSSFP), DE (LGE), T2")
    print(f"  Label values      : 0=BG, 200=Myo, 500=LV, 600=RV, 1220=Edema, 2221=Scar")

    # Count test files
    test_dir = os.path.join(data_dir, "test20")
    if os.path.isdir(test_dir):
        test_ids = set()
        for f in os.listdir(test_dir):
            m = re.match(r"myops_test_(\d+)_", f)
            if m:
                test_ids.add(int(m.group(1)))
        print(f"  Test patients     : {len(test_ids)}  (IDs: {min(test_ids)}–{max(test_ids)})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Visualize raw MyoPS2020 data")
    parser.add_argument("--data-dir", type=str, default="data/MyoPS2020",
                        help="Path to MyoPS2020 data root")
    parser.add_argument("--patient", type=int, default=None,
                        help="Visualize a specific patient ID (e.g. 101)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to disk instead of showing")
    parser.add_argument("--save-dir", type=str, default="data/MyoPS2020/visualizations",
                        help="Directory to save figures")
    parser.add_argument("--max-patients", type=int, default=5,
                        help="Max patients to visualize when --patient is not set")
    args = parser.parse_args()

    print_dataset_summary(args.data_dir)

    patient_ids = discover_patients(args.data_dir)
    if args.patient:
        if args.patient not in patient_ids:
            print(f"Error: Patient {args.patient} not found. Available: {patient_ids}")
            return
        patient_ids = [args.patient]
    else:
        patient_ids = patient_ids[:args.max_patients]

    print(f"\nVisualizing {len(patient_ids)} patient(s)...")
    save_dir = args.save_dir if args.save else None

    for pid in patient_ids:
        visualize_patient(args.data_dir, pid, save_dir=save_dir)


if __name__ == "__main__":
    main()
