"""
Visualize raw MyoPS2020 data before preprocessing.

Shows static QA grids for all 3 modalities (C0/bSSFP, DE/LGE, T2) with ground
truth overlays for each patient.

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
import tempfile
import numpy as np
import nibabel as nib

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "specumamba_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "specumamba_xdg_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from preprocess_myops2020 import (
        ALLOWED_LABELS,
        MODALITIES,
        MODALITY_NAMES,
        _histogram,
        _validate_geometry,
    )
except ImportError:  # pragma: no cover - supports `python -m scripts.visualize_myops2020`
    from scripts.preprocess_myops2020 import (
        ALLOWED_LABELS,
        MODALITIES,
        MODALITY_NAMES,
        _histogram,
        _validate_geometry,
    )


# ─── Label mapping ───────────────────────────────────────────────────────────
LABEL_MAP = {
    0:    ("Background",  [0.0, 0.0, 0.0, 0.0]),   # transparent
    200:  ("Myo",         [0.0, 0.8, 0.0, 0.6]),    # green
    500:  ("LV",          [1.0, 0.2, 0.2, 0.6]),     # red
    600:  ("RV",          [0.2, 0.4, 1.0, 0.6]),     # blue
    1220: ("Edema",       [1.0, 1.0, 0.0, 0.7]),     # yellow
    2221: ("Scar",        [1.0, 0.0, 1.0, 0.7]),     # magenta
}

DISPLAY_MODALITY_NAMES = {"C0": "bSSFP (C0)", "DE": "LGE (DE)", "T2": "T2-weighted"}


def discover_patients(data_dir):
    """Find all patient IDs from the train25 folder."""
    train_dir = os.path.join(data_dir, "train25")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing MyoPS train25 directory: {train_dir}")
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
    for mod in MODALITIES:
        path = os.path.join(train_dir, f"myops_training_{pid}_{mod}.nii.gz")
        if os.path.exists(path):
            nii = nib.load(path)
            images[mod] = {
                "nii": nii,
                "data": nii.get_fdata(),
                "spacing": nii.header.get_zooms(),
                "affine": nii.affine,
            }

    gd_path = os.path.join(gd_dir, f"myops_training_{pid}_gd.nii.gz")
    gd = None
    gd_nii = None
    if os.path.exists(gd_path):
        gd_nii = nib.load(gd_path)
        gd = gd_nii.get_fdata()

    return images, gd, gd_nii


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


def display_window(sl):
    """Return robust display limits for one image slice."""
    positive = sl[np.isfinite(sl) & (sl > 0)]
    if positive.size:
        vmin, vmax = np.percentile(positive, [1, 99])
    else:
        finite = sl[np.isfinite(sl)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    return vmin, vmax


def validate_loaded_patient(pid, images, gd, gd_nii):
    """Validate raw geometry and label set before visualization overlays."""
    missing = [mod for mod in MODALITIES if mod not in images]
    if missing:
        raise FileNotFoundError(f"Patient {pid}: missing modalities {missing}")
    if gd is None or gd_nii is None:
        raise FileNotFoundError(f"Patient {pid}: missing ground truth")

    ref_nii = images["C0"]["nii"]
    _validate_geometry(pid, ref_nii, gd_nii, "ground truth")
    for mod in MODALITIES:
        _validate_geometry(pid, ref_nii, images[mod]["nii"], mod)

    labels = np.rint(gd).astype(np.int32)
    unknown = sorted(set(np.unique(labels).tolist()) - ALLOWED_LABELS)
    if unknown:
        raise ValueError(
            f"Patient {pid}: unknown labels {unknown}. Expected only {sorted(ALLOWED_LABELS)}."
        )


def visualize_patient(data_dir, pid, save_dir=None):
    """Visualize all slices for a single patient."""
    images, gd, gd_nii = load_patient(data_dir, pid)

    if not images:
        print(f"  [SKIP] Patient {pid}: no image data found")
        return
    validate_loaded_patient(pid, images, gd, gd_nii)

    num_slices = images["C0"]["data"].shape[2]
    spacing = images["C0"]["spacing"]

    print(f"  Patient {pid}: shape={images['C0']['data'].shape}, "
          f"spacing={tuple(round(s, 3) for s in spacing)}, slices={num_slices}")

    # Print label distribution
    if gd is not None:
        label_hist = _histogram(np.rint(gd).astype(np.int32))
        total = gd.size
        print(f"    Labels: " + ", ".join(
            f"{LABEL_MAP.get(int(v), ('?',))[0]}({int(v)})={c/total*100:.1f}%"
            for v, c in label_hist.items()
        ))

    n_cols = len(MODALITIES) * 2
    fig, axes = plt.subplots(num_slices, n_cols, figsize=(4 * n_cols, 4 * num_slices))
    if num_slices == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"MyoPS2020 — Patient {pid}  |  Shape: {images['C0']['data'].shape}  |  "
        f"Spacing: {tuple(round(s, 2) for s in spacing)} mm",
        fontsize=14, fontweight="bold", y=1.0
    )

    for s in range(num_slices):
        overlay = make_label_overlay(gd[:, :, s]) if gd is not None else None
        for mod_idx, mod in enumerate(MODALITIES):
            sl = images[mod]["data"][:, :, s]
            vmin, vmax = display_window(sl)

            ax = axes[s, mod_idx * 2]
            ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(f"{DISPLAY_MODALITY_NAMES.get(mod, mod)} — Slice {s}", fontsize=10)
            ax.axis("off")

            ax = axes[s, mod_idx * 2 + 1]
            ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            if overlay is not None:
                ax.imshow(overlay, aspect="equal")
            ax.set_title(f"GT on {mod} — Slice {s}", fontsize=10)
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
    if patient_ids:
        print(f"  Training patients : {len(patient_ids)}  (IDs: {patient_ids[0]}-{patient_ids[-1]})")
    else:
        print("  Training patients : 0")
    print(f"  Modalities        : {', '.join(f'{m} ({MODALITY_NAMES[m]})' for m in MODALITIES)}")
    print(f"  Label values      : 0=BG, 200=Myo, 500=LV, 600=RV, 1220=Edema, 2221=Scar")

    # Count test files
    test_dir = os.path.join(data_dir, "test20")
    if os.path.isdir(test_dir):
        test_ids = set()
        for f in os.listdir(test_dir):
            m = re.match(r"myops_test_(\d+)_", f)
            if m:
                test_ids.add(int(m.group(1)))
        if test_ids:
            print(f"  Test patients     : {len(test_ids)}  (IDs: {min(test_ids)}-{max(test_ids)})")
        else:
            print("  Test patients     : 0")
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
