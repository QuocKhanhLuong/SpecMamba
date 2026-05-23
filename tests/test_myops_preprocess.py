from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_myops2020 import main as preprocess_main
from scripts.preprocess_myops2020 import preprocess_patient


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    affine = np.eye(4, dtype=np.float32) if affine is None else affine
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header.set_zooms((0.75, 0.75, 12.0))
    nib.save(img, str(path))


def _make_myops_case(root: Path, labels: np.ndarray | None = None) -> None:
    train_dir = root / "train25"
    gd_dir = root / "train25_myops_gd"
    train_dir.mkdir(parents=True)
    gd_dir.mkdir(parents=True)

    shape = (6, 5, 2)
    for idx, mod in enumerate(("C0", "DE", "T2")):
        data = np.full(shape, fill_value=float(idx + 1), dtype=np.float32)
        data[1:4, 1:4, :] += np.arange(np.prod((3, 3, 2)), dtype=np.float32).reshape(3, 3, 2)
        _write_nifti(train_dir / f"myops_training_101_{mod}.nii.gz", data)

    if labels is None:
        labels = np.zeros(shape, dtype=np.int16)
        labels[1:3, 1:3, 0] = 200
        labels[3:5, 2:4, 0] = 500
        labels[2:4, 1:3, 1] = 600
        labels[4:6, 3:5, 1] = 1220
        labels[0:2, 3:5, 1] = 2221
    _write_nifti(gd_dir / "myops_training_101_gd.nii.gz", labels)


def test_preprocess_patient_keeps_modalities_paired(tmp_path: Path) -> None:
    _make_myops_case(tmp_path)

    records = preprocess_patient(str(tmp_path), 101, target_size=(4, 4))

    assert len(records) == 1
    volume, mask, volume_id, info = records[0]
    assert volume_id == "patient101"
    assert volume.shape == (3, 4, 4, 2)
    assert mask.shape == (4, 4, 2)
    assert volume.dtype == np.float32
    assert mask.dtype == np.uint8
    assert info["modalities"] == ["C0", "DE", "T2"]
    assert info["num_modalities"] == 3
    assert set(np.unique(mask)).issubset({0, 1, 2, 3, 4, 5})
    assert info["label_histogram_original"]["200"] > 0
    assert info["label_histogram_remapped"]["1"] > 0


def test_preprocess_patient_rejects_unknown_labels(tmp_path: Path) -> None:
    labels = np.zeros((6, 5, 2), dtype=np.int16)
    labels[1, 1, 0] = 999
    _make_myops_case(tmp_path, labels=labels)

    with pytest.raises(ValueError, match="Unknown MyoPS labels"):
        preprocess_patient(str(tmp_path), 101, target_size=(4, 4))


def test_preprocess_cli_writes_one_paired_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "preprocessed"
    _make_myops_case(raw_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_myops2020.py",
            "--input",
            str(raw_root),
            "--output",
            str(out_root),
            "--size",
            "4",
        ],
    )

    preprocess_main()

    assert sorted(p.name for p in (out_root / "volumes").glob("*.npy")) == ["patient101.npy"]
    assert sorted(p.name for p in (out_root / "masks").glob("*.npy")) == ["patient101.npy"]
    volume = np.load(out_root / "volumes" / "patient101.npy")
    mask = np.load(out_root / "masks" / "patient101.npy")
    assert volume.shape == (3, 4, 4, 2)
    assert mask.shape == (4, 4, 2)
