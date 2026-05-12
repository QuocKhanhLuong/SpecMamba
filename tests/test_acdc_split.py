import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from acdc_split import build_split_manifest, group_by_patient, validate_manifest


def test_group_by_patient_keeps_ed_es_together():
    grouped = group_by_patient([
        "patient001_ED.npy",
        "patient001_ES.npy",
        "patient002_ED.npy",
    ])
    assert grouped["patient001"] == ["patient001_ED", "patient001_ES"]
    assert grouped["patient002"] == ["patient002_ED"]


def test_validate_manifest_rejects_patient_overlap():
    manifest = {
        "train_patients": ["patient001"],
        "val_patients": ["patient001"],
        "train_volumes": ["patient001_ED"],
        "val_volumes": ["patient001_ES"],
    }
    try:
        validate_manifest(manifest)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected patient overlap to fail validation")


def test_real_acdc_split_has_no_patient_overlap_if_data_present():
    data_dir = ROOT / "preprocessed_data" / "ACDC"
    if not (data_dir / "volumes").exists():
        return
    manifest = build_split_manifest(data_dir, train_ratio=0.8, seed=42)
    assert set(manifest["train_patients"]).isdisjoint(manifest["val_patients"])
    assert len(manifest["train_patients"]) + len(manifest["val_patients"]) == manifest["n_patients"]
