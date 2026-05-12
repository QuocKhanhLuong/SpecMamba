# CHANGELOG.md

## 2026-05-11

### Added

- Added `scripts/acdc_split.py` for patient-level ACDC split creation and
  validation.
- Added `splits/acdc_patient_split_seed42.json` with a saved 80/20 patient
  split.
- Added `configs/acdc_asym_v31.yaml` and aligned `configs/acdc_reproducible.yaml`
  for config-driven AsymSpecMambaDCN v3.1 training.
- Added `scripts/train_acdc_reproducible.sh` wrapper for config-driven training.
- Added split/reproducibility tests in `tests/test_acdc_split.py` and
  `tests/test_acdc_training_reproducibility.py`.
- Added `KNOWLEDGE.md` with research critique, external evidence, implementation
  notes, and remaining paper-readiness work.

### Changed

- Updated `src/training/train_acdc.py` to support YAML/JSON configs, seeded
  training setup, patient-level split manifests, seeded DataLoader workers, and
  resolved run-argument export.
- Moved `src/losses/physics_loss.py` demo code behind a `__main__` guard so
  importing training modules no longer prints random demo loss values.
- Updated `report.md` to use the AsymSpecMambaDCN v3.1 paper narrative instead
  of the legacy 3-stream Spec-HRNet narrative.
- Updated `ARCHITECTURE.md` to qualify Mamba/DCNv3/DCNv4/HDC claims, remove
  unsupported SOTA-style language, and point to config-driven training.
- Softened source docstrings in `src/models/specmamba_net.py` so comments match
  the conservative paper terminology.

### Validation

- Generated patient-level split manifest:
  `python scripts/acdc_split.py --data_dir preprocessed_data/ACDC --output splits/acdc_patient_split_seed42.json --seed 42 --train_ratio 0.8`
- Passed focused tests:
  `python -m pytest tests/test_acdc_training_reproducibility.py tests/test_acdc_split.py -q`
- Passed compile check:
  `python -m py_compile src/training/train_acdc.py scripts/acdc_split.py tests/test_acdc_split.py tests/test_acdc_training_reproducibility.py`
- Passed wrapper shell syntax:
  `bash -n scripts/train_acdc_reproducible.sh`
- Confirmed `conda run -n alvin python src/training/train_acdc.py --help`
  reaches argparse successfully. It still reports missing optional
  `albumentations` in that environment.
- Confirmed split manifest has 80 train patients, 20 validation patients, and
  zero patient overlap.

### Not Done

- Full training was not started.
- Baselines and ablations remain to be implemented and rerun.
