# KNOWLEDGE.md

## Phase 1 — Research Critique and Paper Direction

### Local Project Understanding

- The paper method is now scoped to `AsymSpecMambaDCN v3.1`, not the older
  3-stream `SpecMambaNet`/Spec-HRNet narrative.
- The coherent contribution is a lightweight asymmetric 2.5D cardiac MRI
  segmentation model with:
  - full-resolution precision branch over center 3 slices,
  - low-resolution Fourier context branch over all 5 slices,
  - ABX cross-branch exchange,
  - SDF-gated frequency-split fusion,
  - CompoundHDLoss-style training.
- The first review blocker was protocol validity: ED and ES phases were split
  by volume, which could leak patient identity into validation.

### External Evidence Retrieved

- ACDC is a cardiac cine-MRI benchmark with ED/ES annotations and labels for
  BG, RV, MYO, and LV. The official challenge pages describe the data and
  evaluation protocol: [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html),
  [ACDC evaluation](https://www.creatis.insa-lyon.fr/Challenge/acdc/evaluation.html).
- ACDC leaderboard-style Hausdorff metrics are protocol and unit sensitive;
  local pixel HD95 must not be directly compared with mm-based paper or
  leaderboard values: [ACDC results](https://www.creatis.insa-lyon.fr/Challenge/acdc/results.html).
- U-Net remains the canonical biomedical segmentation baseline:
  [U-Net MICCAI 2015](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
- nnU-Net is a required modern medical segmentation baseline because it
  self-configures preprocessing, architecture, training, and post-processing:
  [nnU-Net Nature Methods](https://www.nature.com/articles/s41592-020-01008-z).
- True Mamba is an input-dependent selective state-space model with selective
  scan; the local `CrossScanGatedMixer` is a Conv1d scan approximation, so the
  paper should say "Mamba-inspired" rather than "Mamba":
  [Mamba](https://arxiv.org/abs/2312.00752).
- DCNv4 is a specific deformable-convolution operator; the local code uses
  `torchvision.ops.deform_conv2d`, so the paper should say
  "spectral-guided deformable convolution" rather than true DCNv4:
  [DCNv4 CVPR 2024](https://cvpr.thecvf.com/virtual/2024/poster/31637).
- Frequency-domain mixing is established in vision and sequence modeling, so
  FFT usage alone is not novel: [GFNet](https://proceedings.neurips.cc/paper_files/paper/2021/hash/07e87c2f4fc7f7c96116d8e2a92790f5-Abstract.html),
  [AFNO](https://iclr.cc/virtual/2022/poster/6073),
  [FNet](https://aclanthology.org/2022.naacl-main.319/).
- Boundary/SDF/HD losses are established; the novelty must be in how boundary
  signals interact with fusion, not the loss terms alone:
  [Boundary loss](https://proceedings.mlr.press/v102/kervadec19a.html),
  [Hausdorff loss](https://doi.org/10.1109/TMI.2019.2930068).

### Proposal Critique

- The proposal should avoid SOTA claims until results are rerun under
  patient-level splits and strong baselines.
- The proposal should not claim true Mamba/SSM or true DCNv4 without replacing
  the operators and adding ablations.
- The old HDC gcd guarantee was invalid for powers-of-two dilations and has
  been removed from the paper narrative.
- Minimum baselines remain: 2D U-Net, 2.5D U-Net/UNet++, nnU-Net, HRNet-style,
  transformer-style, and medical Mamba/SSM baselines.
- Minimum ablations remain: 1/3/5-slice input, deformable vs non-deformable,
  spectral guidance on/off, FFT context vs alternatives, ABX directions,
  SDF gate on/off, frequency split vs additive fusion, and loss/deep supervision
  variants.

## Phase 2 — Implementation Knowledge

### Patient-Level Split

- Added `scripts/acdc_split.py` to create and audit patient-level ACDC split
  manifests.
- Added `splits/acdc_patient_split_seed42.json`.
- Local generated split: 80 train patients, 20 validation patients, 160 train
  volumes, 40 validation volumes.
- The manifest keeps ED and ES phases from the same `patientXXX` in the same
  split.

### Reproducible Training

- `src/training/train_acdc.py` now supports `--config`.
- Paper-facing config: `configs/acdc_asym_v31.yaml`.
- Wrapper script: `scripts/train_acdc_reproducible.sh`.
- Training now records seed settings, can reuse saved split manifests, and
  writes resolved run arguments.
- `physics_loss.py` no longer prints demo loss values during import, so CLI
  checks and training logs are less noisy.

### Paper Narrative Alignment

- `report.md` now describes the current `AsymSpecMambaDCN v3.1` proposal.
- `ARCHITECTURE.md` no longer presents historical pixel-HD95 numbers as
  paper-ready results.
- Mamba/DCNv3/DCNv4/HDC language has been softened or qualified.

## Phase 3 — Remaining Research Work

- Implement and run baselines.
- Implement and run ablation matrix.
- Validate HD95 in physical units when spacing metadata is complete.
- Add strict final evaluation checkpoint loading.
- Save full run metadata: split, config, commit hash, environment, and results.
