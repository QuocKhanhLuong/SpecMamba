# AsymSpecMambaDCN v3.1: Paper Proposal and Readiness Notes

## 1. Working Thesis

AsymSpecMambaDCN v3.1 is a lightweight asymmetric 2.5D cardiac MRI segmentation
model for ACDC-style short-axis cine MRI. The method segments the center slice
from a 5-slice input stack by separating high-resolution boundary processing
from low-resolution global context modeling.

The current paper should be framed as a proposal plus implementation plan until
patient-level validation, baselines, ablations, and metric-unit checks are
rerun.

## 2. Current Method Narrative

The paper method is the dual-branch `AsymSpecMambaDCN` implementation in
`src/models/specmamba_net.py`, not the older 3-stream `SpecMambaNet` baseline.

1. **Precision branch:** the center 3 slices `[k-1, k, k+1]` are processed at
   full resolution by spectral-guided deformable-convolution blocks. The
   implementation uses `torchvision.ops.deform_conv2d`, so claims should say
   "deformable convolution" or "DCN-inspired", not true DCNv3/DCNv4.
2. **Global context branch:** all 5 slices `[k-2, ..., k+2]` are downsampled to
   56x56 and processed by `Global2DFFTMixer` blocks for low-resolution global
   context.
3. **ABX exchange:** after each stage, context-to-precision channel scaling and
   precision-to-context spatial residual exchange share information without
   directly injecting low-resolution spatial maps into full-resolution features.
4. **SDF-gated frequency-split fusion:** the precision feature map is split into
   low/high frequency components; context is injected into the low-frequency
   component under an SDF-derived gate while high-frequency boundary detail is
   preserved.
5. **Compound objective:** Dice, CE, focal, HD-style distance, SDF, context-head,
   and deep-supervision terms are used during training.

## 3. Terminology Rules

Use these terms in the paper and documentation:

- **Use:** AsymSpecMambaDCN v3.1, asymmetric 2.5D routing, precision branch,
  global Fourier context branch, spectral-guided deformable convolution,
  Mamba-inspired cross-scan gated mixer, ABX exchange, SDF-gated
  frequency-split fusion.
- **Avoid as claims:** true Mamba, selective SSM, DCNv3, DCNv4, mathematical HDC
  guarantee, SOTA, artifact elimination, or guaranteed anti-ringing.

The older 3-stream Spec-HRNet / `SpecMambaNet` material should be treated as
legacy background or an internal baseline, not the submitted method.

## 4. Training Protocol

The paper-facing run is config driven:

```bash
python src/training/train_acdc.py --config configs/acdc_asym_v31.yaml
```

The split must be patient-level. ED and ES phases from the same `patientXXX`
must stay in the same fold. The current manifest is:

```text
splits/acdc_patient_split_seed42.json
```

The default local ACDC data directory is:

```text
preprocessed_data/ACDC
```

## 5. Reviewer-Critical Risks

1. **Patient-level leakage:** fixed in the training pipeline by patient split
   manifests, but all old metrics must be rerun.
2. **Metric unit mismatch:** ACDC leaderboard and many papers report Hausdorff in
   physical units. Local documented values are pixel HD95 unless spacing
   metadata is complete and `--hd95_unit mm` is used.
3. **Terminology overclaiming:** Mamba and DCNv4 terminology must stay qualified
   unless the actual operators are implemented and ablated.
4. **Missing baselines:** U-Net, 2.5D U-Net/UNet++, nnU-Net, HRNet-style,
   transformer, and medical Mamba baselines are still needed.
5. **Missing ablations:** ABX, FFT context, SDF gate, frequency split, 2.5D
   routing, deep supervision, class weights, and HD/SDF losses must be isolated.

## 6. Minimum Evidence Before MICCAI Submission

- Patient-level train/val/test or cross-validation split table.
- Per-class RV/MYO/LV Dice and HD95 with units stated.
- Mean and standard deviation over folds or seeds.
- Strict checkpoint loading for final evaluation.
- Config, split manifest, commit hash, environment, and resolved args saved
  with each run.
- Baselines trained under the same preprocessing and split.
- Ablations for every claimed contribution.
- Failure-case visualization focused on RV boundary false positives and HD95
  outliers.

## 7. Current Recommendation

Proceed with AsymSpecMambaDCN v3.1 as a lightweight, conservative contribution:

> An asymmetric 2.5D cardiac MRI segmentation architecture that combines
> high-resolution deformable boundary features with low-resolution Fourier
> context and SDF-gated frequency-split fusion.

Do not claim SOTA until the patient-level protocol, baselines, and ablations are
complete.
