# JBHI Submission Plan: AsymSpecMambaDCN v3.1

## 1. Target Paper Claim

AsymSpecMambaDCN v3.1 is a lightweight asymmetric 2.5D cine-CMR
segmentation model that produces reproducible LV/RV/myocardium masks and
derived patient-specific cardiac indices from ACDC using full-resolution local
precision plus low-resolution spectral context.

The defensible claim is:

> Efficient, clinically oriented, reproducible segmentation for personalized
> cardiac function quantification.

Do not claim SOTA, true Mamba, DCNv4, or a complete personalized medicine
system.

## 2. Recommended Paper Framing

Recommended title:

> Lightweight Asymmetric Spectral 2.5D Segmentation for Patient-Specific
> Cardiac Function Quantification in Cine MRI

Alternative title:

> AsymSpecMambaDCN v3.1: Lightweight Asymmetric Spectral 2.5D Cine-MRI
> Segmentation for Personalized Cardiac Indices

The JBHI special-issue angle should be a compact segmentation-to-measurement
pipeline for cardiovascular imaging. The method details support the clinical
claim, but the paper should emphasize image-derived cardiac indices:

- LV/RV end-diastolic volume (EDV)
- LV/RV end-systolic volume (ESV)
- Stroke volume (SV)
- Ejection fraction (EF)
- Myocardial mass
- Bias, limits of agreement, and failure cases

## 3. Current Evidence

User-provided off-server ACDC results:

| Checkpoint | Params | Test Volumes | Avg Dice | Avg HD95 | HD95 Unit |
| --- | ---: | ---: | ---: | ---: | --- |
| `acdc_asym_v31_c48_best_dice.pt` | 673,089 | 100 | 0.8853 | 2.1158 | pixel |
| `acdc_asym_v31_c48_best_balanced.pt` | 673,089 | 100 | 0.8845 | 2.3787 | pixel |
| `acdc_asym_v31_c48_best_hd95.pt` | 673,089 | 100 | 0.8845 | 2.3787 | pixel |

These numbers are promising but not yet paper-grade until:

- The exact patient/volume split is documented.
- The run is reproducible from saved logs, weights, config, split, and commit.
- HD95 is converted from pixels to physical units when spacing metadata is
  available.
- Baselines and ablations are run under the same preprocessing and split.

## 4. Minimum Viable Paper

### Method

- Full-resolution precision branch over center 3 slices.
- Low-resolution Fourier context branch over all 5 slices.
- Asymmetric bidirectional exchange (ABX).
- SDF-gated frequency-split fusion.
- Compound segmentation objective with Dice/CE/focal/HD/SDF/context terms.
- C48 parameter count: 673,089.

### Dataset

- ACDC cine MRI.
- Patient-level train/validation/test policy.
- ED and ES phases from the same patient must stay in the same fold.
- Resolve whether "100 test volumes" means 50 patients with ED/ES or another
  held-out protocol.

### Metrics

- Dice per class: RV, MYO, LV.
- HD95 in mm, not only pixels.
- ASSD if available.
- ED/ES-specific segmentation metrics.
- EDV, ESV, SV, EF, and myocardial mass error.
- Parameters, FLOPs/MACs, latency, and GPU memory.

### Analysis

- Disease subgroup performance if labels exist: NOR, DCM, HCM, MINF, RV.
- Bland-Altman and correlation for EF/volume estimates.
- Failure cases for basal/apical slices, RV boundary errors, and MYO outliers.

## 5. Experiment Priorities

| Priority | Experiment | Purpose | Expected Evidence | Risk | Effort |
| --- | --- | --- | --- | --- | --- |
| P0 | Reproduce C48 result with archived artifacts | Establish credibility | Same Dice/HD95 within tolerance | Prior run may not reproduce | Medium |
| P0 | Patient-level split audit | Remove leakage concern | Zero patient overlap | Split naming may be inconsistent | Low |
| P0 | Convert HD95 from pixels to mm | Make metric comparable | Class-wise mm HD95 | Spacing metadata errors | Low |
| P0 | Clinical-index evaluation | Fit JBHI special issue | EDV/ESV/SV/EF/mass errors | Postprocessing bugs | Medium |
| P1 | Baseline reproduction | Show method value | Performance/efficiency table | Baselines may outperform | Medium |
| P1 | Core ablations | Prove architecture matters | Module-level gains | Small effect sizes | Medium |
| P1 | Multi-seed stability | Address overfitting | Mean/std over 3 seeds | Compute cost | Medium |
| P2 | Disease subgroup analysis | Translational evidence | Subgroup errors | Small subgroup N | Low |
| P2 | Inference efficiency | Support lightweight claim | Params/FLOPs/latency/memory | Hardware variability | Low |

## 6. Baseline Plan

Required fair baselines:

- 2D U-Net: minimum classical baseline.
- 2.5D U-Net: tests whether gains come from multi-slice input alone.
- UNet++ or Attention U-Net: stronger CNN baseline.
- nnU-Net-style reference: anchors reviewer expectations.
- Lightweight matched-parameter baseline: supports the efficiency claim.

Optional if time and compute allow:

- TransUNet or SwinUNETR.
- U-Mamba, VM-UNet, MSVM-UNet, or a close public Mamba-family baseline.

If Mamba-family baselines are not run fairly, avoid making Mamba-comparison
claims.

## 7. Ablation Plan

Must-have ablations:

- Full model.
- Remove ABX exchange.
- One-way ABX: context-to-precision only.
- One-way ABX: precision-to-context only.
- Remove SDF gate.
- SDF gate without detach, if stable.
- Replace frequency-split fusion with plain additive fusion.
- Replace frequency-split fusion with concat + convolution.
- Precision branch only.
- Context branch only.
- 1-slice, 3-slice, and 5-slice inputs.
- C32, C48, and C64 scaling.
- Loss ablations: no HD term, no SDF term, no context-head term, no deep
  supervision.

Each ablation table should include:

- RV/MYO/LV Dice.
- RV/MYO/LV HD95 in mm.
- Average foreground Dice and HD95.
- EDV/ESV/EF/mass errors if available.
- Parameter count and FLOPs.

## 8. Overfitting Diagnostics

Minimum diagnostics:

- Train/validation loss curves.
- Train/validation Dice curves.
- Validation HD95 or boundary metric curves.
- Best epoch for `best_dice`, `best_hd95`, and `best_balanced`.
- 3-seed mean/std for the final selected config.
- Per-patient outlier list.
- Basal/apical failure cases.
- Checkpoint selection rule fixed before test evaluation.
- Confirmation that no ED/ES or patient identity leaks across splits.

Reviewer-facing rule:

> If logs, weights, and configs cannot reproduce the off-server result, treat the
> current numbers as preliminary and rerun before writing final claims.

## 9. Reproducibility Requirements

For every paper-facing run, archive:

- Git commit hash.
- Full training command.
- Resolved config JSON.
- Split manifest.
- Environment file.
- Training log.
- Validation history JSON/CSV.
- Best checkpoint.
- Evaluation command.
- Evaluation result JSON.
- Per-volume predictions or per-volume metrics.
- Hardware details.

Evaluation should use strict checkpoint loading, or explicitly report missing and
unexpected keys if a non-strict load is unavoidable.

## 10. Six-Week Milestone Plan

### Week 1: Protocol Lock

- Freeze paper title, claim, split policy, and metrics.
- Confirm JBHI special issue deadline and formatting.
- Convert HD95 to mm.
- Define cardiac-index computation.
- Decide exact baseline list.

### Week 2: Reproducibility

- Rerun C48 with fixed config.
- Save logs, weights, resolved config, split, environment, and commit hash.
- Validate patient-level split.
- Verify test evaluation with strict checkpoint loading.

### Week 3: Baselines

- Run 2D U-Net.
- Run 2.5D U-Net.
- Run UNet++ or Attention U-Net.
- Add nnU-Net-style reference if feasible.

### Week 4: Ablations

- Run ABX ablations.
- Run SDF-gate ablations.
- Run frequency-fusion ablations.
- Run branch-only and slice-count ablations.
- Run C32/C48/C64 scaling.

### Week 5: Clinical Analysis

- Compute EDV, ESV, SV, EF, and myocardial mass.
- Add subgroup analysis if disease labels are available.
- Add Bland-Altman/correlation plots.
- Select failure-case visualizations.

### Week 6: Paper Draft

- Write methods and experiments.
- Build main segmentation, clinical-index, ablation, and efficiency tables.
- Write limitations explicitly.
- Remove SOTA, true Mamba, and DCNv4 language.

## 11. Main Tables and Figures

Recommended tables:

- Table 1: method comparison and parameter/FLOP summary.
- Table 2: ACDC segmentation results with Dice and HD95-mm.
- Table 3: cardiac-index errors for EDV, ESV, SV, EF, and myocardial mass.
- Table 4: ablation study.
- Table 5: efficiency and deployment cost.

Recommended figures:

- Architecture overview.
- Segmentation examples at ED/ES.
- Failure cases and boundary outliers.
- Bland-Altman plots for EF/volume estimates.
- Training/validation curves for overfitting analysis.

## 12. Immediate Checklist

- Confirm exact patient/volume split behind the 100 test volumes.
- Retrieve off-server logs, weights, config, and split files.
- Convert HD95 from pixels to mm.
- Implement or verify cardiac-index extraction from masks.
- Run 2D U-Net and 2.5D U-Net under the identical split.
- Run core ablations: ABX, SDF gate, frequency fusion, branch-only.
- Build one main JBHI table: segmentation + cardiac indices + efficiency.
- Write limitations: single dataset, no external clinical validation yet, and
  no claim of true Mamba or DCNv4.

## 13. Next Handoff

Next step: `@Architect` should convert this research plan into a concrete
experiment matrix, metric scripts, and implementation design.
