# SSRBlockV3 ACDC Debug Experiment

This folder contains isolated Selective Spectral Retention experiments. It does
not modify the main SpecMamba training pipeline, models, or paper-facing code.

The experiment uses the existing preprocessed ACDC data under:

```bash
preprocessed_data/ACDC
```

Supported layouts:

```bash
preprocessed_data/ACDC/volumes/*.npy
preprocessed_data/ACDC/masks/*.npy
```

or:

```bash
preprocessed_data/ACDC/training/volumes/*.npy
preprocessed_data/ACDC/training/masks/*.npy
preprocessed_data/ACDC/testing/volumes/*.npy
preprocessed_data/ACDC/testing/masks/*.npy
```

When `training/` and `testing/` are present, train/val splitting is made from
`training/` only. To point directly at a split directory, pass:

```bash
python test/train_ssr_acdc.py --config test/configs/ssr_v3_acdc.yaml --data_root preprocessed_data/ACDC/training
```

Do not download Kaggle data for this debug harness.

## Train

Run from the repository root:

```bash
python test/train_ssr_acdc.py --config test/configs/ssr_v3_acdc.yaml
```

Quick smoke run:

```bash
python test/train_ssr_acdc.py --config test/configs/ssr_v3_acdc.yaml --epochs 2
```

The script supports CLI overrides for `--epochs`, `--run_name`, `--device`,
`--data_root`, `--output_root`, `--batch_size`, and `--num_workers`.

## Analyze

```bash
python test/analyze_ssr_logs.py --run_dir test/outputs/ssr_v3_acdc_debug
```

## Logs To Inspect

- `training_log.csv`: loss, validation foreground Dice, and per-class Dice.
- `ssr_logs.csv`: per-block spectral gate and contribution diagnostics.
- update gate collapse: band 0 taking most of the update budget.
- suppress saturation: high-band suppress gates pinned near their configured max.
- high-frequency ratio: collapse below 0.75 or amplification risk above 1.7.
- boundary to non-boundary high ratio: should improve if boundary-relevant high
  frequencies are retained better.
- validation foreground Dice: sanity metric only; do not treat this prototype as
  a final performance claim.

Generated artifacts are written to `test/outputs/<run_name>/`.

## Phase Test 2

Phase Test 2 adds config-controlled ablations for channel-aware residual gating,
bounded residual strength, high-frequency ratio regularization, suppress floors,
and optional geometry refinement. It is still a controlled experiment, not a
final paper model or performance claim.

Default phase-2 run:

```bash
python test/train_ssr_acdc.py --config test/configs/ssr_phase2_acdc_224.yaml
```

Run a variant:

```bash
python test/train_ssr_acdc.py \
  --config test/configs/ssr_phase2_acdc_224.yaml \
  --variant ssr_se_bounded
```

Smoke test:

```bash
python test/train_ssr_acdc.py \
  --config test/configs/ssr_phase2_acdc_224.yaml \
  --epochs 2 \
  --batch_size 8
```

Run all block-architecture ablations:

```bash
bash test/run_phase2_variants.sh
```

Optional quick ablation run:

```bash
EPOCHS=20 BATCH_SIZE=4 DEVICE=cuda bash test/run_phase2_variants.sh
```

Analyze:

```bash
python test/analyze_ssr_logs.py --run_dir test/outputs/ssr_phase2_acdc_224
```

CLI overrides:

```bash
--epochs --batch_size --image_size --run_name --device --variant
```

Variants:

- `baseline_ssr`: current SSR without SE, gamma bound, or high-frequency ratio penalty.
- `ssr_se`: adds SE gating on the spectral update.
- `ssr_se_bounded`: adds SE, bounded gamma, and high-frequency ratio penalty.
- `ssr_se_lk`: adds large-kernel geometry refinement.
- `ssr_se_dcn`: adds a pure-PyTorch DCNv4-style geometry refinement path based
  on the official module structure. It does not require compiling the external
  DCNv4 CUDA extension, and it is not a speed-equivalent official operator.
- `ssr_se_deformable`: legacy torchvision deformable-conv refinement using
  `torchvision.ops.deform_conv2d`; this is not DCNv4.
- `ssr_full`: uses residual channel gate plus large-kernel geometry refinement.

Expected behavior to inspect:

- `high_freq_ratio` should decrease compared with Phase 1.
- `boundary_to_nonboundary_high_ratio` should stay above 1.5.
- suppress gates should not stay all zero and should not saturate.
- validation Dice should remain competitive.
- prediction grids should show fewer LV holes and fewer small false positives.

Notes:

- The trainer automatically retries CUDA OOM by halving batch size: 8 to 4 to 2 to 1.
- If CUDA is unavailable, it falls back to CPU and uses `num_workers=0` for local stability.
- `geometry_refine: dcnv4` now uses the local pure-PyTorch DCNv4-style block.
- The local DCNv4-style block follows the upstream channel constraint: `channels / dcnv4_group`
  must be divisible by 16. With the default `base_channels: 32`, keep
  `dcnv4_group: 2`; if you increase `base_channels` to 64, `dcnv4_group: 4`
  is also valid.
- `geometry_refine: deformable` uses torchvision `deform_conv2d`, which is a
  modulated deformable-conv path and should not be described as DCNv4.
