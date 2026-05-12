# AsymSpecMambaDCN v3.1 — Full Architecture Documentation

> **Asymmetric 2.5D Routing with Global FFT Context & Bidirectional Exchange**
> for 3D Cardiac MRI Segmentation on the ACDC Benchmark.

---

## 1. High-Level Overview

AsymSpecMambaDCN v3.1 is an asymmetric dual-branch architecture that segments the
**center slice** of a 2.5D input stack (5 consecutive cardiac MRI slices). The
key insight is that **boundary precision** and **global semantic context**
require fundamentally different compute budgets and receptive field strategies.

**v3.1 upgrades over v3:**
- `Global2DFFTMixer` replaces `CrossScanGatedMixer` in the context branch,
  providing **full spatial receptive field** via learned 2D spectral filters
- **Asymmetric Bidirectional Exchange (ABX)** after every stage enables
  inter-branch information flow during feature extraction
- **Progressive 2-step downsampling** (224→112→56) replaces 1-step stride-4
- **Auxiliary context segmentation head** ensures direct gradient to the
  context branch, preventing gradient bottleneck

```
Input: X ∈ ℝ^(B × 5 × 224 × 224)
       [k-2, k-1, k, k+1, k+2]  — 5 consecutive 2D slices

                ┌──────────────────────────────────────┐
                │            INPUT (B,5,H,W)           │
                └───────┬──────────────────┬───────────┘
                        │                  │
           x_local = x[:,1:4]        x (all 5 slices)
            (B,3,224,224)              (B,5,224,224)
                        │                  │
                   stem_init()          down()
                        │                  │
                ┌───────▼───────┐  ┌───────▼───────────┐
                │   STAGE 1     │  │     STAGE 1        │
                │ 2× SG-Deform │  │  Global2DFFTMixer  │
                │ + scan mixer │  │  @ 56²             │
                └───────┬───────┘  └───────┬───────────┘
                        │                  │
                   ┌────▼──────────────────▼────┐
                   │    ABX Exchange #1          │
                   │  ctx→prec: channel scale    │
                   │  prec→ctx: spatial residual │
                   └────┬──────────────────┬────┘
                        │                  │
                ┌───────▼───────┐  ┌───────▼───────────┐
                │   STAGE 2     │  │     STAGE 2        │
                │ 4× SG-Deform │  │  Global2DFFTMixer  │
                │ + scan mixer │  │  @ 56²             │
                └───────┬───────┘  └───────┬───────────┘
                        │                  │
                   ┌────▼──────────────────▼────┐
                   │    ABX Exchange #2          │
                   └────┬──────────────────┬────┘
                        │                  │
                ┌───────▼───────┐  ┌───────▼───────────┐
                │   STAGE 3     │  │     STAGE 3        │
                │ 6× SG-Deform │  │  Global2DFFTMixer  │
                │ + scan mixer │  │  @ 56²             │
                └───────┬───────┘  └───────┬───────────┘
                        │                  │
                   ┌────▼──────────────────▼────┐
                   │    ABX Exchange #3          │
                   └────┬──────────────────┬────┘
                        │                  │
                  feat_center          feat_z_ctx
                  (B,C,224,224)        (B,C,56,56)
                        │                  │
                        │          ┌───────▼───────────┐
                        │          │ CascadedPixelShuffle│
                        │          │ 56² → 112² → 224²  │
                        │          └───────┬───────────┘
                        │                  │
                        │              ctx_up
                        │              (B,C,224,224)
                        │                  │
                ┌───────▼───────┐          │
                │   SDF Head    │          │
                │ Conv1×1→Tanh  │          │
                └───────┬───────┘          │
                        │                  │
                   sdf_pred                │
                   (B,3,H,W)               │
                        │                  │
                ┌───────▼───────┐          │
                │   SDF Gate    │          │
                │ sigmoid(conv  │          │
                │  (detach(sdf)))│         │
                └───────┬───────┘          │
                        │                  │
                      gate                 │
                      (B,1,H,W)            │
                        │                  │
                ┌───────▼──────────────────▼───────────┐
                │      BRANCH C: Frequency-Split Fusion │
                │                                       │
                │  FFT(feat_center)                     │
                │    → Gaussian LP mask → feat_low      │
                │    → feat_high = feat_center - feat_low│
                │                                       │
                │  α = σ(Conv1×1([feat_low, ctx_up]))   │
                │  fused_low = feat_low + gate·α·ctx_up │
                │  feat_fused = fused_low + feat_high   │
                └───────────────────┬───────────────────┘
                                    │
                              feat_fused
                              (B,C,224,224)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼───────┐ ┌────▼────┐  ┌───────▼──────────┐
            │   Seg Head    │ │ctx_logits│  │  Aux DS Heads    │
            │ Conv3→GN→GELU │ │(training)│  │  (training only) │
            │ → Conv1×1     │ │          │  │                  │
            └───────┬───────┘ └────┬────┘  └───────┬──────────┘
                    │              │                │
                 logits       ctx_logits       aux_outputs
              (B,4,224,224)  (B,4,224,224)    [stage1, stage2]
```

**Output dictionary:**

| Key | Shape | Description |
|-----|-------|-------------|
| `output` | `(B, 4, H, W)` | Segmentation logits (BG, RV, MYO, LV) |
| `sdf` | `(B, 3, H, W)` | Per-class Signed Distance Field in \[-1, 1\] |
| `ctx_logits` | `(B, 4, H, W)` | Context branch seg logits (training only) |
| `aux_outputs` | list of `(B, 4, H, W)` | Deep supervision logits (training only) |

**Parameter count:** ~673K (base_ch=48)

---

## 2. Branch A — Precision Stem

**Goal:** Extract boundary-precise features at full resolution (224x224).

**Input:** `x_local = x[:, 1:4, :, :]` — center 3 slices \[k-1, k, k+1\].
Using 3 slices (not all 5) provides local Z-gradients for boundary detection
without bloating the DCN compute.

### 2.1 Stem Initializer

```
Conv2d(3, C, 3, padding=1) → GroupNorm(C) → GELU
```

Maps 3-channel input to `C`-channel feature space.

### 2.2 Three Stages with HDC Dilation Pyramids

Each stage consists of spectral-guided deformable-convolution blocks followed
by a Mamba-inspired `CrossScanGatedMixer` scan approximation. The number of
blocks and dilation values increase per stage:

| Stage | Deformable Blocks | Dilations | Scan Approximation | Receptive Field |
|-------|---------------|-----------|------------|-----------------|
| 1 | 2 | \[1, 2\] | 1-pass (forward H) | Local |
| 2 | 4 | \[1, 2, 4, 8\] | 2-pass (bidirectional H) | Medium |
| 3 | 6 | \[1, 2, 4, 8, 16, 32\] | 4-pass (cross-scan H+W) | Global |

**Total: 12 spectral-guided deformable blocks + 3 scan-mixer blocks**

The dilation pyramid is intended to expand the receptive field within each
stage. It should not be described as a mathematical guarantee against gridding
without a separate proof or ablation.

After each stage, an **ABX exchange** occurs before proceeding to the next
stage. Intermediates from stages 1 and 2 are captured for deep supervision.

### 2.3 SGDCNv4Block — Spectral-Guided Deformable Convolution

`SGDCNv4Block` is the current code name, but the paper wording should be
"spectral-guided deformable-convolution block" because the implementation uses
`torchvision.ops.deform_conv2d`, not a verified DCNv4 operator. Each block
contains two residual connections:

```
Input x
  │
  ├──── Residual 1 (DCN path): ──────────────────────────────────────┐
  │   x_norm = GroupNorm(x)                                          │
  │   x_spec = SpectralGuidanceBlock(GroupNorm(x))                   │
  │                                                                  │
  │   [x_norm, x_spec] ──flatten──→ Linear → offset + mask          │
  │                                                                  │
  │   x_dcn = deform_conv2d(x_norm, offset, weight,                 │
  │                         padding, dilation, mask)                 │
  │   x = x + GroupNorm(Conv1×1(x_dcn))  ◄───────────────────────────┘
  │
  ├──── Residual 2 (FFN path): ──────────────────────────────────────┐
  │   ffn = GroupNorm → Conv1×1(C→4C) → GELU → Conv1×1(4C→C)       │
  │   x = x + ffn(x)  ◄─────────────────────────────────────────────┘
  │
Output x
```

**SpectralGuidanceBlock** operates in the frequency domain:

```
rfft2(x, norm='ortho')
  → Conv2d(C, C, 1) on real part
  → Conv2d(C, C, 1) on imaginary part
  → irfft2(complex(real, imag), s=(H,W), norm='ortho')
  → GroupNorm → GELU
```

Key design decisions:
- Uses `Conv2d(1x1)` (not `nn.Linear`) in frequency domain
- Always passes `s=(H, W)` to `irfft2` to prevent spatial size mismatches
- Disables AMP (`autocast(enabled=False)`) for numerical stability in FFT
- Offset/mask initialized to zero (identity deformation at init)
- `num_groups=4` for deformable conv, `expansion=4` for FFN

### 2.4 CrossScanGatedMixer (Mamba-Inspired Approximation)

Approximates directional sequence mixing with linear layers, gates, and
depthwise `Conv1d`; it is not a true selective state-space Mamba kernel.

| `num_passes` | Scan Pattern | Description |
|-------------|--------------|-------------|
| 1 | H→ | Forward H-scan only |
| 2 | H↕ | Bidirectional H-scan |
| 4 | H↕ + W↔ | Full cross-scan (both axes, bidirectional) |

```
scan(x) → h
gate = sigmoid(Linear_gate(x))
output = GELU(GroupNorm(Linear_out(h * gate)))
```

The scan uses depthwise 1D convolution (`groups=dim`) as the recurrence
approximation, making it pure PyTorch with no custom CUDA kernels.

---

## 3. Branch B — Global Context Encoder (v2, FFT-based)

**Goal:** Capture cross-slice 3D semantic context at low resolution with
**full spatial receptive field**.

**Input:** All 5 slices `x ∈ (B, 5, 224, 224)`.

### 3.1 Progressive Downsampling (NEW in v3.1)

```
Conv2d(5, C//2, 3, stride=2, padding=1) → GroupNorm → GELU    (224² → 112²)
Conv2d(C//2, C, 3, stride=2, padding=1) → GroupNorm → GELU    (112² → 56²)
```

**v3 used 1-step `Conv(stride=4)`** which lost too much spatial detail.
The 2-step progressive approach retains intermediate structural information
and provides an extra nonlinearity for richer feature extraction.

### 3.2 Global2DFFTMixer Blocks (NEW in v3.1)

3 × `Global2DFFTMixer(dim=C)` with residual connections, replacing the
old `CrossScanGatedMixer` that had only a local receptive field (kernel=3).

**Global2DFFTMixer** architecture:

```
Input x (B, C, H, W)
  │
  ├── FFT path:
  │   rfft2(x, norm='ortho')
  │   → Conv2d(C,C,1) on real part       (learned frequency filter)
  │   → Conv2d(C,C,1) on imaginary part
  │   → irfft2(complex(real, imag), s=(H,W), norm='ortho')
  │   → y (global features)
  │
  ├── Gating path:
  │   g = sigmoid(Linear(x))             (spatial gating)
  │
  ├── Combine:
  │   out = Linear(y × g)
  │   → GroupNorm → GELU
  │
Output (B, C, H, W)
```

**Why FFT over Conv1d scan approximation?**
- The old `CrossScanGatedMixer` used `Conv1d(kernel_size=3)` — effectively
  a 3-pixel sliding window with no true long-range dependency
- `Global2DFFTMixer` operates in the 2D frequency domain, so every output
  pixel depends on **every** input pixel — truly global receptive field
- Complexity: O(N log N) via FFT, vs O(N×k) for Conv1d
- At 56×56, the FFT is extremely efficient (~3136 pixels)

### 3.3 Stage-wise Execution

Each `Global2DFFTMixer` block corresponds to one PrecisionStem stage:

| Context Stage | Matching Precision Stage | ABX Exchange |
|--------------|-------------------------|--------------|
| FFTMixer #1 | Stage 1 (2 deformable blocks) | ABX #1 |
| FFTMixer #2 | Stage 2 (4 deformable blocks) | ABX #2 |
| FFTMixer #3 | Stage 3 (6 deformable blocks) | ABX #3 |

**Output:** `feat_z_ctx ∈ (B, C, 56, 56)`

---

## 4. Asymmetric Bidirectional Exchange (ABX) — NEW in v3.1

**Goal:** Enable inter-branch information flow after each stage to
prevent gradient bottleneck and enrich both branches with complementary
information.

**Applied 3 times** — once after each matched stage pair.

### 4.1 Direction 1: Context → Precision (Channel Reweighting)

```
scale = GAP(ctx_feat) → Linear(C→C/4) → GELU → Linear(C/4→C) → Tanh
prec_out = prec_feat × (1 + scale)
```

SE-style channel-wise scaling: the context branch tells the precision
branch **which channels are important** based on global semantics.

**Why channel-only (no spatial)?** Injecting spatial information from
the low-resolution context would blur precision features — the same
problem FrequencySplitFusion is designed to solve. Channel reweighting
provides semantic guidance without spatial contamination.

### 4.2 Direction 2: Precision → Context (Spatial Residual)

```
prec_down = AvgPool(prec_feat, factor=4)      (224² → 56²)
delta = prec_down − ctx_feat                  (difference signal)
gate = sigmoid(Conv1×1(ctx_feat))
ctx_out = ctx_feat + gate × Conv3×3(delta)
```

The **difference signal** `(prec_down - ctx_feat)` forces the context
branch to learn what the precision branch knows that it doesn't. This
is more informative than simply adding the precision features, because:
- If both branches agree → delta ≈ 0 → no change
- If precision has boundary info that context lacks → delta highlights
  exactly what's missing

The gating mechanism (`sigmoid(Conv1×1)`) lets the context branch
selectively accept or reject the injected information.

### 4.3 ABX vs Traditional Skip Connections

| Property | Skip Connection | ABX |
|----------|----------------|-----|
| Direction | One-way (encoder→decoder) | Bidirectional |
| Timing | Between encoder and decoder | Within encoder (per-stage) |
| Mechanism | Concatenation or addition | Channel scale + spatial residual |
| Spatial blur risk | High | Low (channel-only for ctx→prec) |

---

## 5. Branch C — Frequency-Split Fusion

**Goal:** Inject global context into precision features without corrupting
high-frequency boundary details.

This is the critical fusion mechanism that prevents the "patch overlay blur"
problem common in naive additive fusion.

### 5.1 CascadedPixelShuffle (56² → 224²)

Two-step learned upsampling with anti-checkerboard smoothing:

```
Step 1: Conv1×1(C→4C) → PixelShuffle(2) → DWConv3×3 → GN → GELU
        (56² → 112²)

Step 2: Conv1×1(C→4C) → PixelShuffle(2) → DWConv3×3 → GN → GELU
        (112² → 224²)
```

The depthwise convolutions (`groups=C`) after each PixelShuffle are intended as
anti-checkerboard smoothing filters; their effect should be verified in
ablation.

**Output:** `ctx_up ∈ (B, C, 224, 224)`

### 5.2 SDF Gate

The SDF Head predicts per-class Signed Distance Fields from `feat_center`:

```
SDF Head: Conv2d(C, 3, 1) → Tanh   →   sdf_pred ∈ (B, 3, H, W), range [-1, 1]
```

The SDF Gate converts SDF predictions into a spatial attention map:

```
gate = sigmoid(Conv3×3 → GN → GELU → Conv1×1)(detach(sdf_pred))
```

`detach()` prevents gradients from the gate flowing back into the SDF head,
ensuring the SDF head is trained only by the SDF MSE loss.

**Output:** `gate ∈ (B, 1, H, W)` — high near boundaries, low in flat regions.

### 5.3 Frequency-Split Fusion

The core fusion algorithm that preserves high-frequency edges:

```
Step 1: FFT Decomposition
    X_f = rfft2(feat_center, norm='ortho')
    LP  = Gaussian radial low-pass mask (σ = cutoff_ratio × min(H, W_freq×2))
    feat_low  = irfft2(X_f × LP, s=(H,W), norm='ortho')
    feat_high = feat_center − feat_low

Step 2: Context Injection (low-frequency only)
    α = sigmoid(Conv1×1([feat_low, ctx_up]))      # learnable channel-wise scale
    fused_low = feat_low + gate × α × ctx_up

Step 3: Reconstruction
    feat_fused = fused_low + feat_high             # high-freq edges untouched
```

**Why Gaussian (not Sigmoid) low-pass?** A sigmoid cutoff in frequency domain
can introduce a sharper transition than a Gaussian roll-off. The Gaussian mask
is a conservative smoothing choice, but the cutoff and artifact behavior still
require ablation.

**Default `cutoff_ratio=0.25`** — the low-pass mask covers the central 25%
of the frequency spectrum.

---

## 6. Branch D — Dual Heads + Auxiliary Context Head

### 6.1 SDF Head

```
Conv2d(C, 3, 1) → Tanh
```

- Input: `feat_center` (not `feat_fused`) — pure boundary features
- Output: `(B, 3, H, W)` — per-class SDF for RV, MYO, LV
- Range: \[-1, 1\] (positive inside, negative outside)
- Trained with: MSE loss against ground-truth SDF computed via EDT

### 6.2 Segmentation Head

```
Conv2d(C, C, 3, padding=1) → GroupNorm(C) → GELU → Conv2d(C, 4, 1)
```

- Input: `feat_fused` — boundary-precise + context-enriched features
- Output: `(B, 4, H, W)` — logits for BG, RV, MYO, LV
- Trained with: CompoundHDLoss

### 6.3 Auxiliary Context Segmentation Head (NEW in v3.1)

```
Conv2d(C, 4, 1) → Interpolate(224²)
```

- Input: `feat_z_ctx` at 56² resolution — upsampled via bilinear interpolation
- Output: `ctx_logits (B, 4, H, W)`
- Trained with: `0.3 × CrossEntropy(ctx_logits, masks)`
- **Only active during training** — zero inference overhead

**Why?** In v3, the context branch only received gradient through the
fusion path, which is heavily gated (SDF gate × frequency mask × alpha).
This created a **gradient bottleneck** — the context encoder struggled to
learn because gradients were attenuated by multiple multiplicative gates.
The auxiliary head provides a direct, unobstructed gradient path.

### 6.4 Deep Supervision (optional)

When `deep_supervision=True` and `model.training`:

```
aux_outputs = [
    Conv1×1(C→4)(stage1_features),   # after ABX exchange #1
    Conv1×1(C→4)(stage2_features),   # after ABX exchange #2
]
```

Auxiliary heads receive intermediate features from PrecisionStem stages 1
and 2 (post-ABX), providing gradient signal to earlier layers.

---

## 7. Loss Function — CompoundHDLoss

A composite loss with epoch-dependent warmup scheduling:

```
L = L_dice + 0.5·L_ce + 0.5·L_focal + w_hd·L_hd + w_sdf·L_sdf + 0.3·L_ctx
```

| Term | Description | Weight | Warmup |
|------|-------------|--------|--------|
| **L_dice** | Weighted soft Dice loss (per-class) | 1.0 | None |
| **L_ce** | Weighted Cross-Entropy | 0.5 | None |
| **L_focal** | Focal Loss (γ=2.0, class-weighted) | 0.5 | None |
| **L_hd** | Karimi-style Hausdorff Distance loss | 0→1.0 | Starts epoch 10, full by epoch 20 |
| **L_sdf** | SDF MSE (pred vs GT distance transform) | 0→0.5 | Linear ramp over 20 epochs |
| **L_ctx** | Context branch CrossEntropy | 0.3 | None |

**Default class weights:** BG=0.1, RV=2.5, MYO=1.5, LV=1.0

**Early stopping:** Based on **best HD95** (lower is better). Dice and
Balanced checkpoints are saved independently but do not affect patience.

### 7.1 Karimi HD Loss

For each foreground class and each sample:

```
dt = EDT(1 - gt) + EDT(gt)           # distance transform of boundary
L_hd = mean((pred_c - gt_c)² × dt)   # weight errors by distance to boundary
```

### 7.2 SDF Ground Truth Generation

At training time, SDF ground truth is computed on-the-fly per batch:

```python
for each class c ∈ {RV, MYO, LV}:
    fg = (mask == c)
    pos = EDT(fg)           # distance inside
    neg = EDT(1 - fg)       # distance outside
    sdf = (pos - neg) / max(|pos|, |neg|)   # normalize to [-1, 1]
```

---

## 8. Data Pipeline

### 8.1 ACDCDataset25D

**Input format:** Preprocessed `.npy` volumes and masks (224×224, z-score
normalized).

For each center slice `k`:
```
channels = [vol[k-2], vol[k-1], vol[k], vol[k+1], vol[k+2]]
```
Boundary slices use mirror padding: `index = reflect(i, 0, n_slices)`.

**Augmentation (training):**
- Random horizontal flip (p=0.5) — applied consistently across all 5 slices
- Random vertical flip (p=0.5)
- Random rotation (0°, 90°, 180°, 270°)

### 8.2 3D Evaluation

Predictions are assembled back into 3D volumes for evaluation:
- **Dice Similarity Coefficient** per class
- **HD95** (95th percentile Hausdorff Distance) per class
- **Precision, Recall, Accuracy, F1** per class
- Supports both **pixel** and **mm** units for HD95

---

## 9. Training Configuration

Paper-facing experiments should be launched from a config file so the split,
hyperparameters, output directory, and seed are captured with the run:

```bash
python src/training/train_acdc.py --config configs/acdc_asym_v31.yaml
```

The wrapper script uses the same config:

```bash
bash scripts/train_acdc_reproducible.sh
```

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-5 |
| LR Schedule | Warmup (10 epochs) -> CosineAnnealing -> ReduceLROnPlateau |
| Batch Size | 4 |
| Image Size | 224 x 224 |
| Mixed Precision | AMP with GradScaler |
| Gradient Clipping | max_norm = 1.0 |
| Early Stopping | patience = 30 epochs |
| Epochs | 250 |
| Input Channels | 5 (2.5D stack) |
| Base Channels | 48 |
| Num Classes | 4 (BG, RV, MYO, LV) |
| Split | patient-level manifest at `splits/acdc_patient_split_seed42.json` |

The local default data path is `preprocessed_data/ACDC`. The previous
`preprocessed_data/ACDC/training` command is not the current workspace path.

---

## 10. Results Status

Historical numbers from prior runs should be treated as **not paper-ready**
until rerun with:

- patient-level train/validation/test or cross-validation split,
- saved split manifest,
- config/resolved args archived with the run,
- strict evaluation checkpoint loading,
- explicit HD95 unit (`pixel` or `mm`),
- baseline and ablation tables.

Do not compare pixel HD95 directly against ACDC leaderboard or paper results
reported in millimeters.

---

## 11. Design Rationale & Required Ablations

### Why Global2DFFTMixer over CrossScanGatedMixer?

The v3 context branch used `CrossScanGatedMixer` with `Conv1d(kernel_size=3)`.
Analysis revealed this was essentially a 3-pixel sliding average — providing
almost no long-range context at 56×56 resolution. The entire purpose of the
context branch is global understanding, which a kernel-3 convolution cannot
provide.

`Global2DFFTMixer` operates in the 2D frequency domain where every output
pixel can depend on global frequency content. At 56x56 resolution, the FFT
overhead is small enough to test as a practical low-resolution context mixer.

### Why ABX (not just end-to-end fusion)?

In v3, the two branches ran independently until the final fusion stage.
This caused:
1. **Gradient bottleneck** — context branch gradients were attenuated by
   SDF gate × frequency mask × alpha (triple multiplicative gating)
2. **Redundant computation** — both branches independently learned
   overlapping representations without sharing information

ABX is intended to address both:
- **ctx→prec** (channel scale): guides which DCN features matter most
- **prec→ctx** (spatial residual): forces context to learn complementary
  info via the difference signal `(prec_down - ctx_feat)`
- **Auxiliary context head**: direct gradient path bypassing all gates

### Why progressive downsampling?

v3 used `Conv2d(5, C, kernel_size=4, stride=4)` — a single step from
224² to 56². This aggressive compression lost intermediate spatial
structure. The 2-step approach (224→112→56) with intermediate GELU
nonlinearity preserves more information and provides better gradient flow.

### Why asymmetric 2.5D routing?

The v2 architecture ran all 5 slices through full-resolution DCN, causing OOM
on 16GB GPUs. Since only the center slice needs a segmentation mask, the
neighboring slices provide context — they don't need pixel-perfect processing.
Routing 3 slices to DCN@224² and 5 slices to FFT@56² reduces DCN FLOPs by
**~40%** while retaining full Z-axis context.

### Why frequency-split fusion (not additive)?

v2 used direct additive fusion (`feat = feat_local + upsample(feat_context)`),
which may blur high-frequency boundary details. The frequency-split approach
is designed to preserve edge features by only modifying the low-frequency
(semantic) component. Its HD95 value must be validated by ablation.

### Why Gaussian LP filter (not learnable sigmoid)?

A sigmoid cutoff in frequency domain can create a sharper transition than a
Gaussian roll-off. The `cutoff_ratio` hyperparameter controls the boundary
between "low" and "high" frequency and should be ablated.

### Why SDF detach in gate?

If gradients flow from the gate through the SDF prediction, the SDF head
would be trained to produce "good gates" rather than "accurate SDFs". The
`detach()` ensures the SDF head is trained purely by the SDF MSE loss,
while the gate learns to leverage whatever SDF the head produces.

Required ablations before paper claims:

- 1-slice, 3-slice, and 5-slice input variants.
- Symmetric full-resolution routing vs asymmetric 3/5-slice routing.
- Deformable convolution vs plain/dilated/depthwise convolution.
- Spectral guidance on/off in the deformable block.
- `Global2DFFTMixer` vs Conv1d scan vs attention or true Mamba/SS2D baseline.
- ABX off, context-to-precision only, precision-to-context only.
- Additive fusion vs frequency-split fusion.
- SDF gate on/off and SDF detach on/off.
- CompoundHDLoss vs Dice+CE+Focal.
- Deep supervision and context-head supervision on/off.

---

## 12. File Structure

```
src/
├── models/
│   └── specmamba_net.py          # Both SpecMambaNet (v1) and AsymSpecMambaDCN (v3.1)
│       ├── SpectralGuidanceBlock    # FFT-based spectral feature extraction
│       ├── SGDCNv4Block             # Spectral-guided deformable-conv block
│       ├── CrossScanGatedMixer      # Mamba-inspired scan approximation
│       ├── PrecisionStem            # Branch A: 12 deformable blocks at full resolution
│       ├── Global2DFFTMixer         # Learned 2D spectral filter (NEW v3.1)
│       ├── AsymBidirectionalExchange# Per-stage cross-branch exchange (NEW v3.1)
│       ├── GlobalContextEncoder     # Branch B: progressive ↓ + 3× FFTMixer
│       ├── CascadedPixelShuffle     # Branch C-1: 56² → 224² upsampling
│       ├── SDFGate                  # Branch C-2: SDF-conditioned spatial gate
│       ├── FrequencySplitFusion     # Branch C-3: FFT low/high split + fusion
│       └── AsymSpecMambaDCN         # Main model (interleaved stages + ABX)
├── training/
│   └── train_acdc.py             # Training script
│       ├── ACDCDataset25D           # 2.5D data loader with augmentation
│       ├── compute_sdf_from_mask    # Per-class SDF ground truth
│       ├── CompoundHDLoss           # Composite loss with warmup
│       └── evaluate_3d              # Volumetric 3D evaluation
├── losses/
│   └── sota_loss.py              # CombinedSOTALoss (for non-hybrid models)
└── data/
    └── acdc_dataset.py           # Base 2D dataset classes
```

---

## 13. Related Work and Baseline Requirements

This section is a checklist, not a SOTA claim. The paper still needs a sourced
related-work table and reproduced comparisons under the same split and metrics.

Minimum comparison set:

- 2D U-Net and 2.5D U-Net/UNet++.
- nnU-Net 2D/3D or cascade configuration.
- HRNet-style segmentation baseline.
- Transformer baseline such as TransUNet/Swin-Unet/SwinUNETR where feasible.
- Medical Mamba/SSM baseline where feasible.
- Internal legacy `SpecMambaNet` and `HRNetDCN` baselines if retained.

Protocol requirements:

- per-class RV/MYO/LV Dice and HD95,
- ED/ES reporting where applicable,
- HD95 unit in mm when spacing metadata is valid,
- parameter count, FLOPs or MACs, latency, and memory,
- mean +/- std over folds or seeds.

---

## 14. Upgrade Roadmap — Evidence-Based Prioritization

### Tier 1: Loss Function (Candidate Experiments)

| # | Upgrade | Expected Impact |
|---|---------|-----------------|
| 1 | Boundary-weighted SDF loss | Test HD95 sensitivity |
| 2 | Regional or erosion-based HD-style loss | Test HD95 sensitivity |
| 3 | **Curvature-Aware Loss** — 2nd-order smoothness | Smoother boundaries |

### Tier 2: Optimizer & Augmentation (Candidate Experiments)

| # | Upgrade | Expected Impact |
|---|---------|-----------------|
| 4 | Sharpness-aware optimizer | Compare stability |
| 5 | CutMix / mix-style augmentation | Compare Dice and HD95 |
| 6 | Elastic deformation across 2.5D stack | Compare robustness |

### Tier 3: Architecture (Candidate Experiments)

| # | Upgrade | Expected Impact |
|---|---------|-----------------|
| 7 | Multi-frequency gated fusion | Compare cutoff/frequency handling |
| 8 | Boundary attention before deformable conv | Compare RV boundary precision |
| 9 | **Topology Loss** (PI-Att) | Correct topology |

### Tier 4: Post-Processing (Candidate Experiments)

| # | Upgrade | Expected Impact |
|---|---------|-----------------|
| 10 | Connected-component filtering | Compare false positives |
| 11 | TTA — flip/rotate ensemble | Compare inference-time tradeoff |

---

## 15. Paper Positioning — Unique Contributions

1. **Spectral-Guided Deformable Conv:** FFT-based offset guidance implemented
   with `torchvision.ops.deform_conv2d`.
2. **Global2DFFTMixer for Context:** Full-spatial-RF frequency-domain context
   encoder replacing the limited-RF Conv1d scan approximation.
3. **Asymmetric Bidirectional Exchange (ABX):** Per-stage cross-branch
   information flow with asymmetric mechanisms (channel scale vs spatial
   residual) designed to preserve boundary precision
4. **Frequency-Split Fusion with SDF Gate:** Combination of FFT decomposition
   and SDF-conditioned gating for controlled context injection
5. **Asymmetric 2.5D Routing:** Explicitly decouples boundary precision from
   context capture with different compute budgets and receptive field strategies
6. **Lightweight model target:** parameter efficiency should be reported against
   baselines once parameter/FLOP/latency measurements are reproduced.
