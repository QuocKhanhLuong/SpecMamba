# 🧠 RESEARCH BLUEPRINT: PCShear-HRNet
## Noise-Robust Frequency-Aware Boundary Representation for HD95-Optimized 3D Medical Image Segmentation

**Version**: 1.0 | **Date**: March 2026 | **Status**: Pre-experiment / Implementation Phase

---

## 1. PROBLEM STATEMENT

**Task**: 3D medical image segmentation (semantic, multi-class)
**Target metric**: HD95 (Hausdorff Distance 95th percentile) — đo độ lệch biên tệ nhất
**Core failure mode**: HD95 cao do outlier boundary points, đặc biệt tại vùng:
  - Biên có curvature cao (lồi/lõm phức tạp)
  - Nhiễu MRI/CT bị nhầm thành biên (noise artifact)
  - Z-axis boundary không liên tục (lỗi do train 2D, eval 3D)

**Câu hỏi nghiên cứu**:
> "Làm thế nào để xây dựng một biểu diễn đặc trưng biên (boundary feature representation)
> vừa NOISE-ROBUST vừa CURVATURE-AWARE, bằng cách kết hợp
> kiến thức từ miền tần số (frequency domain) với spatial intensity,
> để trực tiếp tối ưu HD95 trong 3D medical segmentation?"

---

## 2. THEORETICAL FOUNDATION

### 2.1 Tại sao High-Frequency ≠ Edge (vấn đề cần giải quyết)

```
High-frequency energy trong FFT
         │
         ├──── Biên thực sự (edge)   → CẦN GIỮ: anisotropic, phase-coherent
         └──── Noise (MRI Rician...) → CẦN BỎ: isotropic, random phase
```

**3 đặc tính phân biệt Edge vs Noise**:
| Đặc tính           | Edge (biên thực)                  | Noise                    |
|--------------------|-----------------------------------|--------------------------|
| Phase              | Đồng pha tại nhiều tần số (coherent)| Phase ngẫu nhiên        |
| Hướng energy       | Tập trung 1-2 hướng (anisotropic) | Phân tán đều (isotropic) |
| Spatial structure  | Liên tục dọc contour              | Độc lập giữa pixels      |

### 2.2 Giải pháp: Phase Congruency (Kovesi, 1999)

Biên thực sự xảy ra khi các thành phần Fourier ở nhiều tần số ĐỒNG PHA tại cùng một điểm:

  PC(x) = [Σ_n A_n * cos(φ_n(x) − φ̄(x))] / [Σ_n A_n + ε]

- PC → 1: tại biên thực sự (phases aligned)
- PC → 0: tại vùng noise (phases random, cancel out)
- **Bất biến với contrast** → phù hợp MRI đa modality, đa máy quét

Implement qua bank of Log-Gabor filters:

  G(f, θ) = exp(−[log(f/f₀)]² / 2σ_f²) × exp(−(θ−θ_k)² / 2σ_θ²)

Noise threshold được ước lượng tự động từ Rayleigh distribution của filter responses
→ KHÔNG cần hyperparameter noise threshold thủ công

### 2.3 Giải pháp: Directional Shearlet Energy (curvature proxy)

Shearlet transform phân tích ảnh theo 8 hướng × 4 scale:

  Sh_ψ f(a, s, t) = <f, ψ_ast>

- Vùng biên thẳng → energy tập trung 1 hướng vuông góc với biên
- Vùng biên cong (high curvature) → energy phân tán nhiều hướng
- Curvature proxy: entropy của directional energy distribution

  E_curv(x) = −Σ_θ p(θ,x) × log(p(θ,x))  [Shannon entropy of directions]

Vùng E_curv cao = biên cong phức tạp = nơi HD95 fail nhất
→ Dùng làm spatial weight cho boundary loss

### 2.4 Tại sao Input 2.5D thay vì 2D thuần

Train 2D, eval 3D → Z-axis boundary không được học → staircase artifact → HD95 cao

Giải pháp: Input = 2k+1 slices liền kề (k=1, tức 3 slices)
- FFT của 2.5D stack encode sự thay đổi boundary qua z-axis
- Nhẹ hơn 3D full volume nhưng có z-context
- Phase Congruency 2.5D detect biên cả in-plane lẫn inter-slice

---

## 3. ARCHITECTURE: PCShear-HRNet

### 3.1 Tổng quan Pipeline

```
Input: 2.5D stack [B, 3, H, W] (3 slices liền kề, bất kỳ modality)
          │
          ├────────────────────────────────────────────────────────────┐
          │                                                            │
   [Branch 1: Spatial]                              [Branch 2: Spectral Boundary]
   Spatial Intensity Features                        Noise-Robust Boundary Features
   Standard 2D→3D encoder                                     │
   HRNet-inspired multi-scale                        ┌─────────┴──────────┐
          │                                   [PC Module]        [Shearlet Module]
   F_spatial [B, C, H, W]              Phase Congruency Map  Directional Energy Map
                                          [B, 1, H, W]          [B, 8, H, W]
                                                   └─────────┬──────────┘
                                               Spectral features [B, C', H, W]
                                               (via lightweight CNN encoder)
          │                                             │
          └──────────────[Cross-Domain Fusion]──────────┘
                    Attention-gated feature fusion
                    [B, C_fused, H, W]
                              │
               ┌──────────────┴──────────────┐
        [Seg Head]                    [Boundary Weight Head]
        Mask output                   Curvature weight map
        [B, N_class, H, W]            [B, 1, H, W]
                                              │
                                    Guide curvature-weighted
                                    boundary loss
```

### 3.2 Component Breakdown

**HRNet Multi-Stream Encoder (Spatial Branch)**
- 3 streams: high-res (1/4), mid-res (1/8), low-res (1/16)
- Channel/Spatial decoupling trong mỗi block:
    - Channel mixing: Linear (1×1 conv) — nhẹ, không spatial coupling
    - Spatial mixing: FFT-based (high-res stream) hoặc depthwise conv (mid/low stream)
- Thay thế conv nặng bằng MetaFormer-style token mixing
- Cross-stream fusion exchange units giữ nguyên từ HRNet gốc

**Phase Congruency Module (PC Module)**
- Input: 2.5D slice stack [B, 3, H, W]
- Process:
    1. FFT2D của mỗi slice
    2. Apply bank of Log-Gabor filters (n_scale=5, n_orient=6)
    3. Tính phase coherence score tại mỗi pixel
    4. Noise threshold tự động qua Rayleigh estimation
- Output: PC map [B, 1, H, W] — 1.0 = biên thực, 0.0 = noise/flat
- Có thể freeze (non-learnable) hoặc fine-tune filter bank parameters
- Dùng thư viện: github.com/Simon-Bertrand/2DPhaseCongruency-PyTorch

**Shearlet Directional Energy Module**
- Input: 2.5D slice stack [B, 3, H, W]
- Process:
    1. Apply digital Shearlet transform (4 scales × 8 hướng = 32 subbands)
    2. Tính energy per orientation per pixel
    3. Normalize → probability distribution over directions
    4. Tính entropy → curvature proxy map
- Output: [B, 9, H, W] = 8 directional energy maps + 1 curvature entropy map
- Dùng thư viện: PyShearLab hoặc custom implementation
- Có thể freeze coefficients, chỉ học fusion weights

**Lightweight Spectral Encoder**
- Input: concatenate(PC_map, Shearlet_maps) [B, 10, H, W]
- Architecture: 3× (Conv 3×3 + BN + GELU) + residual
- Output: F_spectral [B, C', H, W], C' = C (match spatial branch channels)
- ~2M params — giữ lightweight để justify efficiency

**Cross-Domain Attention Fusion**
- Input: F_spatial [B, C, H/4, W/4], F_spectral [B, C, H/4, W/4]
- Process: Cross-attention (F_spatial as Query, F_spectral as Key/Value)
    Q = W_q × F_spatial
    K = W_k × F_spectral
    V = W_v × F_spectral
    F_fused = F_spatial + softmax(QK^T / √d) × V
- Lý do: Spatial features "hỏi" spectral features về boundary location
- Output: F_fused [B, C, H/4, W/4]
- Có thể simplify thành SE-block weighted addition nếu memory hạn chế

**Segmentation Head**
- Standard: Conv 1×1 → upsample → N_class logits

**Boundary Weight Head**
- Input: F_fused
- Output: spatial weight map w(x) [B, 1, H, W]
  w(x) = 1 + α × E_curv(x) + β × PC(x)
- α, β là learnable scalars
- Dùng để weight boundary loss

---

## 4. LOSS FUNCTION

### 4.1 Curvature-Weighted Boundary Loss (Đóng góp chính)

L_cwb = (1/|Ω_boundary|) × Σ_{x ∈ Ω} w(x) × |p(x) − t(x)| × D(t(x))

Trong đó:
- w(x) = curvature weight từ Shearlet entropy map
- D(t(x)) = distance transform của ground truth mask (SDF proxy)
- p(x) = predicted probability
- Ý nghĩa: penalize nặng hơn tại vùng biên cong phức tạp (nơi HD95 fail)

### 4.2 Phase Congruency Consistency Loss

L_pc = BCE(predicted_boundary_map, PC_map_from_GT)
- GT boundary map được tạo tự động từ Phase Congruency của GT mask
- Không cần manual boundary annotation

### 4.3 Combined Loss

L_total = λ₁×L_dice + λ₂×L_ce + λ₃×L_cwb + λ₄×L_pc

Warmup schedule:
- Epoch 0–10: chỉ L_dice + L_ce (ổn định model trước)
- Epoch 10+: thêm L_cwb với warmup_factor tăng dần
- Epoch 20+: thêm L_pc

Hyperparameters mặc định: λ₁=1.0, λ₂=1.0, λ₃=0.5, λ₄=0.3

---

## 5. DATASETS & EVALUATION

### 5.1 Datasets

| Dataset | Modality | Classes | Mục đích |
|---------|----------|---------|----------|
| ACDC | Cardiac MRI | 4 (BG, RV, MYO, LV) | Primary benchmark (2D→3D) |
| BraTS21 | Brain MRI 4-mod | 4 (BG, NCR, ED, ET) | Multi-modal validation |
| Synapse | CT Multi-organ | 14 organs | Generalizability test |
| M&M | Cardiac MRI multi-vendor | 4 | Cross-domain robustness |

### 5.2 Evaluation Metrics

| Metric | Đo cái gì | Priority |
|--------|-----------|----------|
| HD95 (mm) | 95th percentile boundary error | ⭐⭐⭐ PRIMARY |
| Dice Score (%) | Volumetric overlap | ⭐⭐ |
| NSD (%) | Normalized Surface Dice (BraTS std) | ⭐⭐ |
| ASD (mm) | Average Surface Distance | ⭐ |

### 5.3 Baseline Comparisons

| Method | Year | Lý do so sánh |
|--------|------|--------------|
| nnU-Net | 2021 | Gold standard tự cấu hình |
| SwinUNETR | 2022 | Best transformer baseline |
| TransUNet | 2021 | Classic CNN-ViT hybrid |
| UNETR++ | 2023 | Efficient transformer |
| U-Mamba | 2024 | Mamba-based SOTA |
| VM-UNet | 2024 | Vision Mamba |
| PFESA | MICCAI 2025 | Closest FFT-boundary paper |
| FFTMed | Nature 2025 | Closest FFT medical paper |

---

## 6. ABLATION STUDY PLAN

| Experiment | Component bỏ | Expected effect |
|------------|--------------|-----------------|
| A1 | Không có PC Module | HD95 tăng (false boundary từ noise) |
| A2 | Không có Shearlet Module | HD95 tăng ở curved regions |
| A3 | PC + Shearlet → Sobel thay thế | HD95 tăng (noise sensitivity) |
| A4 | Không có L_cwb | HD95 tăng, Dice stable |
| A5 | Không có L_pc | Boundary precision giảm |
| A6 | 2D input → 2.5D input | Z-axis HD95 giảm rõ |
| A7 | Cross-attention → concat | Fusion quality giảm |
| A8 | FFT spatial mixing → Conv | Efficiency giảm, quality similar |

---

## 7. IMPLEMENTATION ROADMAP

### Phase 1 — Proof of Concept (Tuần 1-2)
```python
# Bước 1: Thêm FFT magnitude làm channel thứ 2
def add_fft_channel(x):  # x: [B, 1, H, W]
    fft = torch.fft.fft2(x)
    magnitude = torch.abs(fft)
    magnitude = torch.fft.fftshift(magnitude)
    magnitude = torch.log1p(magnitude)  # log scale
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    return torch.cat([x, magnitude], dim=1)  # [B, 2, H, W]
# → Train baseline với input này, measure HD95 change
```

### Phase 2 — PC Module (Tuần 3-4)
```python
# Dùng thư viện sẵn có:
# pip install torch-phasecong  (Simon-Bertrand/2DPhaseCongruency-PyTorch)
from torch_phasecong import phasecong
# Integrate vào DataLoader như transform
# Output: PC_map [H, W] → additional input channel
```

### Phase 3 — Shearlet Module (Tuần 5-6)
```python
# Custom implementation hoặc dùng PyShearLab
# Tính 8-direction energy, compute entropy map
# Thêm vào spectral branch
```

### Phase 4 — Fusion + Loss (Tuần 7-8)
- Implement Cross-Domain Attention Fusion
- Implement L_cwb (Curvature-Weighted Boundary Loss)
- Implement warmup schedule

### Phase 5 — Experiments (Tuần 9-12)
- Full ablation study
- Baseline comparisons
- Hyperparameter search

### Phase 6 — Writing (Tuần 13-16)
- Target: MICCAI 2026 hoặc TMI/MedIA

---

## 8. CONTRIBUTION STATEMENT (Draft)

> We propose **PCShear-HRNet**, a noise-robust frequency-aware segmentation framework that introduces:
>
> 1. **Phase Congruency Boundary Module**: Detects anatomical boundaries using phase coherence
>    across frequency scales — inherently separating true edges from noise without manual thresholding,
>    contrast-invariant across MRI scanners and protocols.
>
> 2. **Directional Shearlet Energy Module**: Constructs a curvature-proxy map from anisotropic
>    frequency energy distribution — identifying high-curvature boundary regions where HD95 fails most.
>
> 3. **Curvature-Weighted Boundary Loss**: Dynamically weights boundary supervision proportional
>    to local curvature, directly targeting the 95th-percentile outlier points responsible for HD95.
>
> 4. **Cross-Domain Attention Fusion**: Enables spatial features to attend to spectral boundary cues,
>    combining pixel-level intensity information with noise-robust frequency-domain boundary knowledge.
>
> Evaluated on ACDC, BraTS21, Synapse and M&M, our method achieves state-of-the-art HD95
> while maintaining competitive Dice scores.

---

## 9. KEY DIFFERENTIATORS vs EXISTING WORK

| Paper | Họ làm | Chúng ta khác |
|-------|--------|---------------|
| PFESA (MICCAI 2025) | FFT decompose features (intermediate) | FFT của RAW INPUT, tách noise qua phase coherence |
| FFTMed (Nature 2025) | Mạng trong freq domain, drop high-freq | KHÔNG drop — tách edge khỏi noise qua phase |
| D²SFNet (2025) | Dual-domain fusion, concat-based | Cross-attention fusion + curvature-aware loss |
| BoundaryLoss (Kervadec) | SDF-based loss, no noise handling | SDF + curvature weighting từ Shearlet entropy |
| PointRend | Uncertainty sampling tại boundary | PC map guide sampling (noise-robust uncertainty) |

---

## 10. THEORETICAL CHAIN (Reviewer Defense)

FFT global receptive field
  → Phase Congruency: separate edge từ noise (proven: Kovesi 1999, signal theory)
    → Noise-robust boundary map
      → Spectral encoder học đặc trưng từ clean boundary signal
        → Cross-attention: spatial features attend to clean boundary cues
          → Better boundary localization (HD95 location accuracy ↑)

Shearlet directional analysis
  → Anisotropic energy distribution (proven: Shearlet theory, Labate et al.)
    → Curvature proxy = entropy of direction distribution
      → Curvature-weighted loss: học harder tại high-curvature regions
        → Reduce outlier boundary errors (HD95 95th percentile ↓)

Combined:
  → Noise-robust (PC) + Curvature-aware (Shearlet) boundary supervision
    → Both Dice (overall) và HD95 (outlier boundary) improve simultaneously
