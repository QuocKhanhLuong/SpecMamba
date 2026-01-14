# HDC-HRNet: Hierarchical Deformable Convolution High-Resolution Network

> **Medical Image Segmentation with Multi-Scale Deformable Convolution and Hybrid Dilation Pyramid**

A state-of-the-art medical image segmentation architecture combining HRNet's high-resolution multi-scale representations with Deformable Convolution Networks (DCNv3) and Hybrid Dilated Convolution (HDC) strategy.

---

## 🏗️ Architecture Overview

![HDC-HRNet Architecture](assets/architecture.png)

<details>
<summary>📐 ASCII Diagram (Click to expand)</summary>

```
Input Image (H×W×C)
        │
        ▼
┌───────────────────┐
│   HRNet Stem      │  ← Full Resolution Mode (stride=1) or Standard (stride=4)
│   (Conv Layers)   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Layer 1         │  ← 4× Bottleneck Blocks (256 channels)
│   (Bottleneck)    │
└───────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              Multi-Resolution Branches                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Stream1 │  │ Stream2 │  │ Stream3 │  │ Stream4 │    │
│  │  1×     │  │  1/2×   │  │  1/4×   │  │  1/8×   │    │
│  │  64ch   │  │  128ch  │  │  256ch  │  │  512ch  │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│  ┌────▼────────────▼────────────▼────────────▼────┐    │
│  │         Multi-Scale Fusion (FuseLayer)          │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
        │
        ▼ (Repeat for Stage 2, 3, 4)
        
┌─────────────────────────────────────────────────────────┐
│  DCN Blocks with Hybrid Dilation Pyramid (HDC)          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ d=1 │→│ d=2 │→│ d=4 │→│ d=8 │→│d=16 │→│d=32 │       │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
│  Asymmetric Depth: Stage2(2) → Stage3(4) → Stage4(6)   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│   Feature Fusion  │  ← Concatenate all streams (960ch for base_ch=64)
│   + Seg Head      │
└───────────────────┘
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Main Output  │  │  PointRend    │  │   Shearlet    │
│  (1×1 Conv)   │  │  (Optional)   │  │   (Optional)  │
└───────────────┘  └───────────────┘  └───────────────┘
```

</details>

---

## 🔑 Key Components

### 1. **Deformable Convolution v3 (DCN) with Dilation Pyramid**
- **Learnable offsets & modulation** for adaptive receptive fields
- **Hybrid Dilated Convolution (HDC)**: Dilation rates \`[1, 2, 4, 8, 16, 32]\` to avoid gridding artifacts
- **Projection shortcuts** for channel dimension changes

### 2. **HRNet Multi-Resolution Backbone**
- **Parallel high-to-low resolution streams** maintained throughout
- **Repeated multi-scale fusion** via \`FuseLayer\`
- **Full resolution mode** option (stride=1 stem for maximum detail)

### 3. **Asymmetric Stage Depth**
| Stage   | DCN Blocks | Dilation Rates |
|---------|------------|----------------|
| Stage 2 | 2          | 1, 2           |
| Stage 3 | 4          | 1, 2, 4, 8     |
| Stage 4 | 6          | 1, 2, 4, 8, 16, 32 |

### 4. **Optional Refinement Heads**
- **PointRend**: Uncertainty-based boundary refinement (samples 2048 most uncertain points)
- **Shearlet Implicit Head**: Multi-orientation wavelet-based boundary enhancement

### 5. **Deep Supervision** (Optional)
- Auxiliary losses from each resolution stream for better gradient flow

---

## 📦 Modular Block Library

| Block Type         | Description                              | Use Case                      |
|--------------------|------------------------------------------|-------------------------------|
| \`basic\`            | Standard ResNet residual block           | Baseline comparison           |
| \`convnext\`         | ConvNeXt block (SOTA CNN, 2022)          | Strong CNN baseline           |
| \`dcn\`              | Deformable Conv v3 + Dilation Pyramid    | **Default for HDC-HRNet**     |
| \`inverted_residual\`| MobileNetV2 inverted bottleneck          | Lightweight models            |
| \`swin\`             | Swin Transformer with shifted windows    | Vision Transformer hybrid     |
| \`fno\`              | Fourier Neural Operator block            | Global frequency processing   |
| \`wavelet\`          | Haar Wavelet transform block             | Multi-resolution analysis     |
| \`rwkv\`             | RWKV/AFT-style linear attention          | Efficient sequence modeling   |

---

## 🚀 Quick Start

\`\`\`bash
# 1. Create Environment
conda env create -f environment.yaml
conda activate hdc-hrnet

# 2. Preprocess Data (ACDC example)
python scripts/preprocess_acdc.py --data_dir data/ACDC --output_dir preprocessed_data/ACDC

# 3. Train
python src/training/train_acdc.py

# 4. Evaluate
python src/evaluate.py --checkpoint results/best_model.pt --data_dir preprocessed_data/ACDC
\`\`\`

---

## 📁 Project Structure

\`\`\`
HDC-HRNet/
├── src/
│   ├── models/
│   │   ├── hrnet_dcn.py         # 🔥 Main HDC-HRNet Model
│   │   └── blocks.py            # Modular building blocks (DCN, ConvNeXt, Swin, FNO, etc.)
│   ├── layers/
│   │   ├── pointrend.py         # PointRend boundary refinement
│   │   ├── shearlet_implicit.py # Shearlet-based implicit head
│   │   ├── constellation_head.py# RBF constellation classifier
│   │   ├── gabor_implicit.py    # Gabor implicit decoder
│   │   └── spectral_layers.py   # Spectral/FFT layers
│   ├── losses/
│   │   ├── physics_loss.py      # Dice, Focal, Frequency losses
│   │   └── sota_loss.py         # Additional loss functions
│   ├── data/
│   │   ├── acdc_dataset.py      # ACDC cardiac dataset
│   │   ├── brats_dataset.py     # BraTS brain tumor dataset
│   │   ├── mnm_dataset.py       # M&M cardiac dataset
│   │   └── synapse_dataset.py   # Synapse multi-organ dataset
│   ├── training/
│   │   ├── train_acdc.py        # ACDC training script
│   │   ├── train_brats.py       # BraTS training script
│   │   └── train_synapse.py     # Synapse training script
│   ├── utils/
│   │   ├── metrics.py           # Dice, IoU, HD95, F1
│   │   └── visualize.py         # Visualization utilities
│   └── evaluate.py              # Evaluation script
├── scripts/
│   ├── preprocess_acdc.py       # ACDC preprocessing
│   ├── preprocess_brats.py      # BraTS preprocessing
│   └── preprocess_synapse.py    # Synapse preprocessing
├── experiments/
│   ├── ablation/                # Ablation studies
│   ├── baselines/               # Baseline comparisons
│   └── comparison/              # Model comparisons
├── notebooks/
│   └── EGM_Net_Demo.ipynb       # Interactive demo
├── config.yaml                  # Model configuration
├── environment.yaml             # Conda environment
└── requirements.txt             # Pip requirements
\`\`\`

---

## ⚙️ Model Configurations

\`\`\`python
# Small (~10M params)
model = hrnet_dcn_small(num_classes=4, in_channels=1)

# Base (~25M params) - Recommended
model = hrnet_dcn_base(num_classes=4, in_channels=1, use_pointrend=True)

# Large (~40M params)
model = hrnet_dcn_large(num_classes=4, in_channels=1, use_pointrend=True)

# Custom Configuration
model = HRNetDCN(
    in_channels=1,
    num_classes=4,
    base_channels=48,
    img_size=224,
    stage_configs=[
        {'blocks': ['dcn'] * 2},  # Stage 2: 2 DCN blocks
        {'blocks': ['dcn'] * 4},  # Stage 3: 4 DCN blocks
        {'blocks': ['dcn'] * 6},  # Stage 4: 6 DCN blocks
    ],
    use_pointrend=True,           # Boundary refinement
    full_resolution_mode=False,   # Use True for max detail (high VRAM)
    deep_supervision=True,        # Auxiliary losses
    use_shearlet=False            # Shearlet implicit head
)
\`\`\`

---

## 📊 Supported Datasets

| Dataset  | Modality | Classes | Task                    |
|----------|----------|---------|-------------------------|
| **ACDC** | MRI      | 4       | Cardiac segmentation    |
| **BraTS**| MRI      | 4       | Brain tumor segmentation|
| **M&M**  | MRI      | 4       | Multi-center cardiac    |
| **Synapse**| CT     | 14      | Multi-organ segmentation|

---

## 📈 Metrics

- **Dice Score** (per-class and mean)
- **IoU / Jaccard Index**
- **HD95** (Hausdorff Distance 95th percentile)
- **Precision, Recall, F1-Score**

---

## 🔧 Requirements

- Python 3.10+
- PyTorch >= 2.0
- torchvision >= 0.15 (for \`deform_conv2d\`)
- MONAI >= 1.2 (optional, for medical imaging utilities)
- numpy, scipy, scikit-image

---

## 📚 Citation

\`\`\`bibtex
@article{hdc-hrnet2025,
  title={HDC-HRNet: Hierarchical Deformable Convolution High-Resolution Network for Medical Image Segmentation},
  author={Your Name},
  year={2025}
}
\`\`\`

---

## 📄 License

MIT License
