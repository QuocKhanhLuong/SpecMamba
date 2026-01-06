# EGM-Net: Energy-Gated Gabor Mamba Network

Medical image segmentation using implicit neural representations with physics-based energy gating.

## Architecture

- **Backbone**: HRNetV2-Mamba (dual-stream with Mamba + Spectral FFT)
- **Coarse Head**: RBF Constellation classifier (2D I/Q embedding)
- **Fine Head**: Energy-gated Gabor implicit decoder (resolution-free)
- **Fusion**: Physics-based energy gating

## Quick Start

```bash
# Install
conda env create -f environment.yaml
conda activate egm-net

# Train
python src/training/train_egm.py

# Evaluate
python src/scripts/evaluate.py --checkpoint model.pt --data_dir data/ --output_dir results/
```

## Project Structure

```
SpecUMamba/
├── src/
│   ├── models/
│   │   ├── egm_net.py           # Main model
│   │   ├── hrnet_mamba.py       # HRNetV2-Mamba backbone
│   │   └── mamba_block.py       # VSS blocks
│   ├── layers/
│   │   ├── constellation_head.py # RBF classifier
│   │   ├── gabor_implicit.py    # Implicit decoder
│   │   └── spectral_layers.py   # FFT gating
│   ├── losses/
│   │   └── physics_loss.py      # Dice, Eye-Opening, Combined loss
│   ├── utils/
│   │   ├── metrics.py           # Dice, IoU, HD95, F1
│   │   └── visualize.py         # Plotting utilities
│   ├── training/
│   │   └── train_egm.py         # Training script
│   └── scripts/
│       └── evaluate.py          # Evaluation script
├── config.yaml                  # Model configuration
├── environment.yaml             # Conda environment
└── notebooks/
    └── EGM_Net_Demo.ipynb       # Standalone demo
```

## Configuration

Edit `config.yaml` to switch between presets:

| Preset   | Backbone     | Coarse Head   | Fine Head | Params |
|----------|--------------|---------------|-----------|--------|
| baseline | Conv only    | Linear        | Disabled  | ~3M    |
| lite     | Conv only    | Constellation | Disabled  | ~4M    |
| sota     | Mamba+FFT    | Constellation | Enabled   | ~7.5M  |

## Metrics

- Dice Score (per-class, mean)
- IoU / Jaccard
- HD95 (Hausdorff Distance 95%)
- Precision, Recall, F1

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- MONAI >= 1.2
- mamba-ssm >= 1.0

## License

MIT
