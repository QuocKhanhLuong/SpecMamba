"""
Model Analysis Script for EGM-Net.

Provides comprehensive analysis of the Energy-Gated Gabor Mamba Network
including parameter profiling, component breakdown, and architecture
comparison with baseline models.

Analysis Features:
    - Per-module parameter breakdown
    - Full vs Lite model comparison
    - Forward pass verification
    - ASCII architecture diagram with innovation highlights
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'per_module': module_params
    }


def format_number(num):
    """Format large numbers for readability."""
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def analyze_egm_net():
    """Analyze EGM-Net architecture."""
    print("=" * 70)
    print("EGM-NET (Energy-Gated Gabor Mamba) - MODEL ANALYSIS")
    print("=" * 70)
    
    from egm_net import EGMNet, EGMNetLite
    
    # Full model
    print("\n[1] EGM-Net Full Configuration")
    print("-" * 70)
    
    model = EGMNet(
        in_channels=1,
        num_classes=3,
        img_size=256,
        base_channels=64,
        num_stages=4,
        encoder_depth=2,
        implicit_hidden=256,
        implicit_layers=4,
        num_frequencies=64
    )
    
    params = count_parameters(model)
    
    print(f"\nTotal Parameters:     {format_number(params['total'])} ({params['total']:,})")
    print(f"Trainable Parameters: {format_number(params['trainable'])} ({params['trainable']:,})")
    
    print(f"\nPer Module Breakdown:")
    for name, count in params['per_module'].items():
        pct = count / params['total'] * 100
        print(f"  {name:25s}: {format_number(count):>10s} ({pct:5.1f}%)")
    
    # Lite model
    print("\n[2] EGM-Net Lite Configuration")
    print("-" * 70)
    
    lite_model = EGMNetLite(in_channels=1, num_classes=3, img_size=256)
    lite_params = count_parameters(lite_model)
    
    print(f"Total Parameters:     {format_number(lite_params['total'])} ({lite_params['total']:,})")
    
    # Test forward pass
    print("\n[3] Forward Pass Test")
    print("-" * 70)
    
    x = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {outputs['output'].shape}")
    print(f"Coarse: {outputs['coarse'].shape}")
    print(f"Fine:   {outputs['fine'].shape}")
    print(f"Energy: {outputs['energy'].shape}")
    
    # Architecture summary
    print("\n[4] Architecture Summary")
    print("-" * 70)
    print("""
    EGM-Net Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         INPUT IMAGE                                  │
    │                        (B, 1, 256, 256)                             │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
    ┌───────────────────────┐       ┌───────────────────────┐
    │   MONOGENIC ENERGY    │       │    PATCH EMBEDDING    │
    │   (Physics-based)     │       │   Conv 4×4, stride 4  │
    │                       │       │   → (B, 64, 64, 64)   │
    │   Riesz Transform     │       └───────────┬───────────┘
    │   → Energy Map        │                   │
    │   (B, 1, 256, 256)    │                   ▼
    └───────────┬───────────┘       ┌───────────────────────┐
                │                   │    MAMBA ENCODER      │
                │                   │   4 Stages × 2 Blocks │
                │                   │   + Downsample        │
                │                   │                       │
                │                   │   64→128→256→512 ch   │
                │                   │   64→32→16→8 spatial  │
                │                   └───────────┬───────────┘
                │                               │
                │                               ▼
                │                   ┌───────────────────────┐
                │                   │      BOTTLENECK       │
                │                   │   Mamba (512ch, 8×8)  │
                │                   └───────────┬───────────┘
                │                               │
                │               ┌───────────────┴───────────────┐
                │               ▼                               ▼
                │   ┌───────────────────────┐   ┌───────────────────────┐
                │   │    COARSE BRANCH      │   │     FINE BRANCH       │
                │   │                       │   │                       │
                │   │   Conv Decoder        │   │   Gabor Implicit      │
                │   │   Upsample ×3         │   │   GaborBasis + SIREN  │
                │   │   512→256→128→64      │   │                       │
                │   │                       │   │   Resolution-Free     │
                │   │   Smooth but blurry   │   │   Sharp boundaries    │
                │   └───────────┬───────────┘   └───────────┬───────────┘
                │               │                           │
                │               ▼                           ▼
                │         (B, 3, 256, 256)           (B, 3, 256, 256)
                │               │                           │
                └───────────────┼───────────────────────────┤
                                ▼                           │
                    ┌───────────────────────────────────────┴───┐
                    │           ENERGY-GATED FUSION             │
                    │                                           │
                    │   Output = Coarse + Energy × (Fine-Coarse)│
                    │                                           │
                    │   Low Energy  → Use Coarse (flat regions) │
                    │   High Energy → Use Fine (boundaries)     │
                    └───────────────────────┬───────────────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │    FINAL OUTPUT       │
                                │   (B, 3, 256, 256)    │
                                │                       │
                                │   Sharp boundaries    │
                                │   No ringing artifacts│
                                └───────────────────────┘
    """)
    
    # Key innovations
    print("\n[5] Key Innovations")
    print("-" * 70)
    print("""
    1. MONOGENIC ENERGY GATING:
       - Physics-based edge detection (Riesz Transform)
       - Acts as "attention" to focus on boundaries
       - Suppresses artifacts in flat regions
    
    2. GABOR BASIS (vs Fourier):
       - Fourier: sin/cos oscillate infinitely → Gibbs ringing
       - Gabor: Gaussian × sin → Localized oscillation
       - Result: Sharp edges WITHOUT ringing artifacts
    
    3. DUAL-PATH ARCHITECTURE:
       - Coarse Branch: Handles smooth body regions
       - Fine Branch: Handles sharp boundary details
       - Energy gating blends them optimally
    
    4. RESOLUTION-FREE INFERENCE:
       - Implicit representation = continuous function
       - Can query at ANY resolution without retraining
       - Zoom into boundaries indefinitely
    
    5. POINT SAMPLING TRAINING:
       - Sample points instead of full images
       - Focus on boundary regions (energy-weighted)
       - More efficient learning of fine details
    """)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def compare_models():
    """Compare EGM-Net with other architectures."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    from egm_net import EGMNet, EGMNetLite
    from spectral_mamba import SpectralVMUNet
    
    models = {
        'SpectralVMUNet': SpectralVMUNet(1, 3, 256, 64, 4, 2),
        'EGM-Net Full': EGMNet(1, 3, 256, 64, 4, 2, 256, 4, 64),
        'EGM-Net Lite': EGMNetLite(1, 3, 256).model,
    }
    
    print(f"\n{'Model':<20} {'Parameters':>15} {'Trainable':>15}")
    print("-" * 50)
    
    for name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<20} {format_number(total):>15} {format_number(trainable):>15}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_egm_net()
    compare_models()
