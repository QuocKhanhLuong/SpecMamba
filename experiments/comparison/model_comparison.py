"""
Model Comparison: EGM-Net vs State-of-the-Art Methods.

Compares EGM-Net against:
1. U-Net (baseline)
2. Attention U-Net
3. TransUNet
4. nnUNet
5. SpectralVMUNet (our other model)

Metrics:
- Dice Score (per-class and mean)
- Hausdorff Distance 95 (HD95)
- IoU / Jaccard
- Inference Time
- Model Parameters / FLOPs

References:
    [1] Ronneberger et al., "U-Net," MICCAI 2015.
    [2] Chen et al., "TransUNet," arXiv 2021.
    [3] Isensee et al., "nnU-Net," Nature Methods 2021.
"""

import sys
sys.path.insert(0, '../..')

import torch
import time
from dataclasses import dataclass
from typing import Dict, List
import json
from pathlib import Path

from src.models import EGMNet, EGMNetLite, SpectralVMUNet
from src.utils import SegmentationMetrics


@dataclass
class ComparisonConfig:
    """Configuration for model comparison."""
    experiment_name: str = "model_comparison"
    dataset: str = "ACDC"
    num_classes: int = 4
    img_size: int = 224
    device: str = "cuda"
    results_dir: str = "../../results/comparison"


# =============================================================================
# Models to Compare
# =============================================================================

def get_models(config: ComparisonConfig) -> Dict[str, torch.nn.Module]:
    """Get all models for comparison."""
    models = {
        "EGM-Net": EGMNet(
            in_channels=1,
            num_classes=config.num_classes,
            img_size=config.img_size,
        ),
        "EGM-Net-Lite": EGMNetLite(
            in_channels=1,
            num_classes=config.num_classes,
            img_size=config.img_size,
        ),
        "SpectralVMUNet": SpectralVMUNet(
            in_channels=1,
            out_channels=config.num_classes,
            img_size=config.img_size,
        ),
        # TODO: Add other baselines (U-Net, TransUNet, etc.)
    }
    return models


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model: torch.nn.Module, input_size: tuple, 
                           device: str, num_runs: int = 100) -> float:
    """Measure average inference time."""
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    return (time.time() - start) / num_runs * 1000  # ms


def run_comparison(config: ComparisonConfig) -> Dict:
    """Run model comparison experiment."""
    models = get_models(config)
    results = {}
    
    for name, model in models.items():
        print(f"\nAnalyzing: {name}")
        
        params = count_parameters(model)
        inf_time = measure_inference_time(
            model, 
            (1, 1, config.img_size, config.img_size),
            config.device
        )
        
        results[name] = {
            "parameters": params,
            "parameters_M": params / 1e6,
            "inference_time_ms": inf_time,
            # TODO: Add trained metrics (Dice, HD95, etc.)
        }
        
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        print(f"  Inference: {inf_time:.2f} ms")
    
    return results


def main():
    config = ComparisonConfig()
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    results = run_comparison(config)
    
    # Save results
    with open(f"{config.results_dir}/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Comparison complete!")
    print(f"Results saved to: {config.results_dir}/comparison_results.json")


if __name__ == "__main__":
    main()
