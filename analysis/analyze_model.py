"""
Model Analysis Script for SpectralVMUNet.

Provides comprehensive model profiling including parameter counts, FLOPs
estimation, memory usage analysis, and architecture visualization.

Analysis Features:
    - Parameter counting (total, trainable, per-module)
    - FLOPs estimation via hook-based forward pass tracing
    - Network depth analysis (layers, stages)
    - Memory footprint estimation
    - ASCII architecture visualization
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count per module
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'per_module': module_params
    }


def count_flops(model: nn.Module, input_size: tuple = (1, 1, 256, 256)) -> dict:
    """
    Estimate FLOPs for the model.
    Uses a simple hook-based approach.
    """
    flops_dict = {}
    
    def conv_flops(module, input, output):
        batch_size = input[0].size(0)
        output_dims = output.shape[2:]
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        output_elements = output.numel() / batch_size
        
        flops = kernel_ops * output_elements * 2  # multiply-add
        return flops
    
    def linear_flops(module, input, output):
        batch_size = input[0].size(0)
        flops = module.in_features * module.out_features * 2
        return flops * batch_size
    
    def gru_flops(module, input, output):
        # GRU: 3 gates, each with input and hidden linear ops
        input_size = module.input_size
        hidden_size = module.hidden_size
        batch_size = input[0].size(0)
        
        # 3 gates: reset, update, new
        gate_flops = 3 * (input_size * hidden_size + hidden_size * hidden_size) * 2
        return gate_flops * batch_size
    
    total_flops = 0
    hooks = []
    
    def make_hook(name, flop_fn):
        def hook(module, input, output):
            nonlocal total_flops
            flops = flop_fn(module, input, output)
            total_flops += flops
            if name not in flops_dict:
                flops_dict[name] = 0
            flops_dict[name] += flops
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name, conv_flops)))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name, linear_flops)))
        elif isinstance(module, nn.GRUCell):
            hooks.append(module.register_forward_hook(make_hook(name, gru_flops)))
    
    # Forward pass
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)
    
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return {
        'total': total_flops,
        'per_layer': flops_dict
    }


def analyze_depth(model: nn.Module) -> dict:
    """Analyze network depth and structure."""
    
    def count_layers(module, layer_types=None):
        if layer_types is None:
            layer_types = (nn.Conv2d, nn.Linear, nn.GRUCell)
        
        count = 0
        for child in module.modules():
            if isinstance(child, layer_types):
                count += 1
        return count
    
    # Count different layer types
    conv_layers = count_layers(model, (nn.Conv2d,))
    linear_layers = count_layers(model, (nn.Linear,))
    gru_layers = count_layers(model, (nn.GRUCell,))
    norm_layers = count_layers(model, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d))
    
    # Estimate effective depth (longest path)
    # For U-Net: encoder_stages + bottleneck + decoder_stages
    num_stages = model.num_stages if hasattr(model, 'num_stages') else 4
    depth_per_stage = 2  # SpectralVSSBlock depth
    
    # Encoder path + bottleneck + decoder path
    effective_depth = num_stages * depth_per_stage + depth_per_stage + num_stages * depth_per_stage
    
    return {
        'conv_layers': conv_layers,
        'linear_layers': linear_layers,
        'gru_layers': gru_layers,
        'norm_layers': norm_layers,
        'total_layers': conv_layers + linear_layers + gru_layers,
        'effective_depth': effective_depth,
        'num_stages': num_stages
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


def main():
    print("=" * 70)
    print("SPECTRAL MAMBA (Spec-VMUNet) - MODEL ANALYSIS")
    print("=" * 70)
    
    # Import model
    from spectral_mamba import SpectralVMUNet
    
    # Create model with default config
    model = SpectralVMUNet(
        in_channels=1,
        out_channels=3,
        img_size=256,
        base_channels=64,
        num_stages=4,
        depth=2
    )
    
    print("\n[MODEL CONFIGURATION]")
    print(f"  Input channels:  1 (grayscale)")
    print(f"  Output channels: 3 (classes)")
    print(f"  Image size:      256 x 256")
    print(f"  Base channels:   64")
    print(f"  Num stages:      4")
    print(f"  Block depth:     2")
    
    # 1. Parameter count
    print("\n" + "-" * 70)
    print("[PARAMETERS]")
    params = count_parameters(model)
    print(f"  Total Parameters:     {format_number(params['total'])} ({params['total']:,})")
    print(f"  Trainable Parameters: {format_number(params['trainable'])} ({params['trainable']:,})")
    print(f"\n  Per Module:")
    for name, count in params['per_module'].items():
        print(f"    {name:25s}: {format_number(count):>10s} ({count:,})")
    
    # 2. FLOPs
    print("\n" + "-" * 70)
    print("[COMPUTATIONAL COST (FLOPs)]")
    flops = count_flops(model, input_size=(1, 1, 256, 256))
    gflops = flops['total'] / 1e9
    print(f"  Total FLOPs:   {format_number(flops['total'])} ({flops['total']:,.0f})")
    print(f"  GFLOPs:        {gflops:.2f}")
    print(f"  TFLOPs:        {gflops/1000:.4f}")
    
    # 3. Network depth
    print("\n" + "-" * 70)
    print("[NETWORK DEPTH]")
    depth = analyze_depth(model)
    print(f"  Convolutional layers: {depth['conv_layers']}")
    print(f"  Linear layers:        {depth['linear_layers']}")
    print(f"  GRU cells:            {depth['gru_layers']}")
    print(f"  Normalization layers: {depth['norm_layers']}")
    print(f"  Total layers:         {depth['total_layers']}")
    print(f"  Encoder stages:       {depth['num_stages']}")
    print(f"  Effective depth:      {depth['effective_depth']} (encoder + bottleneck + decoder)")
    
    # 4. Memory estimation
    print("\n" + "-" * 70)
    print("[MEMORY ESTIMATION]")
    param_memory = params['total'] * 4 / (1024 ** 2)  # Float32 = 4 bytes
    
    # Estimate activation memory (rough)
    input_size = (8, 1, 256, 256)  # Batch size 8
    activation_memory = 0
    
    # Forward pass to estimate
    x = torch.randn(*input_size)
    with torch.no_grad():
        # Track tensor sizes through forward
        activations = []
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.numel() * 4)  # Float32
        
        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook))
        
        _ = model(x)
        
        for h in hooks:
            h.remove()
        
        activation_memory = sum(activations) / (1024 ** 2)
    
    print(f"  Parameter memory:   {param_memory:.2f} MB")
    print(f"  Activation memory:  {activation_memory:.2f} MB (batch_size=8)")
    print(f"  Total (estimated):  {param_memory + activation_memory:.2f} MB")
    
    # 5. Layer breakdown
    print("\n" + "-" * 70)
    print("[ARCHITECTURE SUMMARY]")
    print("""
    SpectralVMUNet Structure:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: (B, 1, 256, 256)                                    │
    ├─────────────────────────────────────────────────────────────┤
    │  Patch Embedding: Conv2d(1→64, k=4, s=4) → (B, 64, 64, 64)  │
    ├─────────────────────────────────────────────────────────────┤
    │  ENCODER                                                     │
    │  ├── Stage 1: SpectralVSSBlock(64)  → (B, 64, 64, 64)       │
    │  │            PatchMerging         → (B, 128, 32, 32)       │
    │  ├── Stage 2: SpectralVSSBlock(128) → (B, 128, 32, 32)      │
    │  │            PatchMerging         → (B, 256, 16, 16)       │
    │  ├── Stage 3: SpectralVSSBlock(256) → (B, 256, 16, 16)      │
    │  │            PatchMerging         → (B, 512, 8, 8)         │
    │  └── Stage 4: SpectralVSSBlock(512) → (B, 512, 8, 8)        │
    ├─────────────────────────────────────────────────────────────┤
    │  BOTTLENECK: SpectralVSSBlock(1024) → (B, 1024, 4, 4)       │
    ├─────────────────────────────────────────────────────────────┤
    │  DECODER (with Skip Connections)                             │
    │  ├── Stage 4: Upsample + Concat + SpectralVSSBlock(512)     │
    │  ├── Stage 3: Upsample + Concat + SpectralVSSBlock(256)     │
    │  ├── Stage 2: Upsample + Concat + SpectralVSSBlock(128)     │
    │  └── Stage 1: Upsample + Concat + SpectralVSSBlock(64)      │
    ├─────────────────────────────────────────────────────────────┤
    │  Segmentation Head: Conv(64→32→3) → (B, 3, 64, 64)          │
    │  Upsample: Bilinear → (B, 3, 256, 256)                      │
    └─────────────────────────────────────────────────────────────┘
    
    SpectralVSSBlock (Dual-Path):
    ┌───────────────────────────────────────┐
    │  Input                                 │
    │    ↓                                   │
    │  ┌─────────┐        ┌──────────────┐  │
    │  │ VSSBlock│        │SpectralGating│  │
    │  │ (Mamba) │        │   (FFT)      │  │
    │  └────┬────┘        └──────┬───────┘  │
    │       ↓                    ↓          │
    │       └────→ Fusion ←──────┘          │
    │              (weighted sum)           │
    │                  ↓                    │
    │              Output                   │
    └───────────────────────────────────────┘
    """)
    
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
