import torch
import torch.nn as nn
import time
import numpy as np
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple
import sys
import os

# Add src to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models.egm_net import EGMNet, EGMNetLite
except ImportError:
    # Fallback to local import if run from src
    try:
        from models.egm_net import EGMNet, EGMNetLite
    except ImportError:
        print("Error: Could not import EGMNet. Make sure you are in the src directory or have setup PYTHONPATH.")
        sys.exit(1)


class FLOPsProfiler:
    """
    Custom FLOPs/MACs profiler using PyTorch hooks.
    Estimates FLOPs for standard layers.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.module_flops = OrderedDict()
        self.module_macs = OrderedDict()
        self.module_params = OrderedDict()
        
        self.total_flops = 0
        self.total_macs = 0
        self.total_params = 0

    def start_profile(self):
        """Register hooks for profiling."""
        self._remove_hooks()
        
        def conv_hook(module, input, output):
            # Conv2d: 2 * Cin * Cout * K * K * H * W / groups
            # FLOPs = MACs * 2 (approx)
            # MACs = Cout * (Cin/groups * K * K) * H * W
            
            x = input[0]
            out = output
            
            batch_size = x.shape[0]
            if batch_size == 0: return

            kh, kw = module.kernel_size
            groups = module.groups
            cin = module.in_channels
            cout = module.out_channels
            hout, wout = out.shape[2], out.shape[3]
            
            # Kernel ops per output pixel
            kernel_ops = (cin // groups) * kh * kw
            
            # Total MACs for the layer
            macs = kernel_ops * cout * hout * wout * batch_size
            
            # Bias adds 1 FLOP per output element
            bias_flops = cout * hout * wout * batch_size if module.bias is not None else 0
            
            flops = 2 * macs + bias_flops
            
            self._record(module, flops, macs)

        def linear_hook(module, input, output):
            # Linear: 2 * Cin * Cout
            # MACs = Cin * Cout
            x = input[0]
            batch_size = x.shape[0]
            # Flatten other dims
            num_elements = x.numel() // x.shape[-1]
            
            cin = module.in_features
            cout = module.out_features
            
            macs = num_elements * cin * cout
            bias_flops = num_elements * cout if module.bias is not None else 0
            
            flops = 2 * macs + bias_flops
            self._record(module, flops, macs)

        def norm_hook(module, input, output):
            # BatchNorm/GroupNorm/LayerNorm: ~4-5 FLOPs per element (mean, var, sub, div, affine)
            # Roughly 4 * num_elements
            x = input[0]
            num_elements = x.numel()
            flops = 4 * num_elements
            macs = num_elements # Approx 1 MAC equivalent
            self._record(module, flops, macs)

        def gelu_hook(module, input, output):
            # GELU: ~8-10 FLOPs per element (tanh approximation)
            x = input[0]
            num_elements = x.numel()
            flops = 8 * num_elements
            macs = 0
            self._record(module, flops, macs)

        def silu_hook(module, input, output):
            # SiLU (Swish): x * sigmoid(x). Sigmoid is exp based. ~6 FLOPs.
            x = input[0]
            num_elements = x.numel()
            flops = 6 * num_elements
            macs = 0
            self._record(module, flops, macs)
            
        def relu_hook(module, input, output):
            # ReLU: 1 comparison per element
            x = input[0]
            num_elements = x.numel()
            flops = 1 * num_elements
            macs = 0
            self._record(module, flops, macs)

        # Register hooks for leaf modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                self.hooks.append(module.register_forward_hook(norm_hook))
            elif isinstance(module, nn.GELU):
                self.hooks.append(module.register_forward_hook(gelu_hook))
            elif isinstance(module, nn.SiLU):
                self.hooks.append(module.register_forward_hook(silu_hook))
            elif isinstance(module, nn.ReLU):
                self.hooks.append(module.register_forward_hook(relu_hook))
            
            # Record params
            params = sum(p.numel() for p in module.parameters(recurse=False))
            self.module_params[module] = params
            self.total_params += params

    def _record(self, module, flops, macs):
        if module not in self.module_flops:
            self.module_flops[module] = 0
            self.module_macs[module] = 0
        self.module_flops[module] += flops
        self.module_macs[module] += macs
        self.total_flops += flops
        self.total_macs += macs

    def stop_profile(self):
        self._remove_hooks()

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def format_flops(flops):
    if flops >= 1e9:
        return f"{flops/1e9:.2f} G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} M"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f} K"
    return f"{flops:.0f}"


def format_params(params):
    if params >= 1e6:
        return f"{params/1e6:.2f} M"
    elif params >= 1e3:
        return f"{params/1e3:.2f} K"
    return f"{params}"


def benchmark_model(model, input_tensor, device='cpu', num_warmup=10, num_runs=50):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Timing
    if device == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        end_event.record()
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event) / 1000.0 # seconds
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        total_time = time.time() - start_time
        
    avg_latency = (total_time / num_runs) * 1000 # ms
    throughput = num_runs / total_time # FPS
    
    return avg_latency, throughput


def inspect_model():
    print("=" * 80)
    print("EGM-Net Detailed Model Inspection")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    
    # Configuration
    configs = [
        {
            'name': 'EGM-Net (SOTA: HRNetV2+Mamba)',
            'in_channels': 3,
            'num_classes': 4,
            'img_size': 256,
            'model_fn': lambda: EGMNet(
                in_channels=3, 
                num_classes=4, 
                img_size=256, 
                use_hrnet=True,
                base_channels=64, # High-Res dim
                num_stages=4
            )
        },
        # {
        #     'name': 'Backbone Only (HRNetV2-Mamba)',
        #     'in_channels': 3,
        #     'num_classes': 4,
        #     'img_size': 256, 
        #     'model_fn': lambda: EGMNet(..., use_hrnet=True).backbone
        # }
    ]
    
    print("Note: GFLOPs estimation primarily covers Conv2d/Linear layers.")
    print("      Complex operations (FFT, Selective Scan) in Mamba/Spectral blocks")
    print("      are not fully captured by standard hooks, so actual compute is higher.")
    
    for config in configs:
        print("\n" + "-" * 80)
        print(f"Analyzing {config['name']} @ {config['img_size']}x{config['img_size']}")
        print("-" * 80)
        
        model = config['model_fn']()
        input_tensor = torch.randn(1, config['in_channels'], config['img_size'], config['img_size'])
        
        # 1. Profiling FLOPs/Params
        profiler = FLOPsProfiler(model)
        
        # Run forward for profiling
        model.eval()
        profiler.start_profile()
        with torch.no_grad():
            model(input_tensor)
        profiler.stop_profile()
        
        # 2. Timing
        avg_latency, throughput = benchmark_model(model, input_tensor, device=device)
        
        # 3. Detailed Report
        print(f"{'Module':<40} | {'Params':<10} | {'GFLOPs':<10} | {'GMACs':<10}")
        print("-" * 80)
        
        # We want to print stats for high-level blocks
        # Iterate over named children
        
        def report_module(name, module, indent=0):
            params = sum(p.numel() for p in module.parameters())
            
            # Aggregate FLOPs for this module subtree
            flops = 0
            macs = 0
            # If leaf used in hooks
            if module in profiler.module_flops:
                flops += profiler.module_flops[module]
                macs += profiler.module_macs[module]
            
            # Recurse for children
            for child in module.modules():
                if child != module:
                    if child in profiler.module_flops:
                        flops += profiler.module_flops[child]
                        macs += profiler.module_macs[child]
            
            name_str = "  " * indent + name
            # Only print if meaningful
            if params > 0 or flops > 0:
                print(f"{name_str:<40} | {format_params(params):<10} | {format_flops(flops):<10} | {format_flops(macs):<10}")
            
            # Recurse children for detailed view (depth limited)
            if indent < 4:
                for child_name, child in module.named_children():
                    report_module(child_name, child, indent + 1)
        
        # Root level
        print(f"{'Total Model':<40} | {format_params(profiler.total_params):<10} | {format_flops(profiler.total_flops):<10} | {format_flops(profiler.total_macs):<10}")
        print("-" * 80)
        
        for name, child in model.named_children():
            report_module(name, child)
            
        print("-" * 80)
        print(f"Inference Latency: {avg_latency:.2f} ms")
        print(f"Throughput:       {throughput:.2f} FPS")
        print(f"Total Params:     {format_params(profiler.total_params)}")
        print(f"Total GFLOPs:     {profiler.total_flops/1e9:.4f}")
        print(f"Total GMACs:      {profiler.total_macs/1e9:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    inspect_model()
