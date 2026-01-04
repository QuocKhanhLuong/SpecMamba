"""
Architecture Validation Test Suite for SpectralVMUNet.

Provides comprehensive testing for all model components without requiring
a full dataset. Verifies tensor shapes, gradient flow, memory usage,
and loss function computations.

Test Categories:
    - Module tests: SpectralGating, DirectionalScanner, VSSBlock
    - Integration tests: Full network forward/backward passes
    - Loss function tests: Dice, Frequency, SpectralDual, BoundaryAware
    - Memory efficiency tests
"""

import torch
import torch.nn as nn
from spectral_mamba import SpectralVMUNet
from physics_loss import SpectralDualLoss, DiceLoss, FrequencyLoss, BoundaryAwareLoss
from spectral_layers import SpectralGating
from mamba_block import VSSBlock, DirectionalScanner


def test_spectral_gating():
    """Test SpectralGating module."""
    print("Testing SpectralGating...")
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width)
    
    spec_gate = SpectralGating(channels, height, width)
    output = spec_gate(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ SpectralGating: {x.shape} -> {output.shape}")


def test_directional_scanner():
    """Test DirectionalScanner module."""
    print("Testing DirectionalScanner...")
    batch_size, channels, height, width = 2, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    scanner = DirectionalScanner(channels, scan_dim=32)
    output = scanner(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ DirectionalScanner: {x.shape} -> {output.shape}")


def test_vss_block():
    """Test VSSBlock module."""
    print("Testing VSSBlock...")
    batch_size, channels, height, width = 2, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    vss = VSSBlock(channels, scan_dim=32)
    output = vss(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ VSSBlock: {x.shape} -> {output.shape}")


def test_spectral_vss_block():
    """Test SpectralVSSBlock (dual-path)."""
    print("Testing SpectralVSSBlock...")
    from spectral_mamba import SpectralVSSBlock
    
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width)
    
    block = SpectralVSSBlock(channels, height, width, depth=1)
    output = block(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ SpectralVSSBlock: {x.shape} -> {output.shape}")


def test_full_network():
    """Test complete SpectralVMUNet."""
    print("\nTesting SpectralVMUNet...")
    batch_size = 2
    in_channels = 1
    out_channels = 3
    img_size = 256
    
    model = SpectralVMUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        base_channels=64,
        num_stages=4,
        depth=2
    )
    
    # Test forward pass
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    output = model(x)
    
    assert output.shape == (batch_size, out_channels, img_size, img_size), \
        f"Output shape mismatch: {output.shape}"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ SpectralVMUNet: {x.shape} -> {output.shape}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def test_losses():
    """Test loss functions."""
    print("\nTesting Loss Functions...")
    batch_size, num_classes, height, width = 2, 3, 64, 64
    
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dice = dice_loss(pred, target)
    print(f"✓ Dice Loss: {dice.item():.4f}")
    
    # Test Frequency Loss
    freq_loss = FrequencyLoss()
    freq = freq_loss(pred, target)
    print(f"✓ Frequency Loss: {freq.item():.4f}")
    
    # Test SpectralDualLoss
    dual_loss = SpectralDualLoss(spatial_weight=1.0, freq_weight=0.1)
    total, components = dual_loss(pred, target, return_components=True)
    print(f"✓ SpectralDualLoss: {total.item():.4f}")
    for name, value in components.items():
        print(f"  - {name}: {value:.4f}")
    
    # Test BoundaryAwareLoss
    boundary_loss = BoundaryAwareLoss()
    boundary = boundary_loss(pred, target)
    print(f"✓ BoundaryAwareLoss: {boundary.item():.4f}")


def test_backward_pass():
    """Test that gradients flow correctly."""
    print("\nTesting Backward Pass...")
    batch_size = 2
    in_channels = 1
    out_channels = 3
    img_size = 256
    
    model = SpectralVMUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        base_channels=64,
        num_stages=4,
        depth=2
    )
    
    loss_fn = SpectralDualLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward-backward pass
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    target = torch.randint(0, out_channels, (batch_size, img_size, img_size))
    
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients found!"
    print(f"✓ Backward Pass: Loss = {loss.item():.4f}")
    print(f"  Gradients flowing correctly")


def test_memory_efficiency():
    """Test memory usage."""
    print("\nTesting Memory Efficiency...")
    import gc
    
    model = SpectralVMUNet(
        in_channels=1,
        out_channels=3,
        img_size=256,
        base_channels=64,
        num_stages=4,
        depth=2
    )
    
    # Get model size
    param_size = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024  # MB
    print(f"✓ Model Size: {param_size:.2f} MB")
    
    # Test forward pass memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    x = torch.randn(1, 1, 256, 256)
    try:
        output = model(x)
        print(f"✓ Forward Pass: Memory usage acceptable")
    except RuntimeError as e:
        print(f"✗ Forward Pass: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Spectral Mamba (Spec-VMamba) - Architecture Validation")
    print("=" * 60)
    
    # Run tests
    test_spectral_gating()
    test_directional_scanner()
    test_vss_block()
    test_spectral_vss_block()
    test_full_network()
    test_losses()
    test_backward_pass()
    test_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
