"""Test suite: 3-Stream Asymmetric Spec-HRNet."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import torch
from models.specmamba_net import (
    PriorKnowledgeConstructor, DWConvBlock, AdaptiveFourierMixer,
    CrossScanGatedMixer, TriFuseLayer, TriStreamFusion, SkipAttention,
    SpecMambaNet,
)

def test_prior():
    print("1. PriorKnowledgeConstructor...")
    m = PriorKnowledgeConstructor()
    o = m(torch.randn(2,3,224,224))
    assert o.shape == (2,3,224,224) and o[:,1].min()>=0
    print(f"   OK: {o.shape}")

def test_dwconv():
    print("2. DWConvBlock (FR)...")
    m = DWConvBlock(48); o = m(torch.randn(2,48,224,224))
    assert o.shape == (2,48,224,224)
    print(f"   OK: {o.shape}, params={sum(p.numel() for p in m.parameters()):,}")

def test_afm():
    print("3. AdaptiveFourierMixer (HR)...")
    m = AdaptiveFourierMixer(48, 16); o = m(torch.randn(2,48,112,112))
    assert o.shape == (2,48,112,112)
    print(f"   OK: {o.shape}")

def test_csgm():
    print("4. CrossScanGatedMixer (LR)...")
    m = CrossScanGatedMixer(96); o = m(torch.randn(2,96,56,56))
    assert o.shape == (2,96,56,56)
    print(f"   OK: {o.shape}")

def test_trifuse():
    print("5. TriFuseLayer (6-path cross-fuse)...")
    m = TriFuseLayer(48, 48, 96)
    fr, hr, lr = torch.randn(2,48,224,224), torch.randn(2,48,112,112), torch.randn(2,96,56,56)
    fr2, hr2, lr2 = m(fr, hr, lr)
    assert fr2.shape == fr.shape and hr2.shape == hr.shape and lr2.shape == lr.shape
    print(f"   OK: FR{fr2.shape} HR{hr2.shape} LR{lr2.shape}")
    print(f"   Params: {sum(p.numel() for p in m.parameters()):,}")

def test_tristream():
    print("6. TriStreamFusion (FR gates HR+LR)...")
    m = TriStreamFusion(48, 48, 96, 48)
    fr, hr, lr = torch.randn(2,48,224,224), torch.randn(2,48,112,112), torch.randn(2,96,56,56)
    o = m(fr, hr, lr)
    assert o.shape == (2,48,224,224)
    print(f"   OK: {o.shape}")

def test_full_eval():
    print("7. SpecMambaNet eval...")
    model = SpecMambaNet(3, 4, 32, num_modes=16, blocks_per_stage=1, num_stages=2)
    model.eval()
    with torch.no_grad():
        o = model(torch.randn(2,3,224,224))
    assert o['output'].shape == (2,4,224,224)
    assert 'aux_outputs' not in o
    print(f"   OK: {o['output'].shape}")

def test_deep_sup():
    print("8. Deep supervision...")
    model = SpecMambaNet(3, 4, 32, deep_supervision=True, num_modes=16,
                         blocks_per_stage=1, num_stages=3)
    model.train()
    o = model(torch.randn(2,3,224,224))
    assert 'aux_outputs' in o and len(o['aux_outputs']) == 2
    for i, a in enumerate(o['aux_outputs']):
        assert a.shape == (2,4,224,224), f"aux {i}: {a.shape}"
    print(f"   OK: {len(o['aux_outputs'])} aux heads, all [2,4,224,224]")

def test_grads():
    print("9. Gradient flow...")
    model = SpecMambaNet(3, 4, 16, num_modes=8, blocks_per_stage=1, num_stages=2)
    model.train()
    o = model(torch.randn(1,3,64,64))
    o['output'].sum().backward()
    bad = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert len(bad) == 0, f"No grad: {bad[:5]}"
    print(f"   OK: all params have gradients")

def test_params():
    print("10. Parameter count (C=48)...")
    model = SpecMambaNet(3, 4, 48, num_modes=32, blocks_per_stage=2, num_stages=3)
    total = sum(p.numel() for p in model.parameters())
    fr = sum(p.numel() for p in model.fr_stages.parameters())
    hr = sum(p.numel() for p in model.hr_stages.parameters())
    lr = sum(p.numel() for p in model.lr_stages.parameters())
    fuse = sum(p.numel() for p in model.tri_fuse.parameters())
    print(f"   Total: {total:,}")
    print(f"   FR(DWConv): {fr:,} | HR(FFT): {hr:,} | LR(Mamba): {lr:,}")
    print(f"   TriFuse: {fuse:,}")

if __name__ == '__main__':
    tests = [test_prior, test_dwconv, test_afm, test_csgm, test_trifuse,
             test_tristream, test_full_eval, test_deep_sup, test_grads, test_params]
    print(f"\n{'='*60}\n3-Stream Asymmetric Spec-HRNet Test Suite\n{'='*60}\n")
    ok = 0
    for t in tests:
        try: t(); ok += 1
        except Exception as e: print(f"   FAILED: {e}")
        print()
    print(f"{'='*60}\n{ok}/{len(tests)} passed\n{'='*60}")
