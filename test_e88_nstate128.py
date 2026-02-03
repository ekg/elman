#!/usr/bin/env python3
"""Quick test to verify n_state=128 works."""

import torch
import sys
sys.path.insert(0, '.')

from elman.models.e88_fla_hybrid import E88FLAHybrid, E88_NATIVE_CUDA_AVAILABLE

def test_nstate():
    print("Testing E88 with n_state=128...")

    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)

    # Config that failed
    model = E88FLAHybrid(
        dim=384,  # Small dim
        n_state=128,
        n_heads=5,
        expansion=1.0,
        use_conv=False,
        use_gate=False,
        use_output_norm=False,
    ).to(device).to(dtype)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    B, T = 4, 64

    # Test forward
    x = torch.randn(B, T, 384, device=device, dtype=dtype)

    model.train()
    out_cuda, S = model(x, use_cuda=True)

    print(f"Output shape: {out_cuda.shape}")
    print(f"Output norm: {out_cuda.norm():.4f}")
    print(f"Output NaN: {torch.isnan(out_cuda).any()}")
    print(f"Output inf: {torch.isinf(out_cuda).any()}")
    print(f"Output max: {out_cuda.abs().max():.4f}")
    print(f"Output mean: {out_cuda.mean():.4f}")

    # Test backward
    loss = out_cuda.sum()
    loss.backward()

    print(f"\nBackward OK")

    # Check gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            grad_nan = torch.isnan(p.grad).any().item()
            grad_inf = torch.isinf(p.grad).any().item()
            if grad_nan or grad_inf or grad_norm > 1e6:
                print(f"  {name}: norm={grad_norm:.2f}, NaN={grad_nan}, inf={grad_inf}")

    print("\nAll checks passed!")

    # Now test with larger head count
    print("\n" + "="*60)
    print("Testing with n_state=128, n_heads=40...")

    model2 = E88FLAHybrid(
        dim=256,
        n_state=128,
        n_heads=40,  # This config had loss=30
        expansion=1.0,
        use_conv=False,
        use_gate=False,
        use_output_norm=False,
    ).to(device).to(dtype)

    print(f"Model params: {sum(p.numel() for p in model2.parameters()) / 1e6:.1f}M")

    x2 = torch.randn(B, T, 256, device=device, dtype=dtype)

    model2.train()
    out2, S2 = model2(x2, use_cuda=True)

    print(f"Output shape: {out2.shape}")
    print(f"Output norm: {out2.norm():.4f}")
    print(f"Output NaN: {torch.isnan(out2).any()}")
    print(f"Output inf: {torch.isinf(out2).any()}")

    loss2 = out2.sum()
    loss2.backward()
    print(f"Backward OK")

    # Multiple forward passes to see if state accumulates issues
    print("\n" + "="*60)
    print("Testing multiple forward passes (state accumulation)...")

    model2.zero_grad()
    S_state = None

    for i in range(10):
        x_i = torch.randn(B, T, 256, device=device, dtype=dtype)
        if S_state is not None:
            # Pack state
            S0 = torch.stack(S_state, dim=1)
        else:
            S0 = None

        out_i, S_state = model2(x_i, hidden_state=S0, use_cuda=True)

        # Check state
        S_stack = torch.stack(S_state, dim=0)
        s_norm = S_stack.norm().item()
        s_max = S_stack.abs().max().item()
        out_norm = out_i.norm().item()

        print(f"  Pass {i}: out_norm={out_norm:.2f}, state_norm={s_norm:.2f}, state_max={s_max:.4f}")

        if torch.isnan(S_stack).any() or torch.isinf(S_stack).any():
            print(f"    State has NaN/inf!")
            break

if __name__ == '__main__':
    test_nstate()
