#!/usr/bin/env python3
"""
Simplified backward test - smaller values, check for NaN sources.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

def test_backward_simple():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.float32  # Start with fp32 to avoid precision issues

    T, B, dim, n = 2, 1, 16, 16  # Small size

    # Scale down inputs to avoid extreme values
    W = torch.randn(n, dim, device=device, dtype=dtype) * 0.1
    W.requires_grad_(True)
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * 0.1
    x.requires_grad_(True)
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    # Manual forward with gradient tracking
    x_flat = x.reshape(T * B, dim)
    k_all = (x_flat @ W.T).reshape(T, B, n)

    S = S0.clone()
    outputs = []

    print("Forward pass values:")
    for t in range(T):
        k_raw = k_all[t]

        # Normalize k
        k_norm_val = k_raw.norm(dim=-1, keepdim=True) + 1e-6
        k_norm = k_raw / k_norm_val

        # retrieved = S @ k_norm
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)

        # delta = v - retrieved
        delta = k_raw - retrieved

        # State update
        outer = torch.einsum('bi,bj->bij', delta, k_norm)
        S = torch.tanh(S + outer)

        # Output
        Sq = torch.einsum('bij,bj->bi', S, k_raw)
        sig = torch.sigmoid(Sq)
        out = Sq * Sq * sig

        print(f"\nt={t}:")
        print(f"  k_raw[:4]: {k_raw[0,:4]}")
        print(f"  k_norm[:4]: {k_norm[0,:4]}")
        print(f"  S[0,:3,:3]:\n{S[0,:3,:3]}")
        print(f"  Sq[:4]: {Sq[0,:4]}")
        print(f"  out[:4]: {out[0,:4]}")
        print(f"  out has NaN: {torch.isnan(out).any()}")

        outputs.append(out)

    output = torch.stack(outputs, dim=0)

    print(f"\nFinal output has NaN: {torch.isnan(output).any()}")

    # Small gradient
    d_output = torch.ones_like(output) * 0.1

    # Backward
    output.backward(d_output)

    print(f"\nd_x has NaN: {torch.isnan(x.grad).any()}")
    print(f"d_W has NaN: {torch.isnan(W.grad).any()}")
    print(f"\nd_x[0,0,:4]: {x.grad[0,0,:4]}")
    print(f"d_W[0,:4]: {W.grad[0,:4]}")


if __name__ == '__main__':
    test_backward_simple()
