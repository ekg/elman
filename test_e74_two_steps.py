#!/usr/bin/env python3
"""
Test with T=2 and verify second timestep matches.
This will expose bugs that only appear when S is non-zero.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

def test_two_steps():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n_state = 2, 1, 32, 32

    W = torch.randn(n_state, dim, device=device, dtype=dtype)
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    # ============ PYTHON VERSION (using fp32 intermediate like CUDA) ============
    print("=" * 60)
    print("PYTHON FORWARD (fp32 intermediate)")
    print("=" * 60)

    x_flat = x.reshape(T * B, dim)
    kvq = (x_flat.float() @ W.float().T).to(dtype)  # Project then back to bf16
    kvq = kvq.reshape(T, B, n_state)

    S_py = S.clone()
    outputs_py = []

    for t in range(T):
        k_raw = kvq[t]  # [B, n] - original (unnormalized)

        # Normalize k using fp32 (like CUDA)
        k_float = k_raw.float()
        k_norm_val = k_float.norm(dim=-1, keepdim=True)
        k_norm = (k_float / (k_norm_val + 1e-6))  # fp32

        # retrieved = S @ k_norm (fp32)
        S_float = S_py.float()
        retrieved = torch.einsum('bij,bj->bi', S_float, k_norm)

        # delta = v - retrieved (v is ORIGINAL k, not normalized!)
        v = k_float  # ORIGINAL (unnormalized) - THIS IS KEY
        delta = v - retrieved

        # outer product (fp32)
        outer = torch.einsum('bi,bj->bij', delta, k_norm)

        # State update with tanh
        S_raw = S_float + outer
        S_py = torch.tanh(S_raw).to(dtype)  # back to bf16

        # Output: Sq = S @ q (q is ORIGINAL k, not normalized!)
        q = k_float  # ORIGINAL (unnormalized)
        Sq = torch.einsum('bij,bj->bi', S_py.float(), q)
        out = (Sq * Sq * torch.sigmoid(Sq)).to(dtype)
        outputs_py.append(out)

        if t == 0:
            print(f"t=0: S[0,0,:3] = {S_py[0,0,:3].float()}")
        else:
            print(f"t=1: S[0,0,:3] = {S_py[0,0,:3].float()}")

    output_py = torch.stack(outputs_py, dim=0)

    # ============ CUDA VERSION ============
    print("\n" + "=" * 60)
    print("CUDA FORWARD")
    print("=" * 60)

    S_cuda = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)
    empty = torch.empty(0, device=device, dtype=dtype)

    results = elman_ladder_cuda.e74_full_matrix_forward(
        True, x, S_cuda, 0, True, W, empty, empty, empty)

    S_final_cuda = results[0].float()
    output_cuda = results[1].float()

    print(f"t=1 CUDA S[0,0,:3] = {S_final_cuda[0,0,:3]}")

    # ============ COMPARE ============
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    S_diff = (S_py.float() - S_final_cuda).abs()
    out_diff = (output_py.float() - output_cuda).abs()

    print(f"Final state diff: max={S_diff.max():.6f}")
    print(f"Output diff: max={out_diff.max():.6f}")

    # Check if diff is from v being normalized incorrectly
    print("\n--- Checking v usage ---")
    # In CUDA, check if k_shared (normalized) is used for v instead of original k
    print("If CUDA uses normalized k for v, the state will differ")
    print(f"Python S[0,0,0] = {S_py[0,0,0].float():.8f}")
    print(f"CUDA S[0,0,0] = {S_final_cuda[0,0,0]:.8f}")

    if S_diff.max() > 0.01:
        print("\n*** SIGNIFICANT DIFFERENCE - likely v normalization bug! ***")
    else:
        print("\n*** State matches! ***")

if __name__ == '__main__':
    test_two_steps()
