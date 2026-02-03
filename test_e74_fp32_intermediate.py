#!/usr/bin/env python3
"""
Test with Python using fp32 intermediates to match CUDA exactly.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

def test_fp32_intermediate():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.bfloat16

    T, B, dim, n_state = 2, 1, 32, 32

    W = torch.randn(n_state, dim, device=device, dtype=dtype)
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    # ============ CUDA VERSION ============
    print("=" * 60)
    print("CUDA FORWARD")
    print("=" * 60)

    S_cuda = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)
    empty = torch.empty(0, device=device, dtype=dtype)

    results = elman_ladder_cuda.e74_full_matrix_forward(
        True, x, S_cuda, 0, True, W, empty, empty, empty)

    S_final_cuda = results[0]
    output_cuda = results[1]
    k_cache = results[2]

    print(f"CUDA S[0,0,:5]: {S_final_cuda[0,0,:5].float()}")
    print(f"CUDA output[0,0,:5]: {output_cuda[0,0,:5].float()}")
    print(f"CUDA output[1,0,:5]: {output_cuda[1,0,:5].float()}")

    # ============ PYTHON VERSION with fp32 intermediate ============
    print("\n" + "=" * 60)
    print("PYTHON FORWARD (fp32 intermediate)")
    print("=" * 60)

    # Match CUDA: projection stays bf16
    x_flat = x.reshape(T * B, dim)
    kvq_py = (x_flat @ W.T)  # bf16
    kvq_py = kvq_py.reshape(T, B, n_state)

    S_py = S.clone()
    outputs_py = []

    for t in range(T):
        k_raw = kvq_py[t]  # bf16
        v_raw = kvq_py[t]
        q_raw = kvq_py[t]

        # Use fp32 for norm computation (like CUDA)
        k_float = k_raw.float()
        k_norm_val = k_float.norm(dim=-1, keepdim=True) + 1e-6
        k_norm = (k_float / k_norm_val)  # stays fp32

        # retrieved = S @ k_norm (use fp32)
        S_float = S_py.float()
        retrieved = torch.einsum('bij,bj->bi', S_float, k_norm)

        # delta = v - retrieved (v is ORIGINAL bf16 converted to fp32)
        delta = v_raw.float() - retrieved

        # outer product (fp32)
        outer = torch.einsum('bi,bj->bij', delta, k_norm)

        # State update with tanh (fp32 then back to bf16)
        S_raw = S_float + outer
        S_py = torch.tanh(S_raw).to(dtype)  # back to bf16

        # Output: Sq = S @ q (use fp32)
        q_float = q_raw.float()
        Sq = torch.einsum('bij,bj->bi', S_py.float(), q_float)
        out = (Sq * F.silu(Sq)).to(dtype)
        outputs_py.append(out)

    output_py = torch.stack(outputs_py, dim=0)

    print(f"Python S[0,0,:5]: {S_py[0,0,:5].float()}")
    print(f"Python output[0,0,:5]: {output_py[0,0,:5].float()}")
    print(f"Python output[1,0,:5]: {output_py[1,0,:5].float()}")

    # ============ COMPARE ============
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    S_diff = (S_py.float() - S_final_cuda.float()).abs()
    out_diff = (output_py.float() - output_cuda.float()).abs()

    print(f"Final state diff: max={S_diff.max():.6f}")
    print(f"Output diff: max={out_diff.max():.6f}")

    # Check if they match within bf16 rounding
    if S_diff.max() < 0.002 and out_diff.max() < 10:
        print("\n*** MATCHES! Differences are within bf16 rounding tolerance ***")
    else:
        # Find where max diff is
        max_idx = out_diff.argmax()
        t_idx = max_idx // (B * n_state)
        b_idx = (max_idx % (B * n_state)) // n_state
        n_idx = max_idx % n_state
        print(f"\nMax output diff at [{t_idx}, {b_idx}, {n_idx}]:")
        print(f"  Python: {output_py[t_idx, b_idx, n_idx].float():.8f}")
        print(f"  CUDA:   {output_cuda[t_idx, b_idx, n_idx].float():.8f}")

if __name__ == '__main__':
    test_fp32_intermediate()
