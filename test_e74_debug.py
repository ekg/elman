#!/usr/bin/env python3
"""
Detailed debug test - trace every value to find output mismatch.
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as elman_ladder_cuda

def test_debug():
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
    k_cache = results[2]  # Projected k values from CUDA

    print(f"CUDA k_cache shape: {k_cache.shape}")
    print(f"CUDA k_cache[0,0,:5]: {k_cache[0,0,:5].float()}")
    print(f"CUDA k_cache[1,0,:5]: {k_cache[1,0,:5].float()}")

    print(f"\nCUDA output shape: {output_cuda.shape}")
    print(f"CUDA output[0,0,:5]: {output_cuda[0,0,:5].float()}")
    print(f"CUDA output[1,0,:5]: {output_cuda[1,0,:5].float()}")

    # ============ PYTHON VERSION ============
    print("\n" + "=" * 60)
    print("PYTHON FORWARD")
    print("=" * 60)

    # Match CUDA projection exactly
    x_flat = x.reshape(T * B, dim)
    # Use exact same computation as cuBLAS: bf16 inputs, fp32 accumulate, bf16 output
    kvq_py = (x_flat @ W.T)  # This stays in bf16 like cuBLAS
    kvq_py = kvq_py.reshape(T, B, n_state)

    print(f"Python kvq shape: {kvq_py.shape}")
    print(f"Python kvq[0,0,:5]: {kvq_py[0,0,:5].float()}")
    print(f"Python kvq[1,0,:5]: {kvq_py[1,0,:5].float()}")

    # Check k projection match
    k_diff = (kvq_py - k_cache).abs()
    print(f"\nk projection diff: max={k_diff.max():.8f}")

    S_py = S.clone()
    outputs_py = []

    for t in range(T):
        k_raw = kvq_py[t]  # [B, n]
        v_raw = kvq_py[t]
        q_raw = kvq_py[t]

        # Normalize k
        k_norm = k_raw / (k_raw.norm(dim=-1, keepdim=True) + 1e-6)

        # retrieved = S @ k_norm
        retrieved = torch.einsum('bij,bj->bi', S_py, k_norm)

        # delta = v - retrieved
        delta = v_raw - retrieved

        # outer product
        outer = torch.einsum('bi,bj->bij', delta, k_norm)

        # State update with tanh
        S_py = torch.tanh(S_py + outer)

        # Output: Sq = S @ q, then Sq * silu(Sq)
        Sq = torch.einsum('bij,bj->bi', S_py, q_raw)  # Use ORIGINAL q (not normalized!)
        out = Sq * F.silu(Sq)
        outputs_py.append(out)

        print(f"\nt={t}:")
        print(f"  k_norm[:5] = {k_norm[0,:5].float()}")
        print(f"  S[0,0,:5] = {S_py[0,0,:5].float()}")
        print(f"  Sq[:5] = {Sq[0,:5].float()}")
        print(f"  out[:5] = {out[0,:5].float()}")

    output_py = torch.stack(outputs_py, dim=0)

    print(f"\nPython output shape: {output_py.shape}")
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

    # Find where max output diff is
    max_idx = out_diff.argmax()
    t_idx = max_idx // (B * n_state)
    b_idx = (max_idx % (B * n_state)) // n_state
    n_idx = max_idx % n_state

    print(f"\nMax output diff at [{t_idx}, {b_idx}, {n_idx}]:")
    print(f"  Python: {output_py[t_idx, b_idx, n_idx].float():.8f}")
    print(f"  CUDA:   {output_cuda[t_idx, b_idx, n_idx].float():.8f}")

if __name__ == '__main__':
    test_debug()
