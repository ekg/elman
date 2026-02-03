#!/usr/bin/env python3
"""
Minimal test - single timestep, trace through every operation.
"""

import torch
import torch.nn.functional as F

# Import CUDA kernel
import hasty_pytorch_lib as elman_ladder_cuda

def test_single_step():
    torch.manual_seed(42)
    device = torch.device('cuda')

    # Use bf16 for fair comparison (CUDA only supports bf16)
    dtype = torch.bfloat16

    T, B, dim, n_state = 1, 1, 32, 32

    # Create weight
    W = torch.randn(n_state, dim, device=device, dtype=dtype)

    # Create input
    x = torch.randn(T, B, dim, device=device, dtype=dtype)

    # Initial state
    S = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    # ============ PYTHON VERSION ============
    print("=" * 60)
    print("PYTHON FORWARD")
    print("=" * 60)

    # Projection: k = v = q = W @ x
    x_flat = x.reshape(T * B, dim)
    kvq_py = x_flat @ W.T  # [T*B, n_state]
    kvq_py = kvq_py.reshape(T, B, n_state)

    k_py = kvq_py[0]  # [B, n]
    v_py = kvq_py[0]
    q_py = kvq_py[0]

    print(f"k_py (first 5): {k_py[0, :5]}")

    # Normalize k
    k_norm_py = k_py / (k_py.norm(dim=-1, keepdim=True) + 1e-6)
    print(f"k_norm (first 5): {k_norm_py[0, :5]}")
    print(f"||k||: {k_py.norm(dim=-1)}")

    # Retrieval: S @ k_norm (S is zero, so retrieved is zero)
    retrieved_py = torch.einsum('bij,bj->bi', S, k_norm_py)
    print(f"retrieved (first 5): {retrieved_py[0, :5]}")

    # Delta
    delta_py = v_py - retrieved_py
    print(f"delta (first 5): {delta_py[0, :5]}")

    # Outer product
    outer_py = torch.einsum('bi,bj->bij', delta_py, k_norm_py)
    print(f"outer[0,0,:5]: {outer_py[0, 0, :5]}")

    # State update
    S_raw_py = S + outer_py
    S_new_py = torch.tanh(S_raw_py)
    print(f"S_new[0,:3,:3]:\n{S_new_py[0, :3, :3]}")

    # Output
    Sq_py = torch.einsum('bij,bj->bi', S_new_py, q_py)
    print(f"Sq (first 5): {Sq_py[0, :5]}")

    out_py = Sq_py * F.silu(Sq_py)
    print(f"output (first 5): {out_py[0, :5]}")

    # ============ CUDA VERSION ============
    print("\n" + "=" * 60)
    print("CUDA FORWARD")
    print("=" * 60)

    # Already bf16, no conversion needed
    x_cuda = x
    S_cuda = S
    W_cuda = W

    # Empty tensors for unused weights
    empty = torch.empty(0, device=device, dtype=torch.bfloat16)

    results = elman_ladder_cuda.e74_full_matrix_forward(
        True,  # training
        x_cuda,
        S_cuda,
        0,  # proj_type = tied_kvq
        True,  # use_tanh
        W_cuda,  # W_kvq
        empty,  # W_k
        empty,  # W_v
        empty,  # W_q
    )

    S_final_cuda = results[0].float()
    output_cuda = results[1].float()
    k_cache = results[2].float()  # This is the projected k (before normalization?)

    print(f"k_cache (first 5): {k_cache[0, 0, :5]}")
    print(f"S_final[0,:3,:3]:\n{S_final_cuda[0, :3, :3]}")
    print(f"output (first 5): {output_cuda[0, 0, :5]}")

    # ============ COMPARE ============
    print("\n" + "=" * 60)
    print("COMPARISON (should be EXACT for same precision)")
    print("=" * 60)

    # Compare k projection (should match exactly since it's just matmul)
    k_diff = (k_py.float() - k_cache[0].float()).abs()
    print(f"k projection: max_diff={k_diff.max():.6f}")

    # Compare state - compute relative error for non-zero elements
    S_py = S_new_py.float()
    S_cu = S_final_cuda.float()
    S_abs_diff = (S_py - S_cu).abs()
    S_rel_diff = S_abs_diff / (S_py.abs() + 1e-10)
    print(f"State: max_abs_diff={S_abs_diff.max():.6f}, max_rel_diff={S_rel_diff.max():.4f}")

    # Compare output
    out_py_f = out_py.float()
    out_cu = output_cuda[0].float()
    out_abs_diff = (out_py_f - out_cu).abs()
    out_rel_diff = out_abs_diff / (out_py_f.abs() + 1e-10)
    print(f"Output: max_abs_diff={out_abs_diff.max():.6f}, max_rel_diff={out_rel_diff.max():.4f}")

    # Find where max relative error is
    idx = S_rel_diff.argmax()
    b, i, j = idx // (n_state * n_state), (idx % (n_state * n_state)) // n_state, idx % n_state
    print(f"\nMax rel error at S[{b},{i},{j}]: py={S_py[b,i,j]:.6f}, cuda={S_cu[b,i,j]:.6f}")

    # Check k_norm computation - CUDA uses fp32 intermediate, Python uses bf16
    print("\n--- Checking k normalization ---")

    # Python (bf16): k_norm = k / (||k|| + 1e-6)
    k_py_flat = k_py[0]  # [n_state]
    k_norm_py_flat = k_norm_py[0]

    # What CUDA computes (fp32 intermediate):
    # 1. Load k as float: k_shared[i] = __bfloat162float(k_all[i])
    # 2. Compute norm in fp32: sum += k_shared[i]^2, then sqrt
    # 3. Divide in fp32: k_shared[i] /= norm
    k_as_float = k_py_flat.float()  # convert bf16 to float like CUDA does
    k_norm_fp32 = k_as_float.norm().item()
    k_normed_fp32 = k_as_float / (k_norm_fp32 + 1e-6)

    print(f"Python ||k|| (bf16): {k_py_flat.norm().item():.6f}")
    print(f"CUDA-style ||k|| (fp32): {k_norm_fp32:.6f}")

    print(f"\nPython k_norm[:5] (bf16): {k_norm_py_flat[:5].float()}")
    print(f"CUDA-style k_norm[:5] (fp32): {k_normed_fp32[:5]}")

    diff = (k_norm_py_flat.float() - k_normed_fp32).abs()
    print(f"k_norm diff (bf16 vs fp32): max={diff.max():.8f}")

    # Now compute what the outer product should be with fp32 normalization
    v_as_float = k_py_flat.float()  # v = original k
    delta_fp32 = v_as_float  # retrieved is 0
    outer_fp32 = torch.outer(delta_fp32, k_normed_fp32)
    S_fp32 = torch.tanh(outer_fp32)

    print(f"\nPython S[0,0,:5] (bf16 all the way): {S_new_py[0, 0, :5].float()}")
    print(f"Expected S[0,0,:5] (fp32 intermediate): {S_fp32[0, :5]}")
    print(f"CUDA S[0,0,:5]: {S_final_cuda[0, 0, :5]}")

    # CUDA stores output as bf16, so convert fp32 expected to bf16 then back
    S_fp32_rounded = S_fp32.to(torch.bfloat16).float()

    diff_py_fp32 = (S_new_py[0].float() - S_fp32).abs()
    diff_cuda_fp32 = (S_final_cuda[0] - S_fp32).abs()
    diff_cuda_rounded = (S_final_cuda[0] - S_fp32_rounded).abs()

    print(f"\nPython (bf16) vs fp32 expected: max_diff={diff_py_fp32.max():.8f}")
    print(f"CUDA vs fp32 expected: max_diff={diff_cuda_fp32.max():.8f}")
    print(f"CUDA vs fp32-rounded-to-bf16: max_diff={diff_cuda_rounded.max():.8f}")

    if diff_cuda_rounded.max() < 1e-6:
        print("\n*** CUDA matches fp32 computation (after bf16 rounding)! ***")
        print("Forward pass is CORRECT. Diff from Python is bf16 vs fp32 precision.")
    else:
        # Find where the diff comes from
        idx = diff_cuda_rounded.argmax()
        i, j = idx // n_state, idx % n_state
        print(f"\n*** BUG: CUDA doesn't match. Max diff at [{i},{j}]:")
        print(f"    CUDA: {S_final_cuda[0, i, j]:.8f}")
        print(f"    fp32 rounded: {S_fp32_rounded[i, j]:.8f}")
        print(f"    fp32 exact: {S_fp32[i, j]:.8f}")

if __name__ == '__main__':
    test_single_step()
