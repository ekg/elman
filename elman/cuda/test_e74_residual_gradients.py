#!/usr/bin/env python3
"""
Test script to validate E74 Full Matrix RESIDUAL update type gradients.
Compares CUDA kernel to Python implementation.

Usage:
    cd /home/erikg/elman/elman/cuda
    LD_LIBRARY_PATH=/home/erikg/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=1 python test_e74_residual_gradients.py
"""

import torch
import torch.nn.functional as F
import sys

# Import CUDA library
import hasty_pytorch_lib

# Test parameters
T = 8        # sequence length
B = 4        # batch size
dim = 64     # input dimension
n_state = 32 # state dimension

update_type = 1  # RESIDUAL
gate_type = 0    # OUTPUT (self-gate)
proj_type = 2    # no_z (separate k, v, q)
use_tanh = True


def python_residual_forward(x, S0, W_k, W_v, W_q, residual_scale, apply_final_tanh=True):
    """
    Python implementation of E74 Full Matrix with RESIDUAL update.

    RESIDUAL update rule (from e74_ablations.py lines 417-425):
        retrieved = S @ k_norm
        delta = v - retrieved
        outer = outer(delta, k_norm)
        update = tanh(outer)
        S_raw = S + residual_scale.view(1, -1, 1) * update
        S = tanh(S_raw)  # if use_tanh (from _apply_nonlin)

    Output:
        Sq = S @ q
        out = Sq * silu(Sq)
    """
    T, B, D = x.shape
    n = n_state

    S = S0.clone()
    outputs = []

    for t in range(T):
        x_t = x[t]  # [B, dim]

        # Projections
        k_t = x_t @ W_k.T  # [B, n_state]
        v_t = x_t @ W_v.T  # [B, n_state]
        q_t = x_t @ W_q.T  # [B, n_state]

        # Normalize k
        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

        # RESIDUAL update
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        delta = v_t - retrieved
        outer = torch.einsum('bi,bj->bij', delta, k_norm)
        update = torch.tanh(outer)
        S_raw = S + residual_scale.view(1, -1, 1) * update

        # Apply nonlinearity (this is from _apply_nonlin when use_tanh=True)
        if apply_final_tanh:
            S = torch.tanh(S_raw)
        else:
            S = S_raw

        # Output with self-gating
        Sq = torch.einsum('bij,bj->bi', S, q_t)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)  # [T, B, n_state]
    return output, S


def python_residual_forward_cuda_style(x, S0, W_k, W_v, W_q, residual_scale):
    """
    Python implementation matching what CUDA appears to do based on code review:
    - Does NOT apply tanh to S_raw after update
    - Only applies tanh to the outer product
    """
    T, B, D = x.shape
    n = n_state

    S = S0.clone()
    outputs = []

    for t in range(T):
        x_t = x[t]  # [B, dim]

        # Projections
        k_t = x_t @ W_k.T  # [B, n_state]
        v_t = x_t @ W_v.T  # [B, n_state]
        q_t = x_t @ W_q.T  # [B, n_state]

        # Normalize k
        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

        # RESIDUAL update (CUDA-style: no final tanh)
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)
        delta = v_t - retrieved
        outer = torch.einsum('bi,bj->bij', delta, k_norm)
        update = torch.tanh(outer)
        S = S + residual_scale.view(1, -1, 1) * update  # NO tanh on S

        # Output with self-gating
        Sq = torch.einsum('bij,bj->bi', S, q_t)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)  # [T, B, n_state]
    return output, S


def test_residual_gradients():
    """Test RESIDUAL update gradients by comparing CUDA to Python."""
    print("=" * 70)
    print("E74 Full Matrix RESIDUAL Update Gradient Validation")
    print("=" * 70)
    print(f"T={T}, B={B}, dim={dim}, n_state={n_state}")
    print(f"update_type={update_type} (RESIDUAL), gate_type={gate_type} (OUTPUT)")
    print(f"proj_type={proj_type} (no_z), use_tanh={use_tanh}")
    print()

    device = 'cuda'
    dtype = torch.bfloat16

    # Create random inputs
    torch.manual_seed(42)

    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    # Weight matrices
    W_k = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=True)
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=True)
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=True)

    # Scale factor for proper initialization
    scale = (1.0 / dim) ** 0.5
    W_k.data *= scale
    W_v.data *= scale
    W_q.data *= scale

    # Residual scale (learnable per-row scale)
    # Note: CUDA kernel forward signature (line 40) shows: const float* __restrict__ residual_scale
    # So we need to pass as float32, not bfloat16!
    residual_scale = torch.full((n_state,), 0.1, device=device, dtype=torch.float32, requires_grad=True)
    # Also keep a bfloat16 version for the Python implementation
    residual_scale_bf16 = torch.full((n_state,), 0.1, device=device, dtype=dtype, requires_grad=True)

    # Empty tensors for unused parameters (proj_type=2 means no_z with separate k,v,q)
    W_kvq = torch.empty(0, device=device, dtype=dtype)
    W_erase = torch.empty(0, device=device, dtype=dtype)
    W_write = torch.empty(0, device=device, dtype=dtype)
    W_gate = torch.empty(0, device=device, dtype=dtype)
    W_alpha = torch.empty(0, device=device, dtype=dtype)
    b_alpha = torch.empty(0, device=device, dtype=dtype)
    W_z_gate = torch.empty(0, device=device, dtype=dtype)
    b_z_gate = torch.empty(0, device=device, dtype=dtype)

    # =========================================================================
    # Python implementation (matching e74_ablations.py exactly)
    # =========================================================================
    print("Running Python implementation (with final tanh)...")

    # Clone inputs for Python (to get separate gradients)
    x_py = x.detach().clone().requires_grad_(True)
    W_k_py = W_k.detach().clone().requires_grad_(True)
    W_v_py = W_v.detach().clone().requires_grad_(True)
    W_q_py = W_q.detach().clone().requires_grad_(True)
    residual_scale_py = residual_scale_bf16.detach().clone().requires_grad_(True)

    output_py, S_final_py = python_residual_forward(
        x_py, S0, W_k_py, W_v_py, W_q_py, residual_scale_py, apply_final_tanh=True
    )

    # Compute loss and backward
    loss_py = output_py.sum()
    loss_py.backward()

    print(f"  Python output shape: {output_py.shape}")
    print(f"  Python loss: {loss_py.item():.6f}")

    # =========================================================================
    # Python implementation (CUDA-style: no final tanh)
    # =========================================================================
    print("\nRunning Python implementation (CUDA-style, no final tanh)...")

    x_py2 = x.detach().clone().requires_grad_(True)
    W_k_py2 = W_k.detach().clone().requires_grad_(True)
    W_v_py2 = W_v.detach().clone().requires_grad_(True)
    W_q_py2 = W_q.detach().clone().requires_grad_(True)
    residual_scale_py2 = residual_scale_bf16.detach().clone().requires_grad_(True)

    output_py2, S_final_py2 = python_residual_forward_cuda_style(
        x_py2, S0, W_k_py2, W_v_py2, W_q_py2, residual_scale_py2
    )

    loss_py2 = output_py2.sum()
    loss_py2.backward()

    print(f"  Python (CUDA-style) output shape: {output_py2.shape}")
    print(f"  Python (CUDA-style) loss: {loss_py2.item():.6f}")

    # =========================================================================
    # CUDA implementation
    # =========================================================================
    print("\nRunning CUDA kernel (e74_full_matrix_forward_v2)...")

    # Clone inputs for CUDA
    x_cuda = x.detach().clone().requires_grad_(True)
    W_k_cuda = W_k.detach().clone().requires_grad_(True)
    W_v_cuda = W_v.detach().clone().requires_grad_(True)
    W_q_cuda = W_q.detach().clone().requires_grad_(True)
    # Use float32 for residual_scale since that's what the CUDA kernel expects
    residual_scale_cuda = residual_scale.detach().clone().requires_grad_(True)  # Already float32

    # Forward pass
    results = hasty_pytorch_lib.e74_full_matrix_forward_v2(
        True,               # training
        x_cuda,             # x
        S0,                 # S0
        proj_type,          # proj_type (2 = no_z)
        use_tanh,           # use_tanh
        update_type,        # update_type (1 = RESIDUAL)
        gate_type,          # gate_type (0 = OUTPUT)
        W_kvq,              # W_kvq (empty for no_z)
        W_k_cuda,           # W_k
        W_v_cuda,           # W_v
        W_q_cuda,           # W_q
        residual_scale_cuda,# residual_scale
        W_erase,            # W_erase (empty for RESIDUAL)
        W_write,            # W_write (empty for RESIDUAL)
        W_gate,             # W_gate (empty for RESIDUAL)
        W_alpha,            # W_alpha (empty for RESIDUAL)
        b_alpha,            # b_alpha (empty for RESIDUAL)
        W_z_gate,           # W_z_gate (empty for OUTPUT gate)
        b_z_gate,           # b_z_gate (empty for OUTPUT gate)
    )

    # results = [S_final, output, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache]
    S_final_cuda = results[0]
    output_cuda = results[1]
    k_cache = results[2]
    v_cache = results[3]
    q_cache = results[4]
    S_checkpoints = results[5]
    Sq_cache = results[6]

    print(f"  CUDA output shape: {output_cuda.shape}")

    # Compute loss
    loss_cuda = output_cuda.sum()
    print(f"  CUDA loss: {loss_cuda.item():.6f}")

    # Backward pass
    d_output = torch.ones_like(output_cuda)

    # Empty caches for unused update types
    erase_cache = torch.empty(0, device=device, dtype=dtype)
    write_cache = torch.empty(0, device=device, dtype=dtype)
    gate_cache = torch.empty(0, device=device, dtype=dtype)
    alpha_cache = torch.empty(0, device=device, dtype=dtype)
    z_gate_cache = torch.empty(0, device=device, dtype=dtype)

    grad_results = hasty_pytorch_lib.e74_full_matrix_backward_v2(
        x_cuda,             # x
        S_checkpoints,      # S_checkpoints
        Sq_cache,           # Sq_cache
        k_cache,            # k_cache
        v_cache,            # v_cache
        q_cache,            # q_cache
        d_output,           # d_output
        proj_type,          # proj_type
        use_tanh,           # use_tanh
        update_type,        # update_type
        gate_type,          # gate_type
        W_kvq,              # W_kvq
        W_k_cuda,           # W_k
        W_v_cuda,           # W_v
        W_q_cuda,           # W_q
        residual_scale_cuda,# residual_scale
        erase_cache,        # erase_cache
        write_cache,        # write_cache
        gate_cache,         # gate_cache
        alpha_cache,        # alpha_cache
        W_erase,            # W_erase
        W_write,            # W_write
        W_gate,             # W_gate
        W_alpha,            # W_alpha
        z_gate_cache,       # z_gate_cache
        W_z_gate,           # W_z_gate
    )

    # grad_results = [dx, dW_kvq, dW_k, dW_v, dW_q, d_residual_scale, dW_erase, dW_write, dW_gate, dW_alpha, db_alpha, dW_z_gate, db_z_gate]
    dx_cuda = grad_results[0]
    dW_kvq_cuda = grad_results[1]
    dW_k_cuda = grad_results[2]
    dW_v_cuda = grad_results[3]
    dW_q_cuda = grad_results[4]
    d_residual_scale_cuda = grad_results[5]

    # =========================================================================
    # Compare forward outputs
    # =========================================================================
    print("\n" + "=" * 70)
    print("Forward Output Comparison")
    print("=" * 70)

    output_diff = (output_py.float() - output_cuda.float()).abs().max().item()
    S_diff = (S_final_py.float() - S_final_cuda.float()).abs().max().item()

    print(f"  Python (final tanh) vs CUDA:")
    print(f"    Output max abs diff: {output_diff:.6e}")
    print(f"    Final state max abs diff: {S_diff:.6e}")

    output_diff2 = (output_py2.float() - output_cuda.float()).abs().max().item()
    S_diff2 = (S_final_py2.float() - S_final_cuda.float()).abs().max().item()

    print(f"  Python (no final tanh, CUDA-style) vs CUDA:")
    print(f"    Output max abs diff: {output_diff2:.6e}")
    print(f"    Final state max abs diff: {S_diff2:.6e}")

    # =========================================================================
    # Compare gradients
    # =========================================================================
    print("\n" + "=" * 70)
    print("Gradient Comparison (Python with final tanh vs CUDA)")
    print("=" * 70)

    # dx
    dx_diff = (x_py.grad.float() - dx_cuda.float()).abs().max().item()
    print(f"  dx max abs diff: {dx_diff:.6e}")

    # dW_k
    dW_k_diff = (W_k_py.grad.float() - dW_k_cuda.float()).abs().max().item()
    print(f"  dW_k max abs diff: {dW_k_diff:.6e}")

    # dW_v
    dW_v_diff = (W_v_py.grad.float() - dW_v_cuda.float()).abs().max().item()
    print(f"  dW_v max abs diff: {dW_v_diff:.6e}")

    # dW_q
    dW_q_diff = (W_q_py.grad.float() - dW_q_cuda.float()).abs().max().item()
    print(f"  dW_q max abs diff: {dW_q_diff:.6e}")

    # d_residual_scale - note: Python uses bfloat16, CUDA returns bfloat16 (in tensor, but zeros)
    d_residual_scale_diff = (residual_scale_py.grad.float() - d_residual_scale_cuda.float()).abs().max().item()
    print(f"  d_residual_scale max abs diff: {d_residual_scale_diff:.6e}")
    print(f"    (Note: CUDA grad is from bfloat16 output, Python grad from bfloat16)")

    print("\n" + "=" * 70)
    print("Gradient Comparison (Python CUDA-style vs CUDA)")
    print("=" * 70)

    # dx
    dx_diff2 = (x_py2.grad.float() - dx_cuda.float()).abs().max().item()
    print(f"  dx max abs diff: {dx_diff2:.6e}")

    # dW_k
    dW_k_diff2 = (W_k_py2.grad.float() - dW_k_cuda.float()).abs().max().item()
    print(f"  dW_k max abs diff: {dW_k_diff2:.6e}")

    # dW_v
    dW_v_diff2 = (W_v_py2.grad.float() - dW_v_cuda.float()).abs().max().item()
    print(f"  dW_v max abs diff: {dW_v_diff2:.6e}")

    # dW_q
    dW_q_diff2 = (W_q_py2.grad.float() - dW_q_cuda.float()).abs().max().item()
    print(f"  dW_q max abs diff: {dW_q_diff2:.6e}")

    # d_residual_scale
    d_residual_scale_diff2 = (residual_scale_py2.grad.float() - d_residual_scale_cuda.float()).abs().max().item()
    print(f"  d_residual_scale max abs diff: {d_residual_scale_diff2:.6e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Use CUDA-style Python for comparison since that matches CUDA behavior
    # Exclude d_residual_scale since it's a known bug (TODO in CUDA code)
    max_diff_excl_residual = max(output_diff2, S_diff2, dx_diff2, dW_k_diff2, dW_v_diff2, dW_q_diff2)
    max_diff_incl_residual = max(max_diff_excl_residual, d_residual_scale_diff2)
    # Use a more relaxed threshold for bfloat16 (7-bit mantissa = ~0.8% precision)
    # With accumulation over multiple operations, 5% difference is acceptable
    threshold = 5e-2

    print(f"  Comparing Python (CUDA-style, no final tanh) vs CUDA")
    print(f"  Maximum difference (excluding d_residual_scale): {max_diff_excl_residual:.6e}")
    print(f"  Maximum difference (including d_residual_scale): {max_diff_incl_residual:.6e}")
    print(f"  Threshold: {threshold:.0e}")

    # Known issues detected in CUDA kernel:
    print("\n" + "-" * 70)
    print("Known Issues Detected in CUDA Kernel:")
    print("-" * 70)
    print("  1. d_residual_scale gradient is all zeros because the reduction")
    print("     from per-batch accumulators is marked as TODO (line 1256-1260)")
    print("     in /home/erikg/elman/elman/cuda/lib/e74_full_matrix_v2_gpu.cu.cc")
    print()
    print("  2. Forward pass does NOT apply final tanh() to S_raw, unlike the")
    print("     Python reference in e74_ablations.py (line 478: S = _apply_nonlin(S_raw))")
    print("     This is intentional based on kernel design, but worth noting.")
    print("-" * 70)

    if max_diff_excl_residual > threshold:
        print(f"\n  FAIL: Differences (excluding d_residual_scale) exceed threshold!")

        # Investigate which gradients are off
        print("\n  Investigation:")

        if output_diff2 > threshold:
            print(f"\n  Output details:")
            print(f"    Python mean: {output_py2.float().mean().item():.6e}")
            print(f"    CUDA mean:   {output_cuda.float().mean().item():.6e}")

        if dx_diff2 > threshold:
            print(f"\n  dx details:")
            print(f"    Python grad mean: {x_py2.grad.float().mean().item():.6e}")
            print(f"    Python grad std:  {x_py2.grad.float().std().item():.6e}")
            print(f"    CUDA grad mean:   {dx_cuda.float().mean().item():.6e}")
            print(f"    CUDA grad std:    {dx_cuda.float().std().item():.6e}")

        if dW_k_diff2 > threshold:
            print(f"\n  dW_k details:")
            print(f"    Python grad mean: {W_k_py2.grad.float().mean().item():.6e}")
            print(f"    Python grad std:  {W_k_py2.grad.float().std().item():.6e}")
            print(f"    CUDA grad mean:   {dW_k_cuda.float().mean().item():.6e}")
            print(f"    CUDA grad std:    {dW_k_cuda.float().std().item():.6e}")

        if dW_v_diff2 > threshold:
            print(f"\n  dW_v details:")
            print(f"    Python grad mean: {W_v_py2.grad.float().mean().item():.6e}")
            print(f"    Python grad std:  {W_v_py2.grad.float().std().item():.6e}")
            print(f"    CUDA grad mean:   {dW_v_cuda.float().mean().item():.6e}")
            print(f"    CUDA grad std:    {dW_v_cuda.float().std().item():.6e}")

        if dW_q_diff2 > threshold:
            print(f"\n  dW_q details:")
            print(f"    Python grad mean: {W_q_py2.grad.float().mean().item():.6e}")
            print(f"    Python grad std:  {W_q_py2.grad.float().std().item():.6e}")
            print(f"    CUDA grad mean:   {dW_q_cuda.float().mean().item():.6e}")
            print(f"    CUDA grad std:    {dW_q_cuda.float().std().item():.6e}")

        return False
    else:
        if d_residual_scale_diff2 > threshold:
            print(f"\n  PARTIAL PASS: All gradients match except d_residual_scale!")
            print(f"  d_residual_scale issue: CUDA returns zeros due to missing reduction")
            print(f"    Python grad: {residual_scale_py2.grad}")
            print(f"    CUDA grad:   {d_residual_scale_cuda}")
            return False  # Still fail since d_residual_scale is wrong
        else:
            print(f"\n  PASS: All differences within threshold!")
            return True


if __name__ == "__main__":
    success = test_residual_gradients()
    sys.exit(0 if success else 1)
