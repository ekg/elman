#!/usr/bin/env python3
"""
E74 Full Matrix EMA Gradient Validation

Compares CUDA kernel e74_full_matrix_forward_v2 (update_type=4) with
Python E74FullMatrixCell (UpdateType.EMA) implementation.

EMA update rule (Python):
    alpha = sigmoid(x[t] @ W_alpha.T + b_alpha)  # [B, n]
    outer = einsum('bi,bj->bij', v_t, k_norm)     # outer(v, k)
    S_raw = alpha.unsqueeze(-1) * S + (1.0 - alpha).unsqueeze(-1) * outer
    S = tanh(S_raw)

EMA update rule (CUDA kernel - from e74_full_matrix_v2_gpu.cu.cc):
    alpha_val = extra_shared[row];  // assumes sigmoid already applied!
    outer_val = v_shared[row] * k_shared[col];
    S_shared[i] = alpha_val * S_shared[i] + (1.0f - alpha_val) * outer_val;
    S_shared[i] = tanhf(S_shared[i]);

NOTE: CUDA forward computes alpha_cache = W_alpha @ x but does NOT add b_alpha
or apply sigmoid. This is a BUG in the CUDA kernel - the kernel expects sigmoid
already applied but the driver code doesn't do it.

Parameters:
- T=8, B=4, dim=64, n_state=32
- update_type=4 (EMA), gate_type=0 (OUTPUT), proj_type=2 (no_z), use_tanh=True
"""

import os
import sys

# Add the parent paths
sys.path.insert(0, '/home/erikg/elman')

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CUDA kernel
import hasty_pytorch_lib as cuda_lib

# Import Python implementation
from elman.models.e74_ablations import (
    E74FullMatrixCell, UpdateType, ProjType, NonlinType, GateType
)


def python_ema_forward(x, S0, W_k, W_v, W_q, W_alpha, b_alpha):
    """
    Pure Python implementation of E74 Full Matrix EMA forward pass.

    Args:
        x: [T, B, dim] input
        S0: [B, n, n] initial state
        W_k, W_v, W_q: [n, dim] projection weights
        W_alpha: [n, dim] alpha projection weight
        b_alpha: [n] alpha bias

    Returns:
        output: [T, B, n]
        S: [B, n, n] final state
    """
    T, B, dim = x.shape
    n = W_k.shape[0]

    S = S0.clone()
    outputs = []

    for t in range(T):
        x_t = x[t]  # [B, dim]

        # Projections
        k_t = x_t @ W_k.T  # [B, n]
        v_t = x_t @ W_v.T  # [B, n]
        q_t = x_t @ W_q.T  # [B, n]

        # Normalize k
        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

        # EMA update
        alpha = torch.sigmoid(x_t @ W_alpha.T + b_alpha)  # [B, n]
        outer = torch.einsum('bi,bj->bij', v_t, k_norm)  # [B, n, n]
        S_raw = alpha.unsqueeze(-1) * S + (1.0 - alpha).unsqueeze(-1) * outer

        # Nonlinearity
        S = torch.tanh(S_raw)

        # Output with self-gating
        Sq = torch.einsum('bij,bj->bi', S, q_t)  # [B, n]
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)  # [T, B, n]
    return output, S


def run_cuda_forward(x, S0, W_k, W_v, W_q, W_alpha, b_alpha, training=True):
    """
    Run CUDA kernel e74_full_matrix_forward_v2 with update_type=4 (EMA).

    Returns: (S_final, output, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache)
    """
    n_state = W_k.shape[0]
    dim = W_k.shape[1]

    # Create empty tensors for unused parameters
    empty = torch.empty(0, dtype=x.dtype, device=x.device)

    results = cuda_lib.e74_full_matrix_forward_v2(
        training,           # training
        x,                  # x: [T, B, dim]
        S0,                 # S0: [B, n_state, n_state]
        2,                  # proj_type=2 (no_z: separate k, v, q)
        True,               # use_tanh
        4,                  # update_type=4 (EMA)
        0,                  # gate_type=0 (output)
        empty,              # W_kvq (not used for proj_type=2)
        W_k,                # W_k: [n_state, dim]
        W_v,                # W_v: [n_state, dim]
        W_q,                # W_q: [n_state, dim]
        empty,              # residual_scale (not used for EMA)
        empty,              # W_erase (not used for EMA)
        empty,              # W_write (not used for EMA)
        empty,              # W_gate (not used for EMA)
        W_alpha,            # W_alpha: [n_state, dim]
        b_alpha,            # b_alpha: [n_state]
        empty,              # W_z_gate (not used for gate_type=0)
        empty,              # b_z_gate (not used for gate_type=0)
    )

    return results


def run_cuda_backward(x, results, d_output, W_k, W_v, W_q, W_alpha):
    """
    Run CUDA kernel e74_full_matrix_backward_v2 for EMA update.

    Returns gradients for: dx, dW_k, dW_v, dW_q, dW_alpha, db_alpha
    """
    S_checkpoints = results[5]
    Sq_cache = results[6]
    k_cache = results[2]
    v_cache = results[3]
    q_cache = results[4]

    n_state = W_k.shape[0]
    dim = W_k.shape[1]
    T = x.shape[0]
    B = x.shape[1]

    empty = torch.empty(0, dtype=x.dtype, device=x.device)

    # For EMA update_type=4, alpha_cache should be extracted from S_cache
    # The forward stored: checkpoints + sq_cache + alpha_cache
    # Let's compute alpha_cache ourselves for the backward
    alpha_cache = torch.empty(T, B, n_state, dtype=x.dtype, device=x.device)
    for t in range(T):
        alpha_cache[t] = torch.sigmoid(x[t] @ W_alpha.T)  # bias is baked into forward

    grad_results = cuda_lib.e74_full_matrix_backward_v2(
        x,                  # x: [T, B, dim]
        S_checkpoints,      # S_checkpoints
        Sq_cache,           # Sq_cache
        k_cache,            # k_cache
        v_cache,            # v_cache
        q_cache,            # q_cache
        d_output,           # d_output: [T, B, n_state]
        2,                  # proj_type=2 (no_z)
        True,               # use_tanh
        4,                  # update_type=4 (EMA)
        0,                  # gate_type=0 (output)
        empty,              # W_kvq
        W_k,                # W_k
        W_v,                # W_v
        W_q,                # W_q
        empty,              # residual_scale
        empty,              # erase_cache
        empty,              # write_cache
        empty,              # gate_cache
        alpha_cache,        # alpha_cache
        empty,              # W_erase
        empty,              # W_write
        empty,              # W_gate
        W_alpha,            # W_alpha
        empty,              # z_gate_cache
        empty,              # W_z_gate
    )

    # Results: [dx, dW_kvq, dW_k, dW_v, dW_q, d_residual_scale, dW_erase, dW_write,
    #           dW_gate_out, dW_alpha, db_alpha, dW_z_gate, db_z_gate]
    dx = grad_results[0]
    dW_k = grad_results[2]
    dW_v = grad_results[3]
    dW_q = grad_results[4]
    dW_alpha = grad_results[9]
    db_alpha = grad_results[10]

    return dx, dW_k, dW_v, dW_q, dW_alpha, db_alpha


def test_forward():
    """Test forward pass comparison."""
    print("=" * 60)
    print("E74 Full Matrix EMA Forward Pass Validation")
    print("=" * 60)

    # Parameters
    T, B, dim, n_state = 8, 4, 64, 32
    device = 'cuda'
    dtype = torch.bfloat16

    # Create random inputs
    torch.manual_seed(42)
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.1

    # Create weights
    W_k = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_alpha = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    b_alpha = torch.full((n_state,), 2.0, device=device, dtype=dtype)  # Bias toward preserve

    print(f"\nConfiguration:")
    print(f"  T={T}, B={B}, dim={dim}, n_state={n_state}")
    print(f"  dtype={dtype}, device={device}")
    print(f"  update_type=4 (EMA), gate_type=0 (OUTPUT), proj_type=2 (no_z)")
    print(f"  use_tanh=True")

    # Run Python implementation
    print("\nRunning Python implementation...")
    py_output, py_S_final = python_ema_forward(
        x.clone(), S0.clone(), W_k.clone(), W_v.clone(), W_q.clone(),
        W_alpha.clone(), b_alpha.clone()
    )

    # Run CUDA implementation
    print("Running CUDA implementation...")
    cuda_results = run_cuda_forward(
        x.clone(), S0.clone(), W_k.clone(), W_v.clone(), W_q.clone(),
        W_alpha.clone(), b_alpha.clone(), training=True
    )
    cuda_S_final = cuda_results[0]
    cuda_output = cuda_results[1]

    # Compare outputs
    output_diff = (py_output.float() - cuda_output.float()).abs()
    output_max_diff = output_diff.max().item()
    output_mean_diff = output_diff.mean().item()

    state_diff = (py_S_final.float() - cuda_S_final.float()).abs()
    state_max_diff = state_diff.max().item()
    state_mean_diff = state_diff.mean().item()

    print(f"\nForward Pass Results:")
    print(f"  Output max diff:  {output_max_diff:.6e}")
    print(f"  Output mean diff: {output_mean_diff:.6e}")
    print(f"  State max diff:   {state_max_diff:.6e}")
    print(f"  State mean diff:  {state_mean_diff:.6e}")

    # Check threshold
    threshold = 1e-2
    forward_pass = output_max_diff < threshold and state_max_diff < threshold

    if forward_pass:
        print(f"\n[PASS] Forward pass matches (threshold={threshold})")
    else:
        print(f"\n[FAIL] Forward pass mismatch!")
        if output_max_diff >= threshold:
            print(f"  Output max diff {output_max_diff:.6e} >= threshold {threshold}")
        if state_max_diff >= threshold:
            print(f"  State max diff {state_max_diff:.6e} >= threshold {threshold}")

    return forward_pass, (x, S0, W_k, W_v, W_q, W_alpha, b_alpha, cuda_results, py_output)


def test_backward(data):
    """Test backward pass comparison using autograd."""
    print("\n" + "=" * 60)
    print("E74 Full Matrix EMA Backward Pass Validation")
    print("=" * 60)

    x, S0, W_k, W_v, W_q, W_alpha, b_alpha, cuda_results, py_output_ref = data
    T, B, dim = x.shape
    n_state = W_k.shape[0]
    device = x.device
    dtype = x.dtype

    # Create random gradient
    torch.manual_seed(123)
    d_output = torch.randn_like(py_output_ref) * 0.1

    # Python backward via autograd
    print("\nRunning Python backward (autograd)...")
    x_py = x.clone().requires_grad_(True)
    S0_py = S0.clone()  # Don't need grad for S0 typically
    W_k_py = W_k.clone().requires_grad_(True)
    W_v_py = W_v.clone().requires_grad_(True)
    W_q_py = W_q.clone().requires_grad_(True)
    W_alpha_py = W_alpha.clone().requires_grad_(True)
    b_alpha_py = b_alpha.clone().requires_grad_(True)

    py_output, _ = python_ema_forward(
        x_py, S0_py, W_k_py, W_v_py, W_q_py, W_alpha_py, b_alpha_py
    )

    # Compute gradients
    py_output.backward(d_output)

    py_dx = x_py.grad.clone()
    py_dW_k = W_k_py.grad.clone()
    py_dW_v = W_v_py.grad.clone()
    py_dW_q = W_q_py.grad.clone()
    py_dW_alpha = W_alpha_py.grad.clone()
    py_db_alpha = b_alpha_py.grad.clone()

    # CUDA backward
    print("Running CUDA backward...")
    try:
        cuda_dx, cuda_dW_k, cuda_dW_v, cuda_dW_q, cuda_dW_alpha, cuda_db_alpha = run_cuda_backward(
            x, cuda_results, d_output.contiguous(), W_k, W_v, W_q, W_alpha
        )
    except Exception as e:
        print(f"[ERROR] CUDA backward failed: {e}")
        return False

    # Compare gradients
    print("\nGradient Comparisons:")
    threshold = 1e-2
    all_pass = True

    def compare_grads(name, py_grad, cuda_grad):
        nonlocal all_pass
        if cuda_grad.numel() == 0:
            print(f"  {name}: CUDA grad empty, skipping")
            return
        diff = (py_grad.float() - cuda_grad.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = max_diff / (py_grad.abs().max().item() + 1e-8)
        status = "PASS" if max_diff < threshold else "FAIL"
        print(f"  {name}:")
        print(f"    max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}, rel diff: {rel_diff:.6e} [{status}]")
        if max_diff >= threshold:
            all_pass = False

    compare_grads("dx", py_dx, cuda_dx)
    compare_grads("dW_k", py_dW_k, cuda_dW_k)
    compare_grads("dW_v", py_dW_v, cuda_dW_v)
    compare_grads("dW_q", py_dW_q, cuda_dW_q)
    compare_grads("dW_alpha", py_dW_alpha, cuda_dW_alpha)
    compare_grads("db_alpha", py_db_alpha, cuda_db_alpha)

    if all_pass:
        print(f"\n[PASS] Backward pass matches (threshold={threshold})")
    else:
        print(f"\n[FAIL] Backward pass has mismatches!")

    return all_pass


def test_e74_cell_comparison():
    """Test using the actual E74FullMatrixCell class."""
    print("\n" + "=" * 60)
    print("E74FullMatrixCell vs CUDA Comparison")
    print("=" * 60)

    T, B, dim, n_state = 8, 4, 64, 32
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)

    # Create Python cell
    cell = E74FullMatrixCell(
        dim=dim,
        n_state=n_state,
        proj_type=ProjType.NO_Z,
        nonlin_type=NonlinType.TANH,
        gate_type=GateType.OUTPUT,
        update_type=UpdateType.EMA,
    ).to(device).to(dtype)

    # Extract weights
    W_k = cell.W_k.data.clone()
    W_v = cell.W_v.data.clone()
    W_q = cell.W_q.data.clone()
    W_alpha = cell.W_alpha.data.clone()
    b_alpha = cell.b_alpha.data.clone()

    # Create inputs
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.zeros(B, n_state, n_state, device=device, dtype=dtype)

    print(f"\nConfiguration:")
    print(f"  T={T}, B={B}, dim={dim}, n_state={n_state}")
    print(f"  Cell: E74FullMatrixCell with UpdateType.EMA")

    # Run cell forward
    print("\nRunning E74FullMatrixCell forward...")
    cell_output, cell_S = cell(x.clone(), S0.clone())

    # Run CUDA forward
    print("Running CUDA forward...")
    cuda_results = run_cuda_forward(
        x.clone(), S0.clone(), W_k, W_v, W_q, W_alpha, b_alpha, training=True
    )
    cuda_output = cuda_results[1]
    cuda_S = cuda_results[0]

    # Compare
    output_diff = (cell_output.float() - cuda_output.float()).abs()
    output_max = output_diff.max().item()
    output_mean = output_diff.mean().item()

    state_diff = (cell_S.float() - cuda_S.float()).abs()
    state_max = state_diff.max().item()
    state_mean = state_diff.mean().item()

    print(f"\nResults:")
    print(f"  Output max diff:  {output_max:.6e}")
    print(f"  Output mean diff: {output_mean:.6e}")
    print(f"  State max diff:   {state_max:.6e}")
    print(f"  State mean diff:  {state_mean:.6e}")

    threshold = 1e-2
    passed = output_max < threshold and state_max < threshold

    if passed:
        print(f"\n[PASS] E74FullMatrixCell matches CUDA (threshold={threshold})")
    else:
        print(f"\n[FAIL] E74FullMatrixCell mismatch with CUDA!")

    return passed


def python_ema_forward_cuda_style(x, S0, W_k, W_v, W_q, W_alpha, b_alpha):
    """
    Python implementation that matches what the CUDA kernel ACTUALLY does.

    The CUDA kernel BUG: it computes alpha_cache = W_alpha @ x but does NOT
    add b_alpha or apply sigmoid before passing to kernel. The kernel then
    uses this raw value directly as alpha (expecting sigmoid already applied).

    This function replicates that buggy behavior for comparison.
    """
    T, B, dim = x.shape
    n = W_k.shape[0]

    S = S0.clone()
    outputs = []

    for t in range(T):
        x_t = x[t]  # [B, dim]

        # Projections
        k_t = x_t @ W_k.T  # [B, n]
        v_t = x_t @ W_v.T  # [B, n]
        q_t = x_t @ W_q.T  # [B, n]

        # Normalize k
        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

        # CUDA kernel BUG: uses raw W_alpha @ x without b_alpha or sigmoid
        # The kernel expects sigmoid-applied alpha but gets raw linear projection
        alpha = x_t @ W_alpha.T  # NO + b_alpha, NO sigmoid

        # EMA update (same as CUDA kernel)
        outer = torch.einsum('bi,bj->bij', v_t, k_norm)  # [B, n, n]
        S_raw = alpha.unsqueeze(-1) * S + (1.0 - alpha).unsqueeze(-1) * outer

        # Nonlinearity
        S = torch.tanh(S_raw)

        # Output with self-gating
        Sq = torch.einsum('bij,bj->bi', S, q_t)  # [B, n]
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)  # [T, B, n]
    return output, S


def test_single_step_debug():
    """
    Debug single timestep to find exact mismatch point.
    """
    print("\n" + "=" * 60)
    print("Single Step Debug")
    print("=" * 60)

    # NOTE: CUDA kernel only supports n_state=32 or n_state=48!
    # Using n_state=32 for this test
    T, B, dim, n_state = 1, 1, 64, 32
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.1

    W_k = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_alpha = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    b_alpha = torch.zeros((n_state,), device=device, dtype=dtype)  # Zero bias for simplicity

    print(f"\nInputs:")
    print(f"  x[0,0,:4] = {x[0, 0, :4].tolist()}")
    print(f"  S0[0,:2,:2] = {S0[0, :2, :2].tolist()}")

    # Compute projections manually
    x_t = x[0]  # [B=1, dim]
    k_t = x_t @ W_k.T  # [B, n]
    v_t = x_t @ W_v.T  # [B, n]
    q_t = x_t @ W_q.T  # [B, n]
    alpha_raw = x_t @ W_alpha.T  # [B, n]

    print(f"\nProjections:")
    print(f"  k_t = {k_t[0].tolist()}")
    print(f"  v_t = {v_t[0].tolist()}")
    print(f"  q_t = {q_t[0].tolist()}")
    print(f"  alpha_raw = {alpha_raw[0].tolist()}")

    # Normalize k
    k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)
    print(f"  k_norm = {k_norm[0].tolist()}")

    # Run CUDA
    cuda_results = run_cuda_forward(
        x.clone(), S0.clone(), W_k.clone(), W_v.clone(), W_q.clone(),
        W_alpha.clone(), b_alpha.clone(), training=True
    )
    cuda_S = cuda_results[0]
    cuda_output = cuda_results[1]

    # Check k_cache
    k_cache = cuda_results[2]
    print(f"\nCUDA caches:")
    print(f"  k_cache[0,0] = {k_cache[0, 0].tolist()}")

    # Python with raw alpha (no sigmoid)
    S = S0.clone()
    alpha = alpha_raw  # No sigmoid
    outer = torch.einsum('bi,bj->bij', v_t, k_norm)
    S_raw = alpha.unsqueeze(-1) * S + (1.0 - alpha).unsqueeze(-1) * outer
    S = torch.tanh(S_raw)

    print(f"\nState comparison:")
    print(f"  Python S[0,:2,:2] = {S[0, :2, :2].tolist()}")
    print(f"  CUDA   S[0,:2,:2] = {cuda_S[0, :2, :2].tolist()}")
    print(f"  diff = {(S - cuda_S).abs().max().item():.6e}")

    # Output
    Sq = torch.einsum('bij,bj->bi', S, q_t)
    out = Sq * F.silu(Sq)

    print(f"\nOutput comparison:")
    print(f"  Python out = {out[0].tolist()}")
    print(f"  CUDA   out = {cuda_output[0, 0].tolist()}")
    print(f"  diff = {(out - cuda_output[0]).abs().max().item():.6e}")


def test_cuda_bug_confirmation():
    """
    Test that confirms the CUDA kernel bug:
    It computes W_alpha @ x but doesn't add b_alpha or apply sigmoid.
    """
    print("\n" + "=" * 60)
    print("CUDA Kernel Bug Confirmation Test")
    print("=" * 60)

    # First run debug on single step
    test_single_step_debug()

    T, B, dim, n_state = 8, 4, 64, 32
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    # Use smaller scale to avoid numerical issues with raw alpha
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * 0.1
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.01

    W_k = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    # Use very small W_alpha so alpha_raw is close to 0
    W_alpha = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.001
    b_alpha = torch.full((n_state,), 2.0, device=device, dtype=dtype)

    print("\n\nTest 2: Full sequence Python with CUDA-style bug (no b_alpha, no sigmoid)")
    py_bug_output, py_bug_S = python_ema_forward_cuda_style(
        x.clone(), S0.clone(), W_k.clone(), W_v.clone(), W_q.clone(),
        W_alpha.clone(), b_alpha.clone()
    )

    print("Running CUDA kernel...")
    cuda_results = run_cuda_forward(
        x.clone(), S0.clone(), W_k.clone(), W_v.clone(), W_q.clone(),
        W_alpha.clone(), b_alpha.clone(), training=True
    )
    cuda_output = cuda_results[1]
    cuda_S = cuda_results[0]

    output_diff = (py_bug_output.float() - cuda_output.float()).abs()
    output_max = output_diff.max().item()
    state_diff = (py_bug_S.float() - cuda_S.float()).abs()
    state_max = state_diff.max().item()

    print(f"\nBuggy Python vs CUDA:")
    print(f"  Output max diff: {output_max:.6e}")
    print(f"  State max diff:  {state_max:.6e}")

    threshold = 1e-2
    if output_max < threshold and state_max < threshold:
        print(f"\n[CONFIRMED] CUDA kernel matches Python with bug!")
        print("BUG: CUDA forward does NOT add b_alpha or apply sigmoid to alpha")
        return True
    else:
        print(f"\n[NOT CONFIRMED] Still mismatching - investigate further")
        return False


def test_backward_with_buggy_forward():
    """
    Test backward pass using the buggy forward implementation
    to verify gradients are at least internally consistent.
    """
    print("\n" + "=" * 60)
    print("Backward Pass Test (with buggy forward, no sigmoid/bias)")
    print("=" * 60)

    T, B, dim, n_state = 8, 4, 64, 32
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    # Use smaller scale to avoid numerical issues with raw alpha
    x = torch.randn(T, B, dim, device=device, dtype=dtype) * 0.1
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.01

    W_k = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.01
    W_alpha = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.001
    b_alpha = torch.zeros((n_state,), device=device, dtype=dtype)  # Zero bias

    # Random gradient
    torch.manual_seed(123)

    # Python backward via autograd (buggy version)
    print("\nRunning Python backward (buggy version)...")
    x_py = x.clone().requires_grad_(True)
    W_k_py = W_k.clone().requires_grad_(True)
    W_v_py = W_v.clone().requires_grad_(True)
    W_q_py = W_q.clone().requires_grad_(True)
    W_alpha_py = W_alpha.clone().requires_grad_(True)

    py_output, _ = python_ema_forward_cuda_style(
        x_py, S0.clone(), W_k_py, W_v_py, W_q_py, W_alpha_py, b_alpha.clone()
    )

    d_output = torch.randn_like(py_output) * 0.1
    py_output.backward(d_output)

    py_dx = x_py.grad.clone()
    py_dW_k = W_k_py.grad.clone()
    py_dW_v = W_v_py.grad.clone()
    py_dW_q = W_q_py.grad.clone()
    py_dW_alpha = W_alpha_py.grad.clone()

    # CUDA forward + backward
    print("Running CUDA forward + backward...")
    cuda_results = run_cuda_forward(
        x.clone(), S0.clone(), W_k.clone(), W_v.clone(), W_q.clone(),
        W_alpha.clone(), b_alpha.clone(), training=True
    )

    try:
        cuda_dx, cuda_dW_k, cuda_dW_v, cuda_dW_q, cuda_dW_alpha, cuda_db_alpha = run_cuda_backward(
            x, cuda_results, d_output.contiguous(), W_k, W_v, W_q, W_alpha
        )
    except Exception as e:
        print(f"[ERROR] CUDA backward failed: {e}")
        return False

    # Compare gradients
    print("\nGradient Comparisons (buggy forward, should match):")
    threshold = 5e-2  # Slightly looser for bfloat16
    all_pass = True

    def compare_grads(name, py_grad, cuda_grad):
        nonlocal all_pass
        if cuda_grad.numel() == 0:
            print(f"  {name}: CUDA grad empty, skipping")
            return
        diff = (py_grad.float() - cuda_grad.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        py_max = py_grad.abs().max().item()
        rel_diff = max_diff / (py_max + 1e-8)
        status = "PASS" if max_diff < threshold or rel_diff < 0.1 else "FAIL"
        print(f"  {name}:")
        print(f"    max diff: {max_diff:.6e}, mean: {mean_diff:.6e}, rel: {rel_diff:.2%} [{status}]")
        if max_diff >= threshold and rel_diff >= 0.1:
            all_pass = False

    compare_grads("dx", py_dx, cuda_dx)
    compare_grads("dW_k", py_dW_k, cuda_dW_k)
    compare_grads("dW_v", py_dW_v, cuda_dW_v)
    compare_grads("dW_q", py_dW_q, cuda_dW_q)
    compare_grads("dW_alpha", py_dW_alpha, cuda_dW_alpha)

    if all_pass:
        print(f"\n[PASS] Backward pass matches (buggy forward version)")
    else:
        print(f"\n[FAIL] Backward pass has mismatches even with buggy forward!")

    return all_pass


def main():
    print("E74 Full Matrix EMA Gradient Validation Script")
    print("=" * 60)

    # First, confirm the CUDA bug
    bug_confirmed = test_cuda_bug_confirmation()

    # Test backward with buggy forward (should pass if gradients are consistent)
    buggy_backward_pass = test_backward_with_buggy_forward()

    # Test forward pass against correct implementation (expected to fail due to bug)
    forward_pass, data = test_forward()

    # Test backward pass against correct implementation
    backward_pass = test_backward(data)

    # Test cell comparison
    cell_pass = test_e74_cell_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  CUDA bug confirmed:       {'YES' if bug_confirmed else 'NO'}")
    print(f"  Buggy backward matches:   {'PASS' if buggy_backward_pass else 'FAIL'}")
    print(f"  Forward pass (correct):   {'PASS' if forward_pass else 'FAIL (expected due to bug)'}")
    print(f"  Backward pass (correct):  {'PASS' if backward_pass else 'FAIL (expected due to bug)'}")
    print(f"  Cell match:               {'PASS' if cell_pass else 'FAIL (expected due to bug)'}")

    if bug_confirmed:
        print("\n" + "=" * 60)
        print("BUG REPORT:")
        print("=" * 60)
        print("File: /home/erikg/elman/elman/cuda/lib/e74_full_matrix_v2_gpu.cu.cc")
        print("Location: E74FullMatrixForwardV2::Run() around line 365-375")
        print("\nIssue: For EMA update (update_type=4), the code computes:")
        print("  alpha_cache = W_alpha @ x")
        print("But it does NOT:")
        print("  1. Add b_alpha bias")
        print("  2. Apply sigmoid")
        print("\nThe kernel at line 184 expects alpha to already have sigmoid applied:")
        print("  float alpha_val = extra_shared[row]; // alpha[row], already sigmoid applied")
        print("\nFix: After line 373, add a kernel to apply: alpha_cache = sigmoid(alpha_cache + b_alpha)")

    # Return success if bug was confirmed and buggy backward matches
    return 0 if (bug_confirmed and buggy_backward_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
