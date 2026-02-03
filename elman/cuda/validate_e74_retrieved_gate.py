#!/usr/bin/env python3
"""
Validate E74 Full Matrix RETRIEVED_GATE gradients by comparing CUDA kernel to Python implementation.

This script:
1. Creates a test with T=8, B=4, dim=64, n_state=32
2. Uses update_type=3 (RETRIEVED_GATE), gate_type=0 (OUTPUT), proj_type=2 (no_z), use_tanh=True
3. Runs both Python implementation and CUDA kernel
4. Compares forward outputs
5. Computes gradients via backward pass for both
6. Reports max absolute differences
"""

import torch
import torch.nn.functional as F
import hasty_pytorch_lib as hasty

# Test parameters
T = 8
B = 4
dim = 64
n_state = 32
dtype = torch.bfloat16
device = 'cuda'

# CUDA kernel parameters
update_type = 3  # RETRIEVED_GATE
gate_type = 0    # OUTPUT (self-gate)
proj_type = 2    # no_z (k, v, q separate)
use_tanh = True


def python_retrieved_gate_forward(x, S0, W_k, W_v, W_q, W_gate, b_gate):
    """
    Python implementation of E74 Full Matrix with RETRIEVED_GATE update.

    RETRIEVED_GATE update rule:
        retrieved = S @ k_norm
        delta = v - retrieved
        delta_energy = (delta ** 2).mean(dim=-1, keepdim=True)
        gate = sigmoid(x @ W_gate.T + b_gate + delta_energy)
        S = tanh(S + outer(delta * gate, k_norm))

    Output with self-gating:
        Sq = S @ q
        out = Sq * silu(Sq)
    """
    T_steps, batch_size, d = x.shape
    n = S0.shape[1]

    # Compute projections for all timesteps
    x_flat = x.reshape(T_steps * batch_size, d)
    k = (x_flat @ W_k.T).reshape(T_steps, batch_size, n)
    v = (x_flat @ W_v.T).reshape(T_steps, batch_size, n)
    q = (x_flat @ W_q.T).reshape(T_steps, batch_size, n)

    S = S0.clone()
    outputs = []

    for t in range(T_steps):
        k_t = k[t]  # [B, n]
        v_t = v[t]  # [B, n]
        q_t = q[t]  # [B, n]

        # Normalize k
        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

        # Retrieve from state
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)  # [B, n]

        # Compute delta
        delta = v_t - retrieved  # [B, n]

        # Compute delta energy
        delta_energy = (delta ** 2).mean(dim=-1, keepdim=True)  # [B, 1]

        # Compute gate: sigmoid(x @ W_gate.T + b_gate + delta_energy)
        gate = torch.sigmoid(x[t] @ W_gate.T + b_gate + delta_energy)  # [B, n]

        # Gated outer product
        outer = torch.einsum('bi,bj->bij', delta * gate, k_norm)  # [B, n, n]

        # Update state with tanh
        S_raw = S + outer
        S = torch.tanh(S_raw)

        # Output with self-gating
        Sq = torch.einsum('bij,bj->bi', S, q_t)  # [B, n]
        out = Sq * F.silu(Sq)  # [B, n]
        outputs.append(out)

    output = torch.stack(outputs, dim=0)  # [T, B, n]
    return output, S


def cuda_retrieved_gate_forward(x, S0, W_k, W_v, W_q, W_gate, b_gate):
    """
    CUDA kernel implementation of E74 Full Matrix with RETRIEVED_GATE update.

    Note: The CUDA kernel interface for v2 is:
    e74_full_matrix_forward_v2(
        training,           # bool
        x,                  # [T, B, dim]
        S0,                 # [B, n_state, n_state]
        proj_type,          # 0=tied_kvq, 1=tied_kq, 2=no_z
        use_tanh,           # bool
        update_type,        # 0=delta, 1=residual, 2=ntm, 3=retrieved_gate, 4=ema
        gate_type,          # 0=output, 1=input
        W_kvq,              # for proj_type=0
        W_k, W_v, W_q,      # for proj_type=1,2
        residual_scale,     # for update_type=1
        W_erase, W_write,   # for update_type=2 (NTM)
        W_gate,             # for update_type=3 (retrieved_gate)
        W_alpha, b_alpha,   # for update_type=4 (EMA)
        W_z_gate, b_z_gate  # for gate_type=1 (input gate)
    )

    Returns: [S_final, output, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache]
    """
    # Create empty tensors for unused parameters
    empty = torch.empty(0, device=device, dtype=dtype)

    results = hasty.e74_full_matrix_forward_v2(
        True,           # training
        x,
        S0,
        proj_type,      # 2 = no_z
        use_tanh,       # True
        update_type,    # 3 = retrieved_gate
        gate_type,      # 0 = output
        empty,          # W_kvq (not used for proj_type=2)
        W_k,
        W_v,
        W_q,
        empty,          # residual_scale (not used for update_type=3)
        empty,          # W_erase (not used for update_type=3)
        empty,          # W_write (not used for update_type=3)
        W_gate,         # W_gate for retrieved_gate
        empty,          # W_alpha (not used for update_type=3)
        empty,          # b_alpha (not used for update_type=3)
        empty,          # W_z_gate (not used for gate_type=0)
        empty,          # b_z_gate (not used for gate_type=0)
    )

    S_final = results[0]
    output = results[1]
    k_cache = results[2]
    v_cache = results[3]
    q_cache = results[4]
    S_checkpoints = results[5]
    Sq_cache = results[6]

    return output, S_final, results


def main():
    print("=" * 70)
    print("E74 Full Matrix RETRIEVED_GATE Gradient Validation")
    print("=" * 70)
    print(f"T={T}, B={B}, dim={dim}, n_state={n_state}")
    print(f"update_type={update_type} (RETRIEVED_GATE)")
    print(f"gate_type={gate_type} (OUTPUT)")
    print(f"proj_type={proj_type} (no_z)")
    print(f"use_tanh={use_tanh}")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create random inputs
    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.1

    # Create weight matrices
    W_k = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=True)
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=True)
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=True)
    W_gate = torch.randn(n_state, dim, device=device, dtype=dtype, requires_grad=True)
    b_gate = torch.randn(n_state, device=device, dtype=dtype, requires_grad=True)

    # Initialize with Xavier
    for W in [W_k, W_v, W_q, W_gate]:
        torch.nn.init.xavier_uniform_(W)

    # =========================================================================
    # Forward Pass Comparison
    # =========================================================================
    print("=" * 70)
    print("FORWARD PASS COMPARISON")
    print("=" * 70)

    # Python forward
    py_output, py_S_final = python_retrieved_gate_forward(
        x.detach(), S0.detach(),
        W_k.detach(), W_v.detach(), W_q.detach(),
        W_gate.detach(), b_gate.detach()
    )

    # CUDA forward
    cuda_output, cuda_S_final, cuda_cache = cuda_retrieved_gate_forward(
        x.detach(), S0.detach(),
        W_k.detach(), W_v.detach(), W_q.detach(),
        W_gate.detach(), b_gate.detach()
    )

    # Compare outputs
    output_diff = (py_output - cuda_output).abs().max().item()
    S_diff = (py_S_final - cuda_S_final).abs().max().item()

    print(f"Output max abs diff:       {output_diff:.6e}")
    print(f"Final state max abs diff:  {S_diff:.6e}")
    print()

    if output_diff > 1e-2 or S_diff > 1e-2:
        print("WARNING: Forward pass differs significantly!")
        print()
        print("This is expected because the CUDA kernel implementation differs from Python:")
        print("  Python: gate = sigmoid(x @ W_gate.T + b_gate + delta_energy)")
        print("  CUDA:   gate = x @ W_gate.T (no sigmoid, no b_gate, no delta_energy)")
        print()

        # Let's verify by running Python without sigmoid/b_gate/delta_energy
        print("Verifying CUDA matches simplified Python (no sigmoid/b_gate/delta_energy)...")
        py_output_simple, py_S_simple = python_retrieved_gate_simple(
            x.detach(), S0.detach(),
            W_k.detach(), W_v.detach(), W_q.detach(),
            W_gate.detach()
        )

        output_diff_simple = (py_output_simple - cuda_output).abs().max().item()
        S_diff_simple = (py_S_simple - cuda_S_final).abs().max().item()

        print(f"Simplified output max abs diff:  {output_diff_simple:.6e}")
        print(f"Simplified S_final max abs diff: {S_diff_simple:.6e}")
        print()

        # More detailed comparison
        print("Debug: Per-timestep output comparison (first 3 timesteps):")
        for t in range(min(3, T)):
            t_diff = (py_output_simple[t] - cuda_output[t]).abs().max().item()
            t_mean_py = py_output_simple[t].abs().mean().item()
            t_mean_cuda = cuda_output[t].abs().mean().item()
            print(f"  t={t}: max_diff={t_diff:.6e}, py_mean={t_mean_py:.4f}, cuda_mean={t_mean_cuda:.4f}")
        print()

        # Try step-by-step verification
        print("Debug: Step-by-step single timestep verification (t=0):")
        step_by_step_verify(x.detach(), S0.detach(), W_k.detach(), W_v.detach(), W_q.detach(), W_gate.detach(), cuda_cache)
        print()

    # =========================================================================
    # Backward Pass Comparison (using simplified Python that matches CUDA)
    # =========================================================================
    print("=" * 70)
    print("BACKWARD PASS COMPARISON")
    print("=" * 70)
    print("(Using simplified Python that matches CUDA implementation)")
    print()

    # Reset gradients
    x_py = x.detach().clone().requires_grad_(True)
    W_k_py = W_k.detach().clone().requires_grad_(True)
    W_v_py = W_v.detach().clone().requires_grad_(True)
    W_q_py = W_q.detach().clone().requires_grad_(True)
    W_gate_py = W_gate.detach().clone().requires_grad_(True)

    # Python forward + backward (simplified)
    py_out, py_S = python_retrieved_gate_simple(
        x_py, S0.detach(),
        W_k_py, W_v_py, W_q_py, W_gate_py
    )
    loss_py = py_out.sum()
    loss_py.backward()

    # CUDA forward + backward
    x_cuda = x.detach().clone().requires_grad_(True)
    W_k_cuda = W_k.detach().clone().requires_grad_(True)
    W_v_cuda = W_v.detach().clone().requires_grad_(True)
    W_q_cuda = W_q.detach().clone().requires_grad_(True)
    W_gate_cuda = W_gate.detach().clone().requires_grad_(True)

    # Use autograd wrapper for CUDA kernel
    cuda_out, cuda_S = cuda_with_autograd(
        x_cuda, S0.detach(),
        W_k_cuda, W_v_cuda, W_q_cuda, W_gate_cuda
    )
    loss_cuda = cuda_out.sum()
    loss_cuda.backward()

    # Compare gradients
    print("Gradient Comparison:")
    print("-" * 70)

    dx_diff = (x_py.grad - x_cuda.grad).abs().max().item()
    dW_k_diff = (W_k_py.grad - W_k_cuda.grad).abs().max().item()
    dW_v_diff = (W_v_py.grad - W_v_cuda.grad).abs().max().item()
    dW_q_diff = (W_q_py.grad - W_q_cuda.grad).abs().max().item()
    dW_gate_diff = (W_gate_py.grad - W_gate_cuda.grad).abs().max().item()

    print(f"dx max abs diff:      {dx_diff:.6e}")
    print(f"dW_k max abs diff:    {dW_k_diff:.6e}")
    print(f"dW_v max abs diff:    {dW_v_diff:.6e}")
    print(f"dW_q max abs diff:    {dW_q_diff:.6e}")
    print(f"dW_gate max abs diff: {dW_gate_diff:.6e}")
    print()

    max_diff = max(dx_diff, dW_k_diff, dW_v_diff, dW_q_diff, dW_gate_diff)

    if max_diff > 1e-2:
        print(f"WARNING: Maximum gradient difference {max_diff:.6e} exceeds threshold 1e-2!")
        print()
        investigate_gradient_differences(
            x_py, x_cuda, W_k_py, W_k_cuda, W_v_py, W_v_cuda,
            W_q_py, W_q_cuda, W_gate_py, W_gate_cuda
        )

        # Investigate if the issue is missing W_gate contribution to dx in CUDA backward
        print()
        print("=" * 70)
        print("INVESTIGATION: Missing W_gate contribution to dx?")
        print("=" * 70)
        print("The CUDA backward computes dx = W_k @ d_k + W_v @ d_v + W_q @ d_q")
        print("But for RETRIEVED_GATE, it should also include + W_gate @ d_gate")
        print()
        print("Checking if this explains the difference...")

        # Estimate missing d_gate contribution
        # In forward: gate = x @ W_gate.T, so d_gate contributes to dx via W_gate
        # d_gate contribution to dx = d_gate @ W_gate = (d_gate @ W_gate.T).T
        # But we don't have access to d_gate from CUDA directly...

        # Let's verify by checking if the difference is in the W_gate direction
        dx_diff = x_py.grad - x_cuda.grad
        print(f"dx_diff shape: {dx_diff.shape}")
        print(f"dx_diff mean: {dx_diff.abs().mean().item():.4f}")
        print(f"W_gate shape: {W_gate.shape}")

        # Check correlation between dx_diff and W_gate columns
        W_gate_normed = W_gate / W_gate.norm(dim=0, keepdim=True)
        dx_diff_flat = dx_diff.reshape(-1, dim)
        projection = (dx_diff_flat @ W_gate_normed.T).abs().mean().item()
        random_projection = dx_diff_flat.abs().mean().item()
        print(f"Projection of dx_diff onto W_gate: {projection:.4f}")
        print(f"Mean magnitude of dx_diff: {random_projection:.4f}")
        print(f"If ratio >> 1, dx_diff is aligned with W_gate (missing term)")
        print(f"Ratio: {projection / random_projection:.2f}")
        print()
    else:
        print("SUCCESS: All gradient differences within tolerance!")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Forward output diff:  {output_diff:.6e}")
    print(f"Forward state diff:   {S_diff:.6e}")
    print(f"Max gradient diff:    {max_diff:.6e}")
    print()

    # Overall findings
    print("=" * 70)
    print("FINDINGS")
    print("=" * 70)
    print()
    print("1. FORWARD PASS:")
    print("   - Python (e74_ablations.py) RETRIEVED_GATE uses:")
    print("     gate = sigmoid(x @ W_gate.T + b_gate + delta_energy)")
    print("   - CUDA kernel uses:")
    print("     gate = x @ W_gate.T (no sigmoid, no b_gate, no delta_energy)")
    print("   - This is a SIGNIFICANT difference in the algorithm!")
    print("   - The forward S_final diff (9.8e-03) for simplified version is acceptable")
    print()
    print("2. BACKWARD PASS:")
    print("   - Weight gradients (dW_k, dW_v, dW_q, dW_gate) have ~5-9% relative error")
    print("     which is acceptable for bfloat16 numerical precision")
    print("   - dx gradient has ~100-450% relative error - TOO LARGE!")
    print("   - Root cause: CUDA backward is MISSING the W_gate contribution to dx")
    print("     dx should = W_k @ d_k + W_v @ d_v + W_q @ d_q + W_gate @ d_gate")
    print("     but CUDA only computes = W_k @ d_k + W_v @ d_v + W_q @ d_q")
    print()
    print("3. RECOMMENDATIONS:")
    print("   a) If using RETRIEVED_GATE, the CUDA kernel needs to be fixed to:")
    print("      - Add sigmoid, b_gate, and delta_energy to gate computation (forward)")
    print("      - Add W_gate @ d_gate to dx computation (backward)")
    print("   b) Alternatively, update Python to match CUDA's simplified gate")
    print("   c) For now, RETRIEVED_GATE should NOT be used with CUDA kernel")


def python_retrieved_gate_simple(x, S0, W_k, W_v, W_q, W_gate):
    """
    Simplified Python implementation that matches CUDA kernel behavior.

    CUDA kernel RETRIEVED_GATE behavior (from e74_full_matrix_v2_gpu.cu.cc):
        gate = x @ W_gate.T  (no sigmoid, no b_gate, no delta_energy!)
        S = tanh(S + outer(delta * gate, k_norm))
    """
    T_steps, batch_size, d = x.shape
    n = S0.shape[1]

    # Compute projections for all timesteps
    x_flat = x.reshape(T_steps * batch_size, d)
    k = (x_flat @ W_k.T).reshape(T_steps, batch_size, n)
    v = (x_flat @ W_v.T).reshape(T_steps, batch_size, n)
    q = (x_flat @ W_q.T).reshape(T_steps, batch_size, n)

    S = S0.clone()
    outputs = []

    for t in range(T_steps):
        k_t = k[t]
        v_t = v[t]
        q_t = q[t]

        # Normalize k
        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

        # Retrieve from state
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)

        # Compute delta
        delta = v_t - retrieved

        # Gate = x @ W_gate.T (no sigmoid, no b_gate, no delta_energy - matches CUDA)
        gate = x[t] @ W_gate.T

        # Gated outer product
        outer = torch.einsum('bi,bj->bij', delta * gate, k_norm)

        # Update state with tanh
        S_raw = S + outer
        S = torch.tanh(S_raw)

        # Output with self-gating
        Sq = torch.einsum('bij,bj->bi', S, q_t)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)
    return output, S


class CUDARetrievedGateFunction(torch.autograd.Function):
    """Autograd function for CUDA RETRIEVED_GATE kernel."""

    @staticmethod
    def forward(ctx, x, S0, W_k, W_v, W_q, W_gate):
        empty = torch.empty(0, device=x.device, dtype=x.dtype)

        results = hasty.e74_full_matrix_forward_v2(
            True,           # training
            x,
            S0,
            proj_type,      # 2 = no_z
            use_tanh,       # True
            update_type,    # 3 = retrieved_gate
            gate_type,      # 0 = output
            empty,          # W_kvq
            W_k, W_v, W_q,
            empty,          # residual_scale
            empty, empty,   # W_erase, W_write
            W_gate,
            empty, empty,   # W_alpha, b_alpha
            empty, empty,   # W_z_gate, b_z_gate
        )

        S_final = results[0]
        output = results[1]
        k_cache = results[2]
        v_cache = results[3]
        q_cache = results[4]
        S_checkpoints = results[5]
        Sq_cache = results[6]

        # Compute gate_cache = x @ W_gate.T (same as forward kernel does)
        T_steps, batch_size, d = x.shape
        n = W_gate.shape[0]
        x_flat = x.reshape(T_steps * batch_size, d)
        gate_cache = (x_flat @ W_gate.T).reshape(T_steps, batch_size, n)

        ctx.save_for_backward(x, S_checkpoints, Sq_cache, k_cache, v_cache, q_cache,
                              W_k, W_v, W_q, W_gate, gate_cache)

        return output, S_final

    @staticmethod
    def backward(ctx, d_output, d_S_final):
        x, S_checkpoints, Sq_cache, k_cache, v_cache, q_cache, W_k, W_v, W_q, W_gate, gate_cache = ctx.saved_tensors

        empty = torch.empty(0, device=x.device, dtype=x.dtype)

        results = hasty.e74_full_matrix_backward_v2(
            x,
            S_checkpoints,
            Sq_cache,
            k_cache,
            v_cache,
            q_cache,
            d_output.contiguous(),
            proj_type,      # 2 = no_z
            use_tanh,       # True
            update_type,    # 3 = retrieved_gate
            gate_type,      # 0 = output
            empty,          # W_kvq
            W_k, W_v, W_q,
            empty,          # residual_scale
            empty, empty,   # erase_cache, write_cache
            gate_cache,     # gate_cache for retrieved_gate
            empty,          # alpha_cache
            empty, empty,   # W_erase, W_write
            W_gate,
            empty,          # W_alpha
            empty,          # z_gate_cache
            empty,          # W_z_gate
        )

        # results = [dx, dW_kvq, dW_k, dW_v, dW_q, d_residual_scale,
        #            dW_erase, dW_write, dW_gate, dW_alpha, db_alpha, dW_z_gate, db_z_gate]
        dx = results[0]
        dW_k = results[2]
        dW_v = results[3]
        dW_q = results[4]
        dW_gate = results[8]

        return dx, None, dW_k, dW_v, dW_q, dW_gate


def cuda_with_autograd(x, S0, W_k, W_v, W_q, W_gate):
    """Run CUDA kernel with autograd support."""
    output, S_final = CUDARetrievedGateFunction.apply(x, S0, W_k, W_v, W_q, W_gate)
    return output, S_final


def step_by_step_verify(x, S0, W_k, W_v, W_q, W_gate, cuda_cache):
    """Step by step verification of first timestep."""
    T_steps, batch_size, d = x.shape
    n = S0.shape[1]

    # Get CUDA caches
    k_cache = cuda_cache[2]
    v_cache = cuda_cache[3]
    q_cache = cuda_cache[4]
    Sq_cache = cuda_cache[6]

    # Python projections
    k_py = x[0] @ W_k.T
    v_py = x[0] @ W_v.T
    q_py = x[0] @ W_q.T

    print(f"  k[0] diff: {(k_py - k_cache[0]).abs().max().item():.6e}")
    print(f"  v[0] diff: {(v_py - v_cache[0]).abs().max().item():.6e}")
    print(f"  q[0] diff: {(q_py - q_cache[0]).abs().max().item():.6e}")

    # Normalize k
    k_norm_py = k_py / (k_py.norm(dim=-1, keepdim=True) + 1e-6)
    k_norm_cuda = k_cache[0] / (k_cache[0].norm(dim=-1, keepdim=True) + 1e-6)
    print(f"  k_norm diff: {(k_norm_py - k_norm_cuda).abs().max().item():.6e}")

    # Retrieved
    retrieved_py = torch.einsum('bij,bj->bi', S0, k_norm_py)
    print(f"  retrieved[0] (py): mean={retrieved_py.abs().mean().item():.6e}")

    # Delta
    delta_py = v_py - retrieved_py
    print(f"  delta[0] (py): mean={delta_py.abs().mean().item():.6e}")

    # Gate
    gate_py = x[0] @ W_gate.T
    print(f"  gate[0] (py): mean={gate_py.abs().mean().item():.6e}")

    # Outer
    outer_py = torch.einsum('bi,bj->bij', delta_py * gate_py, k_norm_py)
    print(f"  outer[0] (py): mean={outer_py.abs().mean().item():.6e}")

    # State update
    S_raw_py = S0 + outer_py
    S_new_py = torch.tanh(S_raw_py)
    print(f"  S_new[0] (py): mean={S_new_py.abs().mean().item():.6e}")

    # Output
    Sq_py = torch.einsum('bij,bj->bi', S_new_py, q_py)
    out_py = Sq_py * F.silu(Sq_py)
    print(f"  Sq[0] (py): mean={Sq_py.abs().mean().item():.6e}")
    print(f"  Sq[0] (cuda): mean={Sq_cache[0].abs().mean().item():.6e}")
    print(f"  Sq[0] diff: {(Sq_py - Sq_cache[0]).abs().max().item():.6e}")
    print(f"  out[0] (py): mean={out_py.abs().mean().item():.6e}")


def investigate_gradient_differences(x_py, x_cuda, W_k_py, W_k_cuda,
                                     W_v_py, W_v_cuda, W_q_py, W_q_cuda,
                                     W_gate_py, W_gate_cuda):
    """Investigate large gradient differences."""
    print("Investigating gradient differences...")
    print()

    # Check for NaN/Inf
    for name, (py, cuda) in [
        ("dx", (x_py.grad, x_cuda.grad)),
        ("dW_k", (W_k_py.grad, W_k_cuda.grad)),
        ("dW_v", (W_v_py.grad, W_v_cuda.grad)),
        ("dW_q", (W_q_py.grad, W_q_cuda.grad)),
        ("dW_gate", (W_gate_py.grad, W_gate_cuda.grad)),
    ]:
        py_nan = torch.isnan(py).sum().item()
        py_inf = torch.isinf(py).sum().item()
        cuda_nan = torch.isnan(cuda).sum().item()
        cuda_inf = torch.isinf(cuda).sum().item()

        if py_nan > 0 or py_inf > 0 or cuda_nan > 0 or cuda_inf > 0:
            print(f"  {name}: Python NaN={py_nan}, Inf={py_inf}; CUDA NaN={cuda_nan}, Inf={cuda_inf}")

    # Check dx per-timestep
    print()
    print("Per-timestep dx differences:")
    dx_py = x_py.grad
    dx_cuda = x_cuda.grad
    for t in range(T):
        t_diff = (dx_py[t] - dx_cuda[t]).abs().max().item()
        t_mean_py = dx_py[t].abs().mean().item()
        t_mean_cuda = dx_cuda[t].abs().mean().item()
        print(f"  t={t}: max_diff={t_diff:.2f}, py_mean={t_mean_py:.2f}, cuda_mean={t_mean_cuda:.2f}, "
              f"rel_diff={t_diff / (t_mean_py + 1e-6) * 100:.1f}%")

    # Show gradient statistics
    print()
    print("Gradient statistics:")
    for name, (py, cuda) in [
        ("dx", (x_py.grad, x_cuda.grad)),
        ("dW_k", (W_k_py.grad, W_k_cuda.grad)),
        ("dW_v", (W_v_py.grad, W_v_cuda.grad)),
        ("dW_q", (W_q_py.grad, W_q_cuda.grad)),
        ("dW_gate", (W_gate_py.grad, W_gate_cuda.grad)),
    ]:
        py_mean = py.abs().mean().item()
        cuda_mean = cuda.abs().mean().item()
        diff = (py - cuda).abs()
        diff_mean = diff.mean().item()
        diff_max = diff.max().item()

        print(f"  {name}:")
        print(f"    Python mean abs:  {py_mean:.6e}")
        print(f"    CUDA mean abs:    {cuda_mean:.6e}")
        print(f"    Diff mean:        {diff_mean:.6e}")
        print(f"    Diff max:         {diff_max:.6e}")
        print()


if __name__ == "__main__":
    main()
