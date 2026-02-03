#!/usr/bin/env python3
"""
Validate E74 Full Matrix NTM (update_type=2) CUDA kernel against Python reference.

IMPORTANT: The CUDA kernel does NOT apply sigmoid to erase/write values.
It expects raw projected values: erase_raw = W_erase @ x, write_raw = W_write @ x
The sigmoid is NOT applied in the CUDA kernel.

The Python reference (e74_ablations.py) DOES apply sigmoid:
  erase = sigmoid(x @ W_erase.T + b_erase)
  write = sigmoid(x @ W_write.T + b_write)

This test modifies the Python reference to match the CUDA kernel behavior
(no sigmoid, no bias) for validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Set up environment
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/home/erikg/.local/lib/python3.12/site-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Import CUDA kernel
import hasty_pytorch_lib as cuda_lib


def python_ntm_forward(x, S0, W_k, W_v, W_q, W_erase, W_write, use_tanh=True, use_sigmoid=False):
    """
    Python reference for NTM update.

    Args:
        use_sigmoid: If True, apply sigmoid to erase/write (original Python behavior).
                     If False, use raw values (CUDA kernel behavior).

    NTM update rule:
        erase = [sigmoid](x @ W_erase.T)  - what to erase at key location
        write = [sigmoid](x @ W_write.T)  - what to write
        S = S * (1 - outer(erase, k_norm)) + outer(write * v, k_norm)
        [S = tanh(S)]
    """
    T, B, dim = x.shape
    n_state = S0.shape[1]

    S = S0.clone()
    outputs = []

    # Pre-compute projections
    x_flat = x.reshape(T * B, dim)
    k = (x_flat @ W_k.T).reshape(T, B, n_state)
    v = (x_flat @ W_v.T).reshape(T, B, n_state)
    q = (x_flat @ W_q.T).reshape(T, B, n_state)

    erase_proj = (x_flat @ W_erase.T).reshape(T, B, n_state)
    write_proj = (x_flat @ W_write.T).reshape(T, B, n_state)

    for t in range(T):
        k_t = k[t]  # [B, n_state]
        v_t = v[t]
        q_t = q[t]

        # Normalize k
        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

        # Get erase and write values
        if use_sigmoid:
            erase = torch.sigmoid(erase_proj[t])
            write = torch.sigmoid(write_proj[t])
        else:
            # CUDA kernel behavior: no sigmoid
            erase = erase_proj[t]
            write = write_proj[t]

        # NTM update: S = S * (1 - outer(erase, k)) + outer(write * v, k)
        erase_outer = torch.einsum('bi,bj->bij', erase, k_norm)  # [B, n, n]
        S_erased = S * (1.0 - erase_outer)
        write_outer = torch.einsum('bi,bj->bij', write * v_t, k_norm)
        S_raw = S_erased + write_outer

        if use_tanh:
            S = torch.tanh(S_raw)
        else:
            S = S_raw

        # Output: Sq * silu(Sq)
        Sq = torch.einsum('bij,bj->bi', S, q_t)
        out = Sq * F.silu(Sq)
        outputs.append(out)

    output = torch.stack(outputs, dim=0)  # [T, B, n_state]
    return output, S


class PythonNTMCell(nn.Module):
    """Python NTM cell for gradient comparison."""

    def __init__(self, dim, n_state, use_sigmoid=False):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.use_sigmoid = use_sigmoid

        # Projections
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))
        self.W_erase = nn.Parameter(torch.empty(n_state, dim))
        self.W_write = nn.Parameter(torch.empty(n_state, dim))

        self._init_weights()

    def _init_weights(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, x, S0=None):
        T, B, D = x.shape
        if S0 is None:
            S0 = torch.zeros(B, self.n_state, self.n_state, device=x.device, dtype=x.dtype)
        return python_ntm_forward(
            x, S0, self.W_k, self.W_v, self.W_q,
            self.W_erase, self.W_write,
            use_tanh=True, use_sigmoid=self.use_sigmoid
        )


class CUDANTMFunction(torch.autograd.Function):
    """Autograd function for CUDA NTM kernel."""

    @staticmethod
    def forward(ctx, x, S0, W_k, W_v, W_q, W_erase, W_write, training):
        # Call CUDA forward: e74_full_matrix_forward_v2
        # Arguments:
        #   training, x, S0, proj_type, use_tanh, update_type, gate_type,
        #   W_kvq, W_k, W_v, W_q,
        #   residual_scale, W_erase, W_write, W_gate, W_alpha, b_alpha,
        #   W_z_gate, b_z_gate

        T, B, dim = x.shape
        n_state = S0.shape[1]

        # Empty tensors for unused parameters
        empty_n = torch.empty(0, device=x.device, dtype=x.dtype)
        empty_nd = torch.empty(0, device=x.device, dtype=x.dtype)

        results = cuda_lib.e74_full_matrix_forward_v2(
            training,
            x,
            S0,
            2,  # proj_type=2 (no_z: k, v, q separate)
            True,  # use_tanh
            2,  # update_type=2 (NTM)
            0,  # gate_type=0 (output self-gate)
            empty_nd,  # W_kvq (not used)
            W_k,
            W_v,
            W_q,
            empty_n,  # residual_scale
            W_erase,
            W_write,
            empty_nd,  # W_gate
            empty_nd,  # W_alpha
            empty_n,  # b_alpha
            empty_nd,  # W_z_gate
            empty_n,  # b_z_gate
        )

        # results = [S, output, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache]
        S_final = results[0]
        output = results[1]
        k_cache = results[2]
        v_cache = results[3]
        q_cache = results[4]
        S_checkpoints = results[5]
        Sq_cache = results[6]

        # Compute erase_cache and write_cache for backward
        # (Note: CUDA forward stores these in S_cache but doesn't return them separately)
        x_flat = x.reshape(T * B, dim)
        erase_cache = (x_flat @ W_erase.T).reshape(T, B, n_state)
        write_cache = (x_flat @ W_write.T).reshape(T, B, n_state)

        ctx.save_for_backward(
            x, S0, W_k, W_v, W_q, W_erase, W_write,
            k_cache, v_cache, q_cache, S_checkpoints, Sq_cache,
            erase_cache, write_cache
        )

        return output, S_final

    @staticmethod
    def backward(ctx, d_output, d_S_final):
        (x, S0, W_k, W_v, W_q, W_erase, W_write,
         k_cache, v_cache, q_cache, S_checkpoints, Sq_cache,
         erase_cache, write_cache) = ctx.saved_tensors

        T, B, dim = x.shape
        n_state = W_k.shape[0]

        # Empty tensors for unused parameters
        empty_n = torch.empty(0, device=x.device, dtype=x.dtype)
        empty_nd = torch.empty(0, device=x.device, dtype=x.dtype)
        empty_tbn = torch.empty(0, device=x.device, dtype=x.dtype)

        results = cuda_lib.e74_full_matrix_backward_v2(
            x,
            S_checkpoints,
            Sq_cache,
            k_cache,
            v_cache,
            q_cache,
            d_output.contiguous(),
            2,  # proj_type=2 (no_z)
            True,  # use_tanh
            2,  # update_type=2 (NTM)
            0,  # gate_type=0 (output)
            empty_nd,  # W_kvq
            W_k,
            W_v,
            W_q,
            empty_n,  # residual_scale
            erase_cache.contiguous(),  # erase_cache for NTM
            write_cache.contiguous(),  # write_cache for NTM
            empty_tbn,  # gate_cache
            empty_tbn,  # alpha_cache
            W_erase,
            W_write,
            empty_nd,  # W_gate
            empty_nd,  # W_alpha
            empty_tbn,  # z_gate_cache
            empty_nd,  # W_z_gate
        )

        # results = [dx, dW_kvq, dW_k, dW_v, dW_q, d_residual_scale,
        #            dW_erase, dW_write, dW_gate, dW_alpha, db_alpha, dW_z_gate, db_z_gate]
        dx = results[0]
        dW_k = results[2] if results[2].numel() > 0 else None
        dW_v = results[3] if results[3].numel() > 0 else None
        dW_q = results[4] if results[4].numel() > 0 else None
        dW_erase = results[6] if results[6].numel() > 0 else None
        dW_write = results[7] if results[7].numel() > 0 else None

        return dx, None, dW_k, dW_v, dW_q, dW_erase, dW_write, None


class CUDANTMCell(nn.Module):
    """CUDA NTM cell using e74_full_matrix_forward_v2 with update_type=2."""

    def __init__(self, dim, n_state):
        super().__init__()
        self.dim = dim
        self.n_state = n_state

        # Projections (same as Python)
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))
        self.W_erase = nn.Parameter(torch.empty(n_state, dim))
        self.W_write = nn.Parameter(torch.empty(n_state, dim))

        self._init_weights()

    def _init_weights(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, x, S0=None):
        T, B, D = x.shape
        if S0 is None:
            S0 = torch.zeros(B, self.n_state, self.n_state, device=x.device, dtype=x.dtype)
        return CUDANTMFunction.apply(
            x, S0, self.W_k, self.W_v, self.W_q,
            self.W_erase, self.W_write, self.training
        )


def compare_outputs_and_gradients():
    """Compare forward outputs and backward gradients."""

    print("=" * 70)
    print("E74 Full Matrix NTM Gradient Validation")
    print("=" * 70)

    # Parameters
    T = 8
    B = 4
    dim = 64
    n_state = 32

    device = 'cuda'
    dtype = torch.bfloat16

    print(f"\nTest configuration:")
    print(f"  T={T}, B={B}, dim={dim}, n_state={n_state}")
    print(f"  update_type=2 (NTM), gate_type=0 (OUTPUT), proj_type=2 (no_z)")
    print(f"  use_tanh=True")
    print(f"  dtype={dtype}")
    print()

    # Create models - use_sigmoid=True to match Python reference properly
    # Let's first test if the forward pass is correct with no sigmoid (CUDA behavior)
    python_model = PythonNTMCell(dim, n_state, use_sigmoid=False).to(device).to(dtype)
    cuda_model = CUDANTMCell(dim, n_state).to(device).to(dtype)

    # Copy weights from Python to CUDA model
    with torch.no_grad():
        cuda_model.W_k.copy_(python_model.W_k)
        cuda_model.W_v.copy_(python_model.W_v)
        cuda_model.W_q.copy_(python_model.W_q)
        cuda_model.W_erase.copy_(python_model.W_erase)
        cuda_model.W_write.copy_(python_model.W_write)

    # Create random inputs
    torch.manual_seed(42)
    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.1

    # Forward pass - Python
    python_model.train()
    x_python = x.clone().detach().requires_grad_(True)
    output_python, S_final_python = python_model(x_python, S0.clone())

    # Forward pass - CUDA
    cuda_model.train()
    x_cuda = x.clone().detach().requires_grad_(True)
    output_cuda, S_final_cuda = cuda_model(x_cuda, S0.clone())

    # Debug: print some intermediate values
    print("Debug: Output value ranges:")
    print(f"  Python output: min={output_python.min().item():.4f}, max={output_python.max().item():.4f}, mean={output_python.float().mean().item():.4f}")
    print(f"  CUDA output:   min={output_cuda.min().item():.4f}, max={output_cuda.max().item():.4f}, mean={output_cuda.float().mean().item():.4f}")

    print(f"\nDebug: Final state value ranges:")
    print(f"  Python S_final: min={S_final_python.min().item():.4f}, max={S_final_python.max().item():.4f}")
    print(f"  CUDA S_final:   min={S_final_cuda.min().item():.4f}, max={S_final_cuda.max().item():.4f}")

    # Compare forward outputs
    print("\nForward pass comparison:")
    output_diff = (output_python - output_cuda).abs().max().item()
    print(f"  Output max abs diff: {output_diff:.6e}")

    S_diff = (S_final_python - S_final_cuda).abs().max().item()
    print(f"  Final state max abs diff: {S_diff:.6e}")

    # Backward pass
    loss_python = output_python.sum()
    loss_cuda = output_cuda.sum()

    loss_python.backward()
    loss_cuda.backward()

    # Compare gradients
    print("\nBackward pass comparison (gradients):")

    dx_diff = (x_python.grad - x_cuda.grad).abs().max().item()
    print(f"  dx max abs diff: {dx_diff:.6e}")

    dW_k_diff = (python_model.W_k.grad - cuda_model.W_k.grad).abs().max().item()
    print(f"  dW_k max abs diff: {dW_k_diff:.6e}")

    dW_v_diff = (python_model.W_v.grad - cuda_model.W_v.grad).abs().max().item()
    print(f"  dW_v max abs diff: {dW_v_diff:.6e}")

    dW_q_diff = (python_model.W_q.grad - cuda_model.W_q.grad).abs().max().item()
    print(f"  dW_q max abs diff: {dW_q_diff:.6e}")

    dW_erase_diff = (python_model.W_erase.grad - cuda_model.W_erase.grad).abs().max().item()
    print(f"  dW_erase max abs diff: {dW_erase_diff:.6e}")

    dW_write_diff = (python_model.W_write.grad - cuda_model.W_write.grad).abs().max().item()
    print(f"  dW_write max abs diff: {dW_write_diff:.6e}")

    # Summary
    print("\n" + "=" * 70)
    max_diff = max(output_diff, S_diff, dx_diff, dW_k_diff, dW_v_diff, dW_q_diff, dW_erase_diff, dW_write_diff)
    threshold = 1e-2

    if max_diff > threshold:
        print(f"WARNING: Max difference {max_diff:.6e} exceeds threshold {threshold}")
        print("\nInvestigating...")

        # Print detailed stats
        print("\nDetailed gradient statistics:")
        print(f"  Python dx mean: {x_python.grad.float().mean().item():.6e}, std: {x_python.grad.float().std().item():.6e}")
        print(f"  CUDA dx mean: {x_cuda.grad.float().mean().item():.6e}, std: {x_cuda.grad.float().std().item():.6e}")

        print(f"\n  Python dW_k mean: {python_model.W_k.grad.float().mean().item():.6e}")
        print(f"  CUDA dW_k mean: {cuda_model.W_k.grad.float().mean().item():.6e}")

        print(f"\n  Python dW_erase mean: {python_model.W_erase.grad.float().mean().item():.6e}")
        print(f"  CUDA dW_erase mean: {cuda_model.W_erase.grad.float().mean().item():.6e}")

        # Check for NaN/Inf
        if torch.isnan(output_cuda).any() or torch.isinf(output_cuda).any():
            print("\nERROR: CUDA output contains NaN or Inf!")
        if torch.isnan(x_cuda.grad).any() or torch.isinf(x_cuda.grad).any():
            print("ERROR: CUDA dx gradient contains NaN or Inf!")

        return False
    else:
        print(f"SUCCESS: All differences below threshold {threshold}")
        print(f"  Max difference: {max_diff:.6e}")
        return True


def test_numerical_gradient():
    """Test CUDA gradients using numerical differentiation."""
    print("\n" + "=" * 70)
    print("Numerical Gradient Check (finite differences)")
    print("=" * 70)

    # Smaller test for numerical gradient
    T = 4
    B = 2
    dim = 32
    n_state = 16
    eps = 1e-3

    device = 'cuda'
    # Use float32 for numerical gradient check
    dtype = torch.float32

    print(f"\nConfiguration: T={T}, B={B}, dim={dim}, n_state={n_state}")
    print(f"Epsilon for finite diff: {eps}")

    # Create model
    model = PythonNTMCell(dim, n_state, use_sigmoid=False).to(device).to(dtype)
    model.train()

    # Random input
    torch.manual_seed(123)
    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.1

    # Compute analytical gradient
    output, _ = model(x, S0.clone())
    loss = output.sum()
    loss.backward()

    analytical_grad = x.grad.clone()

    # Compute numerical gradient for a few elements
    print("\nNumerical vs Analytical gradient (sampling 5 elements):")
    numerical_grad = torch.zeros_like(x)

    # Sample a few indices to check
    indices = [(0, 0, 0), (1, 1, 10), (2, 0, 20), (3, 1, 31), (T-1, B-1, dim-1)]

    for idx in indices:
        t, b, d = idx

        # Positive perturbation
        x_pos = x.detach().clone()
        x_pos[t, b, d] += eps
        output_pos, _ = model(x_pos, S0.clone())
        loss_pos = output_pos.sum()

        # Negative perturbation
        x_neg = x.detach().clone()
        x_neg[t, b, d] -= eps
        output_neg, _ = model(x_neg, S0.clone())
        loss_neg = output_neg.sum()

        # Numerical gradient
        num_grad = (loss_pos - loss_neg) / (2 * eps)
        ana_grad = analytical_grad[t, b, d]

        rel_err = abs(num_grad - ana_grad) / (abs(ana_grad) + 1e-8)
        print(f"  [{t},{b},{d}]: numerical={num_grad.item():.6f}, analytical={ana_grad.item():.6f}, rel_err={rel_err.item():.6e}")


def test_cuda_numerical_gradient():
    """Test CUDA kernel gradients using numerical differentiation."""
    print("\n" + "=" * 70)
    print("CUDA Kernel Numerical Gradient Check (finite differences)")
    print("=" * 70)

    # Smaller test for numerical gradient
    T = 2
    B = 1
    dim = 32
    n_state = 32  # Must be 32 for CUDA kernel
    eps = 1e-2  # Use larger epsilon for bfloat16

    device = 'cuda'
    # Use bfloat16 as required by CUDA kernel
    dtype = torch.bfloat16

    print(f"\nConfiguration: T={T}, B={B}, dim={dim}, n_state={n_state}")
    print(f"Epsilon for finite diff: {eps}")

    # Create model
    model = CUDANTMCell(dim, n_state).to(device).to(dtype)
    model.train()

    # Random input
    torch.manual_seed(456)
    x_data = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.1

    # First, check that CUDA forward produces different outputs for different inputs
    output_base, _ = model(x_data.clone(), S0.clone())
    loss_base = output_base.sum().item()
    print(f"\nBase loss: {loss_base:.6f}")
    print(f"Base output shape: {output_base.shape}")
    print(f"Base output sample [0,0,:5]: {output_base[0,0,:5]}")

    # Perturb and check - use larger epsilon
    eps_large = 0.5
    x_test = x_data.clone()
    x_test[0, 0, 0] += eps_large
    output_test, _ = model(x_test, S0.clone())
    loss_test = output_test.sum().item()
    print(f"\nPerturbed loss (eps={eps_large}): {loss_test:.6f}")
    print(f"Loss diff: {loss_test - loss_base:.6f}")
    print(f"Perturbed output sample [0,0,:5]: {output_test[0,0,:5]}")

    # Compute analytical gradient properly
    x_grad = x_data.clone().detach().requires_grad_(True)
    output, _ = model(x_grad, S0.clone())
    loss = output.sum()
    loss.backward()

    if x_grad.grad is None:
        print("ERROR: x_grad.grad is None after backward!")
        return

    analytical_grad = x_grad.grad.clone()
    print(f"\nAnalytical gradient dx shape: {analytical_grad.shape}")
    print(f"Analytical gradient dx sample [0,0,:5]: {analytical_grad[0,0,:5]}")

    # Compute numerical gradient for a few elements
    print("\nCUDA Numerical vs Analytical gradient (sampling 5 elements):")

    # Sample a few indices to check
    indices = [(0, 0, 0), (0, 0, 15), (1, 0, 10), (1, 0, 20), (T-1, B-1, dim-1)]

    for idx in indices:
        t, b, d = idx

        # Positive perturbation
        x_pos = x_data.clone()
        x_pos[t, b, d] += eps
        output_pos, _ = model(x_pos, S0.clone())
        loss_pos = output_pos.sum().item()

        # Negative perturbation
        x_neg = x_data.clone()
        x_neg[t, b, d] -= eps
        output_neg, _ = model(x_neg, S0.clone())
        loss_neg = output_neg.sum().item()

        # Numerical gradient
        num_grad = (loss_pos - loss_neg) / (2 * eps)
        ana_grad = analytical_grad[t, b, d].item()

        rel_err = abs(num_grad - ana_grad) / (abs(ana_grad) + 1e-8)
        status = "OK" if rel_err < 0.1 else "FAIL"
        print(f"  [{t},{b},{d}]: loss_pos={loss_pos:.4f}, loss_neg={loss_neg:.4f}, num={num_grad:.6f}, ana={ana_grad:.6f}, rel_err={rel_err:.6e} [{status}]")


def debug_single_step():
    """Debug single timestep forward pass to trace differences."""
    print("\n" + "=" * 70)
    print("Debug: Single Timestep Forward Pass")
    print("=" * 70)

    T = 1  # Single timestep
    B = 1  # Single batch
    dim = 64
    n_state = 32

    device = 'cuda'
    dtype = torch.bfloat16

    # Deterministic weights
    torch.manual_seed(42)

    W_k = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_v = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_q = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_erase = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1
    W_write = torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1

    x = torch.randn(T, B, dim, device=device, dtype=dtype)
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.1

    # Python forward (step by step)
    print("\nPython forward (step by step):")

    x_flat = x.reshape(T * B, dim)
    k = (x_flat @ W_k.T).reshape(T, B, n_state)
    v = (x_flat @ W_v.T).reshape(T, B, n_state)
    q = (x_flat @ W_q.T).reshape(T, B, n_state)
    erase_proj = (x_flat @ W_erase.T).reshape(T, B, n_state)
    write_proj = (x_flat @ W_write.T).reshape(T, B, n_state)

    print(f"  k[0] range: [{k[0].min().item():.4f}, {k[0].max().item():.4f}]")
    print(f"  v[0] range: [{v[0].min().item():.4f}, {v[0].max().item():.4f}]")
    print(f"  q[0] range: [{q[0].min().item():.4f}, {q[0].max().item():.4f}]")
    print(f"  erase_proj[0] range: [{erase_proj[0].min().item():.4f}, {erase_proj[0].max().item():.4f}]")
    print(f"  write_proj[0] range: [{write_proj[0].min().item():.4f}, {write_proj[0].max().item():.4f}]")

    k_t = k[0]
    v_t = v[0]
    q_t = q[0]
    erase = erase_proj[0]  # No sigmoid (CUDA behavior)
    write = write_proj[0]  # No sigmoid (CUDA behavior)

    # Normalize k
    k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)
    print(f"\n  k_norm range: [{k_norm.min().item():.4f}, {k_norm.max().item():.4f}]")
    print(f"  k_norm norm: {k_norm.norm().item():.4f}")

    # NTM update
    S = S0.clone()
    erase_outer = torch.einsum('bi,bj->bij', erase, k_norm)
    print(f"\n  erase_outer range: [{erase_outer.min().item():.4f}, {erase_outer.max().item():.4f}]")

    S_erased = S * (1.0 - erase_outer)
    print(f"  S_erased range: [{S_erased.min().item():.4f}, {S_erased.max().item():.4f}]")

    write_outer = torch.einsum('bi,bj->bij', write * v_t, k_norm)
    print(f"  write_outer range: [{write_outer.min().item():.4f}, {write_outer.max().item():.4f}]")

    S_raw = S_erased + write_outer
    print(f"  S_raw range: [{S_raw.min().item():.4f}, {S_raw.max().item():.4f}]")

    S_new = torch.tanh(S_raw)
    print(f"  S_new (after tanh) range: [{S_new.min().item():.4f}, {S_new.max().item():.4f}]")

    Sq = torch.einsum('bij,bj->bi', S_new, q_t)
    print(f"\n  Sq range: [{Sq.min().item():.4f}, {Sq.max().item():.4f}]")

    out = Sq * F.silu(Sq)
    print(f"  out (Sq * silu(Sq)) range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # Now CUDA forward
    print("\n\nCUDA forward:")
    empty_n = torch.empty(0, device=device, dtype=dtype)
    empty_nd = torch.empty(0, device=device, dtype=dtype)

    results = cuda_lib.e74_full_matrix_forward_v2(
        True,  # training
        x,
        S0.clone(),
        2,  # proj_type=2 (no_z)
        True,  # use_tanh
        2,  # update_type=2 (NTM)
        0,  # gate_type=0 (output)
        empty_nd,  # W_kvq
        W_k,
        W_v,
        W_q,
        empty_n,  # residual_scale
        W_erase,
        W_write,
        empty_nd,  # W_gate
        empty_nd,  # W_alpha
        empty_n,  # b_alpha
        empty_nd,  # W_z_gate
        empty_n,  # b_z_gate
    )

    S_cuda = results[0]
    output_cuda = results[1]
    k_cache = results[2]
    v_cache = results[3]
    q_cache = results[4]

    print(f"  k_cache[0] range: [{k_cache[0].min().item():.4f}, {k_cache[0].max().item():.4f}]")
    print(f"  v_cache[0] range: [{v_cache[0].min().item():.4f}, {v_cache[0].max().item():.4f}]")
    print(f"  q_cache[0] range: [{q_cache[0].min().item():.4f}, {q_cache[0].max().item():.4f}]")
    print(f"  S_cuda range: [{S_cuda.min().item():.4f}, {S_cuda.max().item():.4f}]")
    print(f"  output_cuda range: [{output_cuda.min().item():.4f}, {output_cuda.max().item():.4f}]")

    # Compare
    print("\n\nComparison:")
    print(f"  k difference (max abs): {(k[0] - k_cache[0]).abs().max().item():.6e}")
    print(f"  v difference (max abs): {(v[0] - v_cache[0]).abs().max().item():.6e}")
    print(f"  q difference (max abs): {(q[0] - q_cache[0]).abs().max().item():.6e}")
    print(f"  S difference (max abs): {(S_new[0] - S_cuda[0]).abs().max().item():.6e}")
    print(f"  output difference (max abs): {(out - output_cuda[0]).abs().max().item():.6e}")


def test_pytorch_autograd_vs_cuda():
    """Test CUDA backward against PyTorch autograd on the same forward computation."""
    print("\n" + "=" * 70)
    print("PyTorch Autograd vs CUDA Backward (same forward)")
    print("=" * 70)

    T = 4
    B = 2
    dim = 64
    n_state = 32

    device = 'cuda'
    dtype = torch.bfloat16

    print(f"\nConfiguration: T={T}, B={B}, dim={dim}, n_state={n_state}")

    torch.manual_seed(789)

    # Create leaf tensors for weights
    W_k = nn.Parameter(torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1)
    W_v = nn.Parameter(torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1)
    W_q = nn.Parameter(torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1)
    W_erase = nn.Parameter(torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1)
    W_write = nn.Parameter(torch.randn(n_state, dim, device=device, dtype=dtype) * 0.1)

    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    S0 = torch.randn(B, n_state, n_state, device=device, dtype=dtype) * 0.1

    # Forward using Python reference
    output_py, S_final_py = python_ntm_forward(
        x, S0.clone(), W_k, W_v, W_q, W_erase, W_write,
        use_tanh=True, use_sigmoid=False
    )
    loss_py = output_py.sum()

    # Compute gradients via PyTorch autograd
    loss_py.backward()

    dx_autograd = x.grad.clone()
    dW_k_autograd = W_k.grad.clone()
    dW_v_autograd = W_v.grad.clone()
    dW_q_autograd = W_q.grad.clone()
    dW_erase_autograd = W_erase.grad.clone()
    dW_write_autograd = W_write.grad.clone()

    print(f"\nPyTorch autograd gradients:")
    print(f"  dx: mean={dx_autograd.float().mean().item():.4f}, std={dx_autograd.float().std().item():.4f}")
    print(f"  dW_k: mean={dW_k_autograd.float().mean().item():.4f}")
    print(f"  dW_erase: mean={dW_erase_autograd.float().mean().item():.4f}")

    # Clear gradients
    x.grad = None
    W_k.grad = None
    W_v.grad = None
    W_q.grad = None
    W_erase.grad = None
    W_write.grad = None

    # Forward using CUDA
    x_cuda = x.detach().clone().requires_grad_(True)
    W_k_cuda = W_k.detach().clone().requires_grad_(True)
    W_v_cuda = W_v.detach().clone().requires_grad_(True)
    W_q_cuda = W_q.detach().clone().requires_grad_(True)
    W_erase_cuda = W_erase.detach().clone().requires_grad_(True)
    W_write_cuda = W_write.detach().clone().requires_grad_(True)

    output_cuda, S_final_cuda = CUDANTMFunction.apply(
        x_cuda, S0.clone(), W_k_cuda, W_v_cuda, W_q_cuda,
        W_erase_cuda, W_write_cuda, True
    )
    loss_cuda = output_cuda.sum()

    # Compare forward
    print(f"\nForward comparison:")
    print(f"  output diff: {(output_py - output_cuda).abs().max().item():.6e}")
    print(f"  S_final diff: {(S_final_py - S_final_cuda).abs().max().item():.6e}")

    # Backward via CUDA
    loss_cuda.backward()

    dx_cuda = x_cuda.grad
    dW_k_cuda = W_k_cuda.grad
    dW_v_cuda = W_v_cuda.grad
    dW_q_cuda = W_q_cuda.grad
    dW_erase_cuda = W_erase_cuda.grad
    dW_write_cuda = W_write_cuda.grad

    print(f"\nCUDA backward gradients:")
    print(f"  dx: mean={dx_cuda.float().mean().item():.4f}, std={dx_cuda.float().std().item():.4f}")
    print(f"  dW_k: mean={dW_k_cuda.float().mean().item():.4f}")
    print(f"  dW_erase: mean={dW_erase_cuda.float().mean().item():.4f}")

    print(f"\nGradient comparison (max abs diff and relative error):")

    def grad_comparison(name, g1, g2):
        abs_diff = (g1 - g2).abs().max().item()
        max_val = max(g1.abs().max().item(), g2.abs().max().item(), 1e-8)
        rel_err = abs_diff / max_val
        mean_diff = (g1.float() - g2.float()).abs().mean().item()
        return abs_diff, rel_err, mean_diff

    for name, g_auto, g_cuda in [
        ("dx", dx_autograd, dx_cuda),
        ("dW_k", dW_k_autograd, dW_k_cuda),
        ("dW_v", dW_v_autograd, dW_v_cuda),
        ("dW_q", dW_q_autograd, dW_q_cuda),
        ("dW_erase", dW_erase_autograd, dW_erase_cuda),
        ("dW_write", dW_write_autograd, dW_write_cuda),
    ]:
        abs_diff, rel_err, mean_diff = grad_comparison(name, g_auto, g_cuda)
        status = "OK" if rel_err < 0.05 else ("WARN" if rel_err < 0.1 else "FAIL")
        print(f"  {name}: max_diff={abs_diff:.4e}, rel_err={rel_err:.4e}, mean_diff={mean_diff:.4e} [{status}]")

    # Summary
    print("\n  Summary:")
    all_ok = True
    for name, g_auto, g_cuda in [
        ("dx", dx_autograd, dx_cuda),
        ("dW_k", dW_k_autograd, dW_k_cuda),
        ("dW_v", dW_v_autograd, dW_v_cuda),
        ("dW_q", dW_q_autograd, dW_q_cuda),
        ("dW_erase", dW_erase_autograd, dW_erase_cuda),
        ("dW_write", dW_write_autograd, dW_write_cuda),
    ]:
        _, rel_err, _ = grad_comparison(name, g_auto, g_cuda)
        if rel_err >= 0.1:
            all_ok = False
            print(f"    {name}: FAILED (rel_err={rel_err:.4e} >= 0.1)")

    if all_ok:
        print("    All gradients within 10% relative error - ACCEPTABLE for bfloat16")
    else:
        print("    Some gradients exceed 10% relative error - NEEDS INVESTIGATION")


if __name__ == "__main__":
    print("Testing E74 Full Matrix NTM (update_type=2) gradient validation")
    print()

    debug_single_step()

    success = compare_outputs_and_gradients()
    test_numerical_gradient()
    test_pytorch_autograd_vs_cuda()

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print("""
FINDINGS:

1. Forward Pass:
   - Single timestep: S difference ~4e-3, output difference ~2e-3 (acceptable for bfloat16)
   - Multiple timesteps: Differences accumulate but remain reasonable

2. Weight Gradients (dW_k, dW_v, dW_q, dW_erase, dW_write):
   - All within 5% relative error - PASS
   - Mean differences are small, suggesting correct gradient flow

3. Input Gradient (dx):
   - 43% relative error - FAIL
   - CUDA backward computes dx = W_k @ d_k + W_v @ d_v + W_q @ d_q
   - BUT for NTM (update_type=2), it's missing:
     dx += W_erase @ d_erase + W_write @ d_write
   - This is a bug in e74_full_matrix_v2_gpu.cu.cc lines 1196-1230

BUG LOCATION: /home/erikg/elman/elman/cuda/lib/e74_full_matrix_v2_gpu.cu.cc
The backward pass needs to add gradient contributions through W_erase and W_write to dx
for update_type=2 (NTM). Currently only dW_erase and dW_write are computed, but the
chain rule also requires dx += W_erase @ d_erase_all + W_write @ d_write_all.

RECOMMENDATION:
Fix the CUDA backward to include:
    if (update_type_ == 2) {
        cublasGemmEx(..., W_erase, ..., d_erase_all, ..., dx, ..., CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(..., W_write, ..., d_write_all, ..., dx, ..., CUBLAS_GEMM_DEFAULT);
    }
""")

    if success:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED - dx gradient missing W_erase and W_write contributions")
    print("=" * 70)
