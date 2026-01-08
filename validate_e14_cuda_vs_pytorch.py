#!/usr/bin/env python3
"""
Validate E14 Matrix State Elman - CUDA vs PyTorch Reference

This script verifies that the CUDA kernel produces mathematically
identical results to the PyTorch reference implementation.
"""

import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import torch
import torch.nn.functional as F
import hasty_pytorch_lib

print("=" * 70)
print("E14 Matrix State Elman: CUDA vs PyTorch Validation")
print("=" * 70)

device = 'cuda'
dtype = torch.bfloat16

# Test dimensions
B, T, d, k = 2, 8, 64, 32

torch.manual_seed(42)

# Create random weights
# CUDA expects W_val, W_query to be [d, k] - will transpose for F.linear
W_key = torch.randn(d, d, device=device, dtype=dtype) * 0.02    # [d, d]
b_key = torch.zeros(d, device=device, dtype=dtype)
W_val = torch.randn(d, k, device=device, dtype=dtype) * 0.02    # [d, k] for CUDA
b_val = torch.zeros(k, device=device, dtype=dtype)
W_query = torch.randn(d, k, device=device, dtype=dtype) * 0.02  # [d, k] for CUDA
b_query = torch.zeros(k, device=device, dtype=dtype)
W_decay = torch.randn(d, d, device=device, dtype=dtype) * 0.02  # [d, d]
b_decay = torch.full((d,), 3.0, device=device, dtype=dtype)  # sigmoid(3) ≈ 0.95

# Create inputs
x = torch.randn(T, B, d, device=device, dtype=dtype)  # pre-activated
z = torch.randn(T, B, d, device=device, dtype=dtype)  # gate
H0 = torch.zeros(B, d, k, device=device, dtype=dtype)

print(f"\nTest configuration:")
print(f"  Batch size: {B}")
print(f"  Sequence length: {T}")
print(f"  Model dimension (d): {d}")
print(f"  State dimension (k): {k}")
print(f"  Dtype: {dtype}")
print()

# =============================================================================
# PyTorch Reference Forward
# =============================================================================
print("Running PyTorch reference forward...")

def pytorch_forward(x, z, H0, W_key, b_key, W_val, b_val, W_query, b_query, W_decay, b_decay):
    """Pure PyTorch implementation of forward pass."""
    T, B, d = x.shape
    k = W_val.shape[1]

    H = H0.clone()
    H_list = [H.clone()]
    output_list = []
    key_list = []
    value_list = []
    decay_list = []
    query_list = []

    for t in range(T):
        x_t = x[t]  # [B, d]
        z_t = z[t]  # [B, d]

        # Projections (transpose W_val, W_query for F.linear which expects [out, in])
        key = torch.tanh(F.linear(x_t, W_key, b_key))  # [B, d]
        value = F.linear(x_t, W_val.t(), b_val)  # [B, k] - W_val is [d,k], need [k,d] for F.linear
        decay = torch.sigmoid(F.linear(x_t, W_decay, b_decay))  # [B, d]
        query = F.linear(x_t, W_query.t(), b_query)  # [B, k] - W_query is [d,k], need [k,d]

        # State update: H_new = decay * H + key ⊗ value
        H_new = decay.unsqueeze(-1) * H + key.unsqueeze(-1) * value.unsqueeze(1)

        # Output: pre_out = H_new @ query, then gate
        pre_out = torch.bmm(H_new, query.unsqueeze(-1)).squeeze(-1)  # [B, d]
        output = pre_out * F.silu(z_t)

        H = H_new
        H_list.append(H.clone())
        output_list.append(output)
        key_list.append(key)
        value_list.append(value)
        decay_list.append(decay)
        query_list.append(query)

    H_all = torch.stack(H_list, dim=0)  # [T+1, B, d, k]
    output_all = torch.stack(output_list, dim=0)  # [T, B, d]

    return H_all, output_all, key_list, value_list, decay_list, query_list

H_all_pt, output_pt, key_pt, value_pt, decay_pt, query_pt = pytorch_forward(
    x, z, H0, W_key, b_key, W_val, b_val, W_query, b_query, W_decay, b_decay
)

print(f"  H_all shape: {H_all_pt.shape}")
print(f"  output shape: {output_pt.shape}")

# =============================================================================
# CUDA Forward
# =============================================================================
print("\nRunning CUDA forward...")

# CUDA expects: training, x, z, H0, W_key, b_key, W_val, b_val, W_query, b_query, W_decay, b_decay
H_all_cuda, output_cuda, key_cache, value_cache, decay_cache, query_cache = \
    hasty_pytorch_lib.matrix_state_elman_forward(
        True,  # training
        x.contiguous(),
        z.contiguous(),
        H0.contiguous(),
        W_key.contiguous(),
        b_key.contiguous(),
        W_val.contiguous(),
        b_val.contiguous(),
        W_query.contiguous(),
        b_query.contiguous(),
        W_decay.contiguous(),
        b_decay.contiguous()
    )

print(f"  H_all shape: {H_all_cuda.shape}")
print(f"  output shape: {output_cuda.shape}")

# =============================================================================
# Compare Forward Results
# =============================================================================
print("\n" + "=" * 70)
print("Forward Pass Comparison")
print("=" * 70)

# Compare outputs
output_diff = (output_cuda - output_pt).abs()
output_max_diff = output_diff.max().item()
output_mean_diff = output_diff.mean().item()
output_rel_diff = (output_diff / (output_pt.abs() + 1e-8)).max().item()

print(f"\nOutput comparison:")
print(f"  Max absolute diff: {output_max_diff:.2e}")
print(f"  Mean absolute diff: {output_mean_diff:.2e}")
print(f"  Max relative diff: {output_rel_diff:.2e}")

# Compare hidden states
H_diff = (H_all_cuda - H_all_pt).abs()
H_max_diff = H_diff.max().item()
H_mean_diff = H_diff.mean().item()

print(f"\nHidden state comparison:")
print(f"  Max absolute diff: {H_max_diff:.2e}")
print(f"  Mean absolute diff: {H_mean_diff:.2e}")

# Compare caches
key_stack = torch.stack(key_pt, dim=0)
value_stack = torch.stack(value_pt, dim=0)
decay_stack = torch.stack(decay_pt, dim=0)
query_stack = torch.stack(query_pt, dim=0)

key_diff = (key_cache - key_stack).abs().max().item()
value_diff = (value_cache - value_stack).abs().max().item()
decay_diff = (decay_cache - decay_stack).abs().max().item()
query_diff = (query_cache - query_stack).abs().max().item()

print(f"\nCache comparison:")
print(f"  Key cache max diff: {key_diff:.2e}")
print(f"  Value cache max diff: {value_diff:.2e}")
print(f"  Decay cache max diff: {decay_diff:.2e}")
print(f"  Query cache max diff: {query_diff:.2e}")

# =============================================================================
# Backward Pass Validation
# =============================================================================
print("\n" + "=" * 70)
print("Backward Pass Comparison")
print("=" * 70)

# Create gradient output
d_output = torch.randn_like(output_pt)

# PyTorch backward
print("\nRunning PyTorch reference backward...")

# Need to recompute with grad tracking
x_pt = x.clone().requires_grad_(True)
z_pt = z.clone().requires_grad_(True)
W_key_pt = W_key.clone().requires_grad_(True)
b_key_pt = b_key.clone().requires_grad_(True)
W_val_pt = W_val.clone().requires_grad_(True)
b_val_pt = b_val.clone().requires_grad_(True)
W_query_pt = W_query.clone().requires_grad_(True)
b_query_pt = b_query.clone().requires_grad_(True)
W_decay_pt = W_decay.clone().requires_grad_(True)
b_decay_pt = b_decay.clone().requires_grad_(True)

# Forward with grad
H = H0.clone()
output_list = []
for t in range(T):
    x_t = x_pt[t]
    z_t = z_pt[t]

    key = torch.tanh(F.linear(x_t, W_key_pt, b_key_pt))
    value = F.linear(x_t, W_val_pt.t(), b_val_pt)  # W_val is [d,k], need [k,d]
    decay = torch.sigmoid(F.linear(x_t, W_decay_pt, b_decay_pt))
    query = F.linear(x_t, W_query_pt.t(), b_query_pt)  # W_query is [d,k], need [k,d]

    H = decay.unsqueeze(-1) * H + key.unsqueeze(-1) * value.unsqueeze(1)
    pre_out = torch.bmm(H, query.unsqueeze(-1)).squeeze(-1)
    output = pre_out * F.silu(z_t)
    output_list.append(output)

output_pt_grad = torch.stack(output_list, dim=0)
loss = (output_pt_grad * d_output).sum()
loss.backward()

print("  PyTorch gradients computed")

# CUDA backward
print("\nRunning CUDA backward...")

dx_cuda, dz_cuda, dW_key_cuda, db_key_cuda, dW_val_cuda, db_val_cuda, \
    dW_query_cuda, db_query_cuda, dW_decay_cuda, db_decay_cuda = \
    hasty_pytorch_lib.matrix_state_elman_backward(
        W_key.contiguous(),
        b_key.contiguous(),
        W_val.contiguous(),
        b_val.contiguous(),
        W_query.contiguous(),
        b_query.contiguous(),
        W_decay.contiguous(),
        b_decay.contiguous(),
        x.contiguous(),
        z.contiguous(),
        H_all_cuda.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        decay_cache.contiguous(),
        query_cache.contiguous(),
        d_output.contiguous()
    )

print("  CUDA gradients computed")

# Compare gradients
print("\nGradient comparison:")

dx_diff = (dx_cuda - x_pt.grad).abs()
print(f"  dx: max={dx_diff.max().item():.2e}, mean={dx_diff.mean().item():.2e}")

dz_diff = (dz_cuda - z_pt.grad).abs()
print(f"  dz: max={dz_diff.max().item():.2e}, mean={dz_diff.mean().item():.2e}")

dW_key_diff = (dW_key_cuda - W_key_pt.grad).abs()
print(f"  dW_key: max={dW_key_diff.max().item():.2e}, mean={dW_key_diff.mean().item():.2e}")

db_key_diff = (db_key_cuda - b_key_pt.grad).abs()
print(f"  db_key: max={db_key_diff.max().item():.2e}, mean={db_key_diff.mean().item():.2e}")

# W_val_pt.grad shape is [d, k], CUDA dW_val is now correctly [d, k] row-major
dW_val_diff = (dW_val_cuda - W_val_pt.grad).abs()
print(f"  dW_val: max={dW_val_diff.max().item():.2e}, mean={dW_val_diff.mean().item():.2e}")

db_val_diff = (db_val_cuda - b_val_pt.grad).abs()
print(f"  db_val: max={db_val_diff.max().item():.2e}, mean={db_val_diff.mean().item():.2e}")

# W_query_pt.grad is [d, k], CUDA dW_query is now correctly [d, k]
dW_query_diff = (dW_query_cuda - W_query_pt.grad).abs()
print(f"  dW_query: max={dW_query_diff.max().item():.2e}, mean={dW_query_diff.mean().item():.2e}")

db_query_diff = (db_query_cuda - b_query_pt.grad).abs()
print(f"  db_query: max={db_query_diff.max().item():.2e}, mean={db_query_diff.mean().item():.2e}")

dW_decay_diff = (dW_decay_cuda - W_decay_pt.grad).abs()
print(f"  dW_decay: max={dW_decay_diff.max().item():.2e}, mean={dW_decay_diff.mean().item():.2e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

# Use BF16 tolerance (about 1e-2 relative error is acceptable)
tol = 1e-2

forward_pass = output_max_diff < tol and H_max_diff < tol
backward_pass = dx_diff.max().item() < tol and dz_diff.max().item() < tol

if forward_pass:
    print("Forward pass:  PASSED")
else:
    print(f"Forward pass:  FAILED (output diff={output_max_diff:.2e}, H diff={H_max_diff:.2e})")

if backward_pass:
    print("Backward pass: PASSED")
else:
    print(f"Backward pass: FAILED (dx diff={dx_diff.max().item():.2e}, dz diff={dz_diff.max().item():.2e})")

if forward_pass and backward_pass:
    print("\nCUDA and PyTorch implementations match mathematically!")
else:
    print("\nWARNING: Implementations differ. Check the kernel.")

print("=" * 70)
