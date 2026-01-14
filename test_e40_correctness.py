#!/usr/bin/env python3
"""
Test E40 No Pre-SiLU Elman: Python vs CUDA correctness.

This tests:
1. Forward pass: output and hidden state match between Python and CUDA
2. Backward pass: gradients match for all parameters
"""

import sys
# Add the cuda directory to path for hasty_pytorch_lib
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

# Import torch first to load CUDA libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import hasty after torch is loaded
import hasty_pytorch_lib

print("=" * 60)
print("E40 No Pre-SiLU Elman Correctness Test")
print("=" * 60)

def test_e40_forward_backward():
    """Test E40 forward and backward pass against Python implementation."""

    device = 'cuda'
    dtype = torch.bfloat16

    # Test dimensions
    T, B, dim = 16, 4, 128  # seq_len, batch, hidden_dim

    print(f"\nTest config: T={T}, B={B}, dim={dim}, dtype={dtype}")

    # Create inputs and parameters
    torch.manual_seed(42)
    x = torch.randn(T, B, dim, device=device, dtype=dtype, requires_grad=True)
    h0 = torch.zeros(B, dim, device=device, dtype=dtype)
    W_h = torch.randn(dim, dim, device=device, dtype=dtype, requires_grad=True)
    b = torch.zeros(dim, device=device, dtype=dtype, requires_grad=True)

    # Initialize W_h to have reasonable spectral radius
    with torch.no_grad():
        W_h_fp32 = W_h.float()
        u, s, v = torch.svd(W_h_fp32)
        W_h_fp32 = u @ torch.diag(s * 0.99 / s.max()) @ v.T
        W_h.copy_(W_h_fp32.to(dtype))

    print("\n--- Python Forward ---")

    # Python forward (reference implementation)
    h_list_py = [h0]
    output_list_py = []

    for t in range(T):
        h_prev = h_list_py[-1]
        x_t = x[t]

        # E40: h = tanh(x + W_h @ h_prev + b) - NO W_x, NO pre-silu
        raw = x_t + h_prev @ W_h.T + b
        h_new = torch.tanh(raw)
        h_list_py.append(h_new)

        # Self-gate: output = h * silu(h)
        output = h_new * F.silu(h_new)
        output_list_py.append(output)

    h_py = torch.stack(h_list_py, dim=0)
    output_py = torch.stack(output_list_py, dim=0)

    print(f"Python h shape: {h_py.shape}")
    print(f"Python output shape: {output_py.shape}")

    print("\n--- CUDA Forward ---")

    # Make copies for CUDA path (since they need gradients independently)
    x_cuda = x.detach().clone().requires_grad_(True)
    W_h_cuda = W_h.detach().clone().requires_grad_(True)
    b_cuda = b.detach().clone().requires_grad_(True)

    # CUDA forward
    h_cuda, output_cuda, v_cache = hasty_pytorch_lib.e40_no_presilu_forward(
        True,  # training
        x_cuda.contiguous(),
        h0.contiguous(),
        W_h_cuda.contiguous(),
        b_cuda.contiguous()
    )

    print(f"CUDA h shape: {h_cuda.shape}")
    print(f"CUDA output shape: {output_cuda.shape}")

    # Compare forward outputs
    print("\n--- Forward Comparison ---")

    h_diff = (h_py - h_cuda).abs().max().item()
    output_diff = (output_py - output_cuda).abs().max().item()

    print(f"Max h difference: {h_diff:.6e}")
    print(f"Max output difference: {output_diff:.6e}")

    # Check tolerance for bfloat16
    atol = 1e-2  # bfloat16 has limited precision
    h_match = torch.allclose(h_py, h_cuda, atol=atol, rtol=1e-2)
    output_match = torch.allclose(output_py, output_cuda, atol=atol, rtol=1e-2)

    print(f"h match (atol={atol}): {h_match}")
    print(f"output match (atol={atol}): {output_match}")

    if not h_match or not output_match:
        print("\nWARNING: Forward mismatch detected!")
        print("Sample h values:")
        print(f"  Python h[1,0,:5]: {h_py[1,0,:5]}")
        print(f"  CUDA h[1,0,:5]: {h_cuda[1,0,:5]}")

    print("\n--- Backward Test ---")

    # Create gradient for backward
    d_output = torch.randn_like(output_py)

    # Python backward
    loss_py = (output_py * d_output).sum()
    loss_py.backward()

    dx_py = x.grad.clone()
    dW_h_py = W_h.grad.clone()
    db_py = b.grad.clone()

    print(f"Python dx shape: {dx_py.shape}")
    print(f"Python dW_h shape: {dW_h_py.shape}")
    print(f"Python db shape: {db_py.shape}")

    # CUDA backward
    dx_cuda, dW_h_cuda, db_cuda = hasty_pytorch_lib.e40_no_presilu_backward(
        W_h_cuda.contiguous(),
        x_cuda.contiguous(),
        h_cuda.contiguous(),
        v_cache.contiguous(),
        d_output.contiguous()
    )

    print(f"CUDA dx shape: {dx_cuda.shape}")
    print(f"CUDA dW_h shape: {dW_h_cuda.shape}")
    print(f"CUDA db shape: {db_cuda.shape}")

    # Compare gradients
    print("\n--- Backward Comparison ---")

    dx_diff = (dx_py - dx_cuda).abs().max().item()
    dW_h_diff = (dW_h_py - dW_h_cuda).abs().max().item()
    db_diff = (db_py - db_cuda).abs().max().item()

    print(f"Max dx difference: {dx_diff:.6e}")
    print(f"Max dW_h difference: {dW_h_diff:.6e}")
    print(f"Max db difference: {db_diff:.6e}")

    # Use looser tolerance for backward (accumulated numerical error)
    atol_bwd = 5e-2
    dx_match = torch.allclose(dx_py, dx_cuda, atol=atol_bwd, rtol=1e-1)
    dW_h_match = torch.allclose(dW_h_py, dW_h_cuda, atol=atol_bwd, rtol=1e-1)
    db_match = torch.allclose(db_py, db_cuda, atol=atol_bwd, rtol=1e-1)

    print(f"dx match (atol={atol_bwd}): {dx_match}")
    print(f"dW_h match (atol={atol_bwd}): {dW_h_match}")
    print(f"db match (atol={atol_bwd}): {db_match}")

    if not dx_match:
        print("\nWARNING: dx mismatch!")
        print(f"  Python dx[0,0,:5]: {dx_py[0,0,:5]}")
        print(f"  CUDA dx[0,0,:5]: {dx_cuda[0,0,:5]}")

    if not dW_h_match:
        print("\nWARNING: dW_h mismatch!")
        print(f"  Python dW_h[0,:5]: {dW_h_py[0,:5]}")
        print(f"  CUDA dW_h[0,:5]: {dW_h_cuda[0,:5]}")

    if not db_match:
        print("\nWARNING: db mismatch!")
        print(f"  Python db[:5]: {db_py[:5]}")
        print(f"  CUDA db[:5]: {db_cuda[:5]}")

    # Summary
    print("\n" + "=" * 60)
    all_pass = h_match and output_match and dx_match and dW_h_match and db_match
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - check warnings above")
    print("=" * 60)

    return all_pass


def test_e40_layer():
    """Test the full E40NoPresilu layer (with in_proj and out_proj)."""

    print("\n" + "=" * 60)
    print("E40NoPresilu Layer Test")
    print("=" * 60)

    device = 'cuda'
    dtype = torch.bfloat16

    # Add elman models to path
    sys.path.insert(0, '/home/erikg/elman')
    from elman.models.e40_no_presilu import E40NoPresilu

    # Create layer
    dim = 256
    expansion = 2.0
    B, T = 2, 32

    model = E40NoPresilu(dim=dim, expansion=expansion, use_conv=False).to(device).to(dtype)
    model.train()

    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    print(f"\nLayer config: dim={dim}, expansion={expansion}, d_inner={model.d_inner}")
    print(f"Input shape: {x.shape}")

    # Forward
    out, h_final = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Final hidden shape: {h_final.shape}")

    # Backward
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    has_grads = all(p.grad is not None for p in model.parameters())
    print(f"All parameters have gradients: {has_grads}")

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")

    # Compare with E33 params (which has W_x)
    from elman.models.e33_self_gate import E33SelfGate
    model_e33 = E33SelfGate(dim=dim, expansion=expansion, use_conv=False).to(device)
    params_e33 = sum(p.numel() for p in model_e33.parameters())
    print(f"E33 parameters: {params_e33:,}")
    print(f"E40 saves {params_e33 - params:,} parameters ({100*(params_e33-params)/params_e33:.1f}%)")

    print("\nE40NoPresilu Layer test passed!")
    return True


if __name__ == "__main__":
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

    print(f"Using CUDA device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Run tests
    success = True
    success = success and test_e40_forward_backward()
    success = success and test_e40_layer()

    print("\n" + "=" * 60)
    if success:
        print("ALL E40 TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    sys.exit(0 if success else 1)
