#!/usr/bin/env python3
"""
Reference backward pass using PyTorch autograd.
This is the ground truth that CUDA must match.
"""

import torch
import torch.nn.functional as F

def e74_forward_with_grad(k_all, S0, return_intermediates=False):
    """
    Forward pass with gradient tracking.

    k_all: [T, B, n] - projected values (k=v=q for tied_kvq)
    S0: [B, n, n] - initial state

    Returns: output [T, B, n], S_final [B, n, n]
    """
    T, B, n = k_all.shape

    S = S0.clone()
    outputs = []

    # Store for backward
    k_norms = []
    deltas = []
    retrieveds = []
    pre_tanhs = []
    Sqs = []

    for t in range(T):
        k_raw = k_all[t]  # [B, n] - original (unnormalized)

        # Normalize k
        k_norm_val = k_raw.norm(dim=-1, keepdim=True) + 1e-6
        k_norm = k_raw / k_norm_val

        # retrieved = S @ k_norm
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)

        # delta = v - retrieved (v = original k for tied_kvq)
        delta = k_raw - retrieved

        # outer product
        outer = torch.einsum('bi,bj->bij', delta, k_norm)

        # State update with tanh
        pre_tanh = S + outer
        S = torch.tanh(pre_tanh)

        # Output: Sq = S @ q (q = original k for tied_kvq)
        Sq = torch.einsum('bij,bj->bi', S, k_raw)

        # Self-gating: out = Sq * Sq * sigmoid(Sq)
        sig = torch.sigmoid(Sq)
        out = Sq * Sq * sig
        outputs.append(out)

        if return_intermediates:
            k_norms.append(k_norm)
            deltas.append(delta)
            retrieveds.append(retrieved)
            pre_tanhs.append(pre_tanh)
            Sqs.append(Sq)

    output = torch.stack(outputs, dim=0)

    if return_intermediates:
        return output, S, k_norms, deltas, retrieveds, pre_tanhs, Sqs
    return output, S


def test_backward():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.float32  # Use fp32 for reference

    T, B, n = 4, 2, 8  # Small for debugging

    # Create inputs with gradients
    k_all = torch.randn(T, B, n, device=device, dtype=dtype, requires_grad=True)
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype, requires_grad=False)

    # Forward
    output, S_final = e74_forward_with_grad(k_all, S0)

    print("Forward pass:")
    print(f"  output shape: {output.shape}")
    print(f"  S_final shape: {S_final.shape}")
    print(f"  output[-1,0,:4]: {output[-1,0,:4]}")

    # Create gradient
    torch.manual_seed(123)
    d_output = torch.randn_like(output)

    # Backward with autograd
    output.backward(d_output)
    d_k_autograd = k_all.grad.clone()

    print("\nBackward pass (autograd):")
    print(f"  d_k shape: {d_k_autograd.shape}")
    print(f"  d_k[0,0,:4]: {d_k_autograd[0,0,:4]}")
    print(f"  d_k[-1,0,:4]: {d_k_autograd[-1,0,:4]}")

    # Now compute manual backward to understand the gradient flow
    print("\n" + "=" * 60)
    print("Manual backward derivation:")
    print("=" * 60)

    k_all.grad.zero_()

    # Forward again to get intermediates
    k_all_detached = k_all.detach().requires_grad_(True)
    output, S_final, k_norms, deltas, retrieveds, pre_tanhs, Sqs = \
        e74_forward_with_grad(k_all_detached, S0, return_intermediates=True)

    # Manual backward
    d_k_manual = torch.zeros_like(k_all)
    d_S = torch.zeros(B, n, n, device=device, dtype=dtype)

    for t in range(T - 1, -1, -1):
        # Load values for this timestep
        k_raw = k_all_detached[t]
        k_norm = k_norms[t]
        delta = deltas[t]
        retrieved = retrieveds[t]
        Sq = Sqs[t]

        # Current state (after update at time t)
        if t == T - 1:
            S_curr = S_final
        else:
            # We need S at time t+1 but with gradients... this is complex
            pass

        # 1. d_out -> d_Sq
        d_out_t = d_output[t]
        sig = torch.sigmoid(Sq)
        d_Sq = d_out_t * (2 * Sq * sig + Sq * Sq * sig * (1 - sig))

        # 2. Sq = S @ q (q = k_raw)
        # d_S += outer(d_Sq, q)
        # d_q = S^T @ d_Sq
        d_S = d_S + torch.einsum('bi,bj->bij', d_Sq, k_raw)
        d_k_from_q = torch.einsum('bji,bj->bi', S_curr, d_Sq)  # S^T @ d_Sq

        d_k_manual[t] = d_k_from_q  # This is partial - need more terms

        print(f"\nt={t}:")
        print(f"  d_Sq[:4]: {d_Sq[0,:4]}")
        print(f"  d_k_from_q[:4]: {d_k_from_q[0,:4]}")

    # Verify against autograd
    output.backward(d_output)
    d_k_auto = k_all_detached.grad

    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)
    print(f"Manual d_k (partial): {d_k_manual[0,0,:4]}")
    print(f"Autograd d_k: {d_k_auto[0,0,:4]}")
    print("Note: Manual is incomplete - missing state update gradients")


if __name__ == '__main__':
    test_backward()
