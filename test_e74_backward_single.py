#!/usr/bin/env python3
"""
Single timestep backward - trace every gradient term.
"""

import torch
import torch.nn.functional as F

def test_single_step_backward():
    torch.manual_seed(42)
    device = torch.device('cuda')
    dtype = torch.float32

    B, n = 1, 4  # Minimal size for debugging

    # Inputs - make sure it's a leaf tensor
    k_raw = (torch.randn(B, n, device=device, dtype=dtype) * 0.1).clone().detach().requires_grad_(True)
    S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

    print("=" * 60)
    print("Single Timestep Forward (T=1, S0=0)")
    print("=" * 60)

    # Step 1: k normalization
    k_norm_val = k_raw.norm(dim=-1, keepdim=True) + 1e-6
    k_norm = k_raw / k_norm_val

    print(f"k_raw: {k_raw[0]}")
    print(f"||k_raw||: {k_norm_val[0].item():.6f}")
    print(f"k_norm: {k_norm[0]}")

    # Step 2: retrieved = S0 @ k_norm = 0 (since S0=0)
    retrieved = torch.einsum('bij,bj->bi', S0, k_norm)
    print(f"retrieved: {retrieved[0]}")

    # Step 3: delta = v - retrieved = k_raw (since v=k_raw and retrieved=0)
    delta = k_raw - retrieved
    print(f"delta (v - retrieved): {delta[0]}")

    # Step 4: outer product
    outer = torch.einsum('bi,bj->bij', delta, k_norm)
    print(f"outer[0,:2,:2]:\n{outer[0,:2,:2]}")

    # Step 5: state update S = tanh(S0 + outer) = tanh(outer)
    S = torch.tanh(S0 + outer)
    print(f"S[0,:2,:2]:\n{S[0,:2,:2]}")

    # Step 6: Sq = S @ q where q = k_raw (original, not normalized)
    Sq = torch.einsum('bij,bj->bi', S, k_raw)
    print(f"Sq: {Sq[0]}")

    # Step 7: output = Sq * Sq * sigmoid(Sq)
    sig = torch.sigmoid(Sq)
    out = Sq * Sq * sig
    print(f"out: {out[0]}")

    # Backward with unit gradient
    d_out = torch.ones_like(out)

    print("\n" + "=" * 60)
    print("Manual Backward Derivation")
    print("=" * 60)

    # d_out -> d_Sq
    # out = Sq^2 * sig
    # d_out/d_Sq = 2*Sq*sig + Sq^2 * sig*(1-sig)
    d_Sq = d_out * (2 * Sq * sig + Sq * Sq * sig * (1 - sig))
    print(f"d_Sq: {d_Sq[0]}")

    # d_Sq -> d_S and d_q
    # Sq[i] = sum_j(S[i,j] * q[j])
    # d_S[i,j] = d_Sq[i] * q[j]
    # d_q[j] = sum_i(d_Sq[i] * S[i,j])
    d_S = torch.einsum('bi,bj->bij', d_Sq, k_raw)
    d_q = torch.einsum('bi,bij->bj', d_Sq, S)
    print(f"d_S[0,:2,:2]:\n{d_S[0,:2,:2]}")
    print(f"d_q: {d_q[0]}")

    # d_S -> d_pre_tanh
    # S = tanh(pre_tanh)
    # d_pre_tanh = d_S * (1 - S^2)
    d_pre_tanh = d_S * (1 - S * S)
    print(f"d_pre_tanh[0,:2,:2]:\n{d_pre_tanh[0,:2,:2]}")

    # d_pre_tanh = d_outer (since pre_tanh = S0 + outer and S0 doesn't depend on k)
    d_outer = d_pre_tanh

    # d_outer -> d_delta and d_k_norm
    # outer[i,j] = delta[i] * k_norm[j]
    # d_delta[i] = sum_j(d_outer[i,j] * k_norm[j])
    # d_k_norm[j] = sum_i(d_outer[i,j] * delta[i])
    d_delta = torch.einsum('bij,bj->bi', d_outer, k_norm)
    d_k_norm_from_outer = torch.einsum('bij,bi->bj', d_outer, delta)
    print(f"d_delta: {d_delta[0]}")
    print(f"d_k_norm (from outer): {d_k_norm_from_outer[0]}")

    # d_delta -> d_v and d_retrieved
    # delta = v - retrieved
    # d_v = d_delta
    # d_retrieved = -d_delta
    d_v = d_delta
    d_retrieved = -d_delta
    print(f"d_v (= d_delta): {d_v[0]}")
    print(f"d_retrieved: {d_retrieved[0]}")

    # d_retrieved -> d_S0 and d_k_norm (from retrieval)
    # But S0 = 0, so retrieved = 0, so d_S0 = 0
    # Also d_k_norm contribution from retrieval = S0^T @ d_retrieved = 0
    d_k_norm_from_retr = torch.zeros_like(k_norm)

    # Total d_k_norm
    d_k_norm = d_k_norm_from_outer + d_k_norm_from_retr
    print(f"d_k_norm (total): {d_k_norm[0]}")

    # d_k_norm -> d_k_raw (through normalization)
    # k_norm = k_raw / ||k_raw||
    # d_k_raw = d_k_norm / ||k_raw|| - k_raw * (k_raw . d_k_norm) / ||k_raw||^3
    norm = k_norm_val
    dot = (k_raw * d_k_norm).sum(dim=-1, keepdim=True)
    d_k_from_norm = d_k_norm / norm - k_raw * dot / (norm ** 3)
    print(f"d_k (from normalization): {d_k_from_norm[0]}")

    # Also d_v = d_delta contributes to d_k_raw (since v = k_raw)
    # And d_q contributes to d_k_raw (since q = k_raw)
    d_k_total = d_k_from_norm + d_v + d_q
    print(f"\nd_k_total (manual): {d_k_total[0]}")

    # Compare with autograd
    out.backward(d_out)
    d_k_autograd = k_raw.grad
    print(f"d_k (autograd):     {d_k_autograd[0]}")

    diff = (d_k_total - d_k_autograd).abs()
    print(f"\nDifference: {diff[0]}")
    print(f"Max diff: {diff.max():.8f}")


if __name__ == '__main__':
    test_single_step_backward()
