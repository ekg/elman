#!/usr/bin/env python3
"""Test B2B GEMM accuracy vs reference across multiple steps."""

import torch
import sys
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, '/home/erikg/elman/elman/cuda')

import hasty_pytorch_lib

def reference_forward(x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b):
    """Reference implementation using PyTorch in float32."""
    steps, batch, dim = x.shape
    rank = U_h.shape[1]
    dtype = x.dtype

    # Work in float32 for accuracy
    h0_f = h0.float()
    U_h_f, V_h_f = U_h.float(), V_h.float()
    U_x_f, V_x_f = U_x.float(), V_x.float()
    U_z_f, V_z_f = U_z.float(), V_z.float()
    b_f = b.float()

    h_all = [h0_f]
    out_all = []

    for t in range(steps):
        h_prev = h_all[-1]
        x_t = x[t].float()

        # h-path: h_prev @ V_h^T @ U_h^T
        h_part = torch.mm(h_prev, V_h_f.T)  # [batch, rank]
        h_part = torch.mm(h_part, U_h_f.T)  # [batch, dim]

        # x-path: x_t @ V_x^T @ U_x^T
        x_part = torch.mm(x_t, V_x_f.T)  # [batch, rank]
        x_part = torch.mm(x_part, U_x_f.T)  # [batch, dim]

        # z-path: x_t @ V_z^T @ U_z^T
        z_part = torch.mm(x_t, V_z_f.T)  # [batch, rank]
        z_part = torch.mm(z_part, U_z_f.T)  # [batch, dim]

        # h = tanh(h_part + x_part + b)
        pre_act = h_part + x_part + b_f
        h_new = torch.tanh(pre_act)
        h_all.append(h_new)

        # output = h * silu(z)
        silu_z = z_part * torch.sigmoid(z_part)
        out = h_new * silu_z
        out_all.append(out)

    # Convert back to original dtype
    h_result = torch.stack(h_all, dim=0).to(dtype)
    out_result = torch.stack(out_all, dim=0).to(dtype)

    return h_result, out_result

def main():
    torch.manual_seed(42)

    batch = 256
    dim = 1536
    rank = 256
    steps = 2
    device = 'cuda'
    dtype = torch.bfloat16

    # Create data
    x = torch.randn(steps, batch, dim, device=device, dtype=dtype)
    h0 = torch.zeros(batch, dim, device=device, dtype=dtype)

    U_h = torch.randn(dim, rank, device=device, dtype=dtype) * 0.1
    V_h = torch.randn(rank, dim, device=device, dtype=dtype) * 0.1
    U_x = torch.randn(dim, rank, device=device, dtype=dtype) * 0.1
    V_x = torch.randn(rank, dim, device=device, dtype=dtype) * 0.1
    U_z = torch.randn(dim, rank, device=device, dtype=dtype) * 0.1
    V_z = torch.randn(rank, dim, device=device, dtype=dtype) * 0.1
    b = torch.zeros(dim, device=device, dtype=dtype)

    print(f"Test: batch={batch}, dim={dim}, rank={rank}, steps={steps}")
    print()

    # Run reference
    h_ref, out_ref = reference_forward(x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)

    # Run FUSED kernel
    h_fused, out_fused, _ = hasty_pytorch_lib.pure_lowrank_elman_forward_fused(
        False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
    torch.cuda.synchronize()

    # Run B2B kernel
    h_b2b, out_b2b, _ = hasty_pytorch_lib.b2b_lowrank_elman_forward(
        False, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
    torch.cuda.synchronize()

    # Compare hidden states vs reference
    print("Hidden states vs REFERENCE:")
    for t in range(steps + 1):
        diff_fused = (h_fused[t] - h_ref[t]).abs()
        diff_b2b = (h_b2b[t] - h_ref[t]).abs()
        print(f"  h[{t}]: FUSED max={diff_fused.max():.6f} mean={diff_fused.mean():.6f}")
        print(f"        B2B   max={diff_b2b.max():.6f} mean={diff_b2b.mean():.6f}")
        print()

    # Summary
    print("SUMMARY:")
    h_fused_err = (h_fused - h_ref).abs().max().item()
    h_b2b_err = (h_b2b - h_ref).abs().max().item()
    print(f"  FUSED max error vs reference: {h_fused_err:.6f}")
    print(f"  B2B max error vs reference:   {h_b2b_err:.6f}")

    if h_b2b_err < h_fused_err:
        print(f"  -> B2B is {h_fused_err/h_b2b_err:.1f}x MORE accurate than FUSED!")
    else:
        print(f"  -> FUSED is {h_b2b_err/h_fused_err:.1f}x more accurate than B2B")

if __name__ == '__main__':
    main()
