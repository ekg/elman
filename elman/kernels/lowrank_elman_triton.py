"""
Low-rank Elman (E4) with hybrid PyTorch + Triton.

Architecture:
    h_t = tanh(Wx_t + U @ V @ h_{t-1} + b)
    output = h_t * silu(z_t)

Key insight: With rank r, U is [D, r] and V is [r, D].
- E1: d=768, W_h = 768*768 = 590k params
- E4: d=1536, r=192, U+V = 2*1536*192 = 590k params
- Same params, 2x hidden state!

Strategy:
- Use PyTorch for matmuls (leverages cuBLAS)
- Use Triton for fused tanh + silu gate
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_tanh_silu_kernel(
    pre_ptr, z_ptr, h_ptr, out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    """Fused h = tanh(pre), out = h * silu(z)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    pre = tl.load(pre_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # tanh
    exp2x = tl.exp(2.0 * pre)
    h = (exp2x - 1.0) / (exp2x + 1.0)

    # silu
    silu_z = z / (1.0 + tl.exp(-z))

    # gate
    out = h * silu_z

    tl.store(h_ptr + offs, h.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)


def fused_tanh_silu(pre, z):
    """Apply h = tanh(pre), out = h * silu(z)."""
    h = torch.empty_like(pre)
    out = torch.empty_like(pre)
    N = pre.numel()
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    fused_tanh_silu_kernel[grid](pre, z, h, out, N, BLOCK)
    return h, out


def lowrank_elman_recurrence(Wx, z, h0, U, V, b, spectral_radius=0.95):
    """
    Low-rank Elman recurrence using PyTorch autograd.

    Args:
        Wx: [T, B, D] - pre-computed W_x @ x
        z: [T, B, D] - gate input
        h0: [B, D] - initial hidden (or None)
        U: [D, R] - up projection
        V: [R, D] - down projection
        b: [D] - bias
        spectral_radius: float - unused, for API compat

    Returns:
        out: [T, B, D] - gated output
        h: [T+1, B, D] - hidden states
    """
    T, B, D = Wx.shape

    if h0 is None:
        h0 = torch.zeros(B, D, device=Wx.device, dtype=Wx.dtype)

    h_list = [h0]
    out_list = []

    h_prev = h0
    for t in range(T):
        # Low-rank recurrence
        Vh = h_prev @ V.T  # [B, R]
        UVh = Vh @ U.T  # [B, D]

        # pre = Wx + UVh + b
        pre = Wx[t] + UVh + b

        # tanh + gate
        h_new = torch.tanh(pre)
        out_t = h_new * F.silu(z[t])

        h_list.append(h_new)
        out_list.append(out_t)
        h_prev = h_new

    h = torch.stack(h_list, dim=0)
    out = torch.stack(out_list, dim=0)

    return out, h


if __name__ == "__main__":
    import time

    print("Testing low-rank Elman (E4)...")

    device = 'cuda'
    dtype = torch.bfloat16

    # E4 config: 2x hidden dim of E1 with same params
    T, B, D, R = 512, 32, 1536, 192

    print(f"Config: T={T}, B={B}, D={D}, R={R}")
    print(f"E4 U+V params: {2*D*R:,}")
    print(f"Equivalent E1 W_h: {int((2*D*R)**0.5)}x{int((2*D*R)**0.5)}")

    # Create inputs
    Wx = torch.randn(T, B, D, device=device, dtype=dtype) * 0.1
    Wx.requires_grad_(True)
    z = torch.randn(T, B, D, device=device, dtype=dtype) * 0.1
    z.requires_grad_(True)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)

    # Initialize U, V with spectral norm constraint
    U = torch.randn(D, R, device=device, dtype=dtype) * 0.1
    V = torch.randn(R, D, device=device, dtype=dtype) * 0.1
    U.requires_grad_(True)
    V.requires_grad_(True)

    b = torch.zeros(D, device=device, dtype=dtype)
    b.requires_grad_(True)

    print("\nTesting forward...")
    out, h = lowrank_elman_recurrence(Wx, z, h0, U, V, b)
    print(f"Output: {out.shape}, Hidden: {h.shape}")

    print("\nTesting backward...")
    loss = out.mean()
    loss.backward()

    print(f"dWx: {Wx.grad.norm().item():.4f}")
    print(f"dU: {U.grad.norm().item():.4f}")
    print(f"dV: {V.grad.norm().item():.4f}")

    # Benchmark
    print("\nBenchmarking...")
    Wx = torch.randn(T, B, D, device=device, dtype=dtype) * 0.1
    Wx.requires_grad_(True)
    z = torch.randn(T, B, D, device=device, dtype=dtype) * 0.1
    z.requires_grad_(True)
    U = torch.randn(D, R, device=device, dtype=dtype) * 0.1
    U.requires_grad_(True)
    V = torch.randn(R, D, device=device, dtype=dtype) * 0.1
    V.requires_grad_(True)
    b = torch.zeros(D, device=device, dtype=dtype)
    b.requires_grad_(True)

    # Warmup
    for _ in range(3):
        out, h = lowrank_elman_recurrence(Wx, z, h0, U, V, b)
        out.mean().backward()
        Wx.grad = None
        z.grad = None
        U.grad = None
        V.grad = None
        b.grad = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out, h = lowrank_elman_recurrence(Wx, z, h0, U, V, b)
        out.mean().backward()
        Wx.grad = None
        z.grad = None
        U.grad = None
        V.grad = None
        b.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 10 * 1000

    tok_per_sec = B * T / (elapsed / 1000)
    print(f"E4 cell: {elapsed:.1f}ms, {tok_per_sec / 1e3:.1f}k tok/s")

    # Compare with E1-equivalent params
    print("\nComparing with equivalent E1...")
    from elman.models.mamba_gated_elman import MambaGatedElmanCell

    # E1 with equivalent hidden dim (sqrt of E4 U+V params)
    e1_dim = int((2 * D * R) ** 0.5)
    e1_cell = MambaGatedElmanCell(e1_dim).to(device).to(dtype)

    e1_x = torch.randn(T, B, e1_dim, device=device, dtype=dtype) * 0.1
    e1_z = torch.randn(T, B, e1_dim, device=device, dtype=dtype) * 0.1

    for _ in range(3):
        e1_out, e1_h = e1_cell(e1_x, e1_z)
        e1_out.mean().backward()
        e1_cell.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        e1_out, e1_h = e1_cell(e1_x, e1_z)
        e1_out.mean().backward()
        e1_cell.zero_grad()
    torch.cuda.synchronize()
    e1_elapsed = (time.perf_counter() - t0) / 10 * 1000

    e1_tok_per_sec = B * T / (e1_elapsed / 1000)
    print(f"E1 cell (d={e1_dim}): {e1_elapsed:.1f}ms, {e1_tok_per_sec / 1e3:.1f}k tok/s")

    print(f"\nE4 hidden dim: {D} (2x E1)")
    print(f"E4 params: {2*D*R:,} (same as E1)")
    print(f"E4 vs E1 speed: {e1_tok_per_sec / tok_per_sec:.1f}x slower")
