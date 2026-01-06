"""
Triton-accelerated low-rank Elman recurrence (E4).

Architecture:
    h_t = tanh(Wx_t + U @ V @ h_{t-1} + b)
    output = h_t * silu(z_t)

Where U is [D, R] and V is [R, D], giving low-rank approximation to W_h.

Strategy: Use PyTorch for the small low-rank matmuls, Triton for fused tanh/gate.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_tanh_silu_gate_kernel(
    pre_ptr, z_ptr, out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    """Fused tanh(pre) * silu(z) kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    pre = tl.load(pre_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # tanh via exp
    exp2x = tl.exp(2.0 * pre)
    h = (exp2x - 1.0) / (exp2x + 1.0)

    # silu = z * sigmoid(z)
    silu_z = z / (1.0 + tl.exp(-z))

    out = (h * silu_z).to(tl.bfloat16)
    tl.store(out_ptr + offs, out, mask=mask)


def fused_tanh_silu_gate(pre, z):
    """Apply fused tanh(pre) * silu(z)."""
    out = torch.empty_like(pre)
    N = pre.numel()
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    fused_tanh_silu_gate_kernel[grid](pre, z, out, N, BLOCK)
    return out


def lowrank_recurrence(Wx, z, h0, U, V, b):
    """
    Low-rank Elman recurrence using PyTorch autograd.

    Args:
        Wx: [T, B, D] - pre-computed W_x @ x
        z: [T, B, D] - gate input
        h0: [B, D] - initial hidden state (or None)
        U: [D, R] - up projection
        V: [R, D] - down projection
        b: [D] - bias

    Returns:
        out: [T, B, D] - gated output
        h: [T+1, B, D] - all hidden states
    """
    T, B, D = Wx.shape

    if h0 is None:
        h0 = torch.zeros(B, D, device=Wx.device, dtype=Wx.dtype)

    h_list = [h0]
    out_list = []

    h_prev = h0
    for t in range(T):
        # Low-rank recurrence: U @ V @ h_prev
        v = h_prev @ V.T  # [B, R]
        Uh = v @ U.T  # [B, D]

        # pre = Wx + Uh + b
        pre = Wx[t] + Uh + b

        # h_new = tanh(pre)
        h_new = torch.tanh(pre)
        h_list.append(h_new)

        # output = h_new * silu(z)
        out = h_new * F.silu(z[t])
        out_list.append(out)

        h_prev = h_new

    h = torch.stack(h_list, dim=0)
    out = torch.stack(out_list, dim=0)

    return out, h


if __name__ == "__main__":
    print("Testing low-rank recurrence...")

    device = 'cuda'
    dtype = torch.bfloat16

    T, B, D, R = 512, 32, 768, 64

    # Create inputs (multiply first, then set requires_grad)
    Wx = torch.randn(T, B, D, device=device, dtype=dtype) * 0.1
    Wx.requires_grad_(True)
    z = torch.randn(T, B, D, device=device, dtype=dtype) * 0.1
    z.requires_grad_(True)
    h0 = torch.zeros(B, D, device=device, dtype=dtype)
    U = torch.randn(D, R, device=device, dtype=dtype) * 0.1
    U.requires_grad_(True)
    V = torch.randn(R, D, device=device, dtype=dtype) * 0.1
    V.requires_grad_(True)
    b = torch.zeros(D, device=device, dtype=dtype)
    b.requires_grad_(True)

    # Forward
    print("Testing forward...")
    out, h = lowrank_recurrence(Wx, z, h0, U, V, b)
    print(f"Output shape: {out.shape}")
    print(f"Hidden shape: {h.shape}")

    # Backward
    print("Testing backward...")
    loss = out.mean()  # Use mean instead of sum to avoid large gradients
    loss.backward()
    print(f"dWx norm: {Wx.grad.norm().item():.4f}")
    print(f"dU norm: {U.grad.norm().item():.4f}")
    print(f"dV norm: {V.grad.norm().item():.4f}")

    # Benchmark
    import time

    # Fresh tensors
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
    for _ in range(5):
        out, h = lowrank_recurrence(Wx, z, h0, U, V, b)
        out.mean().backward()
        Wx.grad = None
        z.grad = None
        U.grad = None
        V.grad = None
        b.grad = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        out, h = lowrank_recurrence(Wx, z, h0, U, V, b)
        out.mean().backward()
        Wx.grad = None
        z.grad = None
        U.grad = None
        V.grad = None
        b.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 20 * 1000

    tok_per_sec = B * T / (elapsed / 1000)
    print(f"\nBenchmark: {elapsed:.1f}ms, {tok_per_sec / 1e6:.2f}M tok/s")
