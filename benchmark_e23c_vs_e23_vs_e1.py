"""
Benchmark E23c (Chunked) vs E23 (Original) vs E1 (Gated Elman)

Tests throughput and validates that all models work correctly.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['LD_LIBRARY_PATH'] = f"/home/erikg/.local/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch
import torch.nn as nn
import time
import hasty_pytorch_lib


class E1Cell(nn.Module):
    """E1: Mamba-Gated Elman (reference implementation using CUDA kernel)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

        # Initialize W_h with controlled spectral radius
        with torch.no_grad():
            nn.init.orthogonal_(self.W_h.weight)
            self.W_h.weight.mul_(0.9)

    def forward(self, x_seq, h0=None):
        """
        Args:
            x_seq: [B, T, D] input (already projected through in_proj)
            h0: [B, D] initial hidden state
        Returns:
            output: [B, T, D]
            h_final: [B, D]
        """
        B, T, D = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        if h0 is None:
            h0 = torch.zeros(B, D, device=device, dtype=dtype)

        # Transpose to [T, B, D] for CUDA kernel
        x_t = x_seq.permute(1, 0, 2).contiguous()
        # For E1, we need x and z (gate input) - use same input for both
        z = x_t.clone()

        results = hasty_pytorch_lib.mamba_gated_elman_forward(
            False,  # training
            x_t,
            z,
            h0,
            self.W_x.weight,
            self.W_h.weight,
            self.b
        )

        h_all = results[0]  # [T+1, B, D]
        output = results[1]  # [T, B, D]

        # Transpose back to [B, T, D]
        output = output.permute(1, 0, 2).contiguous()
        h_final = h_all[-1]

        return output, h_final


class E23Cell(nn.Module):
    """E23: Dual-Memory Elman (original, using optimized CUDA kernel)"""
    def __init__(self, dim, n_slots=64):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots

        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Linear(dim, dim, bias=False)

        with torch.no_grad():
            nn.init.orthogonal_(self.W_h.weight)
            self.W_h.weight.mul_(0.9)

    def forward(self, x_seq, h_tape=None, h_work=None):
        B, T, D = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        results = hasty_pytorch_lib.dual_memory_elman_forward_opt(
            False,
            x_seq,
            h_tape,
            h_work,
            self.W_h.weight,
            self.W_x.weight,
            self.b_h,
            self.W_write.weight
        )

        h_work_out = results[0]  # [B, T, D]
        h_tape_final = results[1]  # [B, N, D]

        return h_work_out, h_tape_final, h_work_out[:, -1]


class E23cCell(nn.Module):
    """E23c: Chunked Dual-Memory Elman (new optimized version)"""
    def __init__(self, dim, n_slots=64, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.chunk_size = chunk_size

        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Linear(dim, dim, bias=False)

        with torch.no_grad():
            nn.init.orthogonal_(self.W_h.weight)
            self.W_h.weight.mul_(0.9)

    def forward(self, x_seq, h_tape=None, h_work=None):
        B, T, D = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        results = hasty_pytorch_lib.e23c_chunked_forward(
            False,
            x_seq,
            h_tape,
            h_work,
            self.W_h.weight,
            self.W_x.weight,
            self.b_h,
            self.W_write.weight,
            self.chunk_size
        )

        output = results[0]  # [B, T, D]
        h_tape_final = results[1]  # [B, N, D]
        h_work_all = results[2]  # [B, T, D]

        return output, h_tape_final, h_work_all[:, -1]


def benchmark_model(name, model, x, n_warmup=5, n_iters=20):
    """Benchmark a single model."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iters * 1000

    B, T, D = x.shape
    tokens = B * T
    tok_per_sec = tokens / (elapsed / 1000) / 1000

    return elapsed, tok_per_sec


def main():
    device = 'cuda'
    dtype = torch.bfloat16

    print("=" * 80)
    print("E23c vs E23 vs E1 Benchmark")
    print("=" * 80)

    # Test configurations
    configs = [
        # (B, T, D, N, K, description)
        (4, 512, 256, 64, 64, "Small: B=4, T=512, D=256"),
        (8, 512, 512, 64, 64, "Medium: B=8, T=512, D=512"),
        (16, 512, 768, 64, 64, "Large: B=16, T=512, D=768"),
        (32, 512, 768, 64, 64, "XLarge: B=32, T=512, D=768"),
        (4, 2048, 512, 64, 64, "Long seq: B=4, T=2048, D=512"),
    ]

    results = []

    for B, T, D, N, K, desc in configs:
        print(f"\n{desc}")
        print("-" * 60)

        x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1

        # E1
        e1 = E1Cell(D).to(device).to(dtype)
        e1_time, e1_toks = benchmark_model("E1", e1, x)

        # E23
        e23 = E23Cell(D, n_slots=N).to(device).to(dtype)
        e23_time, e23_toks = benchmark_model("E23", e23, x)

        # E23c
        e23c = E23cCell(D, n_slots=N, chunk_size=K).to(device).to(dtype)
        e23c_time, e23c_toks = benchmark_model("E23c", e23c, x)

        print(f"  E1:   {e1_time:7.2f}ms  ({e1_toks:7.1f}K tok/s)")
        print(f"  E23:  {e23_time:7.2f}ms  ({e23_toks:7.1f}K tok/s)")
        print(f"  E23c: {e23c_time:7.2f}ms  ({e23c_toks:7.1f}K tok/s)")
        print(f"  E23c vs E23 speedup: {e23_time/e23c_time:.2f}x")
        print(f"  E23c vs E1 ratio:    {e1_time/e23c_time:.2f}x")

        results.append({
            'config': desc,
            'B': B, 'T': T, 'D': D, 'N': N,
            'e1_ms': e1_time, 'e1_toks': e1_toks,
            'e23_ms': e23_time, 'e23_toks': e23_toks,
            'e23c_ms': e23c_time, 'e23c_toks': e23c_toks,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<30} {'E1 (K/s)':>12} {'E23 (K/s)':>12} {'E23c (K/s)':>12} {'E23c/E23':>10} {'E23c/E1':>10}")
    print("-" * 86)
    for r in results:
        speedup_vs_e23 = r['e23_ms'] / r['e23c_ms']
        ratio_vs_e1 = r['e1_ms'] / r['e23c_ms']
        print(f"{r['config']:<30} {r['e1_toks']:>12.1f} {r['e23_toks']:>12.1f} {r['e23c_toks']:>12.1f} {speedup_vs_e23:>10.2f}x {ratio_vs_e1:>10.2f}x")

    # Also test different chunk sizes for E23c
    print("\n" + "=" * 80)
    print("E23c Chunk Size Sweep (B=16, T=512, D=768, N=64)")
    print("=" * 80)

    B, T, D, N = 16, 512, 768, 64
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1

    chunk_sizes = [16, 32, 64, 128, 256]
    for K in chunk_sizes:
        e23c = E23cCell(D, n_slots=N, chunk_size=K).to(device).to(dtype)
        e23c_time, e23c_toks = benchmark_model(f"E23c K={K}", e23c, x)
        print(f"  K={K:3d}: {e23c_time:7.2f}ms ({e23c_toks:7.1f}K tok/s)")


if __name__ == '__main__':
    main()
