"""Baseline: measure the permute/contiguous overhead in install_hybrid's
patched_optimized path at production shapes.

Breakdown:
  1. Adapter permutes in install_hybrid (transpose(0,1).contiguous() × 5)
  2. Internal permutes in PararnnHybridE88V2.forward (permute + contiguous × 5)
  3. Output conversion permutes (2 total)

Goal: establish budget for v3 optimization.
"""

import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, '/home/erikg/elman')


def make_inputs(B, T, H, N, dtype=torch.bfloat16):
    g = torch.Generator(device='cuda').manual_seed(0)
    k = 0.3 * torch.randn(B, T, H, N, generator=g, dtype=dtype, device='cuda')
    v = 0.3 * torch.randn(B, T, H, N, generator=g, dtype=dtype, device='cuda')
    q = 0.3 * torch.randn(B, T, H, N, generator=g, dtype=dtype, device='cuda')
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, generator=g, dtype=dtype, device='cuda'))
    g_gate = torch.randn(B, T, H, N, generator=g, dtype=dtype, device='cuda')
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dtype, device='cuda')
    return k, v, q, decay, g_gate, S0


def time_fn(fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000


def bench_adapter_permutes_only(k, v, q, decay, g_gate):
    """Time just the [B,T,H,N] -> [T,B,H,N] transposes."""
    def run():
        k_tb = k.transpose(0, 1).contiguous()
        v_tb = v.transpose(0, 1).contiguous()
        q_tb = q.transpose(0, 1).contiguous()
        decay_tb = decay.transpose(0, 1).contiguous()
        g_tb = g_gate.transpose(0, 1).contiguous()
        return k_tb, v_tb, q_tb, decay_tb, g_tb
    return time_fn(run)


def bench_internal_permutes(k_tb, v_tb, q_tb, decay_tb, S0):
    """Time the [T,B,H,N] -> [B,H,T,N] internal permutes in PararnnHybridE88V2."""
    def run():
        K_p = k_tb.permute(1, 2, 0, 3).contiguous()
        V_p = v_tb.permute(1, 2, 0, 3).contiguous()
        Q_p = q_tb.permute(1, 2, 0, 3).contiguous()
        decay_p = decay_tb.permute(1, 2, 0).contiguous()
        S0_p = S0.transpose(-1, -2).contiguous()
        return K_p, V_p, Q_p, decay_p, S0_p
    return time_fn(run)


def bench_output_permutes(B, T, H, N, dtype=torch.bfloat16):
    """Time output [B,H,T,M] -> [T,B,H,M] and back -> [B,T,H,M]."""
    Sq_p = torch.randn(B, H, T, N, dtype=dtype, device='cuda')
    S_final_p = torch.randn(B, H, N, N, dtype=dtype, device='cuda')
    def run():
        S_final = S_final_p.transpose(-1, -2).contiguous()
        output_tb = Sq_p.permute(2, 0, 1, 3).contiguous()
        output_bt = output_tb.transpose(0, 1).contiguous()
        return S_final, output_bt
    return time_fn(run)


def bench_full_patched_optimized(k, v, q, decay, g_gate, S0, H):
    """Time the full patched_optimized call (end-to-end)."""
    from install_hybrid import install
    import elman.models.e88_fla_hybrid as e88m
    # Ensure patched
    install()
    def run():
        S_f, out = e88m.E88OptimizedCUDAFunction.apply(
            True, k, v, q, decay, g_gate, S0, H, True, False, 16,
        )
        return S_f, out
    return time_fn(run)


def bench_full_original(k, v, q, decay, g_gate, S0, H):
    """Time the original CUDA (non-patched) at same shape for reference."""
    import elman.models.e88_fla_hybrid as e88m
    # We need the ORIGINAL apply — grab it before install overrides.
    # Use a fresh import tactic: call the underlying CUDA path.
    # The original apply is `e88m.E88OptimizedCUDAFunction.apply` BEFORE
    # install().  If we've already installed, it's patched.  Save the
    # original first.
    orig = getattr(e88m, '_ORIGINAL_OPTIMIZED_APPLY', None)
    if orig is None:
        return -1.0
    def run():
        return orig(True, k, v, q, decay, g_gate, S0, H, True, False, 16)
    return time_fn(run)


def main():
    print("GPU:", torch.cuda.get_device_name())
    print()
    print(f"{'Shape':<40} {'adapter_ms':>10} {'internal_ms':>11} {'output_ms':>10} {'total_permutes_ms':>18}")
    print("-" * 100)

    # Production training shapes
    cases = [
        ("B=1 T=32K H=141 N=16", 1, 32768, 141, 16),
        ("B=16 T=512 H=141 N=16", 16, 512, 141, 16),
        ("B=1 T=32K H=83 N=32", 1, 32768, 83, 32),
        ("B=16 T=512 H=83 N=32", 16, 512, 83, 32),
    ]
    results = []
    for label, B, T, H, N in cases:
        k, v, q, decay, g_gate, S0 = make_inputs(B, T, H, N)
        # Measure adapter permutes
        adapter_ms = bench_adapter_permutes_only(k, v, q, decay, g_gate)
        # After adapter: k_tb etc. are in [T, B, H, N] layout
        k_tb = k.transpose(0, 1).contiguous()
        v_tb = v.transpose(0, 1).contiguous()
        q_tb = q.transpose(0, 1).contiguous()
        decay_tb = decay.transpose(0, 1).contiguous()
        internal_ms = bench_internal_permutes(k_tb, v_tb, q_tb, decay_tb, S0)
        output_ms = bench_output_permutes(B, T, H, N)
        total = adapter_ms + internal_ms + output_ms
        results.append((label, adapter_ms, internal_ms, output_ms, total))
        print(f"{label:<40} {adapter_ms:>10.3f} {internal_ms:>11.3f} {output_ms:>10.3f} {total:>18.3f}")

    # Total end-to-end for the patched path
    print()
    print(f"{'Shape':<40} {'full_patched_ms':>17} {'orig_cuda_ms':>14}")
    print("-" * 100)
    import elman.models.e88_fla_hybrid as e88m
    if not hasattr(e88m, '_ORIGINAL_OPTIMIZED_APPLY'):
        e88m._ORIGINAL_OPTIMIZED_APPLY = e88m.E88OptimizedCUDAFunction.apply
    for label, B, T, H, N in cases:
        k, v, q, decay, g_gate, S0 = make_inputs(B, T, H, N)
        try:
            full_ms = bench_full_patched_optimized(k, v, q, decay, g_gate, S0, H)
        except Exception as e:
            full_ms = -1.0
            print(f"  full FAIL: {e}")
        try:
            orig_ms = bench_full_original(k, v, q, decay, g_gate, S0, H)
        except Exception as e:
            orig_ms = -1.0
        print(f"{label:<40} {full_ms:>17.3f} {orig_ms:>14.3f}")

    # Summary
    print()
    print("Summary — permute overhead as % of full_patched:")
    for (label, a, i, o, tot), (_, B, T, H, N) in zip(results, cases):
        k, v, q, decay, g_gate, S0 = make_inputs(B, T, H, N)
        try:
            full_ms = bench_full_patched_optimized(k, v, q, decay, g_gate, S0, H)
            pct = 100.0 * tot / full_ms if full_ms > 0 else -1
            print(f"  {label:<40}  permutes={tot:>6.2f}ms  full={full_ms:>6.2f}ms  permute_frac={pct:>4.1f}%")
        except Exception:
            pass


if __name__ == '__main__':
    main()
