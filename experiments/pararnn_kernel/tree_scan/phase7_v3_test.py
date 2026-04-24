"""Comprehensive correctness + benchmark: V3 vs V2 at production shapes.

Tests:
  1. Correctness at fp32 (exact match) and bf16 (noise bound)
     Grid: production shapes that training actually uses.
  2. Benchmark: full patched-optimized path on V3 vs V2 (fwd+bwd with gate)
  3. Benchmark: raw kernel-call only (no autograd) for V3 vs V2

Target: V3 ≥ 15% faster than V2 at B=16 T=512 due to no permutes.
"""

import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, '/home/erikg/elman')

from phase6_hybrid import PararnnHybridE88V2
from phase6_hybrid_v3 import PararnnHybridE88V3, hybrid_v3_with_fused_gate
from phase7_fused_gate_hybrid import hybrid_with_fused_gate  # V2 with gate


def rel(a, b):
    return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-10)


def make_prod_inputs(B, T, H, N, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    k = (0.3 * torch.randn(B, T, H, N, dtype=dtype, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(B, T, H, N, dtype=dtype, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(B, T, H, N, dtype=dtype, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dtype, device='cuda')).detach().requires_grad_(True)
    g = torch.randn(B, T, H, N, dtype=dtype, device='cuda').requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, N, N, dtype=dtype, device='cuda')).requires_grad_(True)
    return k, v, q, decay, g, S0


def test_correctness():
    print("=" * 70)
    print("Correctness V3 vs V2 (with fused silu gate)")
    print("=" * 70)
    shapes = [
        (1, 4096, 4, 16),
        (1, 4096, 4, 32),
        (2, 2048, 8, 16),
        (4, 512, 16, 16),
        (4, 512, 16, 32),
        (1, 512, 141, 16),  # mini-training
        (1, 512, 83, 32),
    ]
    for dt_label, dt, thresh in [("fp32", torch.float32, 1e-5), ("bf16", torch.bfloat16, 1e-2)]:
        print(f"\n--- {dt_label} (threshold {thresh:.0e}) ---")
        for B, T, H, N in shapes:
            k, v, q, decay, g, S0 = make_prod_inputs(B, T, H, N, dtype=dt)

            # V3 direct
            k3, v3, q3, decay3, g3, S03 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, g, S0]]
            S_f3, out3 = hybrid_v3_with_fused_gate(True, k3, v3, q3, decay3, g3, S03, H)

            # V2: [B,T,H,N] -> [T,B,H,N] transpose wrapper
            k_tb = k.detach().transpose(0, 1).contiguous().requires_grad_(True)
            v_tb = v.detach().transpose(0, 1).contiguous().requires_grad_(True)
            q_tb = q.detach().transpose(0, 1).contiguous().requires_grad_(True)
            decay_tb = decay.detach().transpose(0, 1).contiguous().requires_grad_(True)
            g_tb = g.detach().transpose(0, 1).contiguous().requires_grad_(True)
            S0_v2 = S0.detach().clone().requires_grad_(True)
            S_f2, out2_tb = hybrid_with_fused_gate(True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0_v2, H)
            out2_bt = out2_tb.transpose(0, 1).contiguous()

            torch.manual_seed(1)
            dL_dout = 0.01 * torch.randn_like(out3)
            dL_dout_tb = dL_dout.transpose(0, 1).contiguous()

            (out3 * dL_dout).sum().backward()
            (out2_tb * dL_dout_tb).sum().backward()

            errs = {
                'output': rel(out3, out2_bt),
                'S_final': rel(S_f3, S_f2),
                'dK': rel(k3.grad, k_tb.grad.transpose(0, 1).contiguous()),
                'dV': rel(v3.grad, v_tb.grad.transpose(0, 1).contiguous()),
                'dQ': rel(q3.grad, q_tb.grad.transpose(0, 1).contiguous()),
                'dg': rel(g3.grad, g_tb.grad.transpose(0, 1).contiguous()),
                'ddec': rel(decay3.grad, decay_tb.grad.transpose(0, 1).contiguous()),
                'dS0': rel(S03.grad, S0_v2.grad),
            }
            w = max(errs.values())
            ok = "PASS" if w < thresh else "FAIL"
            details = "  ".join(f"{k_}={v_:.1e}" for k_, v_ in errs.items())
            print(f"  B={B} T={T:>5d} H={H:>3d} N={N}:  worst={w:.2e}  [{ok}]")
            if w > thresh:
                print(f"    details: {details}")


def time_fn(fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000


def bench_v3_vs_v2(B, T, H, N):
    """Fwd + bwd, with fused gate (production pattern)."""
    dt = torch.bfloat16
    k, v, q, decay, g, S0 = make_prod_inputs(B, T, H, N, dtype=dt)

    def run_v3():
        k.grad = None; v.grad = None; q.grad = None
        decay.grad = None; g.grad = None; S0.grad = None
        S_f, out = hybrid_v3_with_fused_gate(True, k, v, q, decay, g, S0, H)
        out.sum().backward()

    # V2 needs [T, B, H, N] inputs. Simulate the patched_optimized adapter:
    # caller has [B, T, H, N] and has to transpose.
    def run_v2_via_adapter():
        k.grad = None; v.grad = None; q.grad = None
        decay.grad = None; g.grad = None; S0.grad = None
        k_tb = k.transpose(0, 1).contiguous()
        v_tb = v.transpose(0, 1).contiguous()
        q_tb = q.transpose(0, 1).contiguous()
        decay_tb = decay.transpose(0, 1).contiguous()
        g_tb = g.transpose(0, 1).contiguous()
        S_f, out_tb = hybrid_with_fused_gate(True, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, H)
        out_bt = out_tb.transpose(0, 1).contiguous()
        out_bt.sum().backward()

    v3_ms = time_fn(run_v3)
    v2_ms = time_fn(run_v2_via_adapter)
    return v2_ms, v3_ms


def bench():
    print()
    print("=" * 70)
    print("Benchmark V3 vs V2 (full fwd+bwd with gate, bf16)")
    print("=" * 70)
    print(f"{'Shape':<40} {'V2 ms':>9} {'V3 ms':>9} {'speedup':>9} {'saved_ms':>10}")
    print("-" * 80)
    # Production training shapes
    shapes = [
        # Training-relevant: batch_size=16, T=512, E88 n16 and n32
        (16, 512, 141, 16),
        (16, 512, 83, 32),
        # 32K context shapes
        (1, 32768, 141, 16),
        (1, 32768, 83, 32),
        # Some mid-sized shapes
        (8, 1024, 141, 16),
        (8, 1024, 83, 32),
        (4, 2048, 141, 16),
        (4, 2048, 83, 32),
    ]
    for B, T, H, N in shapes:
        try:
            torch.cuda.empty_cache()
            v2_ms, v3_ms = bench_v3_vs_v2(B, T, H, N)
            spd = v2_ms / v3_ms
            saved = v2_ms - v3_ms
            label = f"B={B} T={T:>5d} H={H:>3d} N={N}"
            print(f"{label:<40} {v2_ms:>9.3f} {v3_ms:>9.3f} {spd:>8.2f}x {saved:>10.3f}")
        except Exception as e:
            print(f"  B={B} T={T} H={H} N={N}:  FAIL: {str(e)[:80]}")


if __name__ == '__main__':
    test_correctness()
    bench()
