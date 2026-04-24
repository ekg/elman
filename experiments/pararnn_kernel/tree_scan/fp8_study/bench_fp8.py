"""Step 5: benchmark fp8 vs bf16 at production shapes on GPU 2."""
import os, sys, time
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2, backward_v2
from pararnn_seq_fwd_v2_fp8 import pararnn_seq_fwd_v2_fp8, backward_v2_fp8


def bench(fn, n_warmup=5, n_rep=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_rep):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_rep * 1000


def bench_pair(B, H, T, N):
    dt = torch.bfloat16
    dev = 'cuda'
    g = torch.Generator(device=dev).manual_seed(0)
    K = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device=dev)
    V = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device=dev)
    q = 0.3 * torch.randn(B, H, T, N, generator=g, dtype=dt, device=dev)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, H, T, generator=g, dtype=dt, device=dev))
    S0 = 0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device=dev)
    dL_dout = 0.01 * torch.randn(B, H, T, N, dtype=dt, device=dev)
    g_T = torch.zeros(B, H, N, N, dtype=dt, device=dev)
    nw_fwd = 1 if N == 16 else 4
    nw_bwd = 1 if N == 16 else 2

    # BF16: fwd only
    def fwd_bf():
        return pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=nw_fwd)

    def fwd_fp8():
        return pararnn_seq_fwd_v2_fp8(S0, K, V, decay, num_warps=nw_fwd)

    # BF16: bwd only (fwd excluded). Pre-alloc S_traj outside timing.
    S_traj_bf = fwd_bf()
    S_traj_fp8 = fwd_fp8()

    def bwd_bf():
        return backward_v2(S0, S_traj_bf, K, V, decay, g_T, dL_dout, q,
                           num_warps=nw_bwd, num_stages=1)

    def bwd_fp8():
        return backward_v2_fp8(S0, S_traj_fp8, K, V, decay, g_T, dL_dout, q,
                               num_warps=nw_bwd, num_stages=1)

    # Full (fwd + Sq einsum + bwd)
    def full_bf():
        st = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=nw_fwd)
        Sq = torch.einsum('bhtpq,bhtq->bhtp', st, q)
        backward_v2(S0, st, K, V, decay, g_T, dL_dout, q,
                    num_warps=nw_bwd, num_stages=1)

    def full_fp8():
        st = pararnn_seq_fwd_v2_fp8(S0, K, V, decay, num_warps=nw_fwd)
        # Note: Sq einsum still needs a fp16/bf16 traj — for now rehydrate
        # or compute Sq inside fwd kernel.  For timing purpose we REHYDRATE
        # the fp8 tensor (this cost would be eliminated by computing Sq
        # inside the fp8 fwd kernel).
        st_bf = st.to(torch.bfloat16)
        Sq = torch.einsum('bhtpq,bhtq->bhtp', st_bf, q)
        backward_v2_fp8(S0, st, K, V, decay, g_T, dL_dout, q,
                        num_warps=nw_bwd, num_stages=1)

    fwd_bf_ms = bench(fwd_bf)
    fwd_fp8_ms = bench(fwd_fp8)
    bwd_bf_ms = bench(bwd_bf)
    bwd_fp8_ms = bench(bwd_fp8)

    # Full roundtrip (fwd+bwd)
    full_bf_ms = bench(full_bf)
    full_fp8_ms = bench(full_fp8)

    return {
        'fwd_bf': fwd_bf_ms, 'fwd_fp8': fwd_fp8_ms,
        'bwd_bf': bwd_bf_ms, 'bwd_fp8': bwd_fp8_ms,
        'full_bf': full_bf_ms, 'full_fp8': full_fp8_ms,
    }


if __name__ == '__main__':
    print("Benchmark: bf16 vs fp8 S_traj, GPU 2\n")
    shapes = [(1, 141, 4096, 16), (1, 141, 16384, 16),
              (1, 141, 32768, 16), (1, 141, 65536, 16),
              (1, 83, 16384, 32), (1, 83, 32768, 32)]
    header = f"{'B':>3} {'H':>4} {'T':>6} {'N':>3}  {'fwd_bf':>7}  {'fwd_fp8':>7}  {'spd':>5}  {'bwd_bf':>7}  {'bwd_fp8':>7}  {'spd':>5}  {'full_bf':>8}  {'full_fp8':>8}  {'spd':>5}"
    print(header)
    print('-' * len(header))
    for (B, H, T, N) in shapes:
        try:
            r = bench_pair(B, H, T, N)
            f_spd = r['fwd_bf'] / r['fwd_fp8']
            b_spd = r['bwd_bf'] / r['bwd_fp8']
            full_spd = r['full_bf'] / r['full_fp8']
            print(f"{B:3d} {H:4d} {T:6d} {N:3d}  "
                  f"{r['fwd_bf']:7.2f}  {r['fwd_fp8']:7.2f}  {f_spd:5.2f}  "
                  f"{r['bwd_bf']:7.2f}  {r['bwd_fp8']:7.2f}  {b_spd:5.2f}  "
                  f"{r['full_bf']:8.2f}  {r['full_fp8']:8.2f}  {full_spd:5.2f}")
        except Exception as e:
            print(f"{B:3d} {H:4d} {T:6d} {N:3d}  FAIL: {str(e)[:80]}")
        torch.cuda.empty_cache()
