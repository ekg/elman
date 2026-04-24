"""Scale T up — find crossover where pure scan beats sequential per iter.

The question: when does the ratio (scan_ms / seq_ms) drop below 1?
"""

import sys, os, time, torch
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2
from quasi_deer_ref import _random_case, build_diag_ingredients
from quasi_deer_triton import qd_diagonal_scan_triton


def bench(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000


def run(H, T, N):
    # Use bfloat16 to save memory at long T
    dtype = torch.bfloat16
    try:
        S0, K, V, decay = _random_case(1, H, T, N, seed=0, dtype=dtype,
                                        l2_normalize_k=True, v_scale=0.3)
    except Exception as e:
        print(f"H={H} T={T} N={N}: failed to allocate inputs: {e}")
        return None

    # Sequential forward
    num_warps = 1 if N == 16 else 4
    def run_seq():
        _ = pararnn_seq_fwd_v2(S0, K, V, decay, num_warps=num_warps)
    try:
        seq_ms = bench(run_seq, repeat=5)
    except Exception as e:
        print(f"H={H} T={T} N={N}: seq OOM: {e}")
        return None

    # Build D, b once (fp32 for correctness)
    S0_f32 = S0.float()
    K_f32 = K.float()
    V_f32 = V.float()
    decay_f32 = decay.float()
    S_var = torch.zeros(1, H, T, N, N, dtype=torch.float32, device='cuda')

    try:
        D, bv, _ = build_diag_ingredients(S0_f32, S_var, K_f32, V_f32, decay_f32)
    except torch.cuda.OutOfMemoryError:
        print(f"H={H} T={T:6d} N={N}:  ingredient build OOM; seq={seq_ms:6.2f} ms")
        return None
    del S_var, S0_f32, K_f32, V_f32, decay_f32

    def run_scan():
        _ = qd_diagonal_scan_triton(D, bv, block_T=512)
    scan_ms = bench(run_scan, repeat=5)

    ratio = scan_ms / seq_ms
    print(f"H={H:3d} T={T:6d} N={N}:  seq={seq_ms:7.2f} ms  scan={scan_ms:7.2f} ms  "
          f"scan/seq={ratio:.2f}×")
    return ratio


if __name__ == '__main__':
    print("scan_ms / seq_ms ratio at various T (smaller is better for quasi-DEER)\n")
    print("-- H=141 N=16 --")
    for T in [4096, 16384, 32768, 65536, 131072]:
        run(141, T, 16)
        torch.cuda.empty_cache()
    print("\n-- H=32 N=16 (less parallelism from H) --")
    for T in [16384, 65536, 131072, 262144]:
        run(32, T, 16)
        torch.cuda.empty_cache()
    print("\n-- H=8 N=16 (very low parallelism) --")
    for T in [65536, 131072, 262144]:
        run(8, T, 16)
        torch.cuda.empty_cache()
    print("\n-- H=1 N=16 (single chain) --")
    for T in [16384, 65536, 131072, 262144, 524288]:
        run(1, T, 16)
        torch.cuda.empty_cache()
