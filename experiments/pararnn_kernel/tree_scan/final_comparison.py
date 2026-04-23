"""Final honest comparison: where does tree scan beat sequential?

Sweep T from small to large, compare:
  - Pararnn r=1 sequential (baseline)
  - Fused tree scan (scalar FMA)
  - WMMA tree scan (TF32 tensor cores)

Quantify over what range (if any) tree scan wins.
"""

import sys, os, time
import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import _random_case
from phase4_newton_driver import sequential_e88_forward
from phase7_fused_iter import fused_newton_iter

_CUDA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brent_kung_scan.cu')
ext = load(name='e88_tree_scan_proto', sources=[_CUDA_FILE],
           extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
                               '-gencode=arch=compute_80,code=sm_80'],
           verbose=False)


def run_fused(S0, S_var, K, V, decay, use_wmma=False, T_BLOCK=8):
    B, H, T, N, _ = S_var.shape
    empty = torch.empty(0, dtype=torch.float32, device=S0.device)
    fn = ext.intra_block_fused_wmma if use_wmma else ext.intra_block_fused
    _, summaries = fn(S0, S_var, K, V, decay, empty, T_BLOCK)
    summary_cum = ext.inclusive_matrix_prefix_scan(summaries.clone())
    M_DIM = summary_cum.shape[-1]
    block_cum_excl = torch.zeros_like(summary_cum)
    eye = torch.eye(M_DIM, dtype=torch.float32, device=S0.device)
    block_cum_excl[:, :, :, 0] = eye
    num_blocks_t = summary_cum.shape[3]
    if num_blocks_t > 1:
        block_cum_excl[:, :, :, 1:] = summary_cum[:, :, :, :-1]
    delta, _ = fn(S0, S_var, K, V, decay, block_cum_excl.contiguous(), T_BLOCK)
    return delta.permute(0, 1, 3, 2, 4).contiguous()


def bench(B, H, T, N, n_repeat=3):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    # Pararnn r=1 sequential
    for _ in range(3): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    par_ms = (time.time() - t0) / n_repeat * 1000

    # Tree scan WMMA (only valid for N=16)
    if N == 16:
        try:
            for _ in range(3): _ = run_fused(S0, S_var, K, V, decay, use_wmma=True, T_BLOCK=8)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_repeat): _ = run_fused(S0, S_var, K, V, decay, use_wmma=True, T_BLOCK=8)
            torch.cuda.synchronize()
            wmma_ms = (time.time() - t0) / n_repeat * 1000
        except Exception:
            wmma_ms = float('nan')
    else:
        wmma_ms = float('nan')

    print(f"  H={H:3d} T={T:6d} N={N}  Pararnn r=1: {par_ms:>7.2f} ms   "
          f"Tree-WMMA: {wmma_ms:>7.2f} ms   "
          f"ratio(par/tree)={par_ms/wmma_ms if wmma_ms == wmma_ms else 0:.3f}×")


if __name__ == '__main__':
    print("Final comparison — tree scan vs sequential across T, H:\n")
    # Small T first, scale up
    print("N=16 (WMMA works):")
    for H in [32, 141]:
        for T in [128, 512, 2048, 8192, 32768, 131072]:
            try:
                bench(1, H, T, 16)
            except Exception as e:
                print(f"  FAIL H={H} T={T}: {str(e)[:80]}")
            torch.cuda.empty_cache()
