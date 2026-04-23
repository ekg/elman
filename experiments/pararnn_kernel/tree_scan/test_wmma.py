"""Stage 6 — WMMA tensor-core variant of fused intra-block scan.

Uses TF32 tensor cores for the 32×32 combine matmul instead of scalar FMAs.
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


def fused_hierarchical_wmma(S0, S_var, K, V, decay, T_BLOCK=8):
    """WMMA-accelerated hierarchical scan."""
    B, H, T, N, _ = S_var.shape
    assert N == 16, "WMMA variant requires N=16"
    num_blocks_t = T // T_BLOCK
    empty = torch.empty(0, dtype=torch.float32, device=S0.device)

    # Pass 1: WMMA intra-block, init=identity
    _, summaries = ext.intra_block_fused_wmma(
        S0.contiguous(), S_var.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        empty, T_BLOCK
    )

    # Pass 2: GPU prefix scan of summaries (uses scalar scan_one_level for now)
    summary_cum = ext.inclusive_matrix_prefix_scan(summaries.clone())

    M_DIM = summary_cum.shape[-1]
    block_cum_excl = torch.zeros_like(summary_cum)
    eye = torch.eye(M_DIM, dtype=torch.float32, device=S0.device)
    block_cum_excl[:, :, :, 0] = eye
    if num_blocks_t > 1:
        block_cum_excl[:, :, :, 1:] = summary_cum[:, :, :, :-1]

    # Pass 3: WMMA intra-block with init
    delta, _ = ext.intra_block_fused_wmma(
        S0.contiguous(), S_var.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        block_cum_excl.contiguous(), T_BLOCK
    )
    return delta.permute(0, 1, 3, 2, 4).contiguous()


def test(B, H, T, N, T_BLOCK=8, seed=0):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    delta_wmma = fused_hierarchical_wmma(S0, S_var, K, V, decay, T_BLOCK=T_BLOCK)
    delta_ref = fused_newton_iter(S0, S_var, K, V, decay)

    diff = (delta_wmma - delta_ref).abs().max().item()
    # TF32 has ~10-bit mantissa precision; accept larger tol
    tol = max(1e-2, 1e-4 * T)
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:3d} T={T:5d} N={N} T_BLOCK={T_BLOCK}  "
          f"max|wmma-ref|={diff:.2e}  (tol {tol:.1e})  [{status}]")


def bench(B, H, T, N, T_BLOCK=8, n_repeat=3):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    for _ in range(3): _ = fused_hierarchical_wmma(S0, S_var, K, V, decay, T_BLOCK=T_BLOCK)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_hierarchical_wmma(S0, S_var, K, V, decay, T_BLOCK=T_BLOCK)
    torch.cuda.synchronize()
    wmma_ms = (time.time() - t0) / n_repeat * 1000

    for _ in range(3): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    par_ms = (time.time() - t0) / n_repeat * 1000

    print(f"  B={B} H={H:3d} T={T:5d} N={N}  Pararnn={par_ms:>6.2f} ms  "
          f"wmma-tree={wmma_ms:>6.2f} ms  ratio={par_ms/wmma_ms:.2f}×")


if __name__ == '__main__':
    print("Stage 6 — WMMA correctness:\n")
    for shape in [(1, 2, 8, 16), (1, 2, 16, 16), (1, 4, 64, 16),
                  (1, 2, 256, 16), (1, 2, 1024, 16)]:
        try:
            test(*shape, T_BLOCK=8)
        except Exception as e:
            print(f"  FAIL {shape}: {str(e)[:150]}")

    print("\nStage 6 — WMMA timing vs Pararnn r=1:\n")
    for shape in [(1, 141, 1024, 16), (1, 141, 4096, 16), (1, 32, 4096, 16)]:
        try:
            bench(*shape, T_BLOCK=8)
        except Exception as e:
            print(f"  FAIL {shape}: {str(e)[:100]}")
