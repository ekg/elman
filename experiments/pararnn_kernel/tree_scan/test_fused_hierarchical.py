"""Stage 7 — fused hierarchical: no Python build_AB, no permute overhead."""

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


def fused_hierarchical_scan(S0, S_var, K, V, decay, T_BLOCK=16):
    """Full hierarchical tree scan with fused (A, b) construction.

    Takes raw Newton-iteration inputs, returns δ per position.
    """
    B, H, T, N, _ = S_var.shape
    num_blocks_t = T // T_BLOCK

    empty = torch.empty(0, dtype=torch.float32, device=S0.device)

    # Pass 1: fused build + intra-block scan, init=identity. Outputs summaries.
    _, summaries = ext.intra_block_fused(
        S0.contiguous(), S_var.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        empty, T_BLOCK
    )

    # Pass 2: GPU prefix scan on summaries
    summary_cum = ext.inclusive_matrix_prefix_scan(summaries.clone())

    # Build exclusive prefix (identity prepended)
    M_DIM = summary_cum.shape[-1]
    block_cum_excl = torch.zeros_like(summary_cum)
    eye = torch.eye(M_DIM, dtype=torch.float32, device=S0.device)
    block_cum_excl[:, :, :, 0] = eye
    if num_blocks_t > 1:
        block_cum_excl[:, :, :, 1:] = summary_cum[:, :, :, :-1]

    # Pass 3: fused intra-block with init=block_cum_excl → final δ
    delta, _ = ext.intra_block_fused(
        S0.contiguous(), S_var.contiguous(),
        K.contiguous(), V.contiguous(), decay.contiguous(),
        block_cum_excl.contiguous(), T_BLOCK
    )
    # delta shape: [B, H, N_row, T, N] — permute to [B, H, T, N_row, N]
    return delta.permute(0, 1, 3, 2, 4).contiguous()


def test(B, H, T, N, T_BLOCK=16, seed=0):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    delta_cuda = fused_hierarchical_scan(S0, S_var, K, V, decay, T_BLOCK=T_BLOCK)
    delta_ref = fused_newton_iter(S0, S_var, K, V, decay)

    diff = (delta_cuda - delta_ref).abs().max().item()
    tol = max(1e-4, 1e-5 * T)
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:3d} T={T:5d} N={N:2d} T_BLOCK={T_BLOCK}  "
          f"max|fused-ref|={diff:.2e}  [{status}]")


def bench(B, H, T, N, T_BLOCK=16, n_repeat=3):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    for _ in range(3): _ = fused_hierarchical_scan(S0, S_var, K, V, decay, T_BLOCK=T_BLOCK)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_hierarchical_scan(S0, S_var, K, V, decay, T_BLOCK=T_BLOCK)
    torch.cuda.synchronize()
    fused_ms = (time.time() - t0) / n_repeat * 1000

    for _ in range(3): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    par_ms = (time.time() - t0) / n_repeat * 1000

    print(f"  B={B} H={H:3d} T={T:5d} N={N}  Pararnn r=1={par_ms:>6.2f} ms  "
          f"fused-tree={fused_ms:>6.2f} ms  ratio={par_ms/fused_ms:.2f}×")


if __name__ == '__main__':
    print("Stage 7 — fused hierarchical correctness:\n")
    for shape, tb in [((1, 2, 16, 16), 16),
                      ((1, 4, 32, 16), 16),
                      ((1, 4, 128, 16), 16),
                      ((1, 2, 1024, 16), 16),
                      ((1, 4, 64, 32), 8)]:
        try:
            test(*shape, T_BLOCK=tb)
        except Exception as e:
            print(f"  FAIL {shape}: {str(e)[:100]}")

    print("\nStage 7 — timing:\n")
    for shape, tb in [((1, 141, 1024, 16), 16),
                      ((1, 141, 4096, 16), 16),
                      ((1, 32, 4096, 16), 16)]:
        try:
            bench(*shape, T_BLOCK=tb)
        except Exception as e:
            print(f"  FAIL {shape}: {str(e)[:100]}")
