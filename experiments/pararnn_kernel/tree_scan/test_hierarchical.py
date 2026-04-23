"""Stage 4 — hierarchical tree scan test.

Three-pass pipeline:
  Pass 1: intra-block with init=identity → collect block summaries.
  Pass 2: (recursive) scan block summaries across blocks.
  Pass 3: intra-block with init=block_cum_excl → produce final δ per position.

For Pass 2, if num_blocks fits in one intra-block scan, we can do it with
a single call. If not, recurse.
"""

import sys
import os
import time

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import build_AB, affine_scan_sequential, _random_case
from phase4_newton_driver import sequential_e88_forward
from phase7_fused_iter import fused_newton_iter


_CUDA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'brent_kung_scan.cu')
ext = load(
    name='e88_tree_scan_proto',
    sources=[_CUDA_FILE],
    extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
                        '-gencode=arch=compute_80,code=sm_80'],
    verbose=False,
)


def pack_augmented(A, b, M_DIM):
    *batch, N, _ = A.shape
    M = torch.zeros(*batch, M_DIM, M_DIM, dtype=A.dtype, device=A.device)
    M[..., :N, :N] = A
    M[..., :N, N] = b
    for i in range(N, M_DIM):
        M[..., i, i] = 1.0
    return M


def hierarchical_scan_cuda(A, b, T_BLOCK=16):
    """Hierarchical tree scan via CUDA extension. Two-level for now."""
    B, H, T, N_row, N, _ = A.shape
    M_DIM = ((N + 1 + 7) // 8) * 8

    assert T % T_BLOCK == 0, f"T={T} must be divisible by T_BLOCK={T_BLOCK}"
    num_blocks_t = T // T_BLOCK

    # Permute to [B, H, N_row, T, ...]
    A_perm = A.permute(0, 1, 3, 2, 4, 5).contiguous()
    b_perm = b.permute(0, 1, 3, 2, 4).contiguous()
    M = pack_augmented(A_perm, b_perm, M_DIM).contiguous()

    # Pass 1: intra-block with init=identity, collect summaries
    empty = torch.empty(0, dtype=torch.float32, device=A.device)
    _, summaries = ext.intra_block_with_init(M, empty, N, T_BLOCK)
    # summaries shape: [B, H, N_row, num_blocks_t, M_DIM, M_DIM]

    # Pass 2: GPU-side inclusive prefix scan of summaries via Hillis-Steele.
    # O(log num_blocks_t) kernel launches, each with num_blocks_t parallel matmuls.
    # summaries is updated in place to hold summary_cum[b].
    summary_cum = ext.inclusive_matrix_prefix_scan(summaries.clone())
    # summary_cum[b] = summaries[b] @ summaries[b-1] @ ... @ summaries[0]

    # block_cum_excl[b] = summary_cum[b-1] for b >= 1, identity for b = 0
    block_cum_excl = torch.zeros(B, H, N_row, num_blocks_t, M_DIM, M_DIM,
                                  dtype=torch.float32, device=A.device)
    eye = torch.eye(M_DIM, dtype=torch.float32, device=A.device)
    block_cum_excl[:, :, :, 0] = eye
    if num_blocks_t > 1:
        block_cum_excl[:, :, :, 1:] = summary_cum[:, :, :, :-1]

    # Pass 3: intra-block with init=block_cum_excl → produce final δ
    delta, _ = ext.intra_block_with_init(M, block_cum_excl.contiguous(), N, T_BLOCK)

    # Reorder back
    return delta.permute(0, 1, 3, 2, 4).contiguous()


def test(B, H, T, N, T_BLOCK=16, seed=0):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    delta_cuda = hierarchical_scan_cuda(A, b, T_BLOCK=T_BLOCK)
    delta_ref = fused_newton_iter(S0, S_var, K, V, decay)

    diff = (delta_cuda - delta_ref).abs().max().item()
    tol = max(1e-4, 1e-5 * T)
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:2d} T={T:4d} N={N:2d} T_BLOCK={T_BLOCK}  "
          f"max|hier-ref|={diff:.2e}  (tol {tol:.1e})  [{status}]")


def bench(B, H, T, N, T_BLOCK=16, n_repeat=3):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    for _ in range(3): _ = hierarchical_scan_cuda(A, b, T_BLOCK=T_BLOCK)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = hierarchical_scan_cuda(A, b, T_BLOCK=T_BLOCK)
    torch.cuda.synchronize()
    hier_ms = (time.time() - t0) / n_repeat * 1000

    for _ in range(3): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    par_ms = (time.time() - t0) / n_repeat * 1000

    print(f"  B={B} H={H:3d} T={T:5d} N={N:2d} T_BLOCK={T_BLOCK}  "
          f"Pararnn r=1={par_ms:>6.2f} ms   hier-tree={hier_ms:>6.2f} ms  "
          f"ratio={par_ms/hier_ms:.2f}×")


if __name__ == '__main__':
    print("Stage 4 — hierarchical scan correctness:\n")
    # T must satisfy: T = num_blocks * T_BLOCK AND num_blocks ≤ T_BLOCK (host pass 2
    # handles any num_blocks, actually — but we keep small for now).
    for shape, tb in [((1, 2, 32, 16), 16),
                      ((1, 4, 128, 16), 16),
                      ((1, 4, 256, 16), 16),
                      ((1, 2, 1024, 16), 16),      # 64 blocks — tests host pass 2
                      ((1, 2, 2048, 16), 16),      # 128 blocks
                      ((1, 4, 64, 32), 8),
                      ((1, 4, 128, 32), 8)]:
        try:
            test(*shape, T_BLOCK=tb)
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}")

    print("\nStage 4 — timing (hierarchical vs Pararnn r=1 sequential):\n")
    # Small T first — overhead-dominated
    for shape, tb in [((1, 141, 256, 16), 16),
                      ((1, 141, 1024, 16), 16),
                      ((1, 141, 4096, 16), 16),
                      ((1, 32, 1024, 16), 16),
                      ((1, 32, 4096, 16), 16)]:
        try:
            bench(*shape, T_BLOCK=tb)
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}")
