"""Test the CUDA tree-scan prototype extension.

Stage 1: verify the sequential-within-block CUDA kernel matches
the PyTorch reference from phase0_pytorch_ref.
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


# JIT-compile the extension
_CUDA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'brent_kung_scan.cu')

print(f"Compiling {_CUDA_FILE}...")
ext = load(
    name='e88_tree_scan_proto',
    sources=[_CUDA_FILE],
    extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
                        '-gencode=arch=compute_80,code=sm_80'],
    verbose=True,
)
print("Compiled.")


def pack_augmented(A, b, M_DIM):
    """Pack (A, b) into augmented matrix M = [[A, b], [0, 1]] padded to M_DIM."""
    *batch, N, _ = A.shape
    M = torch.zeros(*batch, M_DIM, M_DIM, dtype=A.dtype, device=A.device)
    M[..., :N, :N] = A
    M[..., :N, N] = b
    for i in range(N, M_DIM):
        M[..., i, i] = 1.0
    return M


def intra_block_scan_cuda(A, b, mode='sequential', T_BLOCK=8):
    """Run our CUDA prototype. A: [B,H,T,N_row,N,N]  b: [B,H,T,N_row,N]."""
    B, H, T, N_row, N, _ = A.shape
    # M_DIM must be strictly > N (so column N holds b). Round up to 8 multiple.
    M_DIM = ((N + 1 + 7) // 8) * 8
    # For N=16: M_DIM=24. For N=32: M_DIM=40.
    # Our Hillis-Steele kernel template supports only M_DIM in {24, 32}.
    # For N=32 we need M_DIM=40, need to add to kernel dispatch (later).

    # Reorder to [B, H, N_row, T, N, N]
    A_perm = A.permute(0, 1, 3, 2, 4, 5).contiguous()
    b_perm = b.permute(0, 1, 3, 2, 4).contiguous()

    M = pack_augmented(A_perm, b_perm, M_DIM).contiguous()
    if mode == 'sequential':
        delta = ext.intra_block_sequential(M, N)
    elif mode == 'hillis':
        delta = ext.intra_block_hillis(M, N, T_BLOCK)
    else:
        raise ValueError(mode)

    # Reorder output [B, H, N_row, T, N] -> [B, H, T, N_row, N]
    return delta.permute(0, 1, 3, 2, 4).contiguous()


def test_correctness(B, H, T, N, seed=0, dtype=torch.float32):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    delta_cuda = intra_block_scan_cuda(A, b)
    delta_ref = affine_scan_sequential(A, b)

    diff = (delta_cuda - delta_ref).abs().max().item()
    tol = max(1e-4, 1e-5 * T)
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:2d} T={T:4d} N={N:2d}  max|cuda-ref|={diff:.2e}  "
          f"(tol {tol:.1e})  [{status}]")
    return diff


def bench(B, H, T, N, n_repeat=3):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    # Warmup
    for _ in range(3): _ = intra_block_scan_cuda(A, b)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_repeat): _ = intra_block_scan_cuda(A, b)
    torch.cuda.synchronize()
    cuda_ms = (time.time() - t0) / n_repeat * 1000

    for _ in range(3): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    par_ms = (time.time() - t0) / n_repeat * 1000

    print(f"  B={B} H={H:3d} T={T:4d} N={N:2d}  "
          f"Pararnn r=1={par_ms:>6.2f} ms   CUDA prototype={cuda_ms:>7.2f} ms  "
          f"ratio={par_ms/cuda_ms:.2f}×")


def test_hillis(B, H, T, N, T_BLOCK=8, seed=0, dtype=torch.float32):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    delta_cuda_hillis = intra_block_scan_cuda(A, b, mode='hillis', T_BLOCK=T_BLOCK)
    delta_cuda_seq = intra_block_scan_cuda(A, b, mode='sequential')

    diff = (delta_cuda_hillis - delta_cuda_seq).abs().max().item()
    tol = max(1e-4, 1e-5 * T)
    # NOTE: Hillis only scans within blocks of T_BLOCK. Full T scan requires inter-block pass.
    # So for T > T_BLOCK, Hillis result is NOT the same as sequential-full-scan.
    # We check match only when T == T_BLOCK.
    if T == T_BLOCK:
        status = "PASS" if diff < tol else "FAIL"
        print(f"  (intra-block only) B={B} H={H:2d} T={T} T_BLOCK={T_BLOCK} N={N}  "
              f"max|hillis-seq|={diff:.2e}  [{status}]")
    else:
        print(f"  (inter-block pending) T={T} T_BLOCK={T_BLOCK}: hillis scans per-block only")


def bench_hillis(B, H, T, N, T_BLOCK=8, n_repeat=3):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    for _ in range(3): _ = intra_block_scan_cuda(A, b, mode='hillis', T_BLOCK=T_BLOCK)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = intra_block_scan_cuda(A, b, mode='hillis', T_BLOCK=T_BLOCK)
    torch.cuda.synchronize()
    hil_ms = (time.time() - t0) / n_repeat * 1000

    for _ in range(3): _ = intra_block_scan_cuda(A, b, mode='sequential')
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = intra_block_scan_cuda(A, b, mode='sequential')
    torch.cuda.synchronize()
    seq_ms = (time.time() - t0) / n_repeat * 1000

    print(f"  B={B} H={H:3d} T={T:4d} N={N} T_BLOCK={T_BLOCK}  "
          f"cuda-seq={seq_ms:>6.2f} ms   cuda-hillis={hil_ms:>6.2f} ms  "
          f"hillis/seq={hil_ms/seq_ms:.2f}×")


if __name__ == '__main__':
    print("Stage 1 — CUDA intra-block sequential scan correctness:\n")
    for shape in [(1, 2, 16, 16), (1, 4, 64, 16), (1, 8, 256, 16), (1, 4, 128, 32)]:
        test_correctness(*shape)

    print("\nStage 2 — Hillis-Steele intra-block scan correctness (block-sized T):\n")
    # T_BLOCK constraints by N: shmem = 2*T_BLOCK*M_DIM²*4. Max 99KB per block on A100.
    # N=16: M_DIM=24, T_BLOCK ≤ 21. Use 16.
    # N=32: M_DIM=40, T_BLOCK ≤ 7. Use 4.
    for shape, tb in [((1, 2, 8, 16), 8),
                      ((1, 4, 16, 16), 16),
                      ((1, 8, 4, 32), 4)]:
        test_hillis(*shape, T_BLOCK=tb)

    print("\nStage 2 — Hillis vs sequential timing at block size:\n")
    for shape, tb in [((1, 141, 16, 16), 16),
                      ((1, 141, 4, 32), 4)]:
        try:
            bench_hillis(*shape, T_BLOCK=tb)
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}")
