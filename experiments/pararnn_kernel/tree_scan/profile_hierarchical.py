"""Profile each pass of hierarchical scan to find the bottleneck."""

import sys, os, time
import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import build_AB, _random_case
from phase4_newton_driver import sequential_e88_forward

_CUDA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brent_kung_scan.cu')
ext = load(name='e88_tree_scan_proto', sources=[_CUDA_FILE],
           extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
                               '-gencode=arch=compute_80,code=sm_80'],
           verbose=False)


def pack_augmented(A, b, M_DIM):
    *batch, N, _ = A.shape
    M = torch.zeros(*batch, M_DIM, M_DIM, dtype=A.dtype, device=A.device)
    M[..., :N, :N] = A
    M[..., :N, N] = b
    for i in range(N, M_DIM):
        M[..., i, i] = 1.0
    return M


def time_gpu(fn, n_warm=3, n_iter=5):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000


def profile(B, H, T, N, T_BLOCK=16):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=torch.float32)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    print(f"\nShape: B={B} H={H} T={T} N={N} T_BLOCK={T_BLOCK}")
    num_blocks_t = T // T_BLOCK
    print(f"  num_blocks_t = {num_blocks_t}")

    # build_AB
    def do_build(): return build_AB(S0, S_var, K, V, decay)
    build_ms = time_gpu(do_build)
    A, b = build_AB(S0, S_var, K, V, decay)
    print(f"  build_AB: {build_ms:.2f} ms")

    # permute + pack
    M_DIM = ((N + 1 + 7) // 8) * 8
    def do_pack():
        Ap = A.permute(0, 1, 3, 2, 4, 5).contiguous()
        bp = b.permute(0, 1, 3, 2, 4).contiguous()
        return pack_augmented(Ap, bp, M_DIM).contiguous()
    pack_ms = time_gpu(do_pack)
    M = do_pack()
    print(f"  permute + pack: {pack_ms:.2f} ms")

    empty = torch.empty(0, dtype=torch.float32, device=A.device)

    # Pass 1: intra-block with init=identity
    def do_pass1():
        return ext.intra_block_with_init(M, empty, N, T_BLOCK)
    pass1_ms = time_gpu(do_pass1)
    _, summaries = do_pass1()
    print(f"  pass 1 (intra-block + summary): {pass1_ms:.2f} ms")

    # Pass 2: GPU prefix scan of summaries
    def do_pass2():
        return ext.inclusive_matrix_prefix_scan(summaries.clone())
    pass2_ms = time_gpu(do_pass2)
    summary_cum = do_pass2()
    print(f"  pass 2 (scan summaries): {pass2_ms:.2f} ms")

    # build block_cum_excl
    def do_build_excl():
        block_cum_excl = torch.zeros(B, H, summaries.shape[2], num_blocks_t, M_DIM, M_DIM,
                                      dtype=torch.float32, device=A.device)
        eye = torch.eye(M_DIM, dtype=torch.float32, device=A.device)
        block_cum_excl[:, :, :, 0] = eye
        if num_blocks_t > 1:
            block_cum_excl[:, :, :, 1:] = summary_cum[:, :, :, :-1]
        return block_cum_excl.contiguous()
    excl_ms = time_gpu(do_build_excl)
    block_cum_excl = do_build_excl()
    print(f"  build block_cum_excl: {excl_ms:.2f} ms")

    # Pass 3: intra-block with init=block_cum_excl
    def do_pass3():
        return ext.intra_block_with_init(M, block_cum_excl, N, T_BLOCK)
    pass3_ms = time_gpu(do_pass3)
    print(f"  pass 3 (intra-block with init): {pass3_ms:.2f} ms")

    total = pass1_ms + pass2_ms + excl_ms + pass3_ms
    print(f"  TOTAL (passes 1+2+excl+3): {total:.2f} ms  "
          f"[build_AB={build_ms:.1f}, pack={pack_ms:.1f} NOT in total]")


if __name__ == '__main__':
    for shape, tb in [((1, 141, 1024, 16), 16),
                      ((1, 141, 4096, 16), 16),
                      ((1, 32, 4096, 16), 16)]:
        profile(*shape, T_BLOCK=tb)
