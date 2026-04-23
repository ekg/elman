"""Torch extension binding for the Brent-Kung tree scan CUDA kernel.

First milestone: a sequential-within-block CUDA kernel that scans T
positions using a single shared-memory (A_cum, A_new) ping-pong. No
tensor cores yet, no tree parallelism yet — this is the reference to
verify the augmented-matrix math and Triton/CUDA build-out.

Once this works and matches Phase 0 PyTorch output, we replace the
sequential inner loop with Brent-Kung (parallel prefix) + WMMA tensor
cores. That's the real 5× path.
"""

import os
import torch
from torch.utils.cpp_extension import load

_CUDA_FILE = os.path.join(os.path.dirname(__file__), 'brent_kung_scan.cu')

# JIT-compile the CUDA extension on first call
_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load(
            name='e88_tree_scan_ext',
            sources=[_CUDA_FILE],
            extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
                                '-gencode=arch=compute_80,code=sm_80'],
            verbose=True,
        )
    return _MODULE


def pack_augmented(A, b, M_DIM):
    """Pack (A, b) into augmented (M_DIM × M_DIM) matrix.

    A: [..., N, N]
    b: [..., N]
    Returns M: [..., M_DIM, M_DIM]
    """
    N = A.shape[-1]
    assert M_DIM > N
    shape = A.shape
    M = torch.zeros(*shape[:-2], M_DIM, M_DIM, dtype=A.dtype, device=A.device)
    M[..., :N, :N] = A
    M[..., :N, N] = b
    # Identity on padded diagonal
    for i in range(N, M_DIM):
        M[..., i, i] = 1.0
    return M


def intra_block_scan_cuda(A, b, M_DIM=None):
    """Run the sequential-within-block CUDA scan. Reorders layout to
    [B, H, N_row, T, ...] for kernel-friendly access.

    Returns δ: [B, H, T, N_row, N]
    """
    B, H, T, N_row, N, _ = A.shape
    if M_DIM is None:
        M_DIM = max(N + 1, 16)
        # Round up to multiple of 4 for alignment
        M_DIM = ((M_DIM + 3) // 4) * 4

    # Permute to [B, H, N_row, T, N, N]
    A_perm = A.permute(0, 1, 3, 2, 4, 5).contiguous()
    b_perm = b.permute(0, 1, 3, 2, 4).contiguous()

    M = pack_augmented(A_perm, b_perm, M_DIM).contiguous()
    # Output in [B, H, N_row, T, N] layout, will permute back
    delta_out = torch.zeros(B, H, N_row, T, N, dtype=A.dtype, device=A.device)

    mod = _get_module()
    # We need to expose a Python-accessible launcher. Adding to .cu file would
    # require pybind11 boilerplate. For a quick prototype, let's invoke via
    # ctypes if the module builds, otherwise fall back to stub.

    # Simpler: use torch's native extension binding. Let me rewrite the .cu
    # with pybind11 so we can call it directly.
    raise NotImplementedError("Pybind11 wrapping pending — see brent_kung_scan.cu launch fn.")

    return delta_out.permute(0, 1, 3, 2, 4).contiguous()


if __name__ == '__main__':
    # Try to compile the extension to verify it builds
    print("Compiling CUDA extension (first run can take a few minutes)...")
    mod = _get_module()
    print(f"Loaded module: {mod}")
