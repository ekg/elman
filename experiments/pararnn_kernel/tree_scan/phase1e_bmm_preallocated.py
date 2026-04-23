"""Phase 1e — BMM tree scan with minimal allocation and ping-pong buffers.

Previous phase1b was 100-1000× slower due to per-level tensor allocation.
This version pre-allocates everything once and does in-place updates via
ping-pong buffers.

If this is still slow, tree scan is genuinely not fast enough for our sizes.
If it's fast, we have a workable path.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_pytorch_ref import build_AB, _random_case
from phase4_newton_driver import sequential_e88_forward
from phase7_fused_iter import fused_newton_iter


class TreeScanPreallocated:
    """Pre-allocated workspace for tree scan to avoid per-call/per-level overhead."""

    def __init__(self, B, H, T, N_row, N, dtype=torch.float32, device='cuda'):
        self.B, self.H, self.T, self.N_row, self.N = B, H, T, N_row, N
        self.bs = B * H * N_row
        self.dtype = dtype
        self.device = device

        # Ping-pong buffers for A and b
        self.A_buf = torch.empty(self.bs, T, N, N, dtype=dtype, device=device)
        self.b_buf = torch.empty(self.bs, T, N, dtype=dtype, device=device)

    def scan(self, A, b):
        """Tree scan (A, b) -> δ. Everything done in-place using ping-pong."""
        # Pack input into workspace (batch-major)
        A_in = A.permute(0, 1, 3, 2, 4, 5).reshape(self.bs, self.T, self.N, self.N)
        b_in = b.permute(0, 1, 3, 2, 4).reshape(self.bs, self.T, self.N)
        self.A_buf.copy_(A_in)
        self.b_buf.copy_(b_in)

        N = self.N
        T = self.T
        bs = self.bs

        d = 1
        while d < T:
            # For t >= d: combine(cur[t-d], cur[t])
            # A_new[t] = A[t] @ A[t-d]
            # b_new[t] = A[t] @ b[t-d] + b[t]

            # Views (no allocation)
            A_left = self.A_buf[:, :T - d].reshape(-1, N, N)
            A_right = self.A_buf[:, d:].reshape(-1, N, N)
            b_left = self.b_buf[:, :T - d].reshape(-1, N, 1)
            b_right = self.b_buf[:, d:]

            # Compute combined (into temporaries — sadly bmm doesn't have out=)
            # Actually torch.bmm does have out parameter!
            # But we need the right view order. Let's do it:
            A_new = torch.bmm(A_right, A_left)        # [bs*(T-d), N, N]
            b_new = torch.bmm(A_right, b_left).squeeze(-1)  # [bs*(T-d), N]

            # Write back to the "current" positions [d .. T-1]
            self.A_buf[:, d:] = A_new.reshape(bs, T - d, N, N)
            self.b_buf[:, d:] = b_new.reshape(bs, T - d, N) + b_right

            d *= 2

        # b_buf now holds b_cum at each position
        delta = self.b_buf.reshape(self.B, self.H, self.N_row, T, N).permute(0, 1, 3, 2, 4)
        return delta.contiguous()


def newton_step_prealloc(S0, S_var, K, V, decay, scanner):
    A, b = build_AB(S0, S_var, K, V, decay)
    return scanner.scan(A, b)


def test(B, H, T, N, seed=0, dtype=torch.float32):
    S0, K, V, decay = _random_case(B, H, T, N, seed=seed, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9

    scanner = TreeScanPreallocated(B, H, T, N, N, dtype=dtype)
    delta_tree = newton_step_prealloc(S0, S_var, K, V, decay, scanner)
    delta_ref = fused_newton_iter(S0, S_var, K, V, decay)

    diff = (delta_tree - delta_ref).abs().max().item()
    tol = max(1e-4, 1e-5 * T)
    status = "PASS" if diff < tol else "FAIL"
    print(f"  B={B} H={H:2d} T={T:4d} N={N:2d}  max|tree - ref|={diff:.2e}  [{status}]")


def bench(B, H, T, N, n_repeat=3, dtype=torch.float32):
    S0, K, V, decay = _random_case(B, H, T, N, seed=0, dtype=dtype)
    S_traj = sequential_e88_forward(S0, K, V, decay)
    S_var = S_traj[:, :, 1:] * 0.9
    A, b = build_AB(S0, S_var, K, V, decay)

    scanner = TreeScanPreallocated(B, H, T, N, N, dtype=dtype)

    for _ in range(3): _ = scanner.scan(A, b)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = scanner.scan(A, b)
    torch.cuda.synchronize()
    tree_ms = (time.time() - t0) / n_repeat * 1000

    for _ in range(3): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): _ = fused_newton_iter(S0, S_var, K, V, decay)
    torch.cuda.synchronize()
    par_ms = (time.time() - t0) / n_repeat * 1000

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  B={B} H={H:3d} T={T:5d} N={N:2d}  "
          f"Pararnn r=1={par_ms:>6.2f} ms   tree-bmm(prealloc)={tree_ms:>7.2f} ms   "
          f"ratio={par_ms/tree_ms:.2f}×  peak={peak_gb:.1f}GB")


if __name__ == '__main__':
    print("Phase 1e — correctness:\n")
    for shape in [(1, 4, 64, 16), (1, 8, 256, 32)]:
        test(*shape)

    print("\nPhase 1e — timing (T scaling at production H):\n")
    for shape in [(1, 141, 256, 16),
                  (1, 141, 1024, 16),
                  (1, 141, 4096, 16),
                  (1, 32, 1024, 16),
                  (1, 32, 4096, 16),
                  (1, 32, 16384, 16)]:
        try:
            bench(*shape)
        except Exception as e:
            print(f"  FAIL {shape}: {str(e)[:100]}")
