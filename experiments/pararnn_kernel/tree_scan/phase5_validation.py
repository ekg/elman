"""Phase 5 — real-layer validation.

Test at the E88FLAHybridCUDAFunction level (what the model actually calls):
  - Run CUDA forward + backward (baseline path)
  - Run hybrid forward (ADMM+warmup+1iter) + Pararnn backward
  - Compare: do gradients match within bf16 tolerance?
  - Benchmark: is the hybrid actually faster end-to-end?

If gradients match and hybrid is faster, we have a drop-in replacement.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_newton_driver import sequential_e88_forward
from phase7_fused_backward import backward_e88_fused_rank1
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction
from phase1_warmstart_bench import admm_forward_fixed_iters
from phase2_warmup_scan import warmup_scan_boundaries
from pararnn_seq_fwd import pararnn_seq_fwd_triton


class PararnnHybridE88Function(torch.autograd.Function):
    """Drop-in replacement for E88FLAHybridCUDAFunction that uses:
      - Forward:  ADMM parallel-in-time (warmup-scan + 1-iter)
      - Backward: Pararnn fused rank-1 kernel
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, S0, n_heads, P=16, W=128):
        """
        Args mirror E88FLAHybridCUDAFunction:
          k, v, q: [T, B, H, N]
          decay: [T, B, H]
          S0: [B, H, N, N]
        Returns: (S_final [B, H, N, N], output [T, B, H, N])
        """
        T, B, H, N = k.shape

        # Guard: fall back to CUDA if we can't chunk cleanly
        if T % P != 0 or T < 4 * W:
            return E88FLAHybridCUDAFunction.apply(training, k, v, q, decay, S0, n_heads)

        # Warmup-scan coarse boundaries
        bd = warmup_scan_boundaries(S0, k, v, q, decay, H, P, W=W)
        # 1-iter ADMM forward — returns S_end per chunk + per-position output
        S_end, _, output = admm_forward_fixed_iters(
            S0, k, v, q, decay, H, P, num_iters=1, init_boundaries=bd
        )
        # S_final is the last chunk's end
        S_final = S_end[:, P - 1]  # [B, H, N, N]

        # For backward, save S_traj in Pararnn convention (which is the
        # transpose of CUDA's S).  CUDA:  S_new[i,j] = tanh(... + delta[j]*k[i])
        # Pararnn: S_new[p,q] = tanh(... + delta[p]*k[q]).  They are
        # equivalent under Pararnn_S[p,q] = CUDA_S[q,p], with the SAME
        # (k, v, q, decay) tensors.  We run the Triton Pararnn forward with
        # S0 transposed so the resulting trajectory is Pararnn-convention
        # and directly usable by backward_e88_fused_rank1.
        K_bhhtn = k.permute(1, 2, 0, 3).contiguous()
        V_bhhtn = v.permute(1, 2, 0, 3).contiguous()
        decay_bhht = decay.permute(1, 2, 0).contiguous()
        S0_pararnn = S0.transpose(-1, -2).contiguous()
        S_traj_bf = pararnn_seq_fwd_triton(
            S0_pararnn, K_bhhtn, V_bhhtn, decay_bhht,
            num_warps=4 if N == 32 else 1,
        )

        ctx.save_for_backward(k, v, q, decay, S0, S_traj_bf)
        ctx.n_heads = n_heads
        ctx.P = P
        ctx.W = W
        return S_final, output

    @staticmethod
    def backward(ctx, dS_final, d_output):
        k, v, q, decay, S0, S_traj = ctx.saved_tensors
        T, B, H, N = k.shape

        # S_traj is already in Pararnn convention (forward-built from S0^T).
        # For the Pararnn backward to compute dL/d(CUDA inputs), k/v/q/decay
        # are shared (same tensors, same gradients).  g_T = dL/dCUDA_S_final
        # needs to be transposed to dL/dPararnn_S_final.
        K_d = k.permute(1, 2, 0, 3).contiguous()      # [B, H, T, N]
        V_d = v.permute(1, 2, 0, 3).contiguous()
        decay_d = decay.permute(1, 2, 0).contiguous()  # [B, H, T]
        q_d = q.permute(1, 2, 0, 3).contiguous()

        # dL_dout from d_output [T, B, H, N] → [B, H, T, N].  CUDA's output
        # is indexed by HEAD_V_DIM (= Pararnn's p = Pararnn output index),
        # so no transpose needed.
        dL_dout = d_output.permute(1, 2, 0, 3).contiguous()
        # g_T = dL/dCUDA_S_final.  Pararnn backward expects dL/dPararnn_S
        # which is CUDA grad transposed on the last two dims.
        g_T = dS_final.transpose(-1, -2).contiguous()

        # Run Pararnn backward on Pararnn-convention S_traj
        dS0_pararnn, dK, dV, ddec = backward_e88_fused_rank1(
            S_traj, K_d, V_d, decay_d, g_T, dL_dout, q_d,
            num_warps=4 if N == 32 else 1, num_stages=1)

        # dQ[t, c] = sum_r dL_dout[t, r] * Pararnn_S[t+1, r, c]  (consistent
        # with Pararnn output formula; produces CUDA-aligned dQ since
        # Pararnn's q dim = CUDA's N_STATE dim).
        dQ = torch.einsum('bhti,bhtij->bhtj', dL_dout, S_traj[:, :, 1:])

        # dS0 comes out in Pararnn convention → transpose back to CUDA.
        dS0_out = dS0_pararnn.transpose(-1, -2).contiguous()

        # dK, dV, dQ, ddec are shared — just permute layout to CUDA's [T,B,H,*]
        dK_out = dK.permute(2, 0, 1, 3).contiguous()
        dV_out = dV.permute(2, 0, 1, 3).contiguous()
        dQ_out = dQ.permute(2, 0, 1, 3).contiguous()
        ddec_out = ddec.permute(2, 0, 1).contiguous()

        # Grad order matches forward args: (training, k, v, q, decay, S0, n_heads, P, W)
        return None, dK_out, dV_out, dQ_out, ddec_out, dS0_out, None, None, None


def test_correctness(B, H, T, N, P=16, W=128, seed=0):
    """Compare gradients between CUDA and hybrid paths."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(seed)
    k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                          ).detach().requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)

    # === CUDA baseline ===
    # NOTE: E88FLAHybridCUDAFunction.backward IGNORES dS_final (only d_output
    # reaches the kernel), so to compare grads fairly both paths must
    # receive zero dL/dS_final.
    k1, v1, q1, decay1, S01 = [x.clone().detach().requires_grad_(True) for x in [k, v, q, decay, S0]]
    S_final_cuda, output_cuda = E88FLAHybridCUDAFunction.apply(True, k1, v1, q1, decay1, S01, H)
    torch.manual_seed(seed + 1)
    fake_dL_dout = 0.01 * torch.randn_like(output_cuda)
    fake_dL_dSfinal = torch.zeros_like(S_final_cuda)
    loss_cuda = (output_cuda * fake_dL_dout).sum() + (S_final_cuda * fake_dL_dSfinal).sum()
    loss_cuda.backward()

    dK_cuda = k1.grad.clone()
    dV_cuda = v1.grad.clone()
    dQ_cuda = q1.grad.clone()
    ddec_cuda = decay1.grad.clone()
    dS0_cuda = S01.grad.clone() if S01.grad is not None else None

    # === Hybrid path ===
    k2, v2, q2, decay2, S02 = [x.clone().detach().requires_grad_(True) for x in [k, v, q, decay, S0]]
    S_final_hyb, output_hyb = PararnnHybridE88Function.apply(True, k2, v2, q2, decay2, S02, H, P, W)
    # Same fake gradients
    loss_hyb = (output_hyb * fake_dL_dout).sum() + (S_final_hyb * fake_dL_dSfinal).sum()
    loss_hyb.backward()

    dK_hyb = k2.grad.clone()
    dV_hyb = v2.grad.clone()
    dQ_hyb = q2.grad.clone()
    ddec_hyb = decay2.grad.clone()
    dS0_hyb = S02.grad.clone() if S02.grad is not None else None

    def rel(a, b, tag):
        num = (a.float() - b.float()).abs().max().item()
        denom = max(b.float().abs().max().item(), 1e-10)
        r = num / denom
        status = "PASS" if r < 0.1 else "FAIL"  # 10% tolerance for bf16
        print(f"    {tag:>8s}: max_rel={r:.2e}  [{status}]")
        return r

    print(f"\n  B={B} H={H} T={T} N={N} P={P} W={W}")
    print(f"    Forward outputs:")
    f_out = rel(output_hyb, output_cuda, "output")
    f_Sf = rel(S_final_hyb, S_final_cuda, "S_final")
    print(f"    Gradients:")
    g_K = rel(dK_hyb, dK_cuda, "dK")
    g_V = rel(dV_hyb, dV_cuda, "dV")
    g_Q = rel(dQ_hyb, dQ_cuda, "dQ")
    g_d = rel(ddec_hyb, ddec_cuda, "ddecay")
    if dS0_cuda is not None and dS0_hyb is not None:
        g_S0 = rel(dS0_hyb, dS0_cuda, "dS0")
    else:
        g_S0 = 0.0
        print(f"    {'dS0':>8s}: (CUDA didn't compute)")
    worst = max(f_out, f_Sf, g_K, g_V, g_Q, g_d, g_S0)
    return worst


def bench(B, H, T, N, P=16, W=128, n_repeat=3):
    """Benchmark full f+b step."""
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)

    def setup(fn_cls):
        k = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
        v = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
        q = (0.3 * torch.randn(T, B, H, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
        decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')
                              ).detach().requires_grad_(True)
        S0 = (0.1 * torch.randn(B, H, N, N, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
        return k, v, q, decay, S0

    def bench_fn(fn_cls, use_hybrid):
        k, v, q, decay, S0 = setup(fn_cls)
        def run():
            if use_hybrid:
                S_final, output = fn_cls.apply(True, k, v, q, decay, S0, H, P, W)
            else:
                S_final, output = fn_cls.apply(True, k, v, q, decay, S0, H)
            loss = output.sum() + S_final.pow(2).sum() * 1e-4
            loss.backward()
            k.grad = None; v.grad = None; q.grad = None; decay.grad = None; S0.grad = None
        for _ in range(3): run()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_repeat): run()
        torch.cuda.synchronize()
        return (time.time() - t0) / n_repeat * 1000

    cuda_ms = bench_fn(E88FLAHybridCUDAFunction, use_hybrid=False)
    hyb_ms = bench_fn(PararnnHybridE88Function, use_hybrid=True)
    speedup = cuda_ms / hyb_ms
    print(f"  B={B} H={H:3d} T={T:6d} N={N}  CUDA={cuda_ms:>7.1f}ms  hybrid={hyb_ms:>7.1f}ms  spd={speedup:>5.2f}×")
    return cuda_ms, hyb_ms, speedup


if __name__ == '__main__':
    print("Phase 5 — real-layer validation\n")

    print("== Correctness: CUDA vs hybrid gradients (bf16 tolerance 10%) ==")
    for shape in [(1, 4, 4096, 16), (1, 8, 8192, 32)]:
        test_correctness(*shape)

    print("\n== Benchmark: full training step at production configs ==")
    for name, H, N in [("E88-n16 480M", 141, 16), ("E88-n32 480M", 83, 32)]:
        print(f"\n  {name}:")
        for T in [16384, 32768, 65536]:
            try:
                bench(1, H, T, N)
            except Exception as e:
                print(f"    FAIL T={T}: {str(e)[:100]}")
