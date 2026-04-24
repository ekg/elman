"""Phase 6 — full Triton hybrid for E88.

Forward: single Triton pass (pararnn_seq_fwd_output_triton) that produces
both dense S_traj and per-step outputs.  Eliminates the ADMM + warmup-scan
CUDA forwards entirely.

Backward: Triton rectangular rank-1 kernel (pararnn_bwd_rect) — same
fix as phase5 but with independent M (HEAD_V_DIM) and N (N_STATE).

Drop-in for E88FLAHybridCUDAFunction: same (training, k, v, q, decay, S0,
n_heads) signature.
"""

import sys, os, time
import torch

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pararnn_seq_fwd_rect import pararnn_seq_fwd_output_triton
from pararnn_seq_fwd import pararnn_seq_fwd_triton  # fast square fwd (legacy, non-contig)
from pararnn_seq_fwd_v2 import pararnn_seq_fwd_v2, backward_v2  # contig S_traj + fused dQ
from pararnn_seq_fwd_v2_fp8 import pararnn_seq_fwd_v2_fp8, backward_v2_fp8  # fp8-E4M3 S_traj
from pararnn_bwd_rect import pararnn_bwd_rect
from pararnn_bwd_fused_dq import backward_with_dq  # square, with dQ fused in (non-contig S_traj)
from phase7_fused_backward import backward_e88_fused_rank1
from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction

# Opt-in flag: set ELMAN_PARARNN_FP8=1 to enable fp8-E4M3 storage of S_traj.
# Saves ~50% activation memory for the recurrence; roughly 1.05× speedup on
# fwd+bwd combined (kernel is compute-bound, not BW-bound, so fp8 mostly
# trades time for memory).  Square state only.
FP8_STORAGE = os.environ.get('ELMAN_PARARNN_FP8') == '1'


class PararnnHybridE88V2(torch.autograd.Function):
    """Single-Triton-pass hybrid E88.  Same signature as E88FLAHybridCUDAFunction."""

    @staticmethod
    def forward(ctx, training, k, v, q, decay, S0, n_heads):
        """
        k:     [T, B, H, Ns]
        v:     [T, B, H, Hv]
        q:     [T, B, H, Ns]
        decay: [T, B, H]
        S0:    [B, H, Ns, Hv]    CUDA convention
        Returns (S_final [B, H, Ns, Hv], output [T, B, H, Hv]).
        """
        T, B, H, Ns = k.shape
        _, _, _, Hv = v.shape
        # Permute to Pararnn layout.
        K_p = k.permute(1, 2, 0, 3).contiguous()       # [B, H, T, Ns=N]
        V_p = v.permute(1, 2, 0, 3).contiguous()       # [B, H, T, Hv=M]
        Q_p = q.permute(1, 2, 0, 3).contiguous()       # [B, H, T, Ns=N]
        decay_p = decay.permute(1, 2, 0).contiguous()  # [B, H, T]
        # Pararnn_S = CUDA_S^T, so shape [B, H, M=Hv, N=Ns]
        S0_p = S0.transpose(-1, -2).contiguous()

        # Forward path selection, empirically tuned per shape:
        # - Square N=16: fused Triton kernel emitting both S_traj AND Sq
        #   (S_traj has a padding slot for S_0, used by backward_with_dq).
        # - Square N>=32: v2 path — contig S_traj [B,H,T,N,N] (no S_0) +
        #   separate Sq einsum on contig data + branch-free backward_v2.
        #   At N=32 the contig einsum is a big win over the fused kernel.
        # - Rectangular: fused rect kernel (no rect v2 variant yet).
        use_v2 = (Ns == Hv) and (Ns >= 32)
        use_fp8 = FP8_STORAGE and (Ns == Hv)  # fp8 path: square state only
        if use_fp8:
            # FP8-storage path: S_traj stored as fp8_e4m3fn (halves memory).
            # Always uses the contig v2 layout (needed for fp8 bwd kernel).
            fwd_nw = 4 if Ns >= 32 else 1
            S_traj = pararnn_seq_fwd_v2_fp8(S0_p, K_p, V_p, decay_p, num_warps=fwd_nw)
            # Sq einsum requires a non-fp8 dtype — rehydrate to bf16 view.
            S_traj_bf16 = S_traj.to(torch.bfloat16)
            Sq = torch.einsum('bhtpq,bhtq->bhtp', S_traj_bf16, Q_p)
            S_final_p = S_traj_bf16[:, :, -1]
            # Free the transient bf16 rehydration; we keep only the fp8 tensor.
            del S_traj_bf16
        elif use_v2:
            fwd_nw = 4
            S_traj = pararnn_seq_fwd_v2(S0_p, K_p, V_p, decay_p, num_warps=fwd_nw)
            Sq = torch.einsum('bhtpq,bhtq->bhtp', S_traj, Q_p)
            S_final_p = S_traj[:, :, -1]
        else:
            fwd_nw = 4 if max(Ns, Hv) >= 24 else 1
            S_traj, Sq = pararnn_seq_fwd_output_triton(
                S0_p, K_p, V_p, Q_p, decay_p, num_warps=fwd_nw,
            )
            S_final_p = S_traj[:, :, -1]
        S_final = S_final_p.transpose(-1, -2).contiguous()
        output = Sq.permute(2, 0, 1, 3).contiguous()

        if use_fp8 or use_v2:
            ctx.save_for_backward(K_p, V_p, Q_p, decay_p, S_traj, S0_p)
        else:
            ctx.save_for_backward(K_p, V_p, Q_p, decay_p, S_traj)
        ctx.dims = (T, B, H, Ns, Hv, fwd_nw, use_v2, use_fp8)
        return S_final, output

    @staticmethod
    def backward(ctx, dS_final, d_output):
        use_v2 = ctx.dims[6]
        use_fp8 = ctx.dims[7] if len(ctx.dims) > 7 else False
        if use_fp8 or use_v2:
            K_p, V_p, Q_p, decay_p, S_traj, S0_p = ctx.saved_tensors
        else:
            K_p, V_p, Q_p, decay_p, S_traj = ctx.saved_tensors
        T, B, H, Ns, Hv = ctx.dims[:5]

        dL_dout_p = d_output.permute(1, 2, 0, 3).contiguous()
        g_T_p = dS_final.transpose(-1, -2).contiguous()
        bwd_nw = 2 if Ns >= 32 else 1

        if use_fp8:
            # fp8 backward: tuned num_warps (N=16 -> 1, N=32 -> 4 per bench)
            fp8_bwd_nw = 1 if Ns <= 16 else 4
            dS0_p, dK_p, dV_p, dQ_p, ddec_p = backward_v2_fp8(
                S0_p, S_traj, K_p, V_p, decay_p, g_T_p, dL_dout_p, Q_p,
                num_warps=fp8_bwd_nw, num_stages=1,
            )
        elif use_v2:
            # v2: contig S_traj [B, H, T, N, N] (S_1..S_T), S0 separate.
            # backward_v2 fuses dQ and handles the S_0 boundary branch-free.
            dS0_p, dK_p, dV_p, dQ_p, ddec_p = backward_v2(
                S0_p, S_traj, K_p, V_p, decay_p, g_T_p, dL_dout_p, Q_p,
                num_warps=bwd_nw, num_stages=1,
            )
        elif Ns == Hv:
            # Square non-v2 (N=16): S_traj from fused fwd+output kernel has
            # shape [B, H, T+1, M, N].  Use backward_with_dq (fuses dQ).
            dS0_p, dK_p, dV_p, dQ_p, ddec_p = backward_with_dq(
                S_traj, K_p, V_p, decay_p, g_T_p, dL_dout_p, Q_p,
                num_warps=bwd_nw, num_stages=1,
            )
        else:
            # Rectangular: S_traj [B, H, T+1, M, N].
            dS0_p, dK_p, dV_p, ddec_p = pararnn_bwd_rect(
                S_traj, K_p, V_p, decay_p, g_T_p, dL_dout_p, Q_p,
                num_warps=bwd_nw, num_stages=1,
            )
            dQ_p = torch.einsum('bhti,bhtij->bhtj', dL_dout_p, S_traj[:, :, 1:])

        # Back to CUDA [T, B, H, *] layout.
        dK_out = dK_p.permute(2, 0, 1, 3).contiguous()
        dV_out = dV_p.permute(2, 0, 1, 3).contiguous()
        dQ_out = dQ_p.permute(2, 0, 1, 3).contiguous()
        ddec_out = ddec_p.permute(2, 0, 1).contiguous()
        dS0_out = dS0_p.transpose(-1, -2).contiguous()

        return None, dK_out, dV_out, dQ_out, ddec_out, dS0_out, None


# ============================================================================
# Correctness: compare v2 hybrid vs CUDA on supported (square) shapes.
# Compare v2 hybrid vs fp32 Python autograd on unsupported rectangular shapes.
# ============================================================================
def cuda_step_py(S_prev, k, v, decay):
    retrieved = torch.einsum('bhij,bhi->bhj', S_prev, k)
    delta = v - retrieved
    outer = torch.einsum('bhj,bhi->bhij', delta, k)
    return torch.tanh(decay[..., None, None] * S_prev + outer)


def cuda_forward_py(S0, K, V, Q, decay):
    T = K.shape[0]
    S = S0
    outputs = []
    for t in range(T):
        S = cuda_step_py(S, K[t], V[t], decay[t])
        outputs.append(torch.einsum('bhij,bhi->bhj', S, Q[t]))
    return S, torch.stack(outputs, 0)


def rel(a, b):
    num = (a.float() - b.float()).abs().max().item()
    denom = max(b.float().abs().max().item(), 1e-10)
    return num / denom


def test_correctness_square(B, H, T, Ns, Hv=None, seed=0):
    if Hv is None:
        Hv = Ns
    dt = torch.bfloat16
    torch.manual_seed(seed)
    k = (0.3 * torch.randn(T, B, H, Ns, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, Hv, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, Ns, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, Ns, Hv, dtype=dt, device='cuda')).requires_grad_(True)

    # CUDA path (square only)
    k1, v1, q1, decay1, S01 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, S0]]
    S_f_c, out_c = E88FLAHybridCUDAFunction.apply(True, k1, v1, q1, decay1, S01, H)
    torch.manual_seed(seed + 1)
    dL_dout = 0.01 * torch.randn_like(out_c)
    # CUDA ignores dS_final; match that.
    loss_c = (out_c * dL_dout).sum()
    loss_c.backward()
    dK_c, dV_c, dQ_c, ddec_c = k1.grad, v1.grad, q1.grad, decay1.grad

    # Hybrid v2
    k2, v2, q2, decay2, S02 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, S0]]
    S_f_h, out_h = PararnnHybridE88V2.apply(True, k2, v2, q2, decay2, S02, H)
    loss_h = (out_h * dL_dout).sum()
    loss_h.backward()

    r = {
        'output': rel(out_h, out_c),
        'S_final': rel(S_f_h, S_f_c),
        'dK': rel(k2.grad, dK_c),
        'dV': rel(v2.grad, dV_c),
        'dQ': rel(q2.grad, dQ_c),
        'ddec': rel(decay2.grad, ddec_c),
    }
    worst = max(r.values())
    ok = "PASS" if worst < 0.05 else "FAIL"
    details = "  ".join(f"{k}={v:.1e}" for k, v in r.items())
    print(f"  SQUARE vs CUDA  B={B} H={H} T={T} Ns={Ns} Hv={Hv}")
    print(f"    {details}  [{ok}]")


def test_correctness_rect_fp32(B, H, T, Ns, Hv, seed=0):
    """Rect: compare hybrid vs fp32 autograd through Python CUDA-conv forward."""
    dt = torch.float32
    torch.manual_seed(seed)
    k = (0.3 * torch.randn(T, B, H, Ns, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, Hv, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, Ns, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, Ns, Hv, dtype=dt, device='cuda')).requires_grad_(True)

    S_f_r, out_r = cuda_forward_py(S0, k, v, q, decay)
    torch.manual_seed(seed + 1)
    dL_dout = 0.01 * torch.randn_like(out_r)
    loss_r = (out_r * dL_dout).sum()
    loss_r.backward()
    dK_r, dV_r, dQ_r, ddec_r, dS0_r = k.grad.clone(), v.grad.clone(), q.grad.clone(), decay.grad.clone(), S0.grad.clone()

    # Hybrid
    k2, v2, q2, decay2, S02 = [x.detach().clone().requires_grad_(True) for x in [k, v, q, decay, S0]]
    S_f_h, out_h = PararnnHybridE88V2.apply(True, k2, v2, q2, decay2, S02, H)
    loss_h = (out_h * dL_dout).sum()
    loss_h.backward()

    r = {
        'output': rel(out_h, out_r),
        'S_final': rel(S_f_h, S_f_r),
        'dK': rel(k2.grad, dK_r),
        'dV': rel(v2.grad, dV_r),
        'dQ': rel(q2.grad, dQ_r),
        'ddec': rel(decay2.grad, ddec_r),
        'dS0': rel(S02.grad, dS0_r),
    }
    worst = max(r.values())
    ok = "PASS" if worst < 1e-4 else "FAIL"
    details = "  ".join(f"{k}={v:.1e}" for k, v in r.items())
    print(f"  RECT vs autograd fp32  B={B} H={H} T={T} Ns={Ns} Hv={Hv}")
    print(f"    {details}  [{ok}]")


def bench_cuda(B, H, T, Ns, Hv, n_repeat=3):
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = (0.3 * torch.randn(T, B, H, Ns, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, Hv, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, Ns, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')).detach().requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, Ns, Hv, generator=g, dtype=dt, device='cuda')).requires_grad_(True)

    def run():
        S_f, out = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
        loss = out.sum()  # CUDA ignores dS_final anyway; match that.
        loss.backward()
        k.grad = None; v.grad = None; q.grad = None; decay.grad = None
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


def bench_hybrid_v2(B, H, T, Ns, Hv, n_repeat=3):
    dt = torch.bfloat16
    g = torch.Generator(device='cuda').manual_seed(0)
    k = (0.3 * torch.randn(T, B, H, Ns, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(T, B, H, Hv, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(T, B, H, Ns, generator=g, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(T, B, H, generator=g, dtype=dt, device='cuda')).detach().requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, Ns, Hv, generator=g, dtype=dt, device='cuda')).requires_grad_(True)

    def run():
        S_f, out = PararnnHybridE88V2.apply(True, k, v, q, decay, S0, H)
        loss = out.sum()  # match CUDA's behavior for fair comparison
        loss.backward()
        k.grad = None; v.grad = None; q.grad = None; decay.grad = None; S0.grad = None
    for _ in range(3): run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat): run()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_repeat * 1000


if __name__ == '__main__':
    print("Phase 6 — fused Triton forward + rectangular backward\n")

    print("=== Correctness: hybrid v2 vs CUDA (square) ===")
    test_correctness_square(1, 4, 4096, 16)
    test_correctness_square(1, 8, 8192, 32)

    print("\n=== Correctness: hybrid v2 vs fp32 autograd (rect) ===")
    for Ns, Hv in [(32, 24), (32, 23), (16, 14)]:
        test_correctness_rect_fp32(1, 4, 512, Ns, Hv)

    print("\n=== Benchmark: hybrid v2 vs CUDA (square configs) ===")
    for name, H, Ns in [("E88-n16 square", 141, 16), ("E88-n32 square", 83, 32)]:
        print(f"\n  {name}:")
        for T in [16384, 32768, 65536]:
            try:
                torch.cuda.empty_cache()
                cuda_ms = bench_cuda(1, H, T, Ns, Ns)
                torch.cuda.empty_cache()
                hyb_ms = bench_hybrid_v2(1, H, T, Ns, Ns)
                spd = cuda_ms / hyb_ms
                print(f"    B=1 H={H:3d} T={T:6d} Ns=Hv={Ns}  CUDA={cuda_ms:>7.1f}ms  hyb_v2={hyb_ms:>7.1f}ms  spd={spd:>5.2f}×")
            except Exception as e:
                print(f"    FAIL T={T}: {str(e)[:120]}")
