"""Monkey-patch E88FLAHybridCUDAFunction.apply to use PararnnHybridE88V2.

Usage in train.py:
    if os.environ.get('ELMAN_PARARNN_HYBRID') == '1':
        from experiments.pararnn_kernel.tree_scan.install_hybrid import install
        install()

The patched `apply` dispatches by shape:
  - Square state (n_state == head_v_dim): use PararnnHybridE88V2
  - Anything else: fall through to original CUDA kernel
"""

import os
import sys
import torch


def install():
    # Add experiment tree to path so imports work.
    THIS = os.path.dirname(os.path.abspath(__file__))
    if THIS not in sys.path:
        sys.path.insert(0, THIS)
    # Also phase 7 fused_backward lives in the parent dir.
    PARENT = os.path.dirname(THIS)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

    import elman.models.e88_fla_hybrid as e88m
    from phase6_hybrid import PararnnHybridE88V2
    from phase7_fused_gate_hybrid import hybrid_with_fused_gate

    # Disable cuBLAS backward path (less common variant).
    if hasattr(e88m, 'USE_CUBLAS_BACKWARD'):
        e88m.USE_CUBLAS_BACKWARD = False

    # 1) Replace the unfused E88FLAHybridCUDAFunction with our hybrid.
    original_unfused = e88m.E88FLAHybridCUDAFunction.apply

    def patched_unfused(training, k, v, q, decay, S0, n_heads):
        Ns = k.size(-1); Hv = v.size(-1)
        if Ns == Hv:
            try:
                return PararnnHybridE88V2.apply(training, k, v, q, decay, S0, n_heads)
            except Exception as e:
                if os.environ.get('ELMAN_PARARNN_HYBRID_DEBUG') == '1':
                    print(f"[hybrid] unfused fell back to CUDA: {e}")
                return original_unfused(training, k, v, q, decay, S0, n_heads)
        return original_unfused(training, k, v, q, decay, S0, n_heads)

    e88m.E88FLAHybridCUDAFunction.apply = staticmethod(patched_unfused)

    # 2) Replace the silu-fused-gate E88FusedGateCUDAFunction with our
    #    hybrid + external gate (PyTorch tracks gate grad automatically).
    #    E88FusedGateCUDAFunction is what --use_gate 1 --gate_activation silu
    #    actually invokes — patching this is what gets the hybrid into the
    #    training path.
    original_fused_gate = e88m.E88FusedGateCUDAFunction.apply

    def patched_fused_gate(training, k, v, q, decay, g, S0, n_heads):
        Ns = k.size(-1); Hv = v.size(-1)
        if Ns == Hv:
            try:
                return hybrid_with_fused_gate(training, k, v, q, decay, g, S0, n_heads)
            except Exception as e:
                if os.environ.get('ELMAN_PARARNN_HYBRID_DEBUG') == '1':
                    print(f"[hybrid] fused-gate fell back to CUDA: {e}")
                return original_fused_gate(training, k, v, q, decay, g, S0, n_heads)
        return original_fused_gate(training, k, v, q, decay, g, S0, n_heads)

    e88m.E88FusedGateCUDAFunction.apply = staticmethod(patched_fused_gate)

    # 3) Replace the [B,T,H,N]-layout E88OptimizedCUDAFunction.  This is the
    #    THIRD entry point — and the one training actually uses when
    #    USE_OPTIMIZED_KERNELS=True (default).  Needs transpose to/from
    #    [T,B,H,N] for the hybrid and adapter for the g+apply_gate args.
    original_optimized = e88m.E88OptimizedCUDAFunction.apply

    def patched_optimized(training, k, v, q, decay, g, S0, n_heads,
                          apply_gate=True, normalize_kq=False,
                          checkpoint_interval=16):
        """Adapter from [B,T,H,N] → [T,B,H,N] → hybrid → back.

        Handles normalize_kq=True by L2-normalizing k and q in Python
        before calling the hybrid (matches what the CUDA kernel does
        internally for use_fused_l2).
        """
        Ns = k.size(-1); Hv = v.size(-1)
        # Rectangular state and no-gate cases: fall back.
        if Ns != Hv or not apply_gate or g is None:
            return original_optimized(training, k, v, q, decay, g, S0, n_heads,
                                       apply_gate, normalize_kq, checkpoint_interval)
        try:
            # Apply L2 norm on k and q if requested.  Matches what the CUDA
            # "fused L2" kernel does in shared memory before the recurrence.
            if normalize_kq:
                k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
                q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            # Transpose to [T, B, H, N]
            k_tb = k.transpose(0, 1).contiguous()
            v_tb = v.transpose(0, 1).contiguous()
            q_tb = q.transpose(0, 1).contiguous()
            decay_tb = decay.transpose(0, 1).contiguous()
            g_tb = g.transpose(0, 1).contiguous()
            S_final, out_tb = hybrid_with_fused_gate(
                training, k_tb, v_tb, q_tb, decay_tb, g_tb, S0, n_heads
            )
            out_bt = out_tb.transpose(0, 1).contiguous()
            return S_final, out_bt
        except Exception as e:
            if os.environ.get('ELMAN_PARARNN_HYBRID_DEBUG') == '1':
                print(f"[hybrid] optimized fell back to CUDA: {e}")
            return original_optimized(training, k, v, q, decay, g, S0, n_heads,
                                       apply_gate, normalize_kq, checkpoint_interval)

    e88m.E88OptimizedCUDAFunction.apply = staticmethod(patched_optimized)

    print("[hybrid] installed PararnnHybridE88V2 for unfused + fused-gate + optimized paths")


if __name__ == '__main__':
    install()
    # Quick smoke test
    from elman.models.e88_fla_hybrid import E88FLAHybridCUDAFunction
    import torch

    torch.manual_seed(0)
    dt = torch.bfloat16
    B, T, H, N = 1, 256, 4, 16
    k = torch.randn(T, B, H, N, dtype=dt, device='cuda', requires_grad=True)
    v = torch.randn(T, B, H, N, dtype=dt, device='cuda', requires_grad=True)
    q = torch.randn(T, B, H, N, dtype=dt, device='cuda', requires_grad=True)
    decay = torch.sigmoid(torch.randn(T, B, H, dtype=dt, device='cuda')).requires_grad_(True)
    S0 = 0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda', requires_grad=True)
    S_final, output = E88FLAHybridCUDAFunction.apply(True, k, v, q, decay, S0, H)
    loss = output.sum()
    loss.backward()
    print(f"[hybrid] smoke test PASS  output={output.shape}  S_final={S_final.shape}  "
          f"dK={k.grad.abs().max().item():.2e}")
