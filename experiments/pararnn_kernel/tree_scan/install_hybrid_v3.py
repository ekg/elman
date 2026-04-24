"""Monkey-patch E88OptimizedCUDAFunction to use the V3 [B,T,H,N]-native hybrid.

Eliminates the adapter permutes that install_hybrid.py does.

Usage in train.py:
    if os.environ.get('ELMAN_PARARNN_HYBRID_V3') == '1':
        from experiments.pararnn_kernel.tree_scan.install_hybrid_v3 import install
        install()

Only patches E88OptimizedCUDAFunction (the path the training code actually
uses via USE_OPTIMIZED_KERNELS=True).  Falls back to CUDA for:
  - Rectangular state (Ns != Hv)
  - No gate / apply_gate=False
"""

import os
import sys


def install():
    THIS = os.path.dirname(os.path.abspath(__file__))
    if THIS not in sys.path:
        sys.path.insert(0, THIS)
    PARENT = os.path.dirname(THIS)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

    import elman.models.e88_fla_hybrid as e88m
    from phase6_hybrid_v3 import PararnnHybridE88V3, hybrid_v3_with_fused_gate

    if hasattr(e88m, 'USE_CUBLAS_BACKWARD'):
        e88m.USE_CUBLAS_BACKWARD = False

    # Patch the [B, T, H, N]-native entry point directly — no transposes!
    original_optimized = e88m.E88OptimizedCUDAFunction.apply

    def patched_optimized_v3(training, k, v, q, decay, g, S0, n_heads,
                              apply_gate=True, normalize_kq=False,
                              checkpoint_interval=16):
        Ns = k.size(-1); Hv = v.size(-1)
        if Ns != Hv or not apply_gate or g is None:
            return original_optimized(training, k, v, q, decay, g, S0, n_heads,
                                       apply_gate, normalize_kq, checkpoint_interval)
        try:
            # Optional L2 norm
            if normalize_kq:
                k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
                q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            # Direct [B, T, H, N] call — no permutes!
            S_final, out = hybrid_v3_with_fused_gate(
                training, k, v, q, decay, g, S0, n_heads,
            )
            return S_final, out
        except Exception as e:
            if os.environ.get('ELMAN_PARARNN_HYBRID_DEBUG') == '1':
                print(f"[hybrid_v3] fell back to CUDA: {e}")
            return original_optimized(training, k, v, q, decay, g, S0, n_heads,
                                       apply_gate, normalize_kq, checkpoint_interval)

    e88m.E88OptimizedCUDAFunction.apply = staticmethod(patched_optimized_v3)

    # Also patch the other two entry points.  For unfused and fused-gate,
    # these take [T, B, H, N] — we can either (a) transpose to [B, T, H, N]
    # and dispatch to V3 or (b) fall back to V2 (no permute savings there
    # since layout is still wrong for V3).  Option (a) is a net win because
    # we save the subsequent V2 internal permutes, but we pay one transpose.
    # For now, leave these using V2 (install_hybrid.py style) — production
    # path is E88OptimizedCUDAFunction.
    try:
        from install_hybrid import install as install_v2
        # This calls full install, which patches E88OptimizedCUDAFunction too
        # We want to re-patch it with v3 AFTER v2 has done its setup.
        # Simpler: don't call install_v2; just patch the three entry points here
        # individually.  But the other two are low priority for eliminating
        # permutes — leaving them as-is (pure CUDA) is fine.  If someone sets
        # BOTH env vars they'll get v2 for non-optimized paths and v3 for
        # the optimized one.
        pass
    except ImportError:
        pass

    print("[hybrid_v3] installed PararnnHybridE88V3 for E88OptimizedCUDAFunction (no permutes)")


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
    install()
    # Smoke test
    import torch
    from elman.models.e88_fla_hybrid import E88OptimizedCUDAFunction

    torch.manual_seed(0)
    dt = torch.bfloat16
    B, T, H, N = 2, 512, 4, 16
    k = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    v = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    q = (0.3 * torch.randn(B, T, H, N, dtype=dt, device='cuda')).requires_grad_(True)
    decay = torch.sigmoid(0.5 + 0.1 * torch.randn(B, T, H, dtype=dt, device='cuda')).detach().requires_grad_(True)
    g = torch.randn(B, T, H, N, dtype=dt, device='cuda', requires_grad=True)
    S0 = (0.1 * torch.randn(B, H, N, N, dtype=dt, device='cuda')).requires_grad_(True)
    S_final, output = E88OptimizedCUDAFunction.apply(True, k, v, q, decay, g, S0, H)
    print(f"[hybrid_v3] smoke test: output={output.shape}  S_final={S_final.shape}")
    loss = output.sum()
    loss.backward()
    print(f"[hybrid_v3] backward ok: dK max={k.grad.abs().max().item():.2e}")
