"""Try CUDA graphs on the E88 fwd+bwd training step.

Strategy: record the entire step into a CUDA graph, then replay.

Known risks:
- CUDA graphs don't work with varying input sizes (must be fixed shape)
- Triton kernels work with CUDA graphs as long as no dynamic stream capture issues.
- AdamWScheduleFree reads its internal state — may not be graph-compatible.

Approach: Try fwd-only with CUDA graph first (most common successful pattern).
"""

import os, sys, time
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))

from install_hybrid import install
install()

from elman.models import LadderLM


def time_fn(fn, n=2, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def build_model():
    torch.manual_seed(42)
    n_state, H, depth, dim = 16, 141, 25, 1536
    model = LadderLM(
        vocab_size=256, dim=dim, depth=depth, level=88,
        n_heads=H, n_state=n_state,
        use_gate=True, gate_activation='silu', expansion=1.0,
        gradient_checkpointing=True,
    ).cuda().to(torch.bfloat16)
    model.train()
    return model


def main():
    B, T = 1, 32768
    x_static = torch.randint(0, 256, (B, T), dtype=torch.int64, device='cuda')

    # ========================================================================
    # Baseline
    # ========================================================================
    print("=" * 72)
    print("Baseline forward+backward")
    print("=" * 72)
    model = build_model()

    def baseline_fwd_bwd():
        for p in model.parameters():
            p.grad = None
        loss = model(x_static, return_loss=True)
        loss.backward()

    ms = time_fn(baseline_fwd_bwd)
    print(f"  mean step: {ms:.1f} ms")

    # ========================================================================
    # CUDA graph: try to capture fwd only (no backward)
    # ========================================================================
    print()
    print("=" * 72)
    print("CUDA graph: forward only")
    print("=" * 72)
    try:
        # Create static input/output buffers
        model.eval()  # disable dropout etc.  Important: eval disables grad checkpointing
        # But grad_ckpt is a problem for CUDA graphs — disable it.

        model_ng = build_model()
        model_ng.gradient_checkpointing = False
        for m in model_ng.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = False
        model_ng.eval()

        # Warmup on side stream first (required for cuda graph capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    out = model_ng(x_static)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                static_out = model_ng(x_static)

        def run_graph():
            g.replay()

        ms = time_fn(run_graph)
        print(f"  mean step (graph replay): {ms:.1f} ms")
    except Exception as e:
        import traceback
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()

    del model_ng
    torch.cuda.empty_cache()

    # ========================================================================
    # CUDA graph: fwd+bwd
    # ========================================================================
    print()
    print("=" * 72)
    print("CUDA graph: forward + backward (training) — usually fails")
    print("=" * 72)
    try:
        model_ngb = build_model()
        model_ngb.gradient_checkpointing = False
        for m in model_ngb.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = False
        model_ngb.train()

        # Warmup on side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for p in model_ngb.parameters():
                    p.grad = None
                loss = model_ngb(x_static, return_loss=True)
                loss.backward()
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        for p in model_ngb.parameters():
            p.grad = None  # IMPORTANT: grads must not be None in capture
        # Run once to allocate grads
        loss = model_ngb(x_static, return_loss=True)
        loss.backward()
        torch.cuda.synchronize()

        # Now zero grads for replay (in-place)
        for p in model_ngb.parameters():
            p.grad.zero_()

        with torch.cuda.graph(g):
            for p in model_ngb.parameters():
                p.grad.zero_()
            loss_captured = model_ngb(x_static, return_loss=True)
            loss_captured.backward()

        def run_graph_train():
            g.replay()

        ms = time_fn(run_graph_train)
        print(f"  mean step (graph replay): {ms:.1f} ms")
    except Exception as e:
        import traceback
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
        # Keep it short - traceback
        lines = traceback.format_exc().splitlines()
        for ln in lines[-10:]:
            print(f"    {ln}")


if __name__ == '__main__':
    main()
