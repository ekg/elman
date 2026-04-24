"""Try torch.compile on the E88 hybrid path to reduce kernel launch overhead."""

import os, sys, time
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))

from install_hybrid import install
install()

import schedulefree
from elman.models import LadderLM


def time_step(model, opt, x, n=3):
    """Time n full training steps; return mean step time in ms and tok/s."""
    # Warmup
    for _ in range(2):
        loss = model(x, return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n):
        loss = model(x, return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt_ms = (time.time() - t0) / n * 1000
    B, T = x.shape
    return dt_ms, B * T / dt_ms * 1000, loss.item()


def build_model(**kwargs):
    torch.manual_seed(42)
    n_state, H, depth, dim = 16, 141, 25, 1536
    model = LadderLM(
        vocab_size=256, dim=dim, depth=depth, level=88,
        n_heads=H, n_state=n_state,
        use_gate=True, gate_activation='silu', expansion=1.0,
        gradient_checkpointing=True,
        **kwargs,
    ).cuda().to(torch.bfloat16)
    model.train()
    return model


def main():
    B, T = 1, 32768
    x = torch.randint(0, 256, (B, T), dtype=torch.int64, device='cuda')

    # Baseline
    print("=" * 72)
    print("Baseline (no compile)")
    print("=" * 72)
    model = build_model()
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=7.9e-4)
    opt.train()
    dt_ms, toks, loss = time_step(model, opt, x)
    print(f"  step time: {dt_ms:.1f} ms   tok/s: {toks:.0f}   loss: {loss:.4f}")
    del model, opt
    torch.cuda.empty_cache()

    # Torch compile modes
    for mode in ['default', 'reduce-overhead', 'max-autotune']:
        print()
        print("=" * 72)
        print(f"torch.compile mode='{mode}'")
        print("=" * 72)
        try:
            model = build_model()
            # Use _orig_mod if needed; compile on the nn.Module.
            # Since custom autograd functions don't compose perfectly, compile module-level.
            compiled = torch.compile(model, mode=mode, dynamic=False)

            opt = schedulefree.AdamWScheduleFree(compiled.parameters(), lr=7.9e-4)
            opt.train()
            dt_ms, toks, loss = time_step(compiled, opt, x, n=3)
            print(f"  step time: {dt_ms:.1f} ms   tok/s: {toks:.0f}   loss: {loss:.4f}")
            del model, compiled, opt
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
