"""Instrumented training step - measure forward/backward/opt/clip separately."""

import sys, os, time
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, '/home/erikg/elman')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get('ELMAN_PARARNN_HYBRID') == '1':
    from install_hybrid import install
    install()

import schedulefree
from elman.models import LadderLM


def build_model_and_opt(n_state=16, H=141, depth=25, dim=1536, lr=7.9e-4):
    model = LadderLM(
        vocab_size=256, dim=dim, depth=depth, level=88,
        n_heads=H, n_state=n_state,
        use_gate=True, gate_activation='silu', expansion=1.0,
    ).cuda().to(torch.bfloat16)
    model.train()
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)
    opt.train()
    return model, opt


def timed(label, fn):
    torch.cuda.synchronize()
    t0 = time.time()
    result = fn()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000, result


def instrumented_step(model, opt, x):
    """Mirror train.py's step, recording component times."""
    times = {}

    # Forward + loss
    def fwd():
        loss = model(x, return_loss=True)
        return loss
    times['forward'], loss = timed('fwd', fwd)

    # Backward
    def bwd():
        loss.backward()
    times['backward'], _ = timed('bwd', bwd)

    # Gradient norm (train.py default path: clip_grad_norm_)
    def grad_norm():
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    times['grad_norm'], gn = timed('gn', grad_norm)

    # Optimizer step
    def opt_step():
        opt.step()
    times['opt_step'], _ = timed('opt', opt_step)

    # Zero grads
    def zero():
        opt.zero_grad(set_to_none=True)
    times['zero_grad'], _ = timed('zero', zero)

    total = sum(times.values())
    times['total'] = total
    return times


def bench(n_state, H, depth, dim, mode, n_steps=20):
    torch.manual_seed(42)
    model, opt = build_model_and_opt(n_state, H, depth, dim)
    B, T = 16, 512
    x = torch.randint(0, 256, (B, T), dtype=torch.int64, device='cuda')

    # Warmup
    for _ in range(3):
        loss = model(x, return_loss=True)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Time each component across N steps
    agg = {}
    for _ in range(n_steps):
        times = instrumented_step(model, opt, x)
        for k, v in times.items():
            agg.setdefault(k, []).append(v)

    # Drop first 2 as warmup-affected
    print(f"\n[{mode}] Per-step breakdown (ms), over {n_steps-2} steady-state steps:")
    for k in ['forward', 'backward', 'grad_norm', 'opt_step', 'zero_grad', 'total']:
        vals = agg[k][2:]
        mean_ms = sum(vals)/len(vals)
        print(f"  {k:12s}: {mean_ms:>7.2f} ms")
    return agg


if __name__ == '__main__':
    mode = 'HYBRID' if os.environ.get('ELMAN_PARARNN_HYBRID') == '1' else 'CUDA'
    # E88-n16 production config
    bench(n_state=16, H=141, depth=25, dim=1536, mode=mode)
