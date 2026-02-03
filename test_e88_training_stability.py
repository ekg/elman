#!/usr/bin/env python3
"""
Test E88 training stability with many heads.

Since CUDA kernel matches Python at the forward/backward level,
test actual training to find where divergence occurs.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from elman.models.e88_fla_hybrid import E88FLAHybrid, E88_NATIVE_CUDA_AVAILABLE


def test_training_stability(n_heads, n_state, dim=512, T=64, B=4, steps=50, lr=3e-4, verbose=False):
    """Train a single E88 layer for a few steps and check for divergence."""
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)

    # Create model
    model = E88FLAHybrid(
        dim=dim,
        n_state=n_state,
        n_heads=n_heads,
        expansion=1.0,
        use_conv=False,
        use_gate=False,
        use_output_norm=False,
    ).to(device).to(dtype)

    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    grad_norms = []
    output_norms = []

    model.train()

    for step in range(steps):
        # Random input each step
        x = torch.randn(B, T, dim, device=device, dtype=dtype)
        target = torch.randn(B, T, dim, device=device, dtype=dtype)

        optimizer.zero_grad()

        try:
            out, _ = model(x, use_cuda=True)
            loss = ((out - target) ** 2).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                return {
                    'n_heads': n_heads,
                    'n_state': n_state,
                    'diverged_at': step,
                    'reason': 'loss_nan_inf',
                    'final_loss': float('inf'),
                    'max_grad_norm': float('inf'),
                    'max_output_norm': float('inf'),
                }

            loss.backward()

            # Check gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if total_norm > 1e6 or torch.isnan(torch.tensor(total_norm)):
                return {
                    'n_heads': n_heads,
                    'n_state': n_state,
                    'diverged_at': step,
                    'reason': 'grad_explosion',
                    'final_loss': loss.item(),
                    'max_grad_norm': total_norm,
                    'max_output_norm': out.norm().item(),
                }

            optimizer.step()

            losses.append(loss.item())
            grad_norms.append(total_norm)
            output_norms.append(out.norm().item())

            if verbose and step % 10 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.2f}, out_norm={out.norm().item():.2f}")

        except RuntimeError as e:
            return {
                'n_heads': n_heads,
                'n_state': n_state,
                'diverged_at': step,
                'reason': f'runtime_error: {str(e)[:50]}',
                'final_loss': float('inf'),
                'max_grad_norm': float('inf'),
                'max_output_norm': float('inf'),
            }

    return {
        'n_heads': n_heads,
        'n_state': n_state,
        'diverged_at': None,  # Didn't diverge
        'reason': 'stable',
        'final_loss': losses[-1],
        'max_grad_norm': max(grad_norms),
        'max_output_norm': max(output_norms),
        'mean_grad_norm': sum(grad_norms) / len(grad_norms),
    }


def main():
    print("=" * 80)
    print("E88 Training Stability Test (many heads)")
    print("=" * 80)

    if not E88_NATIVE_CUDA_AVAILABLE:
        print("ERROR: E88 CUDA kernel not available!")
        return

    # Test configurations - focus on the boundary where training fails
    configs = [
        # Working configs
        (8, 32, 512),    # 8 heads, n_state=32, dim=512
        (8, 48, 512),    # 8 heads, n_state=48
        (12, 32, 512),   # 12 heads
        (16, 32, 512),   # 16 heads - boundary?

        # Potentially failing
        (24, 32, 512),   # 24 heads
        (32, 32, 512),   # 32 heads
        (36, 32, 384),   # 36 heads - suspected failure point
        (36, 48, 384),   # 36 heads, n_state=48
        (48, 32, 384),   # 48 heads
        (64, 32, 256),   # 64 heads

        # Larger configs
        (72, 48, 256),   # 72 heads, n_state=48
        (96, 32, 256),   # 96 heads
        (128, 32, 256),  # 128 heads
    ]

    print(f"\n{'n_heads':>8} {'n_state':>8} {'dim':>6} {'Status':>12} {'Diverge@':>10} {'FinalLoss':>12} {'MaxGrad':>12} {'Reason'}")
    print("-" * 100)

    for n_heads, n_state, dim in configs:
        result = test_training_stability(n_heads, n_state, dim=dim, steps=50, verbose=False)

        status = "STABLE" if result['diverged_at'] is None else "DIVERGED"
        diverge_at = str(result['diverged_at']) if result['diverged_at'] is not None else "-"
        final_loss = f"{result['final_loss']:.4f}" if result['final_loss'] < 1e6 else "INF"
        max_grad = f"{result['max_grad_norm']:.2f}" if result['max_grad_norm'] < 1e6 else "INF"
        reason = result['reason'][:20]

        marker = " ***" if result['diverged_at'] is not None else ""
        print(f"{n_heads:>8} {n_state:>8} {dim:>6} {status:>12} {diverge_at:>10} {final_loss:>12} {max_grad:>12} {reason}{marker}")

    # Test with different learning rates for failing config
    print("\n" + "=" * 80)
    print("Learning Rate Sensitivity (n_heads=36, n_state=48)")
    print("=" * 80)

    for lr in [3e-4, 1e-4, 3e-5, 1e-5]:
        result = test_training_stability(36, 48, dim=384, steps=50, lr=lr, verbose=False)
        status = "STABLE" if result['diverged_at'] is None else "DIVERGED"
        final_loss = f"{result['final_loss']:.4f}" if result['final_loss'] < 1e6 else "INF"
        max_grad = f"{result['max_grad_norm']:.2f}" if result['max_grad_norm'] < 1e6 else "INF"
        print(f"  lr={lr:.0e}: {status}, loss={final_loss}, max_grad={max_grad}")

    # Test with gradient clipping
    print("\n" + "=" * 80)
    print("Gradient Clipping Test (n_heads=36, n_state=48)")
    print("=" * 80)

    for clip_val in [1.0, 0.5, 0.1]:
        result = test_training_with_clipping(36, 48, dim=384, steps=50, clip_val=clip_val)
        status = "STABLE" if result['diverged_at'] is None else "DIVERGED"
        final_loss = f"{result['final_loss']:.4f}" if result['final_loss'] < 1e6 else "INF"
        print(f"  clip={clip_val}: {status}, loss={final_loss}")


def test_training_with_clipping(n_heads, n_state, dim=512, T=64, B=4, steps=50, clip_val=1.0):
    """Train with gradient clipping."""
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)

    model = E88FLAHybrid(
        dim=dim,
        n_state=n_state,
        n_heads=n_heads,
        expansion=1.0,
        use_conv=False,
        use_gate=False,
        use_output_norm=False,
    ).to(device).to(dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()

    for step in range(steps):
        x = torch.randn(B, T, dim, device=device, dtype=dtype)
        target = torch.randn(B, T, dim, device=device, dtype=dtype)

        optimizer.zero_grad()

        try:
            out, _ = model(x, use_cuda=True)
            loss = ((out - target) ** 2).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                return {'diverged_at': step, 'final_loss': float('inf'), 'n_heads': n_heads, 'n_state': n_state}

            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            optimizer.step()

        except RuntimeError:
            return {'diverged_at': step, 'final_loss': float('inf'), 'n_heads': n_heads, 'n_state': n_state}

    return {'diverged_at': None, 'final_loss': loss.item(), 'n_heads': n_heads, 'n_state': n_state}


if __name__ == '__main__':
    main()
