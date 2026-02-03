#!/usr/bin/env python3
"""
Test E88 stability within LadderLM (full model stack).
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from elman.models import LadderLM


def test_ladderlm_training(level, dim, depth, steps=100, lr=3e-4, verbose=False):
    """Train full LadderLM for a few steps."""
    device = 'cuda'

    torch.manual_seed(42)

    # Create model
    try:
        model = LadderLM(
            vocab_size=256,
            dim=dim,
            depth=depth,
            level=level,
        ).to(device)
    except Exception as e:
        return {
            'level': level,
            'dim': dim,
            'depth': depth,
            'diverged_at': 0,
            'reason': f'init_error: {str(e)[:50]}',
            'final_loss': float('inf'),
        }

    # Use bf16 training
    model = model.to(torch.bfloat16)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {level} (dim={dim}, depth={depth}): {num_params:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    B = 4
    T = 256  # Shorter sequence for quick test

    losses = []

    model.train()

    for step in range(steps):
        # Random token input
        x = torch.randint(0, 256, (B, T), device=device)

        optimizer.zero_grad()

        try:
            loss = model(x, return_loss=True)

            if torch.isnan(loss) or torch.isinf(loss):
                return {
                    'level': level,
                    'dim': dim,
                    'depth': depth,
                    'diverged_at': step,
                    'reason': 'loss_nan_inf',
                    'final_loss': float('inf'),
                }

            loss.backward()

            # Clip gradients (like train.py)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            losses.append(loss.item())

            if verbose and step % 20 == 0:
                print(f"    Step {step}: loss={loss.item():.4f}")

        except RuntimeError as e:
            return {
                'level': level,
                'dim': dim,
                'depth': depth,
                'diverged_at': step,
                'reason': f'runtime_error: {str(e)[:50]}',
                'final_loss': float('inf'),
            }

    return {
        'level': level,
        'dim': dim,
        'depth': depth,
        'diverged_at': None,
        'reason': 'stable',
        'final_loss': losses[-1] if losses else float('inf'),
        'min_loss': min(losses) if losses else float('inf'),
    }


def main():
    print("=" * 80)
    print("E88 LadderLM Training Stability Test")
    print("=" * 80)

    # Test configs - mix of working and failing
    configs = [
        # Working small head counts
        ('E88_d', 512, 6),       # 8 heads, n_state=32, small model
        ('E88_d', 768, 12),      # 8 heads, medium model

        # Larger head counts (from state-matched configs)
        ('E88_h20n32', 384, 6),  # 20 heads
        ('E88_h36n48', 384, 6),  # 36 heads - suspected failure
        ('E88_h48n32', 384, 6),  # 48 heads
        ('E88_h64n32', 256, 6),  # 64 heads
        ('E88_h72n48', 256, 6),  # 72 heads
        ('E88_h96n32', 256, 6),  # 96 heads
        ('E88_h128n32', 256, 6), # 128 heads

        # At 500M scale configs
        ('E88_h36n48', 2304, 32),  # 500M scale config that failed
        ('E88_h72n48', 1152, 32),  # 500M scale config
    ]

    print(f"\n{'Level':>20} {'Dim':>6} {'Depth':>6} {'Status':>10} {'Diverge@':>10} {'FinalLoss':>12} {'Reason'}")
    print("-" * 100)

    for level, dim, depth in configs:
        try:
            result = test_ladderlm_training(level, dim, depth, steps=100, verbose=False)

            status = "STABLE" if result['diverged_at'] is None else "DIVERGED"
            diverge_at = str(result['diverged_at']) if result['diverged_at'] is not None else "-"
            final_loss = f"{result['final_loss']:.4f}" if result['final_loss'] < 100 else "INF"
            reason = result['reason'][:25]

            marker = " ***" if result['diverged_at'] is not None else ""
            print(f"{level:>20} {dim:>6} {depth:>6} {status:>10} {diverge_at:>10} {final_loss:>12} {reason}{marker}")
        except Exception as e:
            print(f"{level:>20} {dim:>6} {depth:>6} ERROR: {str(e)[:40]}")

        # Clear GPU memory
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
