#!/usr/bin/env python3
"""
E74 Ablation Runner

Runs systematic ablation experiments to find optimal architecture.

Usage:
    python run_e74_ablations.py --phase 1  # State structure ablations
    python run_e74_ablations.py --phase 2  # Projection ablations
    python run_e74_ablations.py --all      # All 20 experiments
    python run_e74_ablations.py --config 5 # Single config by ID

Output saved to benchmark_results/e74_ablations/
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import ablation framework
from elman.models.e74_ablations import (
    E74Ablation,
    get_ablation_configs,
    create_model_from_config,
)

# Try to import checkpointed version
try:
    from elman.kernels.e74_checkpointed_triton import E74CheckpointedAblation
    CHECKPOINTED_AVAILABLE = True
except ImportError:
    CHECKPOINTED_AVAILABLE = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def create_synthetic_data(batch_size: int, seq_len: int, dim: int, device: str, dtype: torch.dtype):
    """Create synthetic training data."""
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    targets = torch.randint(0, dim, (batch_size, seq_len), device=device)
    return x, targets


def run_training_iteration(model: nn.Module, x: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Run one forward-backward pass, return loss and time."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    # Forward
    out, _ = model(x)

    # Simple cross-entropy loss (project to vocab)
    vocab_size = 1000
    logits = F.linear(out, torch.randn(vocab_size, out.shape[-1], device=out.device, dtype=out.dtype))
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    # Backward
    loss.backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    return loss.item(), elapsed


def benchmark_config(
    config: dict,
    dim: int = 512,
    n_state: int = 64,
    batch_size: int = 4,
    seq_len: int = 256,
    num_iters: int = 10,
    warmup_iters: int = 3,
    use_checkpointing: bool = False,
    checkpoint_interval: int = 32,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Benchmark a single configuration."""
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    # Create model
    if use_checkpointing and CHECKPOINTED_AVAILABLE and config['state'] == 'diagonal':
        model = E74CheckpointedAblation(
            dim=dim,
            n_state=n_state,
            checkpoint_interval=checkpoint_interval,
            use_tanh=(config['nonlin'] == 'tanh'),
            tied_kq=(config['proj'] in ['tied_kq', 'tied_kvq']),
        )
    else:
        model = create_model_from_config(config, dim=dim, n_state=n_state)

    model = model.to(device).to(dtype)
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create data
    x, targets = create_synthetic_data(batch_size, seq_len, dim, device, dtype)

    # Warmup
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        try:
            _, _ = run_training_iteration(model, x, targets)
            optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                return {"error": "OOM", "config": config}
            raise

    # Clear memory stats after warmup
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    # Benchmark
    losses = []
    times = []
    grad_norms = []

    for i in range(num_iters):
        optimizer.zero_grad()
        try:
            loss, elapsed = run_training_iteration(model, x, targets)
            losses.append(loss)
            times.append(elapsed)

            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)

            optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                return {"error": "OOM", "config": config}
            raise

    # Collect results
    memory_mb = get_memory_mb()
    params = count_parameters(model)
    tokens_per_sec = (batch_size * seq_len) / (sum(times) / len(times))

    results = {
        "config_id": config['id'],
        "config_desc": config['desc'],
        "state": config['state'],
        "proj": config['proj'],
        "nonlin": config['nonlin'],
        "gate": config['gate'],
        "parameters": params,
        "memory_mb": memory_mb,
        "mean_loss": sum(losses) / len(losses),
        "final_loss": losses[-1],
        "mean_time_ms": (sum(times) / len(times)) * 1000,
        "tokens_per_sec": tokens_per_sec,
        "mean_grad_norm": sum(grad_norms) / len(grad_norms),
        "max_grad_norm": max(grad_norms),
        "checkpointed": use_checkpointing,
    }

    # Cleanup
    del model, optimizer, x, targets
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def run_ablations(
    config_ids: list = None,
    output_dir: str = "benchmark_results/e74_ablations",
    **kwargs
):
    """Run ablation experiments."""
    configs = get_ablation_configs()

    if config_ids is not None:
        configs = [c for c in configs if c['id'] in config_ids]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"results_{timestamp}.jsonl"

    print(f"Running {len(configs)} ablation experiments")
    print(f"Results will be saved to: {results_file}")
    print("=" * 70)

    all_results = []

    for config in configs:
        print(f"\n[{config['id']:2d}] {config['desc']}")
        print(f"    state={config['state']}, proj={config['proj']}, "
              f"nonlin={config['nonlin']}, gate={config['gate']}")

        result = benchmark_config(config, **kwargs)

        if "error" in result:
            print(f"    ERROR: {result['error']}")
        else:
            print(f"    params={result['parameters']:,}")
            print(f"    memory={result['memory_mb']:.1f}MB")
            print(f"    throughput={result['tokens_per_sec']:.0f} tok/s")
            print(f"    loss={result['final_loss']:.4f}")
            print(f"    grad_norm={result['mean_grad_norm']:.4f}")

        all_results.append(result)

        # Save incrementally
        with open(results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ID':>3} {'Description':<30} {'Params':>10} {'Mem MB':>8} {'Tok/s':>8} {'Loss':>8}")
    print("-" * 70)

    for r in all_results:
        if "error" in r:
            print(f"{r['config']['id']:>3} {r['config']['desc']:<30} {'ERROR':<10}")
        else:
            print(f"{r['config_id']:>3} {r['config_desc']:<30} "
                  f"{r['parameters']:>10,} {r['memory_mb']:>8.1f} "
                  f"{r['tokens_per_sec']:>8.0f} {r['final_loss']:>8.4f}")

    # Save summary
    summary_file = Path(output_dir) / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "num_configs": len(configs),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="E74 Ablation Runner")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="Run specific phase (1=state, 2=proj, 3=nonlin, 4=gate)")
    parser.add_argument("--config", type=int, help="Run single config by ID")
    parser.add_argument("--all", action="store_true", help="Run all 20 configs")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer iters)")
    parser.add_argument("--checkpointed", action="store_true", help="Use checkpointing")

    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_state", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="benchmark_results/e74_ablations")

    args = parser.parse_args()

    # Determine which configs to run
    if args.config is not None:
        config_ids = [args.config]
    elif args.phase == 1:
        config_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # State structure
    elif args.phase == 2:
        config_ids = [6, 7, 8, 12, 13]  # Projections on diagonal
    elif args.phase == 3:
        config_ids = [6, 12, 13, 14]  # Nonlinearity
    elif args.phase == 4:
        config_ids = [6, 15, 16, 17]  # Gates
    elif args.all:
        config_ids = list(range(1, 21))
    else:
        # Default: quick test of key configs
        config_ids = [1, 5, 6, 8, 12, 20]

    num_iters = 3 if args.quick else args.num_iters

    print(f"E74 Ablation Study")
    print(f"Configs: {config_ids}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Checkpointed: {args.checkpointed}")
    print()

    run_ablations(
        config_ids=config_ids,
        output_dir=args.output_dir,
        dim=args.dim,
        n_state=args.n_state,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iters=num_iters,
        use_checkpointing=args.checkpointed,
    )


if __name__ == "__main__":
    main()
