#!/usr/bin/env python3
"""
Spectral Norm Ablation Study: Compare E1 and E33 with/without spectral normalization.

This script tests whether spectral normalization is necessary for nonlinear Elman models
(E1 and E33) where tanh already bounds activations to [-1, 1].

Hypothesis: tanh acts as implicit regularization, making spectral norm optional.
"""

import argparse
import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.models import LadderLM
from elman.data import DocumentStreamDataset

try:
    from schedulefree import AdamWScheduleFree
    SCHEDULEFREE_AVAILABLE = True
except ImportError:
    SCHEDULEFREE_AVAILABLE = False
    from torch.optim import AdamW


def format_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n:,}"


def get_lr(step, warmup_steps, max_lr, max_steps):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def compute_spectral_radius(W):
    """Compute spectral radius using power iteration (approximate)."""
    with torch.no_grad():
        W_fp32 = W.float()
        u = torch.randn(W.size(0), device=W.device)
        u = u / u.norm()
        for _ in range(20):  # More iterations for accuracy
            v = W_fp32.T @ u
            v = v / (v.norm() + 1e-8)
            u = W_fp32 @ v
            u = u / (u.norm() + 1e-8)
        sigma = (u @ W_fp32 @ v).abs()
        return sigma.item()


def train_model(model, dataset, args, model_name, output_dir, track_spectral_radius=False):
    """Train a model and return final metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model = model.to(device=device, dtype=dtype)
    num_params = model.get_num_params()

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"Parameters: {format_params(num_params)}")
    print(f"Training time: {args.train_minutes} minutes")
    print(f"Track spectral radius: {track_spectral_radius}")
    print(f"{'='*70}")

    if SCHEDULEFREE_AVAILABLE:
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.1,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.1,
        )

    # Training metrics
    results = {
        "model": model_name,
        "num_params": num_params,
        "losses": [],
        "grad_norms": [],
        "step_times_ms": [],
        "spectral_radii": [] if track_spectral_radius else None,
    }

    log_path = output_dir / f"{model_name}_steps.jsonl"
    with open(log_path, 'w') as f:
        header = {
            "type": "header",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "num_params": num_params,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "lr": args.lr,
            "train_minutes": args.train_minutes,
        }
        f.write(json.dumps(header) + "\n")

    # Training loop
    model.train()
    if SCHEDULEFREE_AVAILABLE:
        optimizer.train()

    start_time = time.time()
    end_time = start_time + args.train_minutes * 60
    tokens_seen = 0
    step = 0
    max_steps = 1000000  # Effectively unlimited (time-based)

    while step < max_steps:
        step += 1
        step_start = time.time()

        # Check timeout
        elapsed = time.time() - start_time
        if time.time() >= end_time:
            print(f"[{model_name}] Time limit reached at {elapsed:.1f}s")
            break

        # Get batch
        batch_chunks = []
        batch_lengths = []
        for _ in range(args.batch_size):
            chunk, _, actual_length = dataset[0]
            batch_chunks.append(chunk)
            batch_lengths.append(actual_length)
        batch = torch.stack(batch_chunks).to(device)
        actual_lengths = torch.tensor(batch_lengths, device=device)

        # Forward
        optimizer.zero_grad()
        loss = model(batch, return_loss=True, actual_length=actual_lengths)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[{model_name}] NaN/Inf loss detected at step {step}! Training unstable.")
            results["unstable"] = True
            results["unstable_step"] = step
            break

        # Backward
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Check for NaN gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"[{model_name}] NaN/Inf gradient at step {step}! Training unstable.")
            results["unstable"] = True
            results["unstable_step"] = step
            break

        # Optimizer step with LR schedule
        lr = get_lr(step, args.warmup_steps, args.lr, max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        step_time_ms = (time.time() - step_start) * 1000
        tokens_seen += args.batch_size * args.chunk_size

        # Log
        if step % args.log_interval == 0 or step == 1:
            loss_val = loss.item()
            grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            perplexity = math.exp(min(loss_val, 20))
            elapsed = time.time() - start_time
            tps = tokens_seen / elapsed

            # Track spectral radius for models without spectral norm
            spectral_radius = None
            if track_spectral_radius and hasattr(model, 'layers'):
                # Get W_h from first layer's cell
                layer = model.layers[0]
                if hasattr(layer, 'cell') and hasattr(layer.cell, 'W_h'):
                    spectral_radius = compute_spectral_radius(layer.cell.W_h)
                    results["spectral_radii"].append(spectral_radius)

            log_msg = (f"[{model_name}] step {step:4d} | loss {loss_val:.4f} | "
                      f"ppl {perplexity:.1f} | grad {grad_val:.2f} | "
                      f"{tps:.0f} tok/s | {elapsed:.1f}s")
            if spectral_radius is not None:
                log_msg += f" | rho={spectral_radius:.4f}"
            print(log_msg)

            results["losses"].append(loss_val)
            results["grad_norms"].append(grad_val)
            results["step_times_ms"].append(step_time_ms)

            with open(log_path, 'a') as f:
                record = {
                    "type": "step",
                    "step": step,
                    "loss": loss_val,
                    "perplexity": perplexity,
                    "grad_norm": grad_val,
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tps,
                    "elapsed_s": elapsed,
                    "step_time_ms": step_time_ms,
                }
                if spectral_radius is not None:
                    record["spectral_radius"] = spectral_radius
                f.write(json.dumps(record) + "\n")

    # Final metrics - average over last 100 log intervals
    last_n = min(100, len(results["losses"]))
    results["final_loss"] = sum(results["losses"][-last_n:]) / last_n if results["losses"] else float('inf')
    results["final_grad_norm"] = sum(results["grad_norms"][-last_n:]) / last_n if results["grad_norms"] else 0
    results["avg_step_time_ms"] = sum(results["step_times_ms"]) / len(results["step_times_ms"]) if results["step_times_ms"] else 0
    results["total_time_s"] = time.time() - start_time
    results["total_steps"] = step
    results["tokens_seen"] = tokens_seen
    results["tokens_per_sec"] = tokens_seen / results["total_time_s"] if results["total_time_s"] > 0 else 0

    if track_spectral_radius and results["spectral_radii"]:
        results["final_spectral_radius"] = results["spectral_radii"][-1]
        results["max_spectral_radius"] = max(results["spectral_radii"])
        results["min_spectral_radius"] = min(results["spectral_radii"])

    print(f"\n{model_name} Final: loss={results['final_loss']:.4f}, "
          f"grad={results['final_grad_norm']:.2f}, "
          f"steps={step}, tokens={tokens_seen:,}, "
          f"tok/s={results['tokens_per_sec']:.0f}, "
          f"time={results['total_time_s']:.1f}s")

    if track_spectral_radius and results.get("spectral_radii"):
        print(f"  Spectral radius: final={results['final_spectral_radius']:.4f}, "
              f"max={results['max_spectral_radius']:.4f}, min={results['min_spectral_radius']:.4f}")

    # Clear GPU memory
    del model
    del optimizer
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Spectral Norm Ablation Study")
    parser.add_argument("--data", type=str, default="data/pile.txt",
                        help="Path to training data")
    parser.add_argument("--dim", type=int, default=1280,
                        help="Model dimension")
    parser.add_argument("--depth", type=int, default=6,
                        help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--train_minutes", type=float, default=10,
                        help="Training time in minutes per model")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--output", type=str, default="benchmark_results/spectral_norm_ablation",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--models", type=str, default="e1,e33",
                        help="Models to test (comma-separated: e1,e33)")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Check data exists
    if not os.path.exists(args.data):
        print(f"Error: Data path does not exist: {args.data}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset (shared across all models)
    dataset = DocumentStreamDataset(
        data_path=args.data,
        chunk_size=args.chunk_size + 1,  # +1 for target
        seed=args.seed,
    )

    # Parse models to test
    models_to_test = [m.strip().lower() for m in args.models.split(",")]

    all_results = []

    # Model configs: (name, level, r_h_mode, track_spectral_radius)
    configs = []

    if "e1" in models_to_test:
        configs.extend([
            ("E1_spectral", 1, "spectral_norm", False),
            ("E1_free", 1, "free", True),
        ])

    if "e33" in models_to_test:
        configs.extend([
            ("E33_spectral", 33, "spectral_norm", False),
            ("E33_free", 33, "free", True),
        ])

    for name, level, r_h_mode, track_spectral in configs:
        print(f"\n{'#'*70}")
        print(f"# Creating model: {name} (level={level}, r_h_mode={r_h_mode})")
        print(f"{'#'*70}")

        # Reset seed for each model
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        model = LadderLM(
            vocab_size=256,
            dim=args.dim,
            depth=args.depth,
            level=level,
            expansion=1.0,
            r_h_mode=r_h_mode,
        )

        print(f"Model parameters: {model.get_num_params():,}")

        # Reset dataset seed
        dataset = DocumentStreamDataset(
            data_path=args.data,
            chunk_size=args.chunk_size + 1,
            seed=args.seed,
        )

        results = train_model(model, dataset, args, name, output_dir, track_spectral_radius=track_spectral)
        all_results.append(results)

    # Summary
    print("\n" + "="*90)
    print("SPECTRAL NORM ABLATION SUMMARY")
    print("="*90)
    print(f"{'Model':<20} {'Params':<12} {'Loss':<10} {'tok/s':<12} {'Time':<8} {'Stable':<8} {'Rho_max':<10}")
    print("-"*90)

    for r in all_results:
        stable = "No" if r.get("unstable", False) else "Yes"
        rho_max = f"{r.get('max_spectral_radius', 'N/A'):.4f}" if r.get('max_spectral_radius') else "N/A"
        tps = r.get('tokens_per_sec', 0)
        print(f"{r['model']:<20} {format_params(r['num_params']):<12} "
              f"{r['final_loss']:<10.4f} {tps:<12.0f} "
              f"{r['total_time_s']:<8.1f}s {stable:<8} {rho_max:<10}")

    # Compute speedup from removing spectral norm
    print("\n" + "-"*90)
    print("THROUGHPUT COMPARISON")
    print("-"*90)

    results_by_base = {}
    for r in all_results:
        base = r['model'].split('_')[0]
        mode = r['model'].split('_')[1] if '_' in r['model'] else 'unknown'
        if base not in results_by_base:
            results_by_base[base] = {}
        results_by_base[base][mode] = r

    for base, modes in results_by_base.items():
        if 'spectral' in modes and 'free' in modes:
            spectral_tps = modes['spectral'].get('tokens_per_sec', 0)
            free_tps = modes['free'].get('tokens_per_sec', 0)
            spectral_loss = modes['spectral']['final_loss']
            free_loss = modes['free']['final_loss']

            speedup = (free_tps / spectral_tps - 1) * 100 if spectral_tps > 0 else 0
            loss_diff = free_loss - spectral_loss

            print(f"{base}: spectral={spectral_tps:.0f} tok/s, free={free_tps:.0f} tok/s, "
                  f"speedup={speedup:+.1f}%")
            print(f"       spectral_loss={spectral_loss:.4f}, free_loss={free_loss:.4f}, "
                  f"diff={loss_diff:+.4f}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "config": {
                "data": args.data,
                "dim": args.dim,
                "depth": args.depth,
                "batch_size": args.batch_size,
                "chunk_size": args.chunk_size,
                "train_minutes": args.train_minutes,
                "lr": args.lr,
                "seed": args.seed,
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
