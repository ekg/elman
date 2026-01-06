#!/usr/bin/env python3
"""
E5 Hidden Dimension Benchmark

Tests E5 (Pure Low-Rank Elman) with different hidden state dimensions.
Key insight: In E5, hidden_state = dim (no projections), so varying dim
directly varies the hidden state size.

E5 params per layer: dim * (6 * rank + 1)
  - 6 low-rank matrices: U_h, V_h, U_x, V_x, U_z, V_z (each dim x rank or rank x dim)
  - 1 bias vector: b (dim)

Configurations at ~50M params:
  dim=512,  rank=64  -> 252 layers (baseline)
  dim=768,  rank=48  -> 168 layers (1.5x hidden)
  dim=1024, rank=32  -> 144 layers (2x hidden)
  dim=1536, rank=24  -> 82 layers  (3x hidden)

Usage:
    python benchmark_e5_dim.py --data data/pile.txt --max_steps 1000
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
from schedulefree import AdamWScheduleFree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.models import LadderLM
from elman.data import DocumentStreamDataset
from elman.data.dataset import FastTokenizedDataset

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark E5 hidden dimensions")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--params", type=str, default="50m", help="Target parameter count")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--chunk_size", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--output", type=str, default="benchmark_results/e5_dim", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="byte",
                        choices=["byte", "p50k_base", "cl100k_base"], help="Tokenizer")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--include_baselines", action="store_true",
                        help="Also run E1 and Mamba2 baselines")
    return parser.parse_args()


def format_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n:,}"


def get_lr(step, warmup_steps, max_lr, max_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def compute_e5_depth(dim, rank, target_params, vocab_size):
    """Compute depth for E5 to hit target params."""
    # E5 params: embed + depth * layer_params + (layernorm per layer) + final_norm
    # layer_params = dim * (6 * rank + 1)  [6 low-rank matrices + bias]
    # layernorm per layer = 2 * dim
    # embedding = vocab_size * dim
    # final norm = 2 * dim
    # lm_head = tied with embedding, no extra

    embed_params = vocab_size * dim
    layer_params = dim * (6 * rank + 1) + 2 * dim  # layer + layernorm
    final_norm = 2 * dim

    remaining = target_params - embed_params - final_norm
    depth = remaining // layer_params

    return max(1, depth)


def create_e5_config(dim, rank, target_params, vocab_size):
    """Create config for E5 with specific dim/rank."""
    depth = compute_e5_depth(dim, rank, target_params, vocab_size)
    return {
        'dim': dim,
        'rank': rank,
        'depth': depth,
        'vocab_size': vocab_size,
    }


def create_e5_model(config):
    """Create E5 model from config."""
    model = LadderLM(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        depth=config['depth'],
        level=5,  # E5: Pure Low-Rank Elman
        rank=config['rank'],
    )
    return model


def train_model(model, dataset, args, model_name, output_dir, use_batched=False):
    """Train a model and return final metrics."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model = model.to(device=device, dtype=dtype)
    num_params = model.get_num_params()

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {format_params(num_params)}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
    )

    results = {
        "model": model_name,
        "num_params": num_params,
        "losses": [],
        "step_times_ms": [],
    }

    log_path = output_dir / f"{model_name}_steps.jsonl"
    with open(log_path, 'w') as f:
        f.write(json.dumps({"type": "header", "model": model_name, "params": num_params}) + "\n")

    model.train()
    optimizer.train()
    start_time = time.time()
    tokens_seen = 0

    for step in range(1, args.max_steps + 1):
        step_start = time.time()

        # Get batch
        if use_batched:
            batch, _, actual_lengths = dataset.get_batch(device=device)
        else:
            batch_chunks = []
            batch_lengths = []
            for _ in range(args.batch_size):
                chunk, _, actual_length = dataset[0]
                batch_chunks.append(chunk)
                batch_lengths.append(actual_length)
            batch = torch.stack(batch_chunks).to(device)
            actual_lengths = torch.tensor(batch_lengths, device=device)

        optimizer.zero_grad()
        loss = model(batch, return_loss=True, actual_length=actual_lengths)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, args.warmup_steps, args.lr, args.max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        step_time_ms = (time.time() - step_start) * 1000
        tokens_seen += args.batch_size * args.chunk_size

        if step % args.log_interval == 0 or step == 1:
            loss_val = loss.item()
            perplexity = math.exp(min(loss_val, 20))
            elapsed = time.time() - start_time
            tps = tokens_seen / elapsed

            print(f"[{model_name}] step {step:4d} | loss {loss_val:.4f} | "
                  f"ppl {perplexity:.1f} | {tps:.0f} tok/s | {step_time_ms:.0f}ms/step")

            results["losses"].append(loss_val)
            results["step_times_ms"].append(step_time_ms)

            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    "step": step, "loss": loss_val, "ppl": perplexity,
                    "tok_s": tps, "elapsed": elapsed
                }) + "\n")

    # Final metrics
    last_n = min(5, len(results["losses"]))
    results["final_loss"] = sum(results["losses"][-last_n:]) / last_n if results["losses"] else float('inf')
    results["total_time_s"] = time.time() - start_time
    results["tokens_seen"] = tokens_seen
    results["tok_per_sec"] = tokens_seen / results["total_time_s"]

    print(f"\n{model_name} Final: loss={results['final_loss']:.4f}, "
          f"tok/s={results['tok_per_sec']:.0f}")

    del model
    del optimizer
    torch.cuda.empty_cache()

    return results


def main():
    args = get_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if not os.path.exists(args.data):
        print(f"Error: Data path does not exist: {args.data}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse target params
    target = args.params.lower()
    if target.endswith('m'):
        target_params = int(float(target[:-1]) * 1e6)
    elif target.endswith('b'):
        target_params = int(float(target[:-1]) * 1e9)
    else:
        target_params = int(target)

    # Setup tokenizer
    if args.tokenizer == "byte":
        tokenizer = None
        vocab_size = 256
    else:
        if not TIKTOKEN_AVAILABLE:
            print("Error: tiktoken not installed")
            sys.exit(1)
        tokenizer = tiktoken.get_encoding(args.tokenizer)
        vocab_size = tokenizer.n_vocab

    print(f"Target params: {format_params(target_params)}")
    print(f"Vocab size: {vocab_size:,}")

    # Create dataset
    print(f"Loading data from {args.data}...")
    if tokenizer is None:
        dataset = DocumentStreamDataset(
            data_path=args.data,
            chunk_size=args.chunk_size + 1,
            seed=42,
        )
        use_batched = False
    else:
        dataset = FastTokenizedDataset(
            data_path=args.data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            seed=42,
        )
        use_batched = True

    # E5 configurations: (dim, rank, name)
    # At ~50M params with vocab=256:
    # Higher dim = more hidden state capacity but fewer layers
    e5_configs = [
        (512, 64, "e5_d512_r64"),   # Baseline: ~252 layers
        (768, 48, "e5_d768_r48"),   # 1.5x hidden: ~168 layers
        (1024, 32, "e5_d1024_r32"), # 2x hidden: ~144 layers
        (1024, 48, "e5_d1024_r48"), # 2x hidden, more rank: ~112 layers
        (1536, 24, "e5_d1536_r24"), # 3x hidden: ~82 layers
    ]

    all_results = []

    # Run E5 dimension experiments
    for dim, rank, name in e5_configs:
        print(f"\n{'='*60}")
        print(f"Creating {name}: dim={dim}, rank={rank}")

        config = create_e5_config(dim, rank, target_params, vocab_size)
        print(f"  Config: depth={config['depth']}, hidden_state={dim}")

        # Compute actual params
        embed_params = vocab_size * dim
        layer_params = dim * (6 * rank + 1) + 2 * dim
        total_params = embed_params + config['depth'] * layer_params + 2 * dim
        print(f"  Params: {format_params(total_params)} (target: {format_params(target_params)})")

        model = create_e5_model(config)
        results = train_model(model, dataset, args, name, output_dir, use_batched)
        results["config"] = config
        all_results.append(results)

    # Optionally run baselines
    if args.include_baselines:
        from elman.models import create_ladder_model

        # E1 baseline
        print("\nCreating E1 baseline...")
        e1_model = create_ladder_model(target_params=args.params, level=1, vocab_size=vocab_size)
        e1_results = train_model(e1_model, dataset, args, "e1_baseline", output_dir, use_batched)
        all_results.append(e1_results)

        # Mamba2 baseline (if available)
        try:
            from elman.models.mamba2_baseline import create_mamba2_model
            print("\nCreating Mamba2 baseline...")
            m2_model = create_mamba2_model(target_params=args.params, vocab_size=vocab_size)
            m2_results = train_model(m2_model, dataset, args, "mamba2", output_dir, use_batched)
            all_results.append(m2_results)
        except ImportError:
            print("Mamba2 not available, skipping")

    # Summary
    print("\n" + "="*90)
    print("E5 HIDDEN DIMENSION BENCHMARK SUMMARY")
    print("="*90)
    print(f"{'Model':<18} {'Params':<12} {'Dim':<8} {'Rank':<6} {'Depth':<7} {'Loss':<10} {'tok/s':<12}")
    print("-"*90)

    for r in sorted(all_results, key=lambda x: x["final_loss"]):
        config = r.get("config", {})
        dim = config.get("dim", "-")
        rank = config.get("rank", "-")
        depth = config.get("depth", "-")
        print(f"{r['model']:<18} {format_params(r['num_params']):<12} "
              f"{dim:<8} {rank:<6} {depth:<7} "
              f"{r['final_loss']:<10.4f} {r['tok_per_sec']:<12.0f}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
