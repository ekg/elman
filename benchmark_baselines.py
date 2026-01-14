#!/usr/bin/env python3
"""
Benchmark script for comparing Elman ladder models against baselines.

Runs GRU, LSTM, and Elman models on the same data with matched parameter counts.

Usage:
    # Run all models
    python benchmark_baselines.py --data /path/to/pile --params 50m --max_steps 1000

    # Run specific model
    python benchmark_baselines.py --data /path/to/pile --model gru --params 50m

    # Include Mamba2 (requires mamba-ssm)
    python benchmark_baselines.py --data /path/to/pile --model mamba2 --params 50m
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

from elman.models import create_ladder_model

# Archived models - may not be available
try:
    from elman.models.archive.gru_baseline import create_gru_model, GRULM
except ImportError:
    create_gru_model = None
    GRULM = None

try:
    from elman.models.archive.lstm_baseline import create_lstm_model, LSTMLM
except ImportError:
    create_lstm_model = None
    LSTMLM = None
from elman.data import DocumentStreamDataset
from elman.data.dataset import FastTokenizedDataset, TokenizedStreamDataset

# Optional tiktoken
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Optional Mamba2
try:
    from elman.models.mamba2_baseline import create_mamba2_model, Mamba2LM, MAMBA2_AVAILABLE
except ImportError:
    MAMBA2_AVAILABLE = False


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark Elman vs baselines")

    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "gru", "lstm", "mamba2", "pure", "auto", "ssm", "x_gated", "diagonal",
                                 "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                 "log_0", "log_1", "log_2", "log_3", "log_4", "log_5", "log_6",
                                 "23"],
                        help="Model to benchmark (default: all)")
    parser.add_argument("--params", type=str, default="50m",
                        help="Target parameter count (e.g., 20m, 50m, 100m)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Max training steps")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Training timeout in seconds (overrides max_steps)")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=256,
                        help="Vocabulary size (default: 256 for byte-level)")
    parser.add_argument("--tokenizer", type=str, default="byte",
                        choices=["byte", "p50k_base", "cl100k_base"],
                        help="Tokenizer (byte=256 vocab, p50k_base=50k, cl100k_base=100k)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming tokenization (CPU parallel) instead of pre-tokenized cache")
    parser.add_argument("--no_spectral_norm", action="store_true",
                        help="Disable spectral normalization on W_h (test stability)")
    parser.add_argument("--use_conv", action="store_true",
                        help="Enable conv1d for local context (like Mamba2)")
    parser.add_argument("--d_conv", type=int, default=4,
                        help="Conv1d kernel size (default: 4)")

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


def train_model(model, dataset, args, model_name, output_dir, use_batched_dataset=False):
    """Train a model and return final metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model = model.to(device=device, dtype=dtype)
    num_params = model.get_num_params()

    timeout = args.timeout
    mode = f"timeout={timeout}s" if timeout else f"steps={args.max_steps}"

    print(f"\n{'='*60}")
    print(f"Training: {model_name} ({mode})")
    print(f"Parameters: {format_params(num_params)}")
    print(f"Vocab size: {args.vocab_size:,}")
    print(f"{'='*60}")

    optimizer = AdamWScheduleFree(
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
        }
        f.write(json.dumps(header) + "\n")

    # Training loop
    model.train()
    optimizer.train()
    start_time = time.time()
    tokens_seen = 0
    step = 0
    max_steps = args.max_steps if not timeout else 1000000  # effectively unlimited if timeout

    while step < max_steps:
        step += 1
        step_start = time.time()

        # Check timeout
        elapsed = time.time() - start_time
        if timeout and elapsed >= timeout:
            print(f"[{model_name}] Timeout reached at {elapsed:.1f}s")
            break

        # Get batch
        if use_batched_dataset:
            # FastTokenizedDataset uses get_batch()
            batch, _, actual_lengths = dataset.get_batch(device=device)
        else:
            # DocumentStreamDataset uses __getitem__ per sample
            batch_chunks = []
            batch_lengths = []
            for _ in range(args.batch_size):
                chunk, _, actual_length = dataset[0]
                batch_chunks.append(chunk)
                batch_lengths.append(actual_length)
            batch = torch.stack(batch_chunks).to(device)
            actual_lengths = torch.tensor(batch_lengths, device=device)

        # Forward - compute loss with padding mask
        optimizer.zero_grad()
        loss = model(batch, return_loss=True, actual_length=actual_lengths)

        # Backward
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step with LR schedule
        lr = get_lr(step, args.warmup_steps, args.lr, args.max_steps)
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

            print(f"[{model_name}] step {step:4d} | loss {loss_val:.4f} | "
                  f"ppl {perplexity:.1f} | grad {grad_val:.2f} | "
                  f"{tps:.0f} tok/s | {elapsed:.1f}s | {step_time_ms:.0f}ms/step")

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
                f.write(json.dumps(record) + "\n")

    # Final metrics - average over last 100 steps for stability
    last_n = min(100, len(results["losses"]))
    results["final_loss"] = sum(results["losses"][-last_n:]) / last_n if results["losses"] else float('inf')
    results["final_grad_norm"] = sum(results["grad_norms"][-last_n:]) / last_n if results["grad_norms"] else 0
    results["avg_step_time_ms"] = sum(results["step_times_ms"]) / len(results["step_times_ms"]) if results["step_times_ms"] else 0
    results["total_time_s"] = time.time() - start_time
    results["total_steps"] = step
    results["tokens_seen"] = tokens_seen

    print(f"\n{model_name} Final: loss={results['final_loss']:.4f}, "
          f"grad={results['final_grad_norm']:.2f}, "
          f"steps={step}, tokens={tokens_seen:,}, "
          f"time={results['total_time_s']:.1f}s")

    # Clear GPU memory
    del model
    del optimizer
    torch.cuda.empty_cache()

    return results


def main():
    args = get_args()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Check data exists
    if not os.path.exists(args.data):
        print(f"Error: Data path does not exist: {args.data}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tokenizer and vocab size
    if args.tokenizer == "byte":
        tokenizer = None
        vocab_size = 256
    else:
        if not TIKTOKEN_AVAILABLE:
            print("Error: tiktoken not installed. Install with: pip install tiktoken")
            sys.exit(1)
        tokenizer = tiktoken.get_encoding(args.tokenizer)
        vocab_size = tokenizer.n_vocab
        print(f"Using {args.tokenizer} tokenizer with vocab size {vocab_size:,}")

    # Override vocab_size if tokenizer is set
    if args.tokenizer != "byte":
        args.vocab_size = vocab_size

    # Create dataset
    print(f"Loading data from {args.data}...")
    if tokenizer is None:
        dataset = DocumentStreamDataset(
            data_path=args.data,
            chunk_size=args.chunk_size + 1,
            seed=42,
        )
    elif args.streaming:
        print("Using streaming tokenization (CPU parallel)")
        dataset = TokenizedStreamDataset(
            data_path=args.data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            seed=42,
        )
    else:
        dataset = FastTokenizedDataset(
            data_path=args.data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            seed=42,
        )

    # Models to benchmark
    if args.model == "all":
        models_to_run = ["gru", "lstm", "log_0", "log_3"]
        if MAMBA2_AVAILABLE:
            models_to_run.insert(0, "mamba2")
        else:
            print("Note: mamba-ssm not installed, skipping Mamba2")
    else:
        models_to_run = [args.model]

    all_results = []

    for model_name in models_to_run:
        print(f"\nCreating {model_name} model with ~{args.params} parameters...")

        if model_name == "gru":
            model = create_gru_model(target_params=args.params, vocab_size=args.vocab_size)
        elif model_name == "lstm":
            model = create_lstm_model(target_params=args.params, vocab_size=args.vocab_size)
        elif model_name == "mamba2":
            if not MAMBA2_AVAILABLE:
                print("Skipping Mamba2 (not installed)")
                continue
            model = create_mamba2_model(target_params=args.params, vocab_size=args.vocab_size)
        elif model_name == "pure":
            # Pure Elman (no output gating)
            model = create_ladder_model(
                target_params=args.params,
                level='pure',
                vocab_size=args.vocab_size,
            )
        elif model_name == "x_gated":
            # X-Gated Elman (x-only output gating)
            model = create_ladder_model(
                target_params=args.params,
                level='x_gated',
                vocab_size=args.vocab_size,
            )
        elif model_name == "diagonal":
            # Diagonal Elman (linear diagonal recurrence + x-only gating)
            model = create_ladder_model(
                target_params=args.params,
                level='diagonal',
                vocab_size=args.vocab_size,
            )
        elif model_name == "auto":
            # Auto Elman (autonomous hidden, input-only gating)
            model = create_ladder_model(
                target_params=args.params,
                level='auto',
                vocab_size=args.vocab_size,
            )
        elif model_name == "ssm":
            # SSM Elman (Mamba2-style diagonal SSM)
            model = create_ladder_model(
                target_params=args.params,
                level='ssm',
                vocab_size=args.vocab_size,
            )
        elif model_name.startswith("log_") or model_name.isdigit():
            # Elman ladder level (log_X or numeric 0-9)
            level = model_name if model_name.startswith("log_") else int(model_name)
            r_h_mode = 'free' if args.no_spectral_norm else 'spectral_norm'
            model = create_ladder_model(
                target_params=args.params,
                level=level,
                vocab_size=args.vocab_size,
                r_h_mode=r_h_mode,
            )
        else:
            print(f"Unknown model: {model_name}")
            continue

        use_batched = (tokenizer is not None)
        results = train_model(model, dataset, args, model_name, output_dir, use_batched_dataset=use_batched)
        all_results.append(results)

    # Summary
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    print(f"{'Model':<15} {'Params':<12} {'Loss':<10} {'Steps':<8} {'Tokens':<12} {'tok/s':<10} {'Time':<8}")
    print("-"*90)

    for r in sorted(all_results, key=lambda x: x["final_loss"]):
        tps = r.get('tokens_seen', 0) / r['total_time_s'] if r['total_time_s'] > 0 else 0
        print(f"{r['model']:<15} {format_params(r['num_params']):<12} "
              f"{r['final_loss']:<10.4f} {r.get('total_steps', 0):<8} "
              f"{r.get('tokens_seen', 0):<12,} {tps:<10.0f} "
              f"{r['total_time_s']:<8.1f}s")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "config": {
                "data": args.data,
                "params": args.params,
                "batch_size": args.batch_size,
                "chunk_size": args.chunk_size,
                "max_steps": args.max_steps,
                "lr": args.lr,
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
