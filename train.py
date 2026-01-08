#!/usr/bin/env python3
"""
Training script for Elman Ladder models.

Simple single-GPU training with:
- Document-aware data loading
- Optional TBPTT with hidden state tracking (--tbptt flag)
- Checkpointing and logging
- Support for all 4 ladder levels (0-3)

Usage:
    python train.py --data /path/to/data.txt --level 3 --params 100m
    python train.py --data /path/to/data.txt --level 3 --params 100m --tbptt  # Enable TBPTT
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
import json
import datetime
import glob
import re

# Add elman package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.models import LadderLM, create_ladder_model
from elman.data import DocumentStreamDataset, BatchedStreamDataset, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train Elman Ladder models')

    # Data
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data file')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data file')

    # Model
    parser.add_argument('--level', type=str, default='3',
                        help='Ladder level: 0-6 (linear) or log_0 to log_5 (log-space)')
    parser.add_argument('--params', type=str, default='100m',
                        help='Target parameter count (e.g., 100m, 500m, 1b)')
    parser.add_argument('--dim', type=int, default=None,
                        help='Model dimension (overrides --params)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Number of layers (overrides --params)')
    parser.add_argument('--expansion', type=float, default=1.0,
                        help='Hidden state expansion factor')
    parser.add_argument('--state_expansion', type=int, default=2,
                        help='State expansion for E16 (d_state = d_inner * state_expansion)')
    parser.add_argument('--n_groups', type=int, default=32,
                        help='Number of groups for compete softmax')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Sequence chunk size (TBPTT)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping (0 to disable)')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Total training steps')
    parser.add_argument('--train_minutes', type=float, default=None,
                        help='Train for N minutes (overrides --steps)')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps for learning rate')

    # Checkpointing
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--val_every', type=int, default=500,
                        help='Validate every N steps')
    parser.add_argument('--keep_checkpoints', type=int, default=5,
                        help='Number of checkpoints to keep')

    # System
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 mixed precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--tbptt', action='store_true',
                        help='Enable TBPTT (carry hidden state across chunks)')

    return parser.parse_args()


def parse_level(level_str):
    """Parse level string to int or keep as string for log-space levels."""
    if level_str.startswith('log_'):
        return level_str  # Keep as string for log-space levels
    try:
        return int(level_str)
    except ValueError:
        return level_str  # Keep as string for any other format


def setup_output_dir(args):
    """Create output directory with run info."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"level{args.level}_{args.params}_{timestamp}"
    output_dir = Path(args.output) / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    return output_dir


def get_lr(step, warmup_steps, max_lr, min_lr=1e-6):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + torch.cos(torch.tensor(step / warmup_steps * 3.14159)))


def save_checkpoint(model, optimizer, step, loss, output_dir, keep_n=5):
    """Save checkpoint and clean up old ones."""
    ckpt_path = output_dir / f'checkpoint_step_{step:06d}_loss_{loss:.4f}.pt'

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, ckpt_path)

    # Update latest symlink
    latest_path = output_dir / 'latest.pt'
    if latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(ckpt_path.name)

    # Clean up old checkpoints
    ckpts = sorted(glob.glob(str(output_dir / 'checkpoint_step_*.pt')))
    for old_ckpt in ckpts[:-keep_n]:
        os.remove(old_ckpt)

    return ckpt_path


def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('step', 0), ckpt.get('loss', float('inf'))


@torch.no_grad()
def validate(model, val_loader, device, max_batches=100):
    """Run validation."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, (chunk, is_doc_end, actual_lengths) in enumerate(val_loader):
        if i >= max_batches:
            break

        chunk = chunk.to(device)
        loss = model(chunk, return_loss=True)

        # Weight by actual tokens
        batch_tokens = actual_lengths.sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


def train(args):
    """Main training loop."""
    # Parse level (convert '3' to 3, keep 'log_5' as string)
    args.level = parse_level(args.level)

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = setup_output_dir(args)
    print(f"Output directory: {output_dir}")

    # Create model
    if args.dim is not None and args.depth is not None:
        model = LadderLM(
            vocab_size=256,
            dim=args.dim,
            depth=args.depth,
            level=args.level,
            expansion=args.expansion,
            n_groups=args.n_groups,
            state_expansion=args.state_expansion,
        )
    else:
        model = create_ladder_model(
            target_params=args.params,
            level=args.level,
            vocab_size=256,
            expansion=args.expansion,
            n_groups=args.n_groups,
            state_expansion=args.state_expansion,
        )

    model = model.to(device)
    if args.bf16:
        model = model.bfloat16()

    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)

    print(f"Model: Level {args.level}, {model.get_num_params():,} parameters")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Resume if requested
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_step, _ = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed at step {start_step}")

    # Create dataset - use BatchedStreamDataset for TBPTT (persistent per-batch streams)
    if args.tbptt:
        print("TBPTT enabled: using BatchedStreamDataset (persistent streams)")
        train_dataset = BatchedStreamDataset(
            data_path=args.data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,  # +1 for target
            seed=args.seed,
        )
    else:
        train_dataset = DocumentStreamDataset(
            data_path=args.data,
            chunk_size=args.chunk_size + 1,  # +1 for target
            seed=args.seed,
        )

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            args.val_data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            device=device,
        )

    # Training state
    hidden_state = None  # Only used if --tbptt
    accumulated_steps = 0
    running_loss = 0
    tokens_processed = 0
    start_time = time.time()

    print(f"\nStarting training from step {start_step}...")
    print(f"Batch size: {args.batch_size}, Chunk size: {args.chunk_size}")
    print(f"Gradient accumulation: {args.grad_accum}, Effective batch: {args.batch_size * args.grad_accum}")
    print()

    model.train()
    step = start_step

    # Time-based training setup
    train_start_time = time.time()
    train_end_time = None
    if args.train_minutes is not None:
        train_end_time = train_start_time + args.train_minutes * 60
        print(f"Time-based training: {args.train_minutes} minutes")

    def should_continue():
        if train_end_time is not None:
            return time.time() < train_end_time
        return step < args.steps

    while should_continue():
        # Get batch - different methods for TBPTT vs non-TBPTT
        if args.tbptt:
            # BatchedStreamDataset: each batch element has its own persistent stream
            chunks, is_doc_end = train_dataset.get_batch(device=device)
            actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
        else:
            # DocumentStreamDataset: single stream, no hidden state persistence
            batch_chunks = []
            batch_doc_ends = []
            batch_lengths = []

            for _ in range(args.batch_size):
                chunk, is_doc_end_single, actual_length = train_dataset[0]
                batch_chunks.append(chunk)
                batch_doc_ends.append(is_doc_end_single)
                batch_lengths.append(actual_length)

            chunks = torch.stack(batch_chunks).to(device)
            is_doc_end = torch.tensor(batch_doc_ends, dtype=torch.bool, device=device)
            actual_lengths = torch.tensor(batch_lengths, dtype=torch.long, device=device)

        # Reset hidden state at document boundaries (only if TBPTT enabled)
        if args.tbptt and hidden_state is not None:
            reset_mask = is_doc_end.view(-1, 1)
            hidden_state = [h * (~reset_mask) if h is not None else None for h in hidden_state]

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
            if args.tbptt:
                result = model(
                    chunks,
                    return_loss=True,
                    return_prev_hiddens=True,
                    prev_hiddens=hidden_state,
                )
            else:
                result = model(
                    chunks,
                    return_loss=True,
                )

            if isinstance(result, tuple):
                loss, (next_hidden, _) = result
            else:
                loss = result
                next_hidden = None

        # Scale for gradient accumulation
        scaled_loss = loss / args.grad_accum
        scaled_loss.backward()

        # Update hidden state (only if TBPTT enabled)
        if args.tbptt and next_hidden is not None:
            hidden_state = [h.detach() if h is not None else None for h in next_hidden]

        accumulated_steps += 1
        running_loss += loss.item()
        tokens_processed += actual_lengths.sum().item()

        # Optimizer step
        if accumulated_steps >= args.grad_accum:
            # Gradient clipping
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

            # Learning rate schedule
            lr = get_lr(step, args.warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            optimizer.zero_grad()

            step += 1
            accumulated_steps = 0

            # Logging
            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed

                print(f"step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                      f"grad {grad_norm:.2f} | tok/s {tokens_per_sec:.0f}")

                running_loss = 0
                tokens_processed = 0
                start_time = time.time()

            # Validation
            if val_loader and step % args.val_every == 0:
                val_loss = validate(model, val_loader, device)
                print(f"  >>> validation loss: {val_loss:.4f}")

            # Checkpointing
            if step % args.save_every == 0:
                ckpt_path = save_checkpoint(
                    model, optimizer, step, avg_loss, output_dir, args.keep_checkpoints
                )
                print(f"  >>> saved checkpoint: {ckpt_path.name}")

    # Final checkpoint
    save_checkpoint(model, optimizer, step, avg_loss, output_dir, args.keep_checkpoints)
    print(f"\nTraining complete! Final step: {step}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
