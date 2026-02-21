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
import schedulefree
from pathlib import Path
import json
import datetime
import glob
import re

# Add elman package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add CUDA extension directory for hasty_pytorch_lib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elman', 'cuda'))

from elman.models import LadderLM, create_ladder_model
from elman.data import DocumentStreamDataset, BatchedStreamDataset, create_dataloader
from elman.models.gru_baseline import GRULM
from elman.models.lstm_baseline import LSTMLM
from elman.models.min_rnn_baseline import MinGRULM, MinLSTMLM
from elman.models.cuda_gru import CudaGRULM
from elman.models.cuda_lstm import CudaLSTMLM
from elman.models.e88_fused import E88FusedLM


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
    parser.add_argument('--n_state', type=int, default=64,
                        help='Matrix state size for E70-E73 (S is n_state x n_state)')
    parser.add_argument('--n_slots', type=int, default=64,
                        help='Number of tape memory slots for E23 DualMemoryElman (default=64)')
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Number of heads for E88 FLA Hybrid')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Number of active heads per token for MoM E88 (sparse routing)')
    parser.add_argument('--k_fast', type=int, default=None,
                        help='Fast state dimension for E90 Dual-Rate (default=16)')
    parser.add_argument('--k_slow', type=int, default=None,
                        help='Slow state dimension for E90 Dual-Rate (default=48)')
    parser.add_argument('--use_gate', type=int, default=1,
                        help='Use output gating for E88 (0=no gate, 1=gate, default=1)')
    parser.add_argument('--gate_activation', type=str, default='sigmoid',
                        help='Gate activation for E88 (sigmoid=E88 original, silu=FLA-GDN style)')
    parser.add_argument('--linear_state', type=int, default=0,
                        help='Use linear state update for E88 (0=tanh, 1=linear)')
    parser.add_argument('--use_write_gate', type=int, default=0,
                        help='Use write gate (beta) for E88 (0=no, 1=yes). Gates delta before writing to memory.')
    parser.add_argument('--r_h_mode', type=str, default='auto',
                        help='W_h constraint mode (spectral_norm, learned, none, auto)')
    # auto: spectral_norm for models with full W_h (1,33,42,51,52,53,56), none for diagonal/scalar
    parser.add_argument('--use_conv', type=int, default=0,
                        help='Use Conv1d before recurrence (0=no, 1=yes)')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Conv kernel size (if use_conv=1)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (0.0 to 0.3)')

    # Mamba2-specific
    parser.add_argument('--mamba_expand', type=int, default=2,
                        help='Mamba2 expansion factor (expand)')
    parser.add_argument('--mamba_d_state', type=int, default=64,
                        help='Mamba2 state dimension (d_state)')

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
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps for learning rate (only for adamw)')
    parser.add_argument('--optimizer', type=str, default='schedulefree',
                        choices=['adamw', 'schedulefree'],
                        help='Optimizer: adamw (with LR schedule) or schedulefree (no schedule needed)')

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
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                        help='torch.compile mode (default, reduce-overhead, max-autotune)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--tbptt', action='store_true',
                        help='Enable TBPTT (carry hidden state across chunks)')
    parser.add_argument('--orth_reg', type=float, default=0.0,
                        help='Orthogonality regularization weight for E79 (0=disabled)')
    parser.add_argument('--orth_sep', type=float, default=0.01,
                        help='Weight for k/m separation in orthogonality loss')
    parser.add_argument('--orth_orth', type=float, default=0.001,
                        help='Weight for key orthogonality in orthogonality loss')

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

    # Resolve 'auto' r_h_mode based on model architecture
    r_h_mode = args.r_h_mode
    if r_h_mode == 'auto' and args.level != 'mamba2':
        # Models with full W_h matrix need spectral norm for stability
        # Models with diagonal/scalar W_h are already bounded
        full_wh_levels = {1, 33, 42, 51, 52, 53, 56, 57, 58, 60}  # Full W_h matrix (E59 is highway, no W_h)
        diagonal_levels = {34, 44, 54}  # Diagonal W_h (already bounded)
        scalar_levels = {43, 55}  # Scalar decay (already bounded)
        no_wh_levels = {45, 46, 48}  # No W_h at all
        # E70-73: Matrix state models use gated updates (alpha*S + (1-alpha)*outer), naturally bounded
        matrix_state_levels = {70, 71, 72, 73}  # No spectral norm needed - gated update is bounded

        level_int = int(args.level) if str(args.level).isdigit() else 0
        if level_int in full_wh_levels:
            r_h_mode = 'spectral_norm'
            print(f"Auto r_h_mode: spectral_norm (level {level_int} has full W_h)")
        elif level_int in matrix_state_levels:
            r_h_mode = 'none'
            print(f"Auto r_h_mode: none (level {level_int} is matrix state - gated update is bounded)")
        else:
            r_h_mode = 'none'
            print(f"Auto r_h_mode: none (level {level_int} has bounded/no W_h)")

    # Create model
    if args.level == 'mamba2':
        # Special handling for Mamba2 - use Mamba2LM directly
        from elman.models.mamba2_baseline import Mamba2LM
        if args.dim is not None and args.depth is not None:
            model = Mamba2LM(
                vocab_size=256,
                dim=args.dim,
                depth=args.depth,
                d_state=args.mamba_d_state,
                expand=args.mamba_expand,
                headdim=64,
            )
        else:
            from elman.models.mamba2_baseline import create_mamba2_model
            model = create_mamba2_model(target_params=args.params, vocab_size=256, expand=args.mamba_expand)
    elif args.level == 'gru':
        # GRU baseline
        if args.dim is not None and args.depth is not None:
            model = GRULM(
                vocab_size=256,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from elman.models.gru_baseline import create_gru_model
            model = create_gru_model(target_params=args.params, vocab_size=256)
    elif args.level == 'lstm':
        # LSTM baseline
        if args.dim is not None and args.depth is not None:
            model = LSTMLM(
                vocab_size=256,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from elman.models.lstm_baseline import create_lstm_model
            model = create_lstm_model(target_params=args.params, vocab_size=256)
    elif args.level == 'mingru':
        # minGRU baseline (parallel)
        if args.dim is not None and args.depth is not None:
            model = MinGRULM(
                vocab_size=256,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from elman.models.min_rnn_baseline import create_mingru_model
            model = create_mingru_model(target_params=args.params, vocab_size=256)
    elif args.level == 'minlstm':
        # minLSTM baseline (parallel)
        if args.dim is not None and args.depth is not None:
            model = MinLSTMLM(
                vocab_size=256,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from elman.models.min_rnn_baseline import create_minlstm_model
            model = create_minlstm_model(target_params=args.params, vocab_size=256)
    elif args.level == 'cudagru':
        # CUDA GRU (avoids cuDNN bfloat16 regression)
        model = CudaGRULM(
            vocab_size=256,
            dim=args.dim,
            depth=args.depth,
            expansion_factor=args.expansion,
        )
    elif args.level == 'cudalstm':
        # CUDA LSTM (avoids cuDNN bfloat16 regression)
        model = CudaLSTMLM(
            vocab_size=256,
            dim=args.dim,
            depth=args.depth,
            expansion_factor=args.expansion,
        )
    elif isinstance(args.level, str) and args.level.lower() == 'e88_fused':
        # E88 Fused: optimized kernel with [B, T, H, dim] layout (no transpose overhead)
        model = E88FusedLM(
            vocab_size=256,
            dim=args.dim,
            depth=args.depth,
            n_heads=args.n_heads,
            n_state=args.n_state,
            expansion=args.expansion,
            use_gate=bool(args.use_gate),
        )
    elif args.dim is not None and args.depth is not None:
        model = LadderLM(
            vocab_size=256,
            dim=args.dim,
            depth=args.depth,
            level=args.level,
            expansion=args.expansion,
            n_groups=args.n_groups,
            n_state=args.n_state,
            n_slots=args.n_slots,
            n_heads=args.n_heads,
            top_k=args.top_k,
            k_fast=args.k_fast,
            k_slow=args.k_slow,
            use_gate=bool(args.use_gate),
            gate_activation=args.gate_activation,
            linear_state=bool(args.linear_state),
            use_write_gate=bool(args.use_write_gate),
            state_expansion=args.state_expansion,
            r_h_mode=r_h_mode,
            use_conv=bool(args.use_conv),
            d_conv=args.d_conv,
            dropout=args.dropout,
        )
    else:
        model = create_ladder_model(
            target_params=args.params,
            level=args.level,
            vocab_size=256,
            expansion=args.expansion,
            n_groups=args.n_groups,
            state_expansion=args.state_expansion,
            r_h_mode=r_h_mode,
        )

    model = model.to(device)
    if args.bf16:
        model = model.bfloat16()

    if args.compile and hasattr(torch, 'compile'):
        print(f"Compiling model (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)

    print(f"Model: Level {args.level}, {model.get_num_params():,} parameters")

    # Create optimizer
    if args.optimizer == 'schedulefree':
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        print(f"Using schedule-free AdamW (lr={args.lr})")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        print(f"Using AdamW with warmup={args.warmup_steps} steps")

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
    avg_loss = 0.0  # Initialize to avoid UnboundLocalError if training ends before first log
    start_time = time.time()

    # Track last 100 step losses for reliable final metric
    from collections import deque
    last_100_losses = deque(maxlen=100)

    print(f"\nStarting training from step {start_step}...")
    print(f"Batch size: {args.batch_size}, Chunk size: {args.chunk_size}")
    print(f"Gradient accumulation: {args.grad_accum}, Effective batch: {args.batch_size * args.grad_accum}")
    print()

    model.train()
    if args.optimizer == 'schedulefree':
        optimizer.train()  # Schedule-free needs this
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

    # Prefetch data function
    import threading
    import queue
    prefetch_queue = queue.Queue(maxsize=2)
    prefetch_stop = threading.Event()

    def prefetch_worker():
        """Background thread to prefetch batches."""
        while not prefetch_stop.is_set():
            try:
                if args.tbptt:
                    chunks, is_doc_end = train_dataset.get_batch(device=device)
                    actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
                else:
                    chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)
                prefetch_queue.put((chunks, is_doc_end, actual_lengths), timeout=1.0)
            except queue.Full:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}")
                break

    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    while should_continue():
        # Get prefetched batch
        try:
            chunks, is_doc_end, actual_lengths = prefetch_queue.get(timeout=5.0)
        except queue.Empty:
            print("Warning: prefetch queue empty, fetching synchronously")
            if args.tbptt:
                chunks, is_doc_end = train_dataset.get_batch(device=device)
                actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
            else:
                chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)

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

        # Add orthogonality regularization for E79 if enabled
        if args.orth_reg > 0:
            orth_loss = 0.0
            for module in model.modules():
                if hasattr(module, 'orthogonality_loss'):
                    orth_loss = orth_loss + module.orthogonality_loss(
                        lambda_sep=args.orth_sep,
                        lambda_orth=args.orth_orth
                    )
            loss = loss + args.orth_reg * orth_loss

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

            # Learning rate schedule (only for standard AdamW)
            if args.optimizer == 'adamw':
                lr = get_lr(step, args.warmup_steps, args.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = args.lr  # Schedule-free handles LR internally

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

                # Track for last-100 average (each entry covers log_every steps)
                last_100_losses.append(avg_loss)

                running_loss = 0
                tokens_processed = 0
                start_time = time.time()

            # Validation
            if val_loader and step % args.val_every == 0:
                if args.optimizer == 'schedulefree':
                    optimizer.eval()  # Get averaged params for eval
                val_loss = validate(model, val_loader, device)
                print(f"  >>> validation loss: {val_loss:.4f}")
                if args.optimizer == 'schedulefree':
                    optimizer.train()  # Back to training mode

            # Checkpointing
            if step % args.save_every == 0:
                if args.optimizer == 'schedulefree':
                    optimizer.eval()  # Get averaged params for checkpoint
                ckpt_path = save_checkpoint(
                    model, optimizer, step, avg_loss, output_dir, args.keep_checkpoints
                )
                print(f"  >>> saved checkpoint: {ckpt_path.name}")
                if args.optimizer == 'schedulefree':
                    optimizer.train()  # Back to training mode

    # Stop prefetch thread
    prefetch_stop.set()
    prefetch_thread.join(timeout=2.0)

    # Compute last-100 average (reliable metric)
    if len(last_100_losses) > 0:
        # Take last 100 log windows (or fewer if not enough data)
        # Each entry covers log_every steps, so 100 entries = 100*log_every steps
        last_n = min(100, len(last_100_losses))
        recent_losses = list(last_100_losses)[-last_n:]
        last_100_avg = sum(recent_losses) / len(recent_losses)
    else:
        last_100_avg = avg_loss  # Fallback if no logging happened

    # Final checkpoint - use last-100 average for reliable metric
    if args.optimizer == 'schedulefree':
        optimizer.eval()  # Get averaged params for final checkpoint
    save_checkpoint(model, optimizer, step, last_100_avg, output_dir, args.keep_checkpoints)

    # Print final metrics in parseable format
    print(f"\nTraining complete! Final step: {step}")
    print(f"FINAL_LOSS_LAST100: {last_100_avg:.4f}")
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"PEAK_MEMORY_MB: {peak_mb:.0f}")
        print(f"RESERVED_MEMORY_MB: {reserved_mb:.0f}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
