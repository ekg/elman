#!/usr/bin/env python3
"""
Training script for Elman Ablation Ladder with DDP support.

Usage:
    # Single GPU
    python train_ladder.py --level 0 --params 100m --data data.txt --output outputs/level0

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 train_ladder.py --level 3 --params 500m --data data.txt --ddp

    # With tiktoken (50k vocab)
    python train_ladder.py --level 3 --tokenizer tiktoken --tokenizer_name p50k_base --data data.txt
"""

import argparse
import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from datetime import datetime
from schedulefree import AdamWScheduleFree

# Add elman package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.models import LadderLM, create_ladder_model, get_available_levels
from elman.data import DocumentStreamDataset, BatchedStreamDataset, TokenizedStreamDataset
from elman.data.tokenizers import get_tokenizer, ByteTokenizer, TikTokenTokenizer


def parse_level(value):
    """Parse level argument - can be int (0-6) or string ('log_0', etc.)"""
    if value.startswith('log_'):
        return value  # Return string for log-space levels
    return int(value)


def get_args():
    parser = argparse.ArgumentParser(description="Train Elman Ablation Ladder models")

    # Model
    parser.add_argument("--level", type=parse_level, default=0,
                        help="Ablation ladder level (0-3 or 'log_0' for log-space polynomial)")
    parser.add_argument("--params", type=str, default="100m",
                        help="Target parameter count (e.g., 100m, 500m, 1b)")
    parser.add_argument("--expansion", type=float, default=1.0,
                        help="Hidden state expansion factor")
    parser.add_argument("--n_groups", type=int, default=32,
                        help="Number of groups for compete softmax (levels 2+)")
    parser.add_argument("--r_h_mode", type=str, default="spectral_norm",
                        choices=["free", "spectral_norm", "scaled_orthogonal"],
                        help="R_h constraint mode for log-space levels (default: spectral_norm)")
    parser.add_argument("--r_h_init_gain", type=float, default=0.1,
                        help="Initial gain for R_h orthogonal initialization")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data (text file)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Sequence length for training")
    parser.add_argument("--tokenizer", type=str, default="tiktoken",
                        choices=["byte", "tiktoken", "sentencepiece", "huggingface"],
                        help="Tokenizer type (default: tiktoken)")
    parser.add_argument("--tokenizer_name", type=str, default="p50k_base",
                        help="Tokenizer name (default: p50k_base ~50k vocab)")

    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--schedulefree", action="store_true", default=True,
                        help="Use AdamWScheduleFree optimizer (default: True)")
    parser.add_argument("--no-schedulefree", dest="schedulefree", action="store_false",
                        help="Use regular AdamW optimizer")
    parser.add_argument("--sf_beta", type=float, default=0.9,
                        help="Schedule-free beta1")
    parser.add_argument("--sf_beta2", type=float, default=0.999,
                        help="Schedule-free beta2")

    # Output
    parser.add_argument("--output", type=str, default="outputs/ladder",
                        help="Output directory")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N steps")

    # Hardware
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Use CUDA (default: True)")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false",
                        help="Disable CUDA")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 (default: True)")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="Use float32")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile")
    parser.add_argument("--ddp", action="store_true", default=True,
                        help="Use DistributedDataParallel (default: True)")
    parser.add_argument("--no-ddp", dest="ddp", action="store_false",
                        help="Disable DDP (single GPU)")
    parser.add_argument("--tbptt", action="store_true",
                        help="Enable TBPTT (carry hidden state across chunks)")

    return parser.parse_args()


def format_params(n):
    """Format parameter count for logging."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


class StepLogger:
    """JSON logger for per-step metrics."""

    def __init__(self, log_path, config):
        self.log_path = log_path
        self.config = config
        self.start_time = time.time()
        self.step_start = None

        # Write header with config
        with open(log_path, 'w') as f:
            header = {
                'type': 'header',
                'timestamp': datetime.now().isoformat(),
                'config': config
            }
            f.write(json.dumps(header) + '\n')

    def start_step(self):
        """Mark the start of a step for timing."""
        self.step_start = time.time()

    def log_step(self, step, metrics):
        """Log metrics for a step."""
        step_time = time.time() - self.step_start if self.step_start else 0
        total_time = time.time() - self.start_time

        record = {
            'type': 'step',
            'step': step,
            'step_time_ms': step_time * 1000,
            'total_time_s': total_time,
            **metrics
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
    return {}


def compute_grad_norms(model):
    """Compute gradient norms for different parameter groups."""
    norms = {}
    total_norm = 0.0

    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2

            # Group by layer type
            if 'embedding' in name:
                group = 'embedding'
            elif 'head' in name:
                group = 'head'
            elif 'layer' in name:
                group = 'layers'
            else:
                group = 'other'

            if group not in norms:
                norms[group] = 0.0
            norms[group] += param_norm ** 2

    norms['total'] = math.sqrt(total_norm)
    for k in norms:
        if k != 'total':
            norms[k] = math.sqrt(norms[k])

    return norms


def get_lr(step, warmup_steps, max_lr, max_steps):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    # DDP setup
    if args.ddp:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Create tokenizer
    if args.tokenizer == "byte":
        tokenizer = ByteTokenizer()
    elif args.tokenizer == "tiktoken":
        tokenizer = TikTokenTokenizer(encoding_name=args.tokenizer_name)
    else:
        tokenizer = get_tokenizer(args.tokenizer, encoding_name=args.tokenizer_name)

    vocab_size = tokenizer.vocab_size

    # Print ablation ladder info (main process only)
    if is_main:
        print("=" * 70)
        print("Elman Ablation Ladder Training")
        print("=" * 70)
        levels = get_available_levels()
        for lvl, (name, available, _) in levels.items():
            marker = " <-- TRAINING" if lvl == args.level else ""
            print(f"  Level {lvl}: {name} {'✓' if available else '✗'}{marker}")
        print("=" * 70)
        print(f"Tokenizer: {tokenizer} (vocab_size={vocab_size:,})")
        print(f"Device: {device}, dtype: {dtype}, world_size: {world_size}")

    # Create model
    model = create_ladder_model(
        target_params=args.params,
        level=args.level,
        vocab_size=vocab_size,
        expansion=args.expansion,
        n_groups=args.n_groups,
        r_h_mode=args.r_h_mode,
        r_h_init_gain=args.r_h_init_gain,
    )
    model = model.to(device=device, dtype=dtype)

    # Log model size
    num_params = model.get_num_params()
    if is_main:
        print(f"\nModel Parameters: {num_params:,} ({format_params(num_params)})")
        print(f"  Embedding: {model.embedding.weight.numel():,}")
        print(f"  Layers: {len(model.layers)} x {model.dim}d")
        for i, layer in enumerate(model.layers[:1]):  # Just first layer as example
            layer_params = sum(p.numel() for p in layer.parameters())
            print(f"  Layer 0: {layer_params:,} params")
        print()

    if args.compile:
        if is_main:
            print("Compiling model...")
        model = torch.compile(model)

    if args.ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Create dataset (each rank gets different data stream)
    use_subword = args.tokenizer != "byte"
    if is_main:
        print(f"Loading data from {args.data}...")
        if args.tbptt:
            print("TBPTT enabled: using persistent streams")
        if use_subword:
            print(f"Using streaming {args.tokenizer} tokenization")

    if use_subword:
        # Use TokenizedStreamDataset for subword tokenizers (tiktoken, etc.)
        dataset = TokenizedStreamDataset(
            data_path=args.data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,  # +1 for targets
            rank=rank,
            world_size=world_size,
            seed=42,
        )
    elif args.tbptt:
        # BatchedStreamDataset: each batch element has its own persistent stream (byte-level)
        dataset = BatchedStreamDataset(
            data_path=args.data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,  # +1 for targets
            rank=rank,
            world_size=world_size,
            seed=42,
        )
    else:
        dataset = DocumentStreamDataset(
            data_path=args.data,
            chunk_size=args.chunk_size + 1,  # +1 for targets
            rank=rank,
            world_size=world_size,
            seed=42,
        )

    # Optimizer
    if args.schedulefree:
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.sf_beta, args.sf_beta2),
        )
        if is_main:
            print("Using AdamWScheduleFree optimizer")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        if is_main:
            print("Using AdamW optimizer")

    # Output directory
    output_dir = Path(args.output)
    step_logger = None
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Config dict for logging
        config = {
            'level': str(args.level),
            'params': args.params,
            'num_params': num_params,
            'vocab_size': vocab_size,
            'tokenizer': args.tokenizer,
            'tokenizer_name': args.tokenizer_name,
            'chunk_size': args.chunk_size,
            'batch_size': args.batch_size,
            'grad_accum': args.grad_accum,
            'lr': args.lr,
            'warmup_steps': args.warmup_steps,
            'max_steps': args.max_steps,
            'weight_decay': args.weight_decay,
            'grad_clip': args.grad_clip,
            'world_size': world_size,
            'dtype': str(dtype),
            'tbptt': args.tbptt,
            'tokens_per_step': args.batch_size * world_size * args.grad_accum * args.chunk_size,
            'r_h_mode': args.r_h_mode,
            'r_h_init_gain': args.r_h_init_gain,
        }

        # Save config
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Create step logger
        log_path = output_dir / 'steps.jsonl'
        step_logger = StepLogger(log_path, config)
        print(f"Logging to: {log_path}")

    # Training loop
    if is_main:
        print(f"\nStarting training for {args.max_steps} steps...")
        print(f"Batch size per GPU: {args.batch_size}, World size: {world_size}")
        print(f"Effective batch size: {args.batch_size * world_size * args.grad_accum}")
        print(f"Tokens per step: {args.batch_size * world_size * args.grad_accum * args.chunk_size:,}")
        print()

    step = 0
    tokens_seen = 0
    start_time = time.time()
    running_loss = 0.0
    avg_loss = float('inf')

    # TBPTT state
    hidden_state = None

    model.train()
    if args.schedulefree:
        optimizer.train()  # Set optimizer to train mode for schedule-free
    optimizer.zero_grad()

    # For per-step timing
    forward_time = 0.0
    backward_time = 0.0

    while step < args.max_steps:
        if step_logger:
            step_logger.start_step()
        # Get batch - different methods based on dataset type
        if use_subword or args.tbptt:
            # TokenizedStreamDataset and BatchedStreamDataset: use get_batch()
            batch, is_doc_end = dataset.get_batch(device=device)
        else:
            # DocumentStreamDataset: single stream, no hidden state persistence
            batch_chunks = []
            batch_doc_ends = []
            for _ in range(args.batch_size):
                chunk, is_doc_end_single, _ = dataset[0]
                batch_chunks.append(chunk)
                batch_doc_ends.append(is_doc_end_single)
            batch = torch.stack(batch_chunks).to(device)
            is_doc_end = torch.tensor(batch_doc_ends, dtype=torch.bool, device=device)

        # Reset hidden state at document boundaries (TBPTT or streaming tokenization)
        # Handle both tensor and tuple (log_h, sign_h) hidden states
        def reset_hidden(h, reset_mask):
            if h is None:
                return None
            if isinstance(h, tuple):
                # Log-space layers return (log_h, sign_h) tuple
                return tuple(elem * (~reset_mask).to(elem.dtype) for elem in h)
            return h * (~reset_mask).to(h.dtype)

        if (args.tbptt or use_subword) and hidden_state is not None:
            reset_mask = is_doc_end.view(-1, 1)
            hidden_state = [reset_hidden(h, reset_mask) for h in hidden_state]

        # Learning rate schedule (only for regular AdamW, not schedule-free)
        if not args.schedulefree:
            lr = get_lr(step, args.warmup_steps, args.lr, args.max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = args.lr  # For logging purposes

        # Forward pass
        x = batch[:, :-1]  # Input tokens
        y = batch[:, 1:]   # Target tokens

        t_forward_start = time.time()
        with torch.amp.autocast('cuda', dtype=dtype, enabled=args.bf16):
            if args.tbptt or use_subword:
                logits, (next_hidden, _) = model(x, return_prev_hiddens=True, prev_hiddens=hidden_state)
            else:
                logits, _ = model(x, return_prev_hiddens=True)
                next_hidden = None
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_time = (time.time() - t_forward_start) * 1000  # ms

        # Backward pass
        t_backward_start = time.time()
        loss_scaled = loss / args.grad_accum
        loss_scaled.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_time = (time.time() - t_backward_start) * 1000  # ms

        # Update hidden state for TBPTT/streaming (detach from graph)
        # Handle both tensor and tuple (log_h, sign_h) hidden states
        def detach_hidden(h):
            if h is None:
                return None
            if isinstance(h, tuple):
                return tuple(elem.detach() for elem in h)
            return h.detach()

        if (args.tbptt or use_subword) and next_hidden is not None:
            hidden_state = [detach_hidden(h) for h in next_hidden]

        # Update
        grad_norms = {}
        if (step + 1) % args.grad_accum == 0:
            # Compute detailed gradient norms before clipping
            model_for_grads = model.module if args.ddp else model
            grad_norms = compute_grad_norms(model_for_grads)
            grad_norm = grad_norms.get('total', 0.0)

            # Only clip if grad_clip > 0
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        else:
            grad_norm = 0.0

        # Logging
        running_loss += loss.item()
        tokens_seen += x.numel() * world_size
        step += 1

        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval

            # Reduce loss across ranks
            if args.ddp:
                loss_tensor = torch.tensor([avg_loss], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss = loss_tensor.item()

            if is_main:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_seen / elapsed
                memory = get_gpu_memory_mb()

                # Perplexity
                ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

                print(f"Step {step:6d} | Loss {avg_loss:.4f} | PPL {ppl:.1f} | LR {lr:.2e} | "
                      f"Grad {grad_norm:.2f} | Tok/s {tokens_per_sec:,.0f} | "
                      f"Mem {memory.get('allocated_mb', 0):.0f}MB | "
                      f"Elapsed {elapsed:.1f}s")

                # Log to JSON
                if step_logger:
                    step_logger.log_step(step, {
                        'loss': avg_loss,
                        'perplexity': ppl if ppl != float('inf') else -1,
                        'lr': lr,
                        'tokens_seen': tokens_seen,
                        'tokens_per_sec': tokens_per_sec,
                        'forward_time_ms': forward_time,
                        'backward_time_ms': backward_time,
                        'grad_norm_total': grad_norms.get('total', 0),
                        'grad_norm_embedding': grad_norms.get('embedding', 0),
                        'grad_norm_layers': grad_norms.get('layers', 0),
                        'grad_norm_head': grad_norms.get('head', 0),
                        **{f'memory_{k}': v for k, v in memory.items()},
                    })

            running_loss = 0.0

        # Save checkpoint (main process only)
        if step % args.save_interval == 0 and is_main:
            model_to_save = model.module if args.ddp else model
            ckpt_path = output_dir / f"level{args.level}_step{step:06d}_loss{avg_loss:.4f}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'tokens_seen': tokens_seen,
                'args': vars(args),
            }, ckpt_path)
            print(f"  >>> Saved: {ckpt_path}")

            # Update latest symlink
            latest = output_dir / 'latest.pt'
            if latest.is_symlink():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)

    if is_main:
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Final loss: {avg_loss:.4f}")
        print(f"Total tokens: {tokens_seen:,}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Parameters: {num_params:,} ({format_params(num_params)})")

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = get_args()
    train(args)
