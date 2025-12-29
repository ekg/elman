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
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

# Add elman package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.models import LadderLM, create_ladder_model, get_available_levels
from elman.data import DocumentStreamDataset, BatchedStreamDataset
from elman.data.tokenizers import get_tokenizer, ByteTokenizer, TikTokenTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Train Elman Ablation Ladder models")

    # Model
    parser.add_argument("--level", type=int, default=0, choices=range(7),
                        help="Ablation ladder level (0-6)")
    parser.add_argument("--params", type=str, default="100m",
                        help="Target parameter count (e.g., 100m, 500m, 1b)")
    parser.add_argument("--expansion", type=float, default=1.0,
                        help="Hidden state expansion factor")
    parser.add_argument("--n_groups", type=int, default=32,
                        help="Number of groups for compete softmax (levels 2+)")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data (text file)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Sequence length for training")
    parser.add_argument("--tokenizer", type=str, default="byte",
                        choices=["byte", "tiktoken", "sentencepiece", "huggingface"],
                        help="Tokenizer type")
    parser.add_argument("--tokenizer_name", type=str, default="p50k_base",
                        help="Tokenizer name (for tiktoken: p50k_base, cl100k_base, etc.)")

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

    # Output
    parser.add_argument("--output", type=str, default="outputs/ladder",
                        help="Output directory")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N steps")

    # Hardware
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile")
    parser.add_argument("--ddp", action="store_true",
                        help="Use DistributedDataParallel")
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
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Create dataset (each rank gets different data stream)
    if is_main:
        print(f"Loading data from {args.data}...")
        if args.tbptt:
            print("TBPTT enabled: using BatchedStreamDataset (persistent streams)")

    if args.tbptt:
        # BatchedStreamDataset: each batch element has its own persistent stream
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Output directory
    output_dir = Path(args.output)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        import json
        with open(output_dir / 'config.json', 'w') as f:
            json.dump({
                'level': args.level,
                'params': args.params,
                'num_params': num_params,
                'vocab_size': vocab_size,
                'tokenizer': args.tokenizer,
                'tokenizer_name': args.tokenizer_name,
                'chunk_size': args.chunk_size,
                'batch_size': args.batch_size,
                'world_size': world_size,
            }, f, indent=2)

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
    optimizer.zero_grad()

    while step < args.max_steps:
        # Get batch - different methods for TBPTT vs non-TBPTT
        if args.tbptt:
            # BatchedStreamDataset: each batch element has its own persistent stream
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

        # Reset hidden state at document boundaries (only if TBPTT enabled)
        if args.tbptt and hidden_state is not None:
            reset_mask = is_doc_end.view(-1, 1)
            hidden_state = [h * (~reset_mask) if h is not None else None for h in hidden_state]

        # Learning rate schedule
        lr = get_lr(step, args.warmup_steps, args.lr, args.max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        x = batch[:, :-1]  # Input tokens
        y = batch[:, 1:]   # Target tokens

        with torch.amp.autocast('cuda', dtype=dtype, enabled=args.bf16):
            if args.tbptt:
                logits, (next_hidden, _) = model(x, return_prev_hiddens=True, prev_hiddens=hidden_state)
            else:
                logits, _ = model(x, return_prev_hiddens=True)
                next_hidden = None
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1))

        # Backward pass
        loss_scaled = loss / args.grad_accum
        loss_scaled.backward()

        # Update hidden state for TBPTT (detach from graph)
        if args.tbptt and next_hidden is not None:
            hidden_state = [h.detach() if h is not None else None for h in next_hidden]

        # Update
        if (step + 1) % args.grad_accum == 0:
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = 0.0
            optimizer.step()
            optimizer.zero_grad()

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
                print(f"Step {step:6d} | Loss {avg_loss:.4f} | LR {lr:.2e} | "
                      f"Grad {grad_norm:.2f} | Tok/s {tokens_per_sec:,.0f} | "
                      f"Elapsed {elapsed:.1f}s")
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
