#!/usr/bin/env python3
"""
Training script for E74 Ablation models.

Uses the E74 checkpointed Triton kernels for memory-efficient training.
This is a minimal LM wrapper around E74 ablation cells.

Usage:
    python train_e74_ablation.py --data data/pile.txt --dim 1408 --depth 20 \
        --state_type diagonal --proj_type tied_kq --nonlin_type tanh
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.data import DocumentStreamDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train E74 Ablation models')

    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to training data')

    # Model architecture
    parser.add_argument('--dim', type=int, required=True, help='Model dimension')
    parser.add_argument('--depth', type=int, required=True, help='Number of layers')
    parser.add_argument('--n_state', type=int, default=96, help='State dimension')
    parser.add_argument('--expansion', type=float, default=2.0, help='Expansion factor')

    # E74 ablation options
    parser.add_argument('--state_type', type=str, default='diagonal',
                        choices=['diagonal', 'full', 'lowrank', 'blockdiag'])
    parser.add_argument('--proj_type', type=str, default='no_z',
                        choices=['full', 'no_z', 'tied_kq', 'tied_kvq'])
    parser.add_argument('--nonlin_type', type=str, default='tanh',
                        choices=['tanh', 'linear', 'rmsnorm', 'frobnorm'])
    parser.add_argument('--gate_type', type=str, default='output',
                        choices=['output', 'retain', 'state', 'input'])
    parser.add_argument('--rank', type=int, default=8, help='Rank for lowrank state')
    parser.add_argument('--block_size', type=int, default=8, help='Block size for blockdiag')
    parser.add_argument('--update_type', type=str, default='delta',
                        choices=['delta', 'simple', 'ema', 'residual', 'ntm', 'retrieved_gate'],
                        help='Update rule: delta, simple, ema, residual, ntm, or retrieved_gate')
    parser.add_argument('--checkpointed', action='store_true', help='Use checkpointing')
    parser.add_argument('--checkpoint_interval', type=int, default=32)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--train_minutes', type=float, default=None)
    parser.add_argument('--warmup_steps', type=int, default=1000)

    # Output
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--keep_checkpoints', type=int, default=3)

    # System
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


class E74AblationLM(nn.Module):
    """Language Model wrapper for E74 ablation layers."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        n_state: int = 96,
        expansion: float = 2.0,
        state_type: str = 'diagonal',
        proj_type: str = 'no_z',
        nonlin_type: str = 'tanh',
        gate_type: str = 'output',
        update_type: str = 'delta',
        rank: int = 8,
        block_size: int = 8,
        use_checkpointing: bool = True,
        checkpoint_interval: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth

        # Embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Layer norm (Mamba-style fused add+norm)
        try:
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
            norm_cls = RMSNorm
        except ImportError:
            norm_cls = nn.LayerNorm

        # Layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Try E74v2 CUDA kernels first for full matrix state
        use_e74v2 = False
        if state_type == 'full' and proj_type == 'no_z':
            try:
                from elman.models.e74_v2 import E74v2, E74V2_CUDA_AVAILABLE
                v2_supported_n = n_state in {1, 2, 4, 8, 16, 28, 32, 48, 64, 96}
                if E74V2_CUDA_AVAILABLE and v2_supported_n:
                    use_tanh = (nonlin_type == 'tanh')
                    for _ in range(depth):
                        layer = E74v2(
                            dim=dim,
                            expansion=expansion,
                            n_state=n_state,
                            proj_type=proj_type,
                            use_tanh=use_tanh,
                            update_type=update_type,
                            gate_type=gate_type,
                        )
                        self.layers.append(layer)
                        self.norms.append(norm_cls(dim))
                    use_e74v2 = True
                    print(f"Using E74v2 CUDA kernels (n_state={n_state}, update={update_type}, gate={gate_type})")
            except ImportError:
                pass

        # Fall back to checkpointed Triton or PyTorch
        if not use_e74v2:
            if use_checkpointing:
                try:
                    from elman.kernels.e74_checkpointed_triton import E74Checkpointed
                    layer_cls = E74Checkpointed
                    use_tanh = (nonlin_type == 'tanh')

                    for _ in range(depth):
                        layer = layer_cls(
                            dim=dim,
                            expansion=expansion,
                            n_state=n_state,
                            state_type=state_type,
                            checkpoint_interval=checkpoint_interval,
                            use_tanh=use_tanh,
                            proj_type=proj_type,
                            rank=rank,
                            block_size=block_size,
                        )
                        self.layers.append(layer)
                        self.norms.append(norm_cls(dim))
                except ImportError:
                    print("Warning: E74 checkpointed Triton not available, using PyTorch")
                    use_checkpointing = False

            if not use_checkpointing:
                from elman.models.e74_ablations import E74Ablation
                for _ in range(depth):
                    layer = E74Ablation(
                        dim=dim,
                        expansion=expansion,
                        n_state=n_state,
                        state_type=state_type,
                        proj_type=proj_type,
                        nonlin_type=nonlin_type,
                        gate_type=gate_type,
                        update_type=update_type,
                        rank=rank,
                        block_size=block_size,
                    )
                    self.layers.append(layer)
                    self.norms.append(norm_cls(dim))

        # Final norm + head
        self.final_norm = norm_cls(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, return_loss=False):
        """
        Args:
            x: [B, T] token ids (T includes target if return_loss=True)
            return_loss: if True, compute cross-entropy loss

        Returns:
            logits [B, T, vocab] or loss scalar
        """
        if return_loss:
            inputs = x[:, :-1]
            targets = x[:, 1:]
        else:
            inputs = x
            targets = None

        h = self.embedding(inputs)  # [B, T, dim]

        # Mamba-style: prenorm + residual
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            out, _ = layer(h)
            h = residual + out

        h = self.final_norm(h)
        logits = self.lm_head(h)

        if return_loss:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
            return loss

        return logits


def setup_output_dir(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_str = f"{args.state_type}_{args.proj_type}_{args.nonlin_type}"
    run_name = f"e74_{config_str}_{timestamp}"
    output_dir = Path(args.output) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    return output_dir


def get_lr(step, warmup_steps, max_lr, min_lr=1e-6):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + torch.cos(torch.tensor(step / warmup_steps * 3.14159)))


def save_checkpoint(model, optimizer, step, loss, output_dir, keep_n=3):
    ckpt_path = output_dir / f'checkpoint_step_{step:06d}_loss_{loss:.4f}.pt'
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, ckpt_path)

    # Clean old
    ckpts = sorted(glob.glob(str(output_dir / 'checkpoint_step_*.pt')))
    for old in ckpts[:-keep_n]:
        os.remove(old)

    return ckpt_path


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = setup_output_dir(args)
    print(f"Output directory: {output_dir}")

    # Create model
    model = E74AblationLM(
        vocab_size=256,
        dim=args.dim,
        depth=args.depth,
        n_state=args.n_state,
        expansion=args.expansion,
        state_type=args.state_type,
        proj_type=args.proj_type,
        nonlin_type=args.nonlin_type,
        gate_type=args.gate_type,
        update_type=args.update_type,
        rank=args.rank,
        block_size=args.block_size,
        use_checkpointing=args.checkpointed,
        checkpoint_interval=args.checkpoint_interval,
    )

    model = model.to(device)
    if args.bf16:
        model = model.bfloat16()

    update_str = f"/{args.update_type}" if args.update_type != 'delta' else ""
    print(f"Model: E74 ({args.state_type}/{args.proj_type}/{args.nonlin_type}{update_str}), {model.get_num_params():,} parameters")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Dataset
    train_dataset = DocumentStreamDataset(
        data_path=args.data,
        chunk_size=args.chunk_size + 1,
        seed=args.seed,
    )

    # Training state
    running_loss = 0
    tokens_processed = 0
    avg_loss = 0.0
    start_time = time.time()

    print(f"\nStarting training from step 0...")
    print(f"Batch size: {args.batch_size}, Chunk size: {args.chunk_size}")
    print()

    model.train()
    step = 0

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
        # Get batch
        batch_chunks = []
        batch_lengths = []

        for _ in range(args.batch_size):
            chunk, _, actual_length = train_dataset[0]
            batch_chunks.append(chunk)
            batch_lengths.append(actual_length)

        chunks = torch.stack(batch_chunks).to(device)
        actual_lengths = torch.tensor(batch_lengths, dtype=torch.long, device=device)

        # Forward
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
            loss = model(chunks, return_loss=True)

        loss.backward()

        running_loss += loss.item()
        tokens_processed += actual_lengths.sum().item()

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

        # Checkpointing
        if step % args.save_every == 0:
            ckpt_path = save_checkpoint(model, optimizer, step, avg_loss, output_dir, args.keep_checkpoints)
            print(f"  >>> saved: {ckpt_path.name}")

    # Final checkpoint
    save_checkpoint(model, optimizer, step, avg_loss, output_dir, args.keep_checkpoints)
    print(f"\nTraining complete! Final step: {step}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
