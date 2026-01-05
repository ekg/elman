#!/usr/bin/env python3
"""
Benchmark E5 backends: FUSED (cuBLAS) vs B2B (WMMA tensor cores)

Trains Pure Low-Rank Elman on byte-level Pile for 1k steps with each backend.
Compares: loss curves, throughput, and memory usage.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elman.data import FastTokenizedDataset
from elman.data.tokenizers import ByteTokenizer


# Import the kernel functions directly
sys.path.insert(0, 'elman/cuda')
import hasty_pytorch_lib


class E5CellOriginal(nn.Module):
    """E5 cell using original kernel (cuBLAS matmuls, separate tanh/gate)."""

    def __init__(self, dim, rank):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.U_h = nn.Parameter(torch.empty(dim, rank))
        self.V_h = nn.Parameter(torch.empty(rank, dim))
        self.U_x = nn.Parameter(torch.empty(dim, rank))
        self.V_x = nn.Parameter(torch.empty(rank, dim))
        self.U_z = nn.Parameter(torch.empty(dim, rank))
        self.V_z = nn.Parameter(torch.empty(rank, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        for U, V in [(self.U_h, self.V_h), (self.U_x, self.V_x), (self.U_z, self.V_z)]:
            nn.init.orthogonal_(U)
            nn.init.orthogonal_(V)
            with torch.no_grad():
                U.mul_(0.5)
                V.mul_(0.5)

    def forward(self, x, h0):
        """x: [T, B, D], h0: [B, D]"""
        from torch.autograd import Function

        class OriginalFn(Function):
            @staticmethod
            def forward(ctx, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b):
                h, output, v = hasty_pytorch_lib.pure_lowrank_elman_forward(
                    True, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
                ctx.save_for_backward(U_h, V_h, U_x, V_x, U_z, V_z, x, h, v)
                return output, h

            @staticmethod
            def backward(ctx, grad_output, grad_h):
                U_h, V_h, U_x, V_x, U_z, V_z, x, h, v = ctx.saved_tensors
                dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db = hasty_pytorch_lib.pure_lowrank_elman_backward(
                    U_h, V_h, U_x, V_x, U_z, V_z, x, h, v, grad_output.contiguous())
                return dx, None, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db

        return OriginalFn.apply(x, h0, self.U_h, self.V_h, self.U_x, self.V_x,
                                self.U_z, self.V_z, self.b)


class E5CellFused(nn.Module):
    """E5 cell using FUSED kernel (cuBLAS matmuls)."""

    def __init__(self, dim, rank):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.U_h = nn.Parameter(torch.empty(dim, rank))
        self.V_h = nn.Parameter(torch.empty(rank, dim))
        self.U_x = nn.Parameter(torch.empty(dim, rank))
        self.V_x = nn.Parameter(torch.empty(rank, dim))
        self.U_z = nn.Parameter(torch.empty(dim, rank))
        self.V_z = nn.Parameter(torch.empty(rank, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        for U, V in [(self.U_h, self.V_h), (self.U_x, self.V_x), (self.U_z, self.V_z)]:
            nn.init.orthogonal_(U)
            nn.init.orthogonal_(V)
            with torch.no_grad():
                U.mul_(0.5)
                V.mul_(0.5)

    def forward(self, x, h0):
        """x: [T, B, D], h0: [B, D]"""
        from torch.autograd import Function

        class FusedFn(Function):
            @staticmethod
            def forward(ctx, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b):
                h, output, v = hasty_pytorch_lib.pure_lowrank_elman_forward_fused(
                    True, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
                ctx.save_for_backward(U_h, V_h, U_x, V_x, U_z, V_z, x, h, v)
                return output, h

            @staticmethod
            def backward(ctx, grad_output, grad_h):
                U_h, V_h, U_x, V_x, U_z, V_z, x, h, v = ctx.saved_tensors
                dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db = hasty_pytorch_lib.pure_lowrank_elman_backward_fused(
                    U_h, V_h, U_x, V_x, U_z, V_z, x, h, v, grad_output.contiguous())
                return dx, None, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db

        return FusedFn.apply(x, h0, self.U_h, self.V_h, self.U_x, self.V_x,
                             self.U_z, self.V_z, self.b)


class E5CellB2B(nn.Module):
    """E5 cell using B2B kernel (WMMA tensor core B2B GEMM)."""

    def __init__(self, dim, rank):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.U_h = nn.Parameter(torch.empty(dim, rank))
        self.V_h = nn.Parameter(torch.empty(rank, dim))
        self.U_x = nn.Parameter(torch.empty(dim, rank))
        self.V_x = nn.Parameter(torch.empty(rank, dim))
        self.U_z = nn.Parameter(torch.empty(dim, rank))
        self.V_z = nn.Parameter(torch.empty(rank, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        for U, V in [(self.U_h, self.V_h), (self.U_x, self.V_x), (self.U_z, self.V_z)]:
            nn.init.orthogonal_(U)
            nn.init.orthogonal_(V)
            with torch.no_grad():
                U.mul_(0.5)
                V.mul_(0.5)

    def forward(self, x, h0):
        """x: [T, B, D], h0: [B, D]"""
        from torch.autograd import Function

        class B2BFn(Function):
            @staticmethod
            def forward(ctx, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b):
                h, output, v = hasty_pytorch_lib.b2b_lowrank_elman_forward(
                    True, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b)
                ctx.save_for_backward(U_h, V_h, U_x, V_x, U_z, V_z, x, h, v)
                return output, h

            @staticmethod
            def backward(ctx, grad_output, grad_h):
                U_h, V_h, U_x, V_x, U_z, V_z, x, h, v = ctx.saved_tensors
                dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db = hasty_pytorch_lib.b2b_lowrank_elman_backward(
                    U_h, V_h, U_x, V_x, U_z, V_z, x, h, v, grad_output.contiguous())
                return dx, None, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db

        return B2BFn.apply(x, h0, self.U_h, self.V_h, self.U_x, self.V_x,
                           self.U_z, self.V_z, self.b)


class E5Layer(nn.Module):
    """E5 layer with selectable backend."""

    def __init__(self, dim, rank, backend='fused'):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.backend = backend

        if backend == 'original':
            self.cell = E5CellOriginal(dim, rank)
        elif backend == 'fused':
            self.cell = E5CellFused(dim, rank)
        elif backend == 'b2b':
            self.cell = E5CellB2B(dim, rank)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def forward(self, x, h0=None):
        """x: [B, T, D] -> [B, T, D]"""
        B, T, D = x.shape
        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        x_rnn = x.permute(1, 0, 2).contiguous()
        out, h = self.cell(x_rnn, h0)
        return out.permute(1, 0, 2).contiguous(), h[-1]


class E5LM(nn.Module):
    """E5 Language Model for benchmarking."""

    def __init__(self, vocab_size, dim, depth, rank, backend='fused'):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.backend = backend

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            E5Layer(dim, rank, backend=backend) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x, return_loss=False):
        """x: [B, T] token ids"""
        if return_loss:
            inputs = x[:, :-1]
            targets = x[:, 1:]
        else:
            inputs = x

        h = self.embed(inputs)

        for layer in self.layers:
            out, _ = layer(h)
            h = h + out  # Residual

        h = self.norm(h)
        logits = self.head(h)

        if return_loss:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return loss
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_model(backend, args):
    """Train a model with specified backend and return metrics."""
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    # Create model
    model = E5LM(
        vocab_size=256,
        dim=args.dim,
        depth=args.depth,
        rank=args.rank,
        backend=backend
    ).to(device).bfloat16()

    print(f"\n{'='*60}")
    print(f"Backend: {backend.upper()}")
    print(f"Model: dim={args.dim}, depth={args.depth}, rank={args.rank}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"{'='*60}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Create dataset
    tokenizer = ByteTokenizer()
    dataset = FastTokenizedDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size + 1,
        seed=args.seed,
    )

    # Training metrics
    losses = []
    times = []
    memory_peak = 0

    torch.cuda.reset_peak_memory_stats()
    model.train()

    start_time = time.time()
    running_loss = 0

    for step in range(1, args.steps + 1):
        step_start = time.time()

        # Get batch
        chunks, _ = dataset.get_batch(device=device)

        # Forward/backward
        optimizer.zero_grad()
        loss = model(chunks, return_loss=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()

        step_time = time.time() - step_start
        times.append(step_time)
        running_loss += loss.item()

        # Log
        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            losses.append((step, avg_loss))
            tok_per_sec = args.batch_size * args.chunk_size / (sum(times[-args.log_every:]) / args.log_every)

            print(f"step {step:5d} | loss {avg_loss:.4f} | tok/s {tok_per_sec:,.0f}")
            running_loss = 0

    # Final metrics
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    memory_peak = torch.cuda.max_memory_allocated() / 1e9
    avg_tok_per_sec = args.batch_size * args.chunk_size * args.steps / total_time

    return {
        'backend': backend,
        'losses': losses,
        'total_time': total_time,
        'avg_tok_per_sec': avg_tok_per_sec,
        'memory_peak_gb': memory_peak,
        'final_loss': losses[-1][1] if losses else float('inf'),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark E5 backends')
    parser.add_argument('--data', type=str, default='data/pile.txt',
                        help='Path to training data')
    parser.add_argument('--dim', type=int, default=1024,
                        help='Model dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of layers')
    parser.add_argument('--rank', type=int, default=128,
                        help='Low-rank dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Training steps')
    parser.add_argument('--log_every', type=int, default=50,
                        help='Log every N steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--backends', type=str, default='fused,b2b',
                        help='Comma-separated backends to test')
    parser.add_argument('--vocab_size', type=int, default=256,
                        help='Vocabulary size (256 for byte-level)')
    args = parser.parse_args()

    backends = args.backends.split(',')
    results = {}

    print(f"\nBenchmarking E5 backends on byte-level Pile")
    print(f"Config: dim={args.dim}, depth={args.depth}, rank={args.rank}")
    print(f"Training: batch={args.batch_size}, seq={args.chunk_size}, steps={args.steps}")

    for backend in backends:
        results[backend] = train_model(backend.strip(), args)
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for backend, r in results.items():
        print(f"\n{backend.upper()}:")
        print(f"  Final loss:    {r['final_loss']:.4f}")
        print(f"  Avg tok/s:     {r['avg_tok_per_sec']:,.0f}")
        print(f"  Total time:    {r['total_time']:.1f}s")
        print(f"  Peak memory:   {r['memory_peak_gb']:.2f} GB")

    if len(results) == 2 and 'fused' in results and 'b2b' in results:
        speedup = results['b2b']['avg_tok_per_sec'] / results['fused']['avg_tok_per_sec']
        loss_diff = results['b2b']['final_loss'] - results['fused']['final_loss']
        print(f"\nB2B vs FUSED:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Loss diff: {loss_diff:+.4f} (negative = B2B better)")

    # Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'benchmark_results/e5_backends_{timestamp}.json'
    os.makedirs('benchmark_results', exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': {k: {**v, 'losses': v['losses']} for k, v in results.items()}
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
