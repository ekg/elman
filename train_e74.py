#!/usr/bin/env python3
"""
E74 CUDA Training Script

Trains E74 diagonal state RNN with CUDA-accelerated kernels.
Supports all projection types, update rules, and nonlinearities.
"""

import argparse
import json
import mmap
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Set library path for CUDA extensions BEFORE importing torch
torch_lib = os.path.expanduser("~/.local/lib/python3.12/site-packages/torch/lib")
if torch_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import E74 ablation framework
from elman.models.e74_ablations import (
    E74DiagonalCell,
    E74CUDADiagonalCell,
    E74FullMatrixCell,
    E74CUDAFullMatrixCell,
    CUDA_AVAILABLE,
    ProjType,
    NonlinType,
    UpdateType,
    GateType,
    StateType,
)

# Try to import fused norm from mamba
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    FUSED_NORM_AVAILABLE = True
except ImportError:
    FUSED_NORM_AVAILABLE = False
    from torch.nn import LayerNorm as RMSNorm


class E74LM(nn.Module):
    """E74 Language Model with proper LadderLM-style architecture."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        n_state: int,
        expansion: float = 2.0,
        update_type: str = 'delta',
        proj_type: str = 'no_z',
        nonlin_type: str = 'tanh',
        state_type: str = 'diagonal',
        use_cuda: bool = True,
        decay: float = 0.9,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.n_state = n_state
        self.d_inner = int(dim * expansion)
        self.use_cuda = use_cuda
        self.state_type = state_type

        # Embedding (tied with lm_head)
        self.embedding = nn.Embedding(vocab_size, dim)

        # Layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Map string to enum
        update_enum = UpdateType.DELTA if update_type == 'delta' else UpdateType.SIMPLE
        proj_enum = {
            'tied_kvq': ProjType.TIED_KVQ,
            'tied_kq': ProjType.TIED_KQ,
            'no_z': ProjType.NO_Z,
            'full': ProjType.FULL,
        }[proj_type]
        nonlin_enum = NonlinType.TANH if nonlin_type == 'tanh' else NonlinType.LINEAR

        for _ in range(depth):
            # E74Layer creates its own projections and cell
            layer = E74Layer(self.d_inner, n_state, dim, use_cuda, update_type, proj_type, nonlin_type, state_type, decay)
            self.layers.append(layer)
            self.norms.append(RMSNorm(dim))

        # Final norm
        self.final_norm = RMSNorm(dim)

        # LM head (tied with embedding)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, return_loss=False):
        """
        Args:
            x: [B, T] input token ids
            return_loss: if True, compute and return loss

        Returns:
            logits or loss
        """
        if return_loss:
            inputs = x[:, :-1]
            targets = x[:, 1:]
        else:
            inputs = x
            targets = None

        B, T = inputs.shape

        # Embedding
        h = self.embedding(inputs)  # [B, T, dim]

        # Transpose to [T, B, dim] for RNN processing
        h = h.transpose(0, 1).contiguous()

        # Residual stream with prenorm
        residual = h
        for layer, norm in zip(self.layers, self.norms):
            # Prenorm
            if FUSED_NORM_AVAILABLE:
                h_normed, residual = layer_norm_fn(
                    h, norm.weight, norm.bias if hasattr(norm, 'bias') else None,
                    residual=residual, prenorm=True, residual_in_fp32=True
                )
            else:
                h_normed = norm(residual)

            # Layer
            out = layer(h_normed)

            # Residual
            if FUSED_NORM_AVAILABLE:
                h = out
            else:
                residual = residual + out
                h = residual

        # Final norm
        if FUSED_NORM_AVAILABLE:
            h = rms_norm_fn(h, self.final_norm.weight, None, residual=residual)
        else:
            h = self.final_norm(residual)

        # Back to [B, T, dim]
        h = h.transpose(0, 1).contiguous()

        # LM head
        logits = self.lm_head(h)

        if return_loss and targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            return loss

        return logits


class E74Layer(nn.Module):
    """Single E74 layer with projections."""

    def __init__(self, d_inner, n_state, dim, use_cuda, update_type, proj_type, nonlin_type, state_type, decay):
        super().__init__()
        self.d_inner = d_inner
        self.n_state = n_state
        self.dim = dim
        self.state_type = state_type
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        # In projection
        self.in_proj = nn.Linear(dim, d_inner, bias=False)

        # Map string to enum
        update_enum = UpdateType.DELTA if update_type == 'delta' else UpdateType.SIMPLE
        proj_enum = {
            'tied_kvq': ProjType.TIED_KVQ,
            'tied_kq': ProjType.TIED_KQ,
            'no_z': ProjType.NO_Z,
            'full': ProjType.FULL,
        }[proj_type]
        nonlin_enum = NonlinType.TANH if nonlin_type == 'tanh' else NonlinType.LINEAR

        # E74 cell - choose diagonal or full matrix, with CUDA if available
        if state_type == 'full':
            # Full matrix state (O(n²) capacity)
            if self.use_cuda and n_state in [1, 2, 4, 8, 16, 32, 48, 64, 96] and proj_type in ['tied_kvq', 'tied_kq', 'no_z']:
                self.cell = E74CUDAFullMatrixCell(
                    dim=d_inner,
                    n_state=n_state,
                    proj_type=proj_enum,
                    nonlin_type=nonlin_enum,
                )
            else:
                self.cell = E74FullMatrixCell(
                    dim=d_inner,
                    n_state=n_state,
                    proj_type=proj_enum,
                    nonlin_type=nonlin_enum,
                    gate_type=GateType.OUTPUT,
                    update_type=update_enum,
                )
        else:
            # Diagonal state (O(n) capacity)
            if self.use_cuda:
                self.cell = E74CUDADiagonalCell(
                    dim=d_inner,
                    n_state=n_state,
                    proj_type=proj_enum,
                    nonlin_type=nonlin_enum,
                    update_type=update_enum,
                    decay=decay,
                )
            else:
                self.cell = E74DiagonalCell(
                    dim=d_inner,
                    n_state=n_state,
                    proj_type=proj_enum,
                    nonlin_type=nonlin_enum,
                    gate_type=GateType.OUTPUT,
                    update_type=update_enum,
                )

        # Out projection
        self.out_proj = nn.Linear(n_state, dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: [T, B, dim]

        Returns:
            output: [T, B, dim]
        """
        # In projection
        h = self.in_proj(x)  # [T, B, d_inner]

        # E74 cell
        out, _ = self.cell(h, None)  # [T, B, n_state]

        # Out projection
        out = self.out_proj(out)  # [T, B, dim]

        return out


def load_data_mmap(path, seq_len, batch_size, device):
    """Load data using memory-mapped file."""
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    data_len = len(mm)

    def get_batch():
        # Random positions
        pos = np.random.randint(0, data_len - seq_len - 1, size=batch_size)
        buf = np.zeros((batch_size, seq_len + 1), dtype=np.int64)
        for i, p in enumerate(pos):
            buf[i] = np.frombuffer(mm[p:p + seq_len + 1], dtype=np.uint8).astype(np.int64)
        return torch.from_numpy(buf).to(device)

    return get_batch, data_len


def main():
    parser = argparse.ArgumentParser(description='Train E74 CUDA model')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--n_state', type=int, default=256)
    parser.add_argument('--expansion', type=float, default=2.0)
    parser.add_argument('--update_type', type=str, default='delta', choices=['delta', 'simple'])
    parser.add_argument('--proj_type', type=str, default='no_z',
                        choices=['tied_kvq', 'tied_kq', 'no_z', 'full'])
    parser.add_argument('--nonlin_type', type=str, default='tanh', choices=['tanh', 'linear'])
    parser.add_argument('--state_type', type=str, default='diagonal', choices=['diagonal', 'full'])
    parser.add_argument('--decay', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--train_minutes', type=float, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--output', type=str, default='output/e74')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA kernels')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / f"e74_{args.update_type}_{args.proj_type}_{args.nonlin_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Model
    model = E74LM(
        vocab_size=256,
        dim=args.dim,
        depth=args.depth,
        n_state=args.n_state,
        expansion=args.expansion,
        update_type=args.update_type,
        proj_type=args.proj_type,
        nonlin_type=args.nonlin_type,
        state_type=args.state_type,
        use_cuda=not args.no_cuda,
        decay=args.decay,
    ).to(device)

    if args.bf16:
        model = model.to(dtype)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: E74 {args.state_type}/{args.update_type}/{args.proj_type}/{args.nonlin_type}")
    print(f"Config: dim={args.dim}, depth={args.depth}, n_state={args.n_state}, state_type={args.state_type}")
    print(f"Parameters: {n_params:,}")

    # Data
    get_batch, data_len = load_data_mmap(args.data, args.chunk_size, args.batch_size, device)
    print(f"Data: {data_len:,} bytes")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    losses = []
    start_time = time.time()
    end_time = start_time + args.train_minutes * 60 if args.train_minutes else None
    tokens_processed = 0

    step = 0
    while True:
        step += 1

        # Check stopping conditions
        if args.train_minutes and time.time() >= end_time:
            break
        if step > args.steps:
            break

        # Get batch
        batch = get_batch()
        if args.bf16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(batch, return_loss=True)
        else:
            loss = model(batch, return_loss=True)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        tokens_processed += args.batch_size * args.chunk_size

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            toks_per_sec = tokens_processed / elapsed
            last_100 = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            print(f"step {step:6d} | loss {loss.item():.4f} | last-100 {last_100:.4f} | "
                  f"{toks_per_sec:,.0f} tok/s | {elapsed/60:.1f}m")

    # Final stats
    elapsed = time.time() - start_time
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    toks_per_sec = tokens_processed / elapsed

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Steps: {step}")
    print(f"Time: {elapsed/60:.2f} minutes")
    print(f"Final last-100 loss: {final_loss:.4f}")
    print(f"Throughput: {toks_per_sec:,.0f} tok/s")
    print(f"{'='*60}")

    # Save results
    results = {
        'config': vars(args),
        'n_params': n_params,
        'final_loss': final_loss,
        'steps': step,
        'elapsed_minutes': elapsed / 60,
        'tok_per_sec': toks_per_sec,
        'all_losses': losses,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pt')


if __name__ == '__main__':
    main()
