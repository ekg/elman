#!/usr/bin/env python3
"""
Find maximum batch size for Mamba2LM (400M params) on GPU 2.
Starts at batch_size=32, runs 3 training steps with adaptive batch search.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import mmap
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from mamba_ssm import Mamba2

# Constants
VOCAB_SIZE = 256
DIM = 3264
DEPTH = 6
D_STATE = 128
HEADDIM = 64
SEQ_LEN = 2048
TRAINING_STEPS = 3
INITIAL_BATCH_SIZE = 32
BATCH_INCREMENT = 8

# Data path
DATA_PATH = '/home/erikg/elman/data/pile.txt'


class Mamba2LM(nn.Module):
    """Mamba2 Language Model with weight tying."""

    def __init__(self, vocab_size, dim, depth, d_state=128, headdim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            Mamba2(dim, d_state=d_state, headdim=headdim)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x, return_loss=False):
        if return_loss:
            x, targets = x[:, :-1], x[:, 1:]
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        h = self.norm(h)
        logits = self.head(h)
        if return_loss:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            return loss
        return logits


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_batch_from_mmap(batch_size, seq_len, data_path=DATA_PATH):
    """Load random batch from mmap file."""
    with open(data_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = len(mm)

        # Random positions for each sample in batch
        max_pos = max(0, file_size - seq_len - 1)
        if max_pos <= 0:
            raise ValueError(f"Data file too small: {file_size} < {seq_len}")

        positions = np.random.randint(0, max_pos, size=batch_size)

        # Load sequences
        batch = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
        for i, pos in enumerate(positions):
            data_slice = mm[pos:pos+seq_len+1]
            if len(data_slice) == seq_len + 1:
                batch[i] = np.frombuffer(data_slice, dtype=np.uint8)
            else:
                # Handle edge case near end of file
                batch[i, :len(data_slice)] = np.frombuffer(data_slice, dtype=np.uint8)

        return torch.from_numpy(batch).long()


def get_peak_memory():
    """Get peak GPU memory usage in GB."""
    return torch.cuda.max_memory_allocated() / 1024**3


def train_step(model, optimizer, batch, device):
    """Single training step."""
    batch = batch.to(device)

    optimizer.zero_grad()
    loss = model(batch, return_loss=True)
    loss.backward()
    optimizer.step()

    return loss.item()


def try_batch_size(batch_size, model, optimizer, device):
    """Try training with given batch size."""
    print(f"\n  Trying batch_size={batch_size}...", end=" ", flush=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    try:
        losses = []
        for step in range(TRAINING_STEPS):
            # Load fresh batch
            batch = load_batch_from_mmap(batch_size, SEQ_LEN)
            loss = train_step(model, optimizer, batch, device)
            losses.append(loss)
            print(f".", end="", flush=True)

        peak_mem = get_peak_memory()
        avg_loss = np.mean(losses)
        print(f" OK (loss={avg_loss:.4f}, peak_mem={peak_mem:.2f}GB)")
        return True, peak_mem, avg_loss

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print(f" OOM")
            return False, 0, float('inf')
        else:
            print(f" ERROR: {e}")
            raise


def main():
    print("=" * 80)
    print("Mamba2LM (400M params) - Max Batch Size Finder")
    print("=" * 80)
    print(f"Config: dim={DIM}, depth={DEPTH}, d_state={D_STATE}, headdim={HEADDIM}")
    print(f"Data: {DATA_PATH}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Training steps per batch: {TRAINING_STEPS}")
    print()

    # Device setup
    device = torch.device('cuda:0')
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    print()

    # Model setup
    print("Creating model...")
    model = Mamba2LM(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        depth=DEPTH,
        d_state=D_STATE,
        headdim=HEADDIM
    ).to(device)

    # Count parameters
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Adaptive batch size search
    batch_size = INITIAL_BATCH_SIZE
    max_batch_size = None
    peak_memory = 0
    best_loss = float('inf')
    consecutive_failures = 0

    print("Adaptive batch size search:")
    print("-" * 80)

    while True:
        success, mem, loss = try_batch_size(batch_size, model, optimizer, device)

        if success:
            consecutive_failures = 0
            max_batch_size = batch_size
            peak_memory = mem
            best_loss = loss
            batch_size += BATCH_INCREMENT
        else:
            consecutive_failures += 1
            # Failed: halve batch size
            if batch_size <= 2:
                print(f"  Cannot reduce batch size further!")
                break
            batch_size = max(1, batch_size // 2)

            # If we've tried twice and failed, give up
            if consecutive_failures >= 2:
                print(f"  Stopping search after consecutive failures")
                break

    # Results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    if max_batch_size is not None:
        print(f"Maximum batch size: {max_batch_size}")
        print(f"Peak GPU memory: {peak_memory:.2f}GB")
        print(f"Average loss (3-step): {best_loss:.4f}")
    else:
        print("ERROR: Could not find working batch size!")
    print("=" * 80)

    return max_batch_size, peak_memory


if __name__ == "__main__":
    max_bs, peak_mem = main()
    sys.exit(0 if max_bs is not None else 1)
