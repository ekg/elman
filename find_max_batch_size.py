#!/usr/bin/env python3
"""
Find maximum batch size for E1 (level=1) at ~400M params on GPU 0.

Config: LadderLM with level=1, dim=3584, depth=6
Target: ~386M params

Strategy:
1. Start with batch_size=64
2. Run 3 training steps
3. If OOM, halve batch size and retry
4. If success, try batch_size + 16
5. Keep narrowing until max working batch size is found
"""

import os
import sys
import mmap
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Force GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add elman package to path
sys.path.insert(0, '/home/erikg/elman')

from elman.models.ladder_lm import LadderLM

def get_batch_from_mmap(data_path, batch_size, seq_len=512):
    """Load a batch using mmap."""
    with open(data_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Get random positions
        max_pos = len(mm) - seq_len - 1
        if max_pos <= 0:
            raise ValueError(f"Data file too small: {len(mm)} bytes")

        positions = np.random.randint(0, max_pos, size=batch_size)

        # Load sequences
        batch = []
        for pos in positions:
            seq = mm[pos:pos + seq_len + 1]  # +1 for target
            tokens = np.frombuffer(seq, dtype=np.uint8)
            batch.append(tokens)

        return torch.tensor(np.stack(batch), dtype=torch.long)


def try_training_step(model, batch, device, batch_size):
    """Try a single training step, return True if successful."""
    try:
        batch = batch.to(device)

        # Forward pass with loss
        loss = model(batch, return_loss=True)

        # Backward pass
        loss.backward()

        # Clear cache to check peak memory
        torch.cuda.synchronize()

        return True, loss.item()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, None
        raise


def find_max_batch_size(data_path, model, device, seq_len=512):
    """Binary search to find maximum batch size."""

    print("=" * 70)
    print("FINDING MAXIMUM BATCH SIZE FOR E1 (level=1) AT ~400M PARAMS")
    print("=" * 70)

    # Model info
    num_params = model.get_num_params()
    print(f"\nModel Config:")
    print(f"  Level: 1 (Mamba-Gated Elman)")
    print(f"  Dim: {model.dim}")
    print(f"  Depth: {model.depth}")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  Device: {device}")
    print(f"  Dtype: bfloat16")

    # Start with batch_size=64
    batch_size = 64
    last_working_batch_size = None

    print(f"\nData: {data_path}")
    print(f"Sequence length: {seq_len}")
    print(f"\nStarting batch size search from {batch_size}...\n")

    step_count = 0
    while True:
        step_count += 1

        print(f"[Attempt {step_count}] Testing batch_size={batch_size}... ", end='', flush=True)

        try:
            # Load batch
            batch = get_batch_from_mmap(data_path, batch_size, seq_len)

            # Reset model gradients
            if model.training:
                model.zero_grad()

            # Try 3 training steps
            success_count = 0
            avg_loss = 0

            for step in range(3):
                torch.cuda.reset_peak_memory_stats(device)

                success, loss = try_training_step(model, batch, device, batch_size)

                if success:
                    success_count += 1
                    avg_loss += loss
                    print('.', end='', flush=True)
                else:
                    print('X', end='', flush=True)
                    break

            if success_count == 3:
                # Success! Record this batch size and try bigger
                last_working_batch_size = batch_size
                avg_loss /= 3

                # Get peak memory
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_mem_gb = peak_mem_mb / 1024

                print(f" SUCCESS")
                print(f"       Loss: {avg_loss:.4f}, Peak Memory: {peak_mem_gb:.2f} GB")

                # Try bigger batch size
                batch_size += 16

            else:
                # Failed - halve batch size
                print(f" FAILED (OOM after {success_count}/3 steps)")
                batch_size = batch_size // 2

                # If we've already found a working size and now halved, we're done
                if last_working_batch_size is not None:
                    if batch_size < last_working_batch_size:
                        print(f"\nSearching between {last_working_batch_size} and {last_working_batch_size + 16}...")

                        # Fine-grained search
                        for test_bs in range(last_working_batch_size + 1, min(last_working_batch_size + 17, batch_size + 17)):
                            print(f"[Fine-grained] Testing batch_size={test_bs}... ", end='', flush=True)

                            batch = get_batch_from_mmap(data_path, test_bs, seq_len)
                            model.zero_grad()

                            for step in range(3):
                                torch.cuda.reset_peak_memory_stats(device)
                                success, loss = try_training_step(model, batch, device, test_bs)
                                if not success:
                                    print('X')
                                    break
                            else:
                                if success:
                                    print('SUCCESS')
                                    last_working_batch_size = test_bs
                                    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                                    peak_mem_gb = peak_mem_mb / 1024
                                    final_avg_loss = loss

                        break

        except Exception as e:
            print(f" ERROR: {e}")
            if last_working_batch_size is not None:
                break
            # Halve and retry
            batch_size = batch_size // 2

        # Safeguard: stop if batch size gets too small
        if batch_size < 1:
            break

    print("\n" + "=" * 70)
    if last_working_batch_size is not None:
        print(f"RESULT: Maximum batch size = {last_working_batch_size}")
        print(f"Peak memory usage: {peak_mem_gb:.2f} GB (out of 48GB available)")
        print("=" * 70)
        return last_working_batch_size, peak_mem_gb
    else:
        print("RESULT: Could not find a working batch size")
        print("=" * 70)
        return None, None


def main():
    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    # Create model
    print("\nCreating E1 model (level=1, dim=3584, depth=6)...")
    model = LadderLM(
        vocab_size=256,
        dim=3584,
        depth=6,
        level=1,  # Mamba-Gated Elman
        expansion=1.0,
        dropout=0.0,
    ).to(device).bfloat16()

    model.train()

    # Data path
    data_path = '/home/erikg/elman/data/pile.txt'
    if not Path(data_path).exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    # Find max batch size
    max_bs, peak_mem_gb = find_max_batch_size(data_path, model, device, seq_len=512)

    if max_bs is not None:
        print(f"\n✓ Test completed successfully")
        print(f"  Max batch size: {max_bs}")
        print(f"  Peak GPU memory: {peak_mem_gb:.2f} GB")
    else:
        print(f"\n✗ Test failed - could not find working batch size")


if __name__ == '__main__':
    main()
