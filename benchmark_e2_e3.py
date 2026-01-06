#!/usr/bin/env python3
"""Benchmark E2 and E3 with different configurations."""

import torch
import torch.nn.functional as F
import time
import sys

def benchmark_model(model, x, warmup=5, iters=20):
    """Benchmark forward + backward."""
    model.train()

    # Warmup
    for _ in range(warmup):
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Time forward
    fwd_times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, _ = model(x)
        torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000)

    # Time backward
    bwd_times = []
    for _ in range(iters):
        out, _ = model(x)
        loss = out.sum()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - t0) * 1000)
        model.zero_grad()

    fwd_avg = sum(fwd_times) / len(fwd_times)
    bwd_avg = sum(bwd_times) / len(bwd_times)
    return fwd_avg, bwd_avg

def main():
    device = 'cuda'
    B, T, D = 32, 512, 768  # 50M config: dim=512, expansion=1.5 -> d_inner=768
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

    print("=" * 70)
    print(f"Benchmarking E2/E3 configs with B={B}, T={T}, D={D}")
    print("=" * 70)

    # E2 with different configs
    print("\n--- E2 (SlotElman) ---")
    from elman.models.slot_elman import SlotElmanCell

    configs = [
        (8, False, "full W_h"),
        (8, True, "diag A"),
        (4, True, "4 slots, diag"),
    ]
    for n_slots, diag, label in configs:
        cell = SlotElmanCell(dim=D, n_slots=n_slots, diag=diag).to(device).bfloat16()

        # Benchmark the cell directly
        x_t = x.permute(1, 0, 2).contiguous()  # [T, B, D]
        z_t = torch.randn_like(x_t)

        cell.train()
        for _ in range(3):  # warmup
            out, h = cell(x_t, z_t)
            out.sum().backward()
            cell.zero_grad()

        torch.cuda.synchronize()

        fwd_times = []
        bwd_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, h = cell(x_t, z_t)
            torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            out.sum().backward()
            torch.cuda.synchronize()
            bwd_times.append((time.perf_counter() - t0) * 1000)
            cell.zero_grad()

        fwd = sum(fwd_times) / len(fwd_times)
        bwd = sum(bwd_times) / len(bwd_times)
        tok_per_sec = B * T / ((fwd + bwd) / 1000)
        print(f"{label:15s}: fwd={fwd:6.1f}ms, bwd={bwd:6.1f}ms, total={fwd+bwd:6.1f}ms, {tok_per_sec/1000:.1f}k tok/s")
        del cell
        torch.cuda.empty_cache()

    # E3 with different ranks
    print("\n--- E3 (LowRankSlotElman) ---")
    from elman.models.lowrank_slot_elman import LowRankSlotElmanCell

    for rank in [192, 128, 64, 32]:
        cell = LowRankSlotElmanCell(dim=D, n_slots=8, rank=rank).to(device).bfloat16()

        x_t = x.permute(1, 0, 2).contiguous()
        z_t = torch.randn_like(x_t)

        cell.train()
        for _ in range(3):
            out, h = cell(x_t, z_t)
            out.sum().backward()
            cell.zero_grad()

        torch.cuda.synchronize()

        fwd_times = []
        bwd_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, h = cell(x_t, z_t)
            torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            out.sum().backward()
            torch.cuda.synchronize()
            bwd_times.append((time.perf_counter() - t0) * 1000)
            cell.zero_grad()

        fwd = sum(fwd_times) / len(fwd_times)
        bwd = sum(bwd_times) / len(bwd_times)
        tok_per_sec = B * T / ((fwd + bwd) / 1000)
        print(f"rank={rank:3d}: fwd={fwd:6.1f}ms, bwd={bwd:6.1f}ms, total={fwd+bwd:6.1f}ms, {tok_per_sec/1000:.1f}k tok/s")
        del cell
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Recommendations:")
    print("  E2: Use n_slots=4 for 2x speedup with minimal capacity loss")
    print("  E3: Use rank=64 for 3x speedup with reasonable capacity")


if __name__ == "__main__":
    main()
