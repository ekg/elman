#!/usr/bin/env python3
"""Profile E88 kernel to understand throughput gap vs Mamba2."""

import torch
import torch.nn as nn
import time

def profile_e88_detailed():
    """Profile E88 with detailed timing breakdown."""
    device = 'cuda'

    configs = [
        {'n_heads': 96, 'n_state': 32, 'name': 'h96n32', 'dim': 2048},
        {'n_heads': 40, 'n_state': 64, 'name': 'h40n64', 'dim': 2432},
        {'n_heads': 80, 'n_state': 32, 'name': 'h80n32', 'dim': 2432},
        {'n_heads': 40, 'n_state': 32, 'name': 'h40n32', 'dim': 2560},
    ]

    B, T = 16, 512  # Match benchmark settings

    for cfg in configs:
        n_heads = cfg['n_heads']
        n_state = cfg['n_state']
        name = cfg['name']
        dim = cfg['dim']
        d_inner = n_heads * n_state

        print(f"\n{'='*70}")
        print(f"Profiling {name}: dim={dim}, n_heads={n_heads}, n_state={n_state}")
        print(f"d_inner={d_inner}, state_per_head={n_state*n_state}, total_state={n_heads*n_state*n_state}")
        print(f"{'='*70}")

        try:
            from elman.models.e88_fla_hybrid import E88FLAHybrid

            model = E88FLAHybrid(
                dim=dim,
                n_heads=n_heads,
                n_state=n_state,
                expansion=1.0,
                use_conv=False,
                use_gate=False,
                use_output_norm=False,
            ).to(device).to(torch.bfloat16)

            x = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16)

            # Warmup
            for _ in range(3):
                out, _ = model(x)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()

            # Profile forward
            n_runs = 10
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                out, _ = model(x)
                torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - start) / n_runs * 1000

            # Profile backward
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                x_grad = x.clone().requires_grad_(True)
                out, _ = model(x_grad)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
            total_time = (time.perf_counter() - start) / n_runs * 1000
            bwd_time = total_time - fwd_time

            tokens_per_batch = B * T
            tok_per_sec = tokens_per_batch / (total_time / 1000)

            print(f"Forward:  {fwd_time:.2f} ms")
            print(f"Backward: {bwd_time:.2f} ms")
            print(f"Total:    {total_time:.2f} ms")
            print(f"Throughput: {tok_per_sec:.0f} tok/s")
            print(f"Fwd/Bwd ratio: {fwd_time/bwd_time:.2f}")

            # Memory usage
            torch.cuda.reset_peak_memory_stats()
            x_grad = x.clone().requires_grad_(True)
            out, _ = model(x_grad)
            loss = out.sum()
            loss.backward()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak memory: {peak_mem:.2f} GB")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def profile_mamba2():
    """Profile Mamba2 for comparison."""
    device = 'cuda'
    B, T = 16, 512

    print(f"\n{'='*70}")
    print(f"Profiling Mamba2 for comparison")
    print(f"{'='*70}")

    try:
        from mamba_ssm import Mamba2

        # Match 500M params config
        model = Mamba2(d_model=1600, d_state=128, d_conv=4, expand=2).to(device).to(torch.bfloat16)
        x = torch.randn(B, T, 1600, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            out = model(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()

        # Profile forward
        n_runs = 10
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            out = model(x)
            torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start) / n_runs * 1000

        # Profile backward
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            x_grad = x.clone().requires_grad_(True)
            out = model(x_grad)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / n_runs * 1000
        bwd_time = total_time - fwd_time

        tokens_per_batch = B * T
        tok_per_sec = tokens_per_batch / (total_time / 1000)

        print(f"Forward:  {fwd_time:.2f} ms")
        print(f"Backward: {bwd_time:.2f} ms")
        print(f"Total:    {total_time:.2f} ms")
        print(f"Throughput: {tok_per_sec:.0f} tok/s")
        print(f"Fwd/Bwd ratio: {fwd_time/bwd_time:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    profile_e88_detailed()
    profile_mamba2()
