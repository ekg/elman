#!/usr/bin/env python3
"""Profile E88 CUDA kernels with torch.profiler - no special permissions needed."""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from elman.models import LadderLM
import json

# CMA-ES optimal config
CONFIG = {
    'dim': 1920,
    'depth': 17,
    'n_heads': 83,
    'n_state': 32,
    'use_gate': True,
    'gate_activation': 'silu',
}

def main():
    device = torch.device('cuda')

    # Create model with optimal config
    model = LadderLM(
        vocab_size=256,
        dim=CONFIG['dim'],
        depth=CONFIG['depth'],
        level='E88',
        n_heads=CONFIG['n_heads'],
        n_state=CONFIG['n_state'],
        use_gate=CONFIG['use_gate'],
        gate_activation=CONFIG['gate_activation'],
    ).to(device).to(torch.bfloat16)

    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params/1e6:.1f}M")
    print(f"Config: {CONFIG}")

    # Create input
    batch_size = 8
    seq_len = 512
    x = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Warmup
    print("Warmup...")
    for _ in range(10):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, return_loss=True)
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Profile with torch.profiler
    print("\nProfiling with torch.profiler...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(5):
            with record_function("forward"):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(x, return_loss=True)

            with record_function("backward"):
                loss.backward()

            model.zero_grad()

    torch.cuda.synchronize()

    # Print CUDA kernel summary
    print("\n" + "="*80)
    print("CUDA KERNEL TIME SUMMARY (sorted by CUDA time)")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n" + "="*80)
    print("CUDA MEMORY SUMMARY")
    print("="*80)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    # Export chrome trace for detailed analysis
    prof.export_chrome_trace("e88_profile_trace.json")
    print("\nChrome trace exported to: e88_profile_trace.json")
    print("Open chrome://tracing and load the file for visual analysis")

    # Get detailed kernel stats
    print("\n" + "="*80)
    print("DETAILED CUDA KERNEL ANALYSIS")
    print("="*80)

    cuda_events = []
    for event in prof.key_averages():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            cuda_events.append({
                'name': event.key,
                'cuda_time_us': event.cuda_time_total,
                'cpu_time_us': event.cpu_time_total,
                'count': event.count,
                'flops': event.flops,
                'self_cuda_mem': event.self_cuda_memory_usage,
            })

    # Sort by CUDA time
    cuda_events.sort(key=lambda x: x['cuda_time_us'], reverse=True)

    total_cuda_time = sum(e['cuda_time_us'] for e in cuda_events)

    print(f"\nTotal CUDA time: {total_cuda_time/1000:.2f} ms")
    print(f"\nTop 15 CUDA kernels by time:")
    print("-"*100)
    print(f"{'Kernel':<60} {'Time (ms)':>10} {'%':>6} {'Count':>6} {'Avg (us)':>10}")
    print("-"*100)

    for e in cuda_events[:15]:
        pct = 100 * e['cuda_time_us'] / total_cuda_time if total_cuda_time > 0 else 0
        avg = e['cuda_time_us'] / e['count'] if e['count'] > 0 else 0
        name = e['name'][:58] if len(e['name']) > 58 else e['name']
        print(f"{name:<60} {e['cuda_time_us']/1000:>10.3f} {pct:>5.1f}% {e['count']:>6} {avg:>10.1f}")

    # Categorize kernels
    print("\n" + "="*80)
    print("KERNEL CATEGORY BREAKDOWN")
    print("="*80)

    categories = {
        'E88 Forward': [],
        'E88 Backward': [],
        'GEMM/MatMul': [],
        'Normalization': [],
        'Elementwise': [],
        'Memory': [],
        'Other': [],
    }

    for e in cuda_events:
        name = e['name'].lower()
        if 'e88' in name and 'backward' in name:
            categories['E88 Backward'].append(e)
        elif 'e88' in name or 'fla_hybrid' in name:
            categories['E88 Forward'].append(e)
        elif 'gemm' in name or 'matmul' in name or 'cutlass' in name or 'cublas' in name:
            categories['GEMM/MatMul'].append(e)
        elif 'norm' in name or 'layer_norm' in name or 'rms' in name:
            categories['Normalization'].append(e)
        elif 'elementwise' in name or 'add' in name or 'mul' in name or 'copy' in name:
            categories['Elementwise'].append(e)
        elif 'memcpy' in name or 'memset' in name:
            categories['Memory'].append(e)
        else:
            categories['Other'].append(e)

    print(f"\n{'Category':<20} {'Time (ms)':>12} {'%':>8} {'Kernels':>10}")
    print("-"*60)
    for cat, events in categories.items():
        cat_time = sum(e['cuda_time_us'] for e in events)
        pct = 100 * cat_time / total_cuda_time if total_cuda_time > 0 else 0
        print(f"{cat:<20} {cat_time/1000:>12.3f} {pct:>7.1f}% {len(events):>10}")

    # E88 kernel specific analysis
    print("\n" + "="*80)
    print("E88 KERNEL DETAILED BREAKDOWN")
    print("="*80)

    e88_kernels = categories['E88 Forward'] + categories['E88 Backward']
    if e88_kernels:
        e88_total = sum(e['cuda_time_us'] for e in e88_kernels)
        print(f"\nE88 kernels total: {e88_total/1000:.3f} ms ({100*e88_total/total_cuda_time:.1f}% of total)")
        print(f"\n{'Kernel':<50} {'Time (ms)':>10} {'%E88':>8} {'Count':>6}")
        print("-"*80)
        for e in sorted(e88_kernels, key=lambda x: x['cuda_time_us'], reverse=True):
            pct = 100 * e['cuda_time_us'] / e88_total if e88_total > 0 else 0
            name = e['name'][:48] if len(e['name']) > 48 else e['name']
            print(f"{name:<50} {e['cuda_time_us']/1000:>10.3f} {pct:>7.1f}% {e['count']:>6}")

if __name__ == '__main__':
    main()
