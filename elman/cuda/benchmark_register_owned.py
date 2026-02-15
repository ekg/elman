#!/usr/bin/env python3
"""
Benchmark E88 Register-Owned Backward Kernel vs Fused Backward

Compares performance across all supported (n_state, head_v_dim) configurations:
- Square: 4×4, 8×8, 16×16, 24×24, 32×32
- Tall (n_state > head_v_dim): various up to 64×32
- Wide (n_state < head_v_dim): various up to 32×32

Usage:
    python benchmark_register_owned.py [--warmup 10] [--iters 100] [--device 0]
"""

import torch
import torch.cuda
import sys
import argparse
import time

sys.path.insert(0, '.')
import hasty_pytorch_lib as lib


def benchmark_kernel(func, *args, warmup=10, iters=100):
    """Benchmark a kernel with warmup and return mean time in ms."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1000  # ms


def create_inputs(B, T, H, n_state, head_v_dim, device, dtype=torch.bfloat16):
    """Create input tensors for backward pass."""
    k = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
    v = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
    q = torch.randn(B, T, H, n_state, device=device, dtype=dtype)
    decay = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    g = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)
    d_output = torch.randn(B, T, H, head_v_dim, device=device, dtype=dtype)

    state_size = n_state * head_v_dim
    checkpoint_interval = 16
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1

    # S_cache: checkpoints + Sq_cache
    s_checkpoints_size = num_checkpoints * B * H * state_size
    sq_cache_size = B * T * H * head_v_dim
    S_cache = torch.zeros(s_checkpoints_size + sq_cache_size, device=device, dtype=dtype)

    # Output tensors
    d_k = torch.zeros_like(k)
    d_v = torch.zeros_like(v)
    d_q = torch.zeros_like(q)
    d_decay = torch.zeros_like(decay)
    d_g = torch.zeros_like(g)

    # Segment cache
    cache_entry_size = state_size + n_state + head_v_dim + 1
    segment_cache = torch.zeros(B * H, checkpoint_interval * cache_entry_size, device=device, dtype=dtype)

    return {
        'k': k.contiguous(),
        'v': v.contiguous(),
        'q': q.contiguous(),
        'decay': decay.contiguous(),
        'g': g.contiguous(),
        'S_cache': S_cache,
        'd_output': d_output.contiguous(),
        'd_k': d_k,
        'd_v': d_v,
        'd_q': d_q,
        'd_decay': d_decay,
        'd_g': d_g,
        'segment_cache': segment_cache,
        'checkpoint_interval': checkpoint_interval,
    }


def run_register_owned(inputs):
    """Run register-owned backward."""
    lib.e88_register_owned_backward(
        inputs['k'], inputs['v'], inputs['q'],
        inputs['decay'], inputs['g'],
        inputs['S_cache'], inputs['d_output'],
        inputs['d_k'], inputs['d_v'], inputs['d_q'],
        inputs['d_decay'], inputs['d_g'],
        inputs['segment_cache'],
        inputs['checkpoint_interval'], True
    )


def run_fused_backward(inputs, n_state, head_v_dim, B, T, H):
    """Run fused backward (baseline)."""
    # Create state tensors for fused backward
    state_size = n_state * head_v_dim
    checkpoint_interval = inputs['checkpoint_interval']
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1

    S_checkpoints = torch.zeros(num_checkpoints, B, H, n_state, head_v_dim,
                                device=inputs['k'].device, dtype=inputs['k'].dtype)
    Sq_cache = torch.zeros(B, T, H, head_v_dim,
                           device=inputs['k'].device, dtype=inputs['k'].dtype)

    # Note: This requires the fused_backward function to be available
    # We'll use e88_fused_backward if available
    try:
        lib.e88_fused_backward(
            inputs['k'], inputs['v'], inputs['q'],
            inputs['decay'], inputs['g'],
            S_checkpoints, Sq_cache,
            inputs['d_output'],
            inputs['d_k'], inputs['d_v'], inputs['d_q'],
            inputs['d_decay'], inputs['d_g'],
            inputs['segment_cache'],
            checkpoint_interval, True
        )
    except Exception as e:
        return None  # Fused backward not available for this config
    return True


def benchmark_config(n_state, head_v_dim, B=16, T=512, H=98, warmup=10, iters=100, device='cuda:0'):
    """Benchmark a single configuration."""
    torch.manual_seed(42)

    inputs = create_inputs(B, T, H, n_state, head_v_dim, device)

    # Test register-owned
    try:
        reg_time = benchmark_kernel(run_register_owned, inputs, warmup=warmup, iters=iters)
    except Exception as e:
        return None, None, str(e)

    # Test fused backward (baseline) - only works for square sizes 16, 24, 32, 48, etc.
    # The fused kernel prints "unsupported" for non-square or small sizes
    fused_time = None

    # Only benchmark fused for square sizes that it actually supports
    fused_supported_sizes = {16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 80, 96, 128}
    if n_state == head_v_dim and n_state in fused_supported_sizes:
        try:
            import sys
            import os

            # Suppress stderr during fused backward call (it prints unsupported warnings)
            inputs_fused = create_inputs(B, T, H, n_state, head_v_dim, device)

            def run_fused():
                # Redirect stderr to suppress "unsupported" messages
                old_stderr = sys.stderr
                sys.stderr = open(os.devnull, 'w')
                try:
                    lib.e88_fused_backward(
                        inputs_fused['k'], inputs_fused['v'], inputs_fused['q'],
                        inputs_fused['decay'], inputs_fused['g'],
                        inputs_fused['S_cache'],
                        inputs_fused['d_output'],
                        inputs_fused['d_k'], inputs_fused['d_v'], inputs_fused['d_q'],
                        inputs_fused['d_decay'], inputs_fused['d_g'],
                        inputs_fused['segment_cache'],
                        H, True
                    )
                finally:
                    sys.stderr.close()
                    sys.stderr = old_stderr

            fused_time = benchmark_kernel(run_fused, warmup=warmup, iters=iters)

            # Check if fused actually ran (output should be non-zero)
            if inputs_fused['d_k'].abs().max() < 1e-6:
                fused_time = None  # Fused didn't actually run
        except Exception as e:
            fused_time = None

    return reg_time, fused_time, None


def main():
    parser = argparse.ArgumentParser(description='Benchmark E88 Register-Owned Kernel')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--seq', type=int, default=512, help='Sequence length')
    parser.add_argument('--heads', type=int, default=98, help='Number of heads')
    args = parser.parse_args()

    device = f'cuda:{args.device}'

    # All supported configurations
    configs = {
        'Square': [
            (4, 4), (8, 8), (16, 16), (24, 24), (32, 32),
        ],
        'Tall (n > v)': [
            (16, 8), (24, 16), (32, 16), (32, 24),
            (36, 32), (40, 32), (48, 32), (64, 32),
        ],
        'Wide (n < v)': [
            (8, 16), (8, 32), (16, 32), (24, 32),
        ],
    }

    print("=" * 80)
    print("E88 Register-Owned Backward Kernel Benchmark")
    print("=" * 80)
    print(f"Config: B={args.batch}, T={args.seq}, H={args.heads}")
    print(f"Device: {device} ({torch.cuda.get_device_name(args.device)})")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print("=" * 80)
    print()

    results = []

    for category, cfgs in configs.items():
        print(f"\n{category}:")
        print("-" * 60)
        print(f"{'n_state':>8} {'head_v_dim':>10} {'Reg-Owned (ms)':>15} {'Fused (ms)':>12} {'Speedup':>10}")
        print("-" * 60)

        for n_state, head_v_dim in cfgs:
            reg_time, fused_time, error = benchmark_config(
                n_state, head_v_dim,
                B=args.batch, T=args.seq, H=args.heads,
                warmup=args.warmup, iters=args.iters,
                device=device
            )

            if error:
                print(f"{n_state:>8} {head_v_dim:>10} {'ERROR':>15} {'-':>12} {'-':>10}")
                print(f"         Error: {error[:50]}")
            else:
                fused_str = f"{fused_time:.3f}" if fused_time else "-"
                speedup_str = f"{fused_time/reg_time:.2f}x" if fused_time else "-"
                print(f"{n_state:>8} {head_v_dim:>10} {reg_time:>15.3f} {fused_str:>12} {speedup_str:>10}")

                results.append({
                    'category': category,
                    'n_state': n_state,
                    'head_v_dim': head_v_dim,
                    'reg_time': reg_time,
                    'fused_time': fused_time,
                    'speedup': fused_time / reg_time if fused_time else None,
                })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find fastest and slowest
    if results:
        fastest = min(results, key=lambda x: x['reg_time'])
        slowest = max(results, key=lambda x: x['reg_time'])

        print(f"Fastest: {fastest['n_state']}×{fastest['head_v_dim']} at {fastest['reg_time']:.3f} ms")
        print(f"Slowest: {slowest['n_state']}×{slowest['head_v_dim']} at {slowest['reg_time']:.3f} ms")

        # Speedup summary for 32×32
        for r in results:
            if r['n_state'] == 32 and r['head_v_dim'] == 32 and r['speedup']:
                print(f"\n32×32 speedup vs fused: {r['speedup']:.2f}x")
                print(f"  Register-owned: {r['reg_time']:.3f} ms")
                print(f"  Fused baseline: {r['fused_time']:.3f} ms")


if __name__ == '__main__':
    main()
