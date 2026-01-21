# E88 FLA Hybrid: Optimal n_state Scaling Analysis

**Date:** January 21, 2026
**Hardware Focus:** RTX 4090, A100, H100

## Executive Summary

Based on analysis of the E88 FLA Hybrid CUDA kernel and benchmark results, smaller n_state (32-48) yields 13-15x faster throughput than larger n_state (72+) on RTX 4090 due to L1 cache fitting.

## Key Finding: Memory Hierarchy Cliff

| n_state | State Size | Fits L1? | Throughput |
|---------|------------|----------|------------|
| 32 | 4 KB | Yes | 18-21K tok/s |
| 64 | 16 KB | Yes | ~10K tok/s |
| 72 | 20 KB | Borderline | 1.4K tok/s |
| 96 | 36 KB | No | <1K tok/s |

## Hardware-Specific Recommendations

### RTX 4090 (Consumer GPU)
| Model Size | n_state | n_heads | Reasoning |
|------------|---------|---------|-----------|
| 100M | 32 | 8-16 | Fast backward, fits shared memory |
| 500M | 32-48 | 40-80 | Balance throughput and capacity |
| 1B+ | 48 | 64-96 | Maximize state without global fallback |

### A100 (Datacenter GPU)
| Model Size | n_state | n_heads | Reasoning |
|------------|---------|---------|-----------|
| 100M | 48 | 8-16 | Good compute utilization |
| 500M | 64 | 32-48 | Balanced for 164KB shared mem |
| 1B+ | 64-96 | 32-64 | Can handle larger states |

### Summary Table by Hardware

| GPU | Memory BW | Tensor TFLOPS | Optimal n_state | Max n_state |
|-----|-----------|---------------|-----------------|-------------|
| RTX 3090 | 936 GB/s | 35.6 | 32 | 48 |
| RTX 4090 | 1008 GB/s | 82.6 | 32-48 | 64 |
| A100 40GB | 1555 GB/s | 312 | 48-64 | 96 |
| A100 80GB | 2039 GB/s | 312 | 64 | 96 |
| H100 | 3350 GB/s | 989 | 64-96 | 128 |

## Heuristic Formula

```python
def optimal_n_state(memory_bandwidth_GBps, tensor_tflops, shared_mem_KB):
    ops_per_byte = tensor_tflops * 1e12 / (memory_bandwidth_GBps * 1e9)
    max_n_from_shared = int(math.sqrt(shared_mem_KB * 1024 / 4 / 2.2))

    if ops_per_byte < 100:  # Memory-bound GPU (RTX 4090)
        optimal_n = min(48, max_n_from_shared)
    else:  # Compute-rich GPU (A100, H100)
        optimal_n = min(64, max_n_from_shared)

    return (optimal_n // 8) * 8  # Round to multiple of 8
```

## General Scaling Rules

1. **Keep n_state <= 48 on consumer GPUs** for fast backward pass
2. **Use n_state=64 on A100/H100** where tensor cores amortize memory latency
3. **Prefer more heads over larger n_state** when throughput matters
4. **Use expansion=1.0** to keep state square (n_state = head_v_dim)
5. **Checkpoint interval of 32** balances memory vs recomputation

## Why This Matters

The backward pass is the bottleneck (4-5x slower than forward). With n_state=32:
- State fits entirely in L1 cache (4KB)
- All reductions stay in registers/shared memory
- No spills to L2 or global memory

With n_state=96:
- State size 72KB+ exceeds preferred shared memory
- Spills cause 13-15x slowdown
