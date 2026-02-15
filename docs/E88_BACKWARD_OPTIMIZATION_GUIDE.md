# E88 Backward Kernel Optimization Guide

**Date**: 2026-02-14
**Synthesized from**: 5 parallel investigation agents (memory layout, algorithmic restructuring, parallelization, numerical precision, hardware-specific optimization)

---

## Executive Summary

The E88 backward kernel (`e88_warp_backward_simple_gpu.cu.cc`) currently accounts for **26.4% of total training step time** (130.3ms per step at 480M scale). Even infinite kernel speedup would yield at most 1.5x overall training speedup (the kernel is 33% of total time when combined with the forward pass). However, the backward kernel is **4x slower than the forward kernel** (130ms vs 31.5ms), suggesting significant optimization headroom.

**Best single approach**: Chunk-based backward prefetching (mirroring forward kernel), expected **20-40% backward kernel speedup**.

**Best combined approach**: Chunk prefetching + async copy + adaptive thread count = **35-55% backward kernel speedup** → **~10-15% end-to-end training speedup**.

**Gradient precision is NOT a concern**: bf16 round-trip errors (~1-5% relative) are 20-100x smaller than mini-batch noise and absorbed by gradient clipping. Both fused and warp_simple kernels produce equivalent training results.

---

## Findings Matrix

| # | Approach | Source | Expected Backward Speedup | Implementation Effort | Risk | Combinable? |
|---|----------|--------|--------------------------|----------------------|------|-------------|
| 1 | Chunk-based backward prefetching | Hardware + Algorithmic | **20-40%** | Medium | Low | Yes |
| 2 | Async copy (`cp.async`) | Hardware | **15-25%** | Medium | Low | Yes (w/ #1) |
| 3 | Parallel reduction for dot products | Parallelization | **20-30%** | Medium | Medium | Yes |
| 4 | Adaptive thread count | Parallelization | **10-15%** | Low | Low | Yes |
| 5 | Double-buffered segment cache | Algorithmic | **50-100%** (for fwd replay) | Medium | Low | Yes (w/ #1) |
| 6 | Fuse d_k + d_k_from_retrieved | Parallelization | **~5%** | Low | Low | Yes |
| 7 | Reduce `__syncthreads()` count | Hardware + Parallelization | **5-10%** | Low | Medium | Yes |
| 8 | L2 cache persistence for checkpoints | Hardware | **5-10%** | Low | Low | Yes |
| 9 | n_state=16 specialized kernel | Parallelization | **2-3x** for n=16 only | Medium | Low | N/A |
| 10 | Tensor core WMMA | Hardware | **<10%** | High | Medium | No (matrices too small) |

---

## Current Kernel Bottlenecks

### Thread Utilization Problem (from Parallelization Investigation)

The backward kernel has **severe thread underutilization**. With n_state=32 and 256 threads:

| Phase | Active Threads | Utilization |
|-------|---------------|-------------|
| State load/save (`for idx = tid; idx < state_size; idx += num_threads`) | 256/256 | 100% |
| k/v/q loads (`if (tid < N_STATE)`) | 32/256 | **12.5%** |
| Dot products for delta, d_q, d_k, d_k_from_retrieved | 32/256 | **12.5%** |
| Decay reduction | 256/256 | 100% |
| dS update | 256/256 | 100% |

**7 of 12 compute phases run at 12.5% utilization** — the dominant bottleneck.

### Memory Access Pattern (from Memory Layout + Hardware)

The backward kernel performs **per-timestep global memory loads** in both forward replay and backward phases. This was a correctness fix (eliminating stale prefetch buffer bugs) but is the primary performance regression vs. the forward kernel which uses chunk prefetching.

Per timestep, the backward kernel issues:
- Forward replay: 3 global loads (k, v, decay) + 3 segment cache writes (k, v, decay) + 1 segment cache write (S)
- Backward: 3 segment cache loads (S, k, v) + 2 global loads (q, d_output) + 1 global load (g) + 5 global writes (d_k, d_v, d_q, d_decay, d_g)

**Total**: ~17 global memory operations per timestep vs. ~3 in the chunk-prefetched forward kernel.

### Synchronization Overhead

Current kernel: **10-12 `__syncthreads()` per timestep** in backward phase (15-19 including forward replay).
Forward kernel: **2-3 `__syncthreads()` per timestep** (chunk-based design).

---

## Implementation Guide

### Approach 1: Chunk-Based Backward Prefetching (Priority: HIGHEST)

**Rationale**: The forward kernel achieved 2.87x speedup primarily through chunk prefetching (16 timesteps loaded at once into shared memory). The backward kernel reverted to per-timestep loads for correctness. The fix is to prefetch correctly, not to abandon prefetching.

**What to change in `e88_warp_backward_simple_gpu.cu.cc`:**

1. **Add shared memory chunk buffers** (matching forward kernel pattern):
```
// New shared memory layout for backward:
float* k_chunk[CHUNK_SIZE * N_STATE]     // 16 × 32 = 512 floats
float* v_chunk[CHUNK_SIZE * HEAD_V_DIM]  // 16 × 32 = 512 floats
float* q_chunk[CHUNK_SIZE * N_STATE]     // 16 × 32 = 512 floats
float* d_out_chunk[CHUNK_SIZE * HEAD_V_DIM]  // 16 × 32 = 512 floats
float* g_chunk[CHUNK_SIZE * HEAD_V_DIM]  // 16 × 32 = 512 floats (if gated)
float* decay_chunk[CHUNK_SIZE]           // 16 floats
// Total additional: ~2560 floats = 10 KB (fits in 48 KB budget)
```

2. **Bulk load all inputs at segment start** (before backward loop):
   - After forward replay completes, all k/v/decay are already in segment cache
   - Bulk load q, d_output, g from global memory for the entire segment
   - This eliminates per-timestep global loads in the backward loop

3. **Backward loop reads from shared memory only**:
   - Replace `q_all[k_offset + tid]` → `q_chunk[local_t * N_STATE + tid]`
   - Replace `d_output[v_offset + tid]` → `d_out_chunk[local_t * HEAD_V_DIM + tid]`
   - Replace `g_all[v_offset + tid]` → `g_chunk[local_t * HEAD_V_DIM + tid]`

4. **Shared memory budget check** (n_state=32, HEAD_V_DIM=32):
   - Current: 4 × 1024 + 32 + 32 + 32 + 32 + 32 + 32 + 32 + 32 + 8 = 4,360 floats = 17.4 KB
   - New chunk buffers: 512 + 512 + 512 + 512 + 512 + 16 = 2,576 floats = 10.3 KB
   - Total: 27.7 KB — fits in 48 KB default shared memory limit

**Expected speedup**: 20-40% for backward kernel (eliminates ~10 per-timestep global memory operations).

**Risk**: Low. Same data, same computation order, just bulk-loaded.

### Approach 2: Adaptive Thread Count (Priority: HIGH, combine with #1)

**Rationale**: 256 threads with n_state=32 means 87.5% of threads idle during guarded phases. Reducing to 128 threads doubles per-thread work for state operations while maintaining full utilization for guarded phases.

**What to change:**

1. **In dispatch function**, change thread count formula:
```cpp
// Current:
int threads_per_block = min(256, state_size);

// New:
int threads_per_block = min(256, max(64, n_state * 4));
// n_state=16 → 64 threads (25% guarded utilization)
// n_state=32 → 128 threads (25% guarded utilization)
// n_state=48 → 192 threads
// n_state=64 → 256 threads
```

2. **No kernel code changes needed** — the `for (int idx = tid; idx < state_size; idx += num_threads)` pattern already handles variable thread counts.

**Expected speedup**: 10-15% for n_state≤32 (eliminates idle thread scheduling overhead).

**Risk**: Very low. Purely a launch configuration change.

### Approach 3: Fuse d_k + d_k_from_retrieved (Priority: MEDIUM, combine with #1 and #2)

**Rationale**: Currently d_k is computed in two phases with a `__syncthreads()` between them. These can be fused into a single loop.

**What to change in backward loop:**

```cpp
// CURRENT (two phases, extra sync):
if (tid < N_STATE) {
    float d_k_local = 0.0f;
    for (int j = 0; j < HEAD_V_DIM; j++) {
        float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh_buf[tid * HEAD_V_DIM + j];
        d_k_local += d_pre * delta_buf[j];
    }
    d_k_buf[tid] = d_k_local;
}
__syncthreads();
// ... decay reduction ...
if (tid < N_STATE) {
    float d_k_from_retrieved = 0.0f;
    for (int j = 0; j < HEAD_V_DIM; j++) {
        d_k_from_retrieved += S[tid * HEAD_V_DIM + j] * (-d_delta_buf[j]);
    }
    d_k_buf[tid] += d_k_from_retrieved;
}

// FUSED (single phase, one fewer sync):
if (tid < N_STATE) {
    float d_k_local = 0.0f;
    for (int j = 0; j < HEAD_V_DIM; j++) {
        float d_pre = dS[tid * HEAD_V_DIM + j] * dtanh_buf[tid * HEAD_V_DIM + j];
        d_k_local += d_pre * delta_buf[j];
        d_k_local += S[tid * HEAD_V_DIM + j] * (-d_delta_buf[j]);
    }
    d_k_buf[tid] = d_k_local;
}
```

**Note**: This requires d_delta_buf to be fully computed before d_k starts, which is already guaranteed by the existing sync at line 271.

**Expected speedup**: ~5% (one fewer sync + better ILP).

---

## What NOT to Pursue

### Tensor Core WMMA (LOW priority)
The 32×32 state matrices are too small for efficient WMMA use. Tensor cores need aligned 16×16 tiles minimum, and the overhead of setup/teardown dominates for such small matrices. Only worthwhile if n_state≥64.

### Tiled/Chunked Processing (Strategy D from Algorithmic report)
Processing 2-4 timesteps together in the backward pass has only 1.1-1.2x expected speedup with medium implementation effort. The math for multi-timestep backward accumulation is complex and error-prone.

### Split Forward Replay (Strategy A from Algorithmic report)
Running forward replay as a separate parallel kernel would use 1.6 GB extra memory — prohibitive at 480M scale.

---

## End-to-End Context

From the REAL_INEFFICIENCIES report, the backward kernel is only **26.4% of training time**. The full optimization picture:

| Optimization | Component | Time Saved | Overall Speedup |
|-------------|-----------|------------|-----------------|
| **Backward kernel chunk prefetch** | E88 backward | 26-52ms | **5-10%** |
| Enable fused projection for training | GEMM overhead | ~40ms | **8%** |
| Fuse L2 norm into E88 kernel | L2 norm | ~15ms | **3%** |
| Fix Triton L2 norm (use PyTorch) | L2 norm | ~10ms | **2%** |
| Fuse a_proj + g_proj with qkv_proj | GEMM launches | ~5ms | **1%** |
| Change layout to [T, B, dim] | Transpose copies | ~12ms | **2.5%** |

**Combined potential**: ~20-25% end-to-end training speedup, from 493ms to ~380ms per step.

**The backward kernel optimization is the single highest-impact change**, but the non-kernel optimizations (fused projection, L2 norm) collectively matter more than the kernel alone.

---

## Recommended Next Steps

### Immediate (create workgraph tasks):

1. **`implement-chunk-backward`**: Implement chunk-based prefetching in backward kernel
   - Copy forward kernel's chunk loading pattern
   - Bulk load q, d_output, g at segment start
   - Backward loop reads from shared memory only
   - Verify against existing kernel with gradient comparison test

2. **`adaptive-thread-backward`**: Change thread count in dispatch function
   - Simple formula: `min(256, max(64, n_state * 4))`
   - Benchmark n_state=16 and n_state=32 before/after

3. **`fuse-dk-computation`**: Merge d_k and d_k_from_retrieved into single loop
   - Simple code change, verify gradients match

### Follow-up (after kernel optimizations):

4. **`enable-fused-projection-training`**: Remove `not self.training` guard from fused projection
   - Requires autograd custom function wrapping
   - 8% end-to-end speedup potential

5. **`fix-l2-norm`**: Replace Triton L2 norm with PyTorch or fuse into E88 kernel
   - Quick win: just use `F.normalize()` instead of Triton kernel

---

## Appendix: Investigation Source Summary

| Investigation | Key Finding |
|--------------|-------------|
| **Memory Layout** | Per-timestep global loads are the correctness fix but major perf regression; chunk prefetching is safe |
| **Algorithmic Restructuring** | Double-buffered segment cache (Strategy B) is recommended; split kernels too memory-heavy |
| **Parallelization** | 87.5% thread idleness during guarded phases; parallel reduction could help dot products |
| **Numerical Precision** | bf16 gradient errors are acceptable noise (20-100x smaller than mini-batch noise); use fast kernel with confidence |
| **Hardware-Specific** | `cp.async` unused in codebase; chunk prefetching is the #1 opportunity; Ada has 100KB shared mem (83KB unused) |
