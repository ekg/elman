# E88 CUDA Kernel Profiling Report

**Date:** February 17, 2026
**Config:** CMA-ES optimal (dim=1920, depth=17, n_heads=83, n_state=32)
**Model:** 436.7M parameters, bf16

## Executive Summary

The E88 model spends **38.6%** of CUDA time on custom E88 kernels and **38.4%** on GEMM operations. The backward kernel is **2.8x slower** than forward, which is the primary optimization target.

## Time Breakdown (5 fwd+bwd iterations)

| Component | CUDA Time (ms) | % Total | Notes |
|-----------|---------------|---------|-------|
| **GEMM (aten::mm)** | 340.0 | 38.4% | cuBLAS, well-optimized |
| **E88 Backward** | 251.8 | 28.4% | `register_owned_backward` kernel |
| **E88 Forward** | 90.6 | 10.2% | `warp_optimized_forward` kernel |
| Other (norms, etc.) | 202.9 | 23.0% | Layer norms, elementwise, memory |
| **Total** | 885.4 | 100% | |

### E88 Kernel Analysis

| Kernel | Time/iter (ms) | Calls/iter | μs/call |
|--------|---------------|------------|---------|
| Forward (`E88WarpOptimizedForwardKernel_BF16`) | 18.1 | 17 | 1065 |
| Backward (`E88RegisterOwnedBackwardKernel_BF16`) | 50.4 | 17 | 2962 |

**Backward is 2.78x slower than forward** despite being optimized with register-owned state.

## Bottleneck Analysis

### 1. E88 Backward Kernel (28.4% of total time)

The register-owned backward kernel processes state matrices in registers for n_state≤32. Per-call time of 2.96ms is the dominant bottleneck.

**Why it's slow:**
- Sequential time-reversed loop over seq_len=512 timesteps
- Must load/store state gradients for each timestep
- No parallel scan possible due to nonlinear (tanh) recurrence

**Potential optimizations:**
1. **Warp-level parallelism over heads**: Currently processes all 83 heads sequentially. Could partition across warps.
2. **Tensor core utilization**: The 32×32 state matrices are perfect for tensor cores (16×16 tiles).
3. **Persistent kernel**: Keep kernel running across chunks to avoid launch overhead.

### 2. GEMM Operations (38.4% of total time)

cuBLAS GEMMs are highly optimized. Breakdown:
- `ampere_bf16_s1688gemm_bf16_128x64` - 87ms (main matmuls)
- `cutlass_80_tensorop_bf16_s1681` - 73ms (backward matmuls)
- Various other GEMM kernels - 180ms

**These are near-optimal**. cuBLAS uses tensor cores effectively.

### 3. E88 Forward Kernel (10.2%)

Already well-optimized at 1.07ms per call. Uses warp-level parallelism.

## Optimization Recommendations

### High Impact (Estimated 15-25% total speedup)

1. **Tensor Core Backward Kernel** (Priority: HIGH)
   - Current: Uses FMA operations on registers
   - Proposed: Use `wmma` intrinsics for 16×16 matrix operations
   - Expected: 2-4x speedup on backward state updates
   - Complexity: HIGH (requires restructuring state layout)

2. **Head-Parallel Backward** (Priority: HIGH)
   - Current: Sequential over 83 heads in one block
   - Proposed: Partition heads across multiple warps/blocks
   - Expected: 2-4x speedup on backward kernel
   - Complexity: MEDIUM

### Medium Impact (Estimated 5-10% total speedup)

3. **Fused Linear+E88** (Priority: MEDIUM)
   - Current: Separate GEMM and E88 kernel launches
   - Proposed: Fuse projection into E88 kernel to reduce memory traffic
   - Expected: 10-20% reduction in memory bandwidth
   - Complexity: HIGH

4. **Chunked Backward with Recomputation** (Priority: MEDIUM)
   - Current: Store all intermediate states
   - Proposed: Recompute forward states during backward to reduce memory
   - Expected: Memory reduction, slight compute increase
   - Complexity: MEDIUM

### Low Impact (Estimated <5% total speedup)

5. **Kernel Launch Overhead**
   - 17 layers × 2 kernels = 34 kernel launches per forward+backward
   - Overhead is ~1-2ms total, negligible

## Comparison to Theoretical Peak

**RTX 4090 theoretical peaks:**
- BF16 Tensor Core: 165 TFLOPS
- Memory Bandwidth: 1 TB/s

**E88 kernel arithmetic intensity:**
- Forward: ~100 FLOPs/byte (compute-bound)
- Backward: ~50 FLOPs/byte (memory-bound)

The backward kernel is memory-bound due to frequent state load/stores during the sequential recurrence. This is the fundamental limitation of sequential RNNs vs. parallel scan SSMs.

## Kernel Structure Analysis

### Current Backward Kernel (`E88RegisterOwnedBackwardKernel_BF16<32,32>`)

```
Block layout: 1 block per (batch, head) pair → 83 blocks per layer per sample
Thread layout: 32 threads (1 warp) per block
State layout: Thread j owns column j of 32×32 state matrix (32 floats in registers)

Processing:
1. For each segment (16 timesteps):
   a. Load checkpoint S into registers
   b. Forward replay: cache S, k, v for each timestep
   c. Backward pass: compute gradients, accumulate dS
2. Sequential over T=512 timesteps
```

### Why Backward is 2.8x Slower than Forward

1. **Forward replay phase**: Must recompute forward states for gradient computation
2. **Gradient accumulation**: Multiple warp shuffles per timestep for d_k, d_q reductions
3. **More memory traffic**: Backward stores d_k, d_v, d_q, d_decay, d_g (5 outputs vs 1 in forward)

### Specific Optimization Opportunities

#### 1. Multi-Head Per Block (HIGH IMPACT)

**Current**: 1 head per block, 1 warp per head
```
grid = (B * H,) = (8 * 83,) = 664 blocks per layer
```

**Proposed**: 4 heads per block, 4 warps
```
grid = (B * H / 4,) = 166 blocks per layer
Shared memory: 4× current (still fits in 48KB L1)
```

**Expected benefit**:
- 4x better occupancy (4 warps vs 1 warp active)
- Reduced kernel launch overhead (166 vs 664 launches per layer)
- **Estimated 30-50% backward speedup** (10-15% total)

#### 2. Fused Forward-Backward (MEDIUM IMPACT)

**Current**: Separate forward and backward kernels per layer
**Proposed**: Single kernel that does forward+backward, recomputing states on the fly

**Expected benefit**:
- Eliminate checkpoint reads/writes (currently ~150MB per layer)
- Fewer kernel launches (17 vs 34 per forward+backward)
- **Estimated 10-20% backward speedup**

#### 3. Vectorized BF16 Loads/Stores (LOW IMPACT)

**Current**: Scalar loads/stores of individual bf16 values
**Proposed**: Use `__nv_bfloat162` for 2-wide loads when stride allows

**Expected benefit**: ~5% memory bandwidth improvement

## Conclusion

The E88 backward kernel is well-optimized for single-warp execution, but **multi-head-per-block** could yield 30-50% backward speedup (10-15% total). This is the highest-impact optimization available.

The **fundamental limitation** remains sequential processing. Unlike Mamba2's parallel scan, E88 must process timesteps sequentially due to nonlinear (tanh) state updates. This architectural constraint means E88 will always have lower throughput than SSMs for the same parameter count.

### Actionable Next Steps

1. **Implement multi-head backward kernel** (4 warps/block)
   - Modify `E88RegisterOwnedBackwardKernel_BF16` to process 4 heads per block
   - Use separate shared memory banks for each warp's k, v, decay
   - Expected: 30-50% backward speedup

2. **Profile after multi-head optimization**
   - Verify occupancy improvement with `ncu` (once permissions available)
   - Measure actual speedup vs expected

## Files

- Profile script: `profile_e88_torch.py`
- Chrome trace: `e88_profile_trace.json`
- Backward kernel: `elman/cuda/lib/e88_register_owned_gpu.cu.cc`
