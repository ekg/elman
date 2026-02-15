# E88 Register-Owned Backward Kernel: Generalization Strategy

## Executive Summary

The register-owned backward kernel (`e88_register_owned_gpu.cu.cc`) achieves 1.7-8.5x speedup over the fused backward kernel for n_state=32, head_v_dim=32 by storing the entire state matrix in registers (one column per thread, 32 threads = 1 warp). Generalizing this to other sizes requires different strategies depending on whether the size fits in a single warp.

**Recommendation: Three-tier kernel dispatch:**
1. **Tier 1 (n ≤ 32):** Single-warp register-owned (current approach, adapted for smaller sizes)
2. **Tier 2 (n = 36-64):** Multi-warp register-tiled with shared memory coordination
3. **Tier 3 (n ≥ 72):** Stay with fused/global-memory backward (register approach loses advantage)

## Current Register-Owned Kernel Architecture

### How It Works (n_state=32, head_v_dim=32)

```
Block: 32 threads (1 warp), 1 block per (batch, head)
Thread j owns: S[:,j] — column j of the 32×32 state matrix

Per-thread register allocation:
  S_reg[32]      = 128 bytes  (state column)
  dS_reg[32]     = 128 bytes  (gradient of state column)
  S_t_reg[32]    = 128 bytes  (post-tanh state)
  dtanh_reg[32]  = 128 bytes  (1 - tanh² derivative)
  ────────────────────────────
  Total:           512 bytes = 128 float registers per thread
```

Shared memory: only 260 bytes (k, v, decay cooperative load buffer).

### Why It's Fast

| Property | Fused Backward | Register-Owned |
|----------|---------------|----------------|
| State storage | Shared memory (16 KB) | Registers (0 KB smem) |
| Synchronization | `__syncthreads()` (barrier) | `__syncwarp()` (free on same warp) |
| Cross-thread comm | Shared memory read/write | Warp shuffle (`__shfl_sync`) |
| Threads per block | 256 (iterate over state) | 32 (each owns a column) |
| Occupancy | High thread count, low occupancy per SM | Low thread count, potentially more blocks per SM |

The key insight: **warp shuffles are 5-10x faster than shared memory round-trips** for small reductions, and `__syncwarp()` is essentially free (warp executes lockstep).

## Tier 1: Small Sizes (n_state ≤ 32)

### Supported Sizes: 4, 8, 16, 24, 32

**Strategy:** Single warp, inactive lanes for n < 32.

For n_state=N where N < 32, launch 32 threads but only N are active:

```
n_state=16:  threads 0-15 own columns, threads 16-31 idle
n_state=24:  threads 0-23 own columns, threads 24-31 idle
n_state=32:  all threads active (current implementation)
```

### Register Budget Per Thread

| n_state | Regs/thread (4 arrays) | Total regs/warp | Feasible? |
|---------|------------------------|-----------------|-----------|
| 4 | 4×4 = 16 floats = 64B | 512 floats | Trivial |
| 8 | 4×8 = 32 floats = 128B | 1024 floats | Trivial |
| 16 | 4×16 = 64 floats = 256B | 2048 floats | Easy |
| 24 | 4×24 = 96 floats = 384B | 3072 floats | Comfortable |
| 32 | 4×32 = 128 floats = 512B | 4096 floats | Current (works) |

GPU limit: 255 registers per thread × 4 bytes = 1020 bytes on most architectures. 512 bytes is well under this, even accounting for temporaries and loop variables (~30-40 additional registers).

### Rectangular States (n_state ≠ head_v_dim)

For rectangular states like (16, 32), (24, 48), (32, 64):
- **Thread count = head_v_dim** (thread j owns column j)
- **Register arrays have length n_state** (S_reg[n_state])
- Only works when head_v_dim ≤ 32 (single warp)

For (n_state=32, head_v_dim=16): 16 threads, each owns a 32-element column. Works fine.
For (n_state=16, head_v_dim=32): 32 threads, each owns a 16-element column. Works fine.
For (n_state=32, head_v_dim=64): **Needs 64 threads = 2 warps → Tier 2.**

### Warp Shuffle Reductions for n < 32

The warp shuffle reduction pattern `__shfl_xor_sync(0xFFFFFFFF, val, offset)` still works with inactive lanes — the mask ensures only active lanes participate. However, the sum includes garbage from inactive lanes unless we mask properly.

**Solution:** Use a lane mask `uint32_t mask = (1u << N) - 1` and `__shfl_xor_sync(mask, ...)`.

For d_q reduction (sum across all j of S_t[i,j] * d_Sq[j]):
- Only need to sum over N active lanes
- Reduction depth: ceil(log2(N)) steps instead of 5

### Template Structure (Tier 1)

```cpp
template<int N_STATE, int HEAD_V_DIM>
__global__ void E88RegisterOwnedBackwardKernel_BF16(...) {
    static_assert(HEAD_V_DIM <= 32,
        "Tier 1 register-owned requires head_v_dim <= 32");

    int tid = threadIdx.x;  // tid in [0, 31]
    constexpr uint32_t ACTIVE_MASK = (1u << HEAD_V_DIM) - 1;

    // Thread tid owns column tid (inactive if tid >= HEAD_V_DIM)
    float S_reg[N_STATE];
    float dS_reg[N_STATE];
    float S_t_reg[N_STATE];
    float dtanh_reg[N_STATE];

    // Shared memory: k[N_STATE] + v[HEAD_V_DIM] + decay
    __shared__ float k_shared[N_STATE];
    __shared__ float v_shared[HEAD_V_DIM];
    __shared__ float decay_shared;

    // ... same algorithm as current, but using ACTIVE_MASK
    // All shuffles use ACTIVE_MASK instead of 0xFFFFFFFF
}
```

### Expected Performance (Tier 1)

| Size | vs Fused Backward | Notes |
|------|-------------------|-------|
| 4×4 | ~8-10x faster | Trivially small, dominated by launch overhead |
| 8×8 | ~5-8x faster | Very small state, huge smem savings |
| 16×16 | ~3-5x faster | Similar to current 32×32 ratio |
| 24×24 | ~2-4x faster | Slight waste from 8 idle lanes |
| 32×32 | ~1.7-8.5x faster | Current measured performance |

## Tier 2: Medium Sizes (n_state = 36-64)

### The Problem

For n_state > 32 (or head_v_dim > 32), we need more than one warp. This breaks the key advantage: warp-lockstep execution without barriers.

### Strategy: Multi-Warp Register-Tiled

Use 2 warps (64 threads) for sizes 33-64:

```
n_state=48:  2 warps × 32 threads
             Warp 0: threads 0-31 own columns 0-31
             Warp 1: threads 32-47 own columns 32-47 (threads 48-63 idle)

n_state=64:  2 warps × 32 threads
             Warp 0: threads 0-31 own columns 0-31
             Warp 1: threads 32-63 own columns 32-63
```

### Register Budget (Tier 2)

| n_state | Regs/thread | Total per 2 warps | Feasible? |
|---------|-------------|-------------------|-----------|
| 36 | 4×36 = 144 floats = 576B | 9216 | OK but tight |
| 40 | 4×40 = 160 floats = 640B | 10240 | Tight |
| 48 | 4×48 = 192 floats = 768B | 12288 | At limit |
| 56 | 4×56 = 224 floats = 896B | 14336 | **Spilling likely** |
| 64 | 4×64 = 256 floats = 1024B | 16384 | **Exceeds 255 reg limit** |

**Critical constraint:** At n_state=64, each thread needs 256 floats just for the 4 state arrays, which hits the hardware limit of 255 registers. Including temporaries, this will spill to local memory (L1 cache), losing much of the register advantage.

### Cross-Warp Communication

The main challenge: operations like d_q[i] = Σ_j S_t[i,j] * d_Sq[j] need data from threads in **both warps**.

**Within a warp:** `__shfl_xor_sync` (fast, ~1 cycle)
**Across warps:** Must use shared memory + `__syncthreads()` (slow, ~20-40 cycles)

**Approach:** Two-phase reduction:
1. Each warp computes its partial sum via warp shuffles
2. Warp 0, lane 0 and Warp 1, lane 0 write partials to shared memory
3. `__syncthreads()`
4. One thread adds the two partials

This is still faster than the fully shared-memory approach because:
- State lives in registers (no shared memory bandwidth for state read/write)
- Only 1-2 `__syncthreads()` per timestep instead of ~8-10

### Practical Tier 2 Sizes

Given register pressure, practical Tier 2 targets are:

| Size | Approach | Expected Speedup vs Fused |
|------|----------|--------------------------|
| 36×36 | 2 warps, 4 idle threads/warp | ~1.5-2.5x |
| 40×40 | 2 warps, 8 idle/24 idle | ~1.3-2.0x |
| 48×48 | 2 warps, 16 idle | ~1.2-1.8x |

**Recommendation:** Only implement Tier 2 for sizes 36 and 40. For 48+, the diminishing returns from register spilling and cross-warp sync make it marginal.

### Alternative for n=48: Reduce Register Arrays

Instead of 4 arrays (S, dS, S_t, dtanh), keep only S and dS in registers, recompute S_t and dtanh on the fly:

```cpp
// Instead of storing S_t_reg and dtanh_reg:
float pre_tanh = decay * S_reg[i] + delta * k_shared[i];
float tanh_val = tanhf(pre_tanh);  // Recompute instead of load
float dtanh_val = 1.0f - tanh_val * tanh_val;  // Recompute
```

This halves register usage: 2 × 48 = 96 floats = 384 bytes (feasible).

**Tradeoff:** Extra tanhf() calls per iteration. But tanhf is ~8 cycles on SM89, while a register spill to L1 is ~20-80 cycles. **Recomputation wins.**

### Template Structure (Tier 2)

```cpp
template<int N_STATE, int HEAD_V_DIM>
__global__ void E88RegisterOwnedBackwardKernel_MultiWarp_BF16(...) {
    static_assert(HEAD_V_DIM > 32 && HEAD_V_DIM <= 64,
        "Tier 2 for head_v_dim in (32, 64]");

    int tid = threadIdx.x;  // tid in [0, 63]
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int col = tid;  // Column owned by this thread

    // Only 2 register arrays (S, dS) — recompute S_t, dtanh
    float S_reg[N_STATE];
    float dS_reg[N_STATE];

    // Small shared memory for cross-warp reduction
    __shared__ float cross_warp[2];  // Partial sums from each warp
    __shared__ float k_shared[N_STATE];
    __shared__ float v_shared[HEAD_V_DIM];
    __shared__ float decay_shared;

    // ... (algorithm with 2-phase reductions)
}
```

## Tier 3: Large Sizes (n_state ≥ 72)

### Why Register-Owned Doesn't Work

| n_state | Regs/thread (2 arrays) | Problem |
|---------|------------------------|---------|
| 72 | 2×72 = 144 = 576B | 3+ warps needed, many cross-warp syncs |
| 80 | 2×80 = 160 = 640B | Near register limit even with 2 arrays |
| 96 | 2×96 = 192 = 768B | **Exceeds register limit** |
| 128 | 2×128 = 256 = 1024B | **Way over limit** |

At n_state ≥ 72:
- 3+ warps needed → multiple `__syncthreads()` per reduction
- Register pressure → spilling to local memory
- Diminishing returns over fused backward with shared memory
- The existing global-memory fallback kernel already handles these sizes

### Recommendation

**Keep the existing fused backward + global memory fallback for n_state ≥ 72.** No benefit from register-owned approach at these sizes.

The performance hierarchy for large sizes:
1. Fused backward with shared memory (n ≤ ~56, fits in 96 KB)
2. Global memory fallback (n ≥ 72, state in global memory)

## Dispatch Logic Design

### Unified Dispatch Function

```cpp
void dispatch_e88_optimized_backward(
    int T, int B, int H, int n_state, int head_v_dim,
    const bf16* k, const bf16* v, const bf16* q,
    const bf16* decay, const bf16* g,
    const bf16* S_checkpoints, const bf16* Sq_cache,
    const bf16* d_output,
    bf16* d_k, bf16* d_v, bf16* d_q, bf16* d_decay, bf16* d_g,
    bf16* segment_cache,
    float* S_global, float* dS_global,  // For Tier 3
    int checkpoint_interval, bool has_gate, cudaStream_t stream
) {
    // Tier 1: Single warp register-owned (head_v_dim ≤ 32)
    if (head_v_dim <= 32 && n_state <= 64) {
        dispatch_register_owned_tier1(/* ... */);
        return;
    }

    // Tier 2: Multi-warp register-tiled (head_v_dim ≤ 64, n_state ≤ 48)
    if (head_v_dim <= 64 && n_state <= 48) {
        dispatch_register_owned_tier2(/* ... */);
        return;
    }

    // Tier 3: Fall through to fused/global-memory backward
    dispatch_e88_fused_backward(/* ... */);
}
```

### Template Instantiations

**Tier 1 (single warp, 10 instantiations):**
```cpp
// Square states
DISPATCH_REG_T1(4, 4)
DISPATCH_REG_T1(8, 8)
DISPATCH_REG_T1(16, 16)
DISPATCH_REG_T1(24, 24)
DISPATCH_REG_T1(32, 32)   // Current implementation

// Common rectangular states (head_v_dim ≤ 32)
DISPATCH_REG_T1(8, 16)
DISPATCH_REG_T1(8, 32)
DISPATCH_REG_T1(16, 32)
DISPATCH_REG_T1(24, 32)
DISPATCH_REG_T1(32, 16)   // Wide key, narrow value
```

**Tier 2 (multi-warp, 6 instantiations):**
```cpp
DISPATCH_REG_T2(36, 36)
DISPATCH_REG_T2(40, 40)
DISPATCH_REG_T2(48, 48)
DISPATCH_REG_T2(36, 48)
DISPATCH_REG_T2(40, 48)
DISPATCH_REG_T2(48, 64)
```

**Tier 3: No new instantiations** — reuses existing fused/global-memory kernels.

## Priority-Ranked Implementation Plan

### Phase 1: Generalize Tier 1 (High Value, Low Risk) ✅ COMPLETED (Feb 15, 2026)

1. ✅ Removed `static_assert(N_STATE == 32 && HEAD_V_DIM == 32)` from kernel
2. ✅ Templated for variable N_STATE (up to 64) and HEAD_V_DIM (up to 32)
3. ✅ Fixed warp synchronization - ALL threads must participate in `__shfl_sync` even if inactive
4. ✅ Added 17 template instantiations:
   - Square: (4,4), (8,8), (16,16), (24,24), (32,32)
   - Tall: (16,8), (24,16), (32,16), (32,24), (36,32), (40,32), (48,32), (64,32)
   - Wide: (8,16), (8,32), (16,32), (24,32)
5. ✅ Updated dispatch function with if/else chain

**Key fix:** Moved `__shfl_sync(0xFFFFFFFF, ...)` outside `if (is_active)` blocks. With partial lane masks, ALL 32 threads must call the shuffle instruction, but only active threads update state.

### Phase 2: Integrate into Training (High Value)

1. Wire register-owned backward into `e88_fla_hybrid.py` as the preferred backward for supported sizes
2. Fall back to fused backward for unsupported sizes
3. Validate with end-to-end training comparison (loss trajectories should match)

**Expected effort:** ~1 hour. Mostly Python dispatch logic.

### Phase 3: Tier 2 Multi-Warp (Medium Value, Medium Risk)

1. Implement 2-warp kernel with recomputation (no S_t_reg, dtanh_reg)
2. Add cross-warp reduction via shared memory
3. Instantiate for (36,36), (40,40), (48,48)
4. Benchmark against fused backward to verify speedup

**Expected effort:** ~4-6 hours. New kernel logic needed for cross-warp coordination.

### Phase 4 (Optional): Register Pressure Optimization

- Profile register usage with `--ptxas-options=-v` to identify spilling
- Experiment with `__launch_bounds__` to tune register allocation
- Consider `__nv_bfloat16` register arrays to halve storage (at precision cost)

## Summary Table

| Size Range | Tier | Threads | Sync | Expected Speedup | Implementation Risk |
|------------|------|---------|------|------------------|-------------------|
| n ≤ 32 | 1 | 32 (1 warp) | `__syncwarp` | 2-8x | Low (generalize existing) |
| n = 36-48 | 2 | 64 (2 warps) | `__syncthreads` (1-2x) | 1.2-2.5x | Medium (new kernel) |
| n ≥ 64 | 3 | 256 | `__syncthreads` (many) | N/A (use fused) | None (existing code) |

## Key Insight

The register-owned approach is fundamentally a **warp-level optimization**. Its power comes from:
1. Warp shuffle replacing shared memory access
2. No synchronization barriers
3. Perfect register locality

Once you exceed one warp (32 threads), you lose benefits #1 and #2. The sweet spot is n_state ≤ 32 where the approach is unambiguously superior. For n=36-48, marginal gains are possible with careful engineering. Beyond that, the existing shared-memory/global-memory kernels are the right tool.

The most impactful work is **Phase 1 + Phase 2**: generalizing the existing 32×32 kernel to all sizes ≤ 32 and wiring it into training. This covers the CMA-ES optimal config (n_state=32) and provides speedups for all small-state experiments.
