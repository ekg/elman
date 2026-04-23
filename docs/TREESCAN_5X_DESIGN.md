# Tree-scan Newton for 5× E88 training — design sketch

## Goal

5× end-to-end training step speedup on single GPU for E88-n16 480M at T=128K, matching FLA-GDN's speed despite being **nonlinear in time**.

Current state: 1.47× e2e via fused backward. Ceiling of the backward-only hybrid is ~3×, bounded by CUDA forward time (Amdahl). Breaking past 3× requires accelerating the forward; the only path with log-depth parallelism is tree-scan of Newton's linearized system.

## The math

E88 recurrence (per row i):
```
S[t, i, :] = tanh(decay_t · S[t-1, i, :] + (V[t, i] - K[t] · S[t-1, i, :]) · K[t])
```

Newton's fixed-point iteration: given iterate `S_var`, compute residual `r[t] = S_var[t] - f(S_var[t-1], …)` and solve `J δ = -r`. Jacobian J is block-bidiagonal with I on the diagonal and `-∂f/∂S[t-1]` on the subdiagonal. Forward substitution gives:
```
δ[0] = -r[0]
δ[t] = A_t · δ[t-1] + b_t          where A_t = ∂f/∂S[t-1],  b_t = -r[t]
```

**A_t is the per-step linearized Jacobian**: `A_t = diag(D_t) − u_t v_t^T` where D_t = decay · tanh′, u_t = tanh′ ⊙ K, v_t = K.

### Key observation: Affine operator composition is associative

```
(A_j, b_j) ∘ (A_i, b_i) = (A_j · A_i,  A_j · b_i + b_j)
```

This is just function composition — always associative. So δ values can be computed as a **parallel prefix scan** in log(T) depth. That's the whole trick.

### Why we don't keep A in rank-1 form

The r=1 structured state `(D, u, v)` is **lossless for SEQUENTIAL composition** (proved in phase 2) but composition of two rank-1 affine operators yields a rank-2 operator in general:
```
(diag(D2) − u2 v2^T)(diag(D1) − u1 v1^T)
  = D2 D1 − D2 u1 v1^T − u2 v2^T D1 + u2 (v2^T u1) v1^T
```
Last term is rank-1 × rank-1 = rank-1 (since `v2^T u1` is scalar), but we still have **two independent rank-1 updates**, not one. Tree-scan via r=1 truncation is LOSSY and non-associative → we proved it diverges.

**Solution: keep A as a full N×N matrix.** Composition is exact. Tree-scan is associative. The cost is memory — 1024 elements per position instead of 64 — but memory is not the binding constraint if we hierarchize.

## Performance math

Per combine: one N×N @ N×N matmul + one N×N matvec = N³ + N² ≈ N³ FLOPs.

**Forward tree scan at T=128K, N=16:**
- Levels: log2(T) = 17
- Combines per level: T/2, T/4, ..., 1. Total ≈ T combines.
- Total FLOPs per head: T × N³ = 128K × 4096 = 500 MFLOPs per Newton iter per head
- For H=141: 70 GFLOPs per iter
- Tensor-core bf16 peak A100: 312 TFLOPs → **0.23 ms compute-bound** per iter
- Realistic 30% TC util: ~1 ms per iter
- Cold-start 3 iters: 3 ms

**At N=32:** 500 MFLOPs × 8 = 4 GFLOPs/head. H=83 → 330 GFLOPs. ~1 ms at peak TC util, ~5 ms realistic. 3 iters = 15 ms.

**vs CUDA forward at T=128K:**
- E88-n16 H=141: 144 ms → **30-50× potential speedup**
- E88-n32 H=83: 228 ms (extrapolated) → **10-15× potential**

### Memory

State per position: (A: N×N, b: N) = N² + N elements.
- N=16: 272 elements × 2B = 544 B/position
- N=32: 1056 elements × 2B = 2.1 KB/position

At T=128K H=141 N=16: 544 B × 128K × 141 = **9.6 GB per level**. Across log T=17 levels with half-halving: ~20 GB. **Tight but fits.**

At T=128K H=83 N=32: 2.1 KB × 128K × 83 = **22 GB per level**. Total **~44 GB** — doesn't fit alongside forward inputs + gradient buffers.

**Mitigation for N=32**: block-hierarchical scan — only store block-level summaries globally; within-block work lives in shared memory / registers.

### End-to-end projection at T=128K E88-n16 H=141

| Stage | Current | With tree scan |
|-------|---------|----------------|
| Forward | CUDA 144 ms | 1-iter: ~3 ms; 3-iter cold: ~10 ms |
| Backward | Pararnn 151 ms | Tree-scan bwd: ~5 ms |
| dQ | 5 ms | 5 ms (unchanged) |
| **Total** | **300 ms** | **13–20 ms** |
| **vs all-CUDA (434 ms)** | 1.47× | **22–33×** |

Even if we lose a factor of 5 to realistic kernel overhead, we're at **4–6×**. The 5× target is realistic if we execute well.

## Architecture

### Hierarchical tree scan (three-level standard)

Standard parallel-prefix-scan architecture for sequences too long for one block:

```
Level 0 (leaf): T positions of (A_0, b_0) built from K, V, decay, S_var
     ↓  within-block scan (shared mem, tree reduction)
Level 1: T/B_T block summaries (A_block, b_block)
     ↓  inter-block scan (global mem, same tree structure recursively)
Level 1 prefix: T/B_T positions with cumulative prefixes
     ↓  within-block scatter (apply block-prefix back to each in-block position)
Output: δ[t] for t=0..T-1
```

Where B_T is the intra-block length, chosen so `B_T × (N² + N) × 2B` fits in shared memory (~48 KB).

- N=16: B_T ≤ 80 positions. Pick 64 (power of 2, good for tree).
- N=32: B_T ≤ 20. Pick 16.

Number of blocks at T=128K, H=141, N=16, B_T=64: **128K / 64 × 141 = 282K blocks** (well into GPU-friendly territory).

### Within-block scan

Classic Kogge-Stone or Brent-Kung on 64 elements, using `tl.dot` for the matmul inside combine. 

For `tl.dot`: each combine is a batched matmul of [64 × 16 × 16] @ [64 × 16 × 16]. Tensor cores happy. One kernel call per scan level inside block (log2(64) = 6 levels).

### Inter-block scan

Over T/B_T = 2048 block summaries, same recursive pattern. Another log2(2048)=11 levels. Fits in one or two kernel calls (each block does many block summaries).

### Scatter / apply phase

Each position in block c gets δ[block_c_start + i] = block_c_prefix ∘ within_block_prefix[i]. Straightforward per-block kernel.

## Backward via reverse tree scan

The adjoint system is linear — the same tree-scan structure works in reverse direction. The Jacobian transpose `A^T_t = diag(D_t) - v_t u_t^T` has the same rank-1 structured form; we again store full N×N for tree associativity.

Single exact pass (no Newton iteration needed — adjoint is linear).

## Numerical stability

Product `A_T · A_{T-1} · ... · A_1` can grow or shrink. For stable E88 (`decay < 1`, `|tanh′| ≤ 1`), Jacobian spectral radius is ≤ 1 near the fixed point → products are bounded, scan is numerically stable.

Far from fixed point (cold start iter 1) Jacobians can be larger, but tree scan is no less stable than sequential scan — in fact it's **better** conditioned because of the "balanced tree" property vs deep sequential chain.

Defensive measure: track a scalar `scale` per node (like log-space), rescale when product magnitude gets extreme. Probably only needed at T > 256K.

## Triton vs CUDA

**Triton preferred for AMD portability.** Concrete plan:
- `tl.dot` uses NVIDIA tensor cores on CUDA backend and AMD matrix cores on ROCm backend — automatically.
- `tl.associative_scan` exists but is **elementwise-only**; not directly usable for N×N matrix combine. We'll write manual Kogge-Stone with explicit shared-mem reductions.
- Block-hierarchical scan is a pattern of ~3-4 kernel launches — Triton handles this cleanly.

**Fall back to CUDA C++ for specific kernels if** tl.dot misses important optimizations (e.g., WGMMA on H100, or MFMA-with-async-copy on MI300). The internal scan math doesn't change.

## Phased implementation plan

### Phase 0: PyTorch verification (~1 day)
Pure-PyTorch tree-scan reference: builds (A, b) tensors, does explicit O(T log T) tree scan via torch ops. Compare output against sequential Pararnn Newton — should be bit-identical. Proves the math is right before kernel work.

### Phase 1: Triton intra-block scan kernel (~3 days)
Single-block scan for T ≤ 64. Kogge-Stone with `tl.dot` for matmul in combine. Proves the in-shared-mem scan math. Memory fits in shared, no hierarchy yet.

### Phase 2: Hierarchical tree scan over full T (~3 days)
Add inter-block pass and scatter phase. Run at T=1K, 8K, 128K. Validate correctness against Phase 1 output (for short T) and sequential Pararnn (for long T).

### Phase 3: Newton iteration wrapper (~1 day)
Wrap the tree scan with Newton iteration: compute residual, build (A, b), tree scan, update S_var. Test convergence on realistic warm-starts. Benchmark vs CUDA forward.

### Phase 4: Reverse tree scan backward (~3 days)
Mirror phases 1-2 in reverse direction. Build (A^T, ext) tensors from forward trajectory, tree scan, produce dS0, dK, dV, ddecay, dQ. Bit-matching test against phase5_backward.

### Phase 5: End-to-end integration + tuning (~3 days)
Replace CUDA fwd with Triton tree-scan fwd. Replace Pararnn Newton bwd with tree-scan bwd. Tune num_warps / num_stages / block sizes. Production benchmark.

**Total: ~2 weeks** of kernel engineering to reach target.

## Risks and mitigations

- **Register pressure from N×N tiles in shared mem**: if 64 × 16×16 bf16 (32 KB) plus working space blows shared mem budget, fall back to B_T=32 and log2(32)=5-level in-block scan.
- **Convergence on cold-start at long T with tree scan**: haven't tested. If problematic, fall back to 1-2 extra outer Newton iters — still faster than sequential.
- **bf16 precision in accumulated A products**: if errors compound over 17 levels, switch to fp32 tensor cores (slower but safer). Benchmark at T=128K before production.
- **tl.dot performance on this shape**: 16×16 matmul with dim=16 is on the small end for tensor cores. May need batch-packing multiple combines per GEMM. Measure in phase 1.

## Success criteria

- Phase 1 kernel: beats sequential r=1 scan by >5× on isolated combine microbenchmark at B_T=64.
- Phase 3 forward: < 20 ms at T=128K H=141 N=16 (vs CUDA 144 ms — 7×).
- Phase 5 e2e: ≥ 5× speedup on full training step at production configs.

## Files to create

- `experiments/pararnn_kernel/tree_scan/phase0_pytorch_ref.py` — PyTorch reference
- `experiments/pararnn_kernel/tree_scan/phase1_intra_block.py` — Triton intra-block
- `experiments/pararnn_kernel/tree_scan/phase2_hierarchical.py` — full T scan
- `experiments/pararnn_kernel/tree_scan/phase3_newton_tree.py` — Newton wrapper
- `experiments/pararnn_kernel/tree_scan/phase4_backward_tree.py` — reverse scan
- `experiments/pararnn_kernel/tree_scan/phase5_e2e_bench.py` — full benchmark
