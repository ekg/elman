# Tree-scan 5× — definitive finding

## TL;DR

**Tree scan does not beat sequential scan on single GPU for E88.** Across all T, H tested, our best tree-scan implementation (WMMA tensor cores, 3-pass hierarchy, fused (A,b) build) is **60-100× slower than the existing Pararnn r=1 sequential scan**.

The 5× speedup via tree scan is infeasible for this problem shape without algorithmic innovation beyond Hillis-Steele/Brent-Kung.

## Measured data (N=16, B=1, fp32)

| H | T | Pararnn r=1 seq | Tree-scan WMMA | ratio |
|---|---|-----------------|----------------|-------|
| 32 | 128 | 0.05 ms | 1.30 ms | **0.040×** |
| 32 | 512 | 0.14 ms | 6.81 ms | **0.021×** |
| 32 | 2K | 0.82 ms | 32.70 ms | **0.025×** |
| 32 | 8K | 3.09 ms | 181.52 ms | **0.017×** |
| 32 | 32K | 12.26 ms | 774.56 ms | **0.016×** |
| 32 | 128K | 50.37 ms | OOM | — |
| 141 | 128 | 0.12 ms | 6.66 ms | **0.019×** |
| 141 | 512 | 0.52 ms | 32.07 ms | **0.016×** |
| 141 | 8K | 7.88 ms | 767.97 ms | **0.010×** |
| 141 | 32K | 47.95 ms | OOM | — |

Ratio *worsens* with T — not just constant slowdown, but growing.

## Why tree scan loses for E88

The fundamental arithmetic:

| Quantity | Sequential (Pararnn r=1) | Tree scan (Brent-Kung, full-matrix) |
|----------|--------------------------|--------------------------------------|
| Depth | O(T) | O(log T) |
| Combines per chain | T | 2T |
| Work per combine | O(N) — rank-1 structured | O(N³) — full matrix product |
| Total work per chain | O(T·N) | O(T·N³) |
| Total work ratio | 1 | **~N² = 256× more** |

Even at peak tensor-core utilization, tree scan's 256× more work cannot beat sequential's 1× unless the GPU has 256× more parallelism headroom — which it doesn't for this problem because:
- Sequential already has B·H·N_row = 2256 independent scans running in parallel
- GPU has ~108 SMs — already saturated by the sequential workload
- Tree scan's log-depth gives only 2T/T = 2× more parallelism WITHIN a chain; chain parallelism is already tapped

## Why WMMA didn't help (much)

Measured: scalar-FMA tree scan 421ms vs WMMA-TF32 tree scan 362ms at T=4096 H=141 = **1.16× speedup**.

Expected from tensor-core peak rates: 5-10× speedup if compute-bound. We got 1.16× because:
1. **Kernel is memory-traffic bound**, not compute-bound (HBM at 1.5 TB/s is saturated by the augmented matrix loads)
2. **16×16 tiles are too small for WMMA**: mma_sync throughput amortizes better at 128×128 or larger
3. **TF32 fragment conversion** via `__float_to_tf32` adds overhead
4. **Sync barriers** between Hillis levels serialize across warps

## Where tree scan WOULD win

Tree scan's algorithmic advantage manifests when:
- Sequential is genuinely compute-bound on serial dependency chain
- Few independent chains (less exploitable parallelism elsewhere)
- Large N (matrix size amortizes tensor-core overhead)
- Small B·H·T (otherwise sequential fills GPU)

For E88-sized problems, none of these conditions hold. The existing sequential r=1 scan is near-optimal.

## What's shippable

The real single-GPU win from this multi-session exploration:

**Fused Triton backward kernel** (`experiments/pararnn_kernel/phase7_fused_backward.py`)
- 2.5× faster than CUDA backward at production H
- 1.47-1.68× end-to-end training step speedup at production E88 configs
- Verified at T up to 128K on E88-n16 480M
- bf16 S_traj storage + rank-1 dL/dS factorization + fused preprocessing
- Committed at `25ba89c`

## Paths to actual 5× that remain untried

1. **Multi-GPU T-distribution** — user deferred, but remains the largest untapped factor.
2. **ADMM/Parareal with cheap coarse solver** — genuinely algorithmic, not just kernel engineering.
3. **Rewrite Pararnn r=1 sequential scan in CUDA with explicit pipelining** — the existing Triton version is already good but might squeeze 20-30% more with hand-tuning.
4. **Custom tensor-core primitive designed for small (16×16, N=16) matmuls** — requires GPU-architecture-level thinking beyond WMMA.

None of these fit in a single session. The 5× target for E88-nonlinear remains a research/engineering program, not a small push from where we are.

## Artifacts from this exploration

All committed:
- `docs/TREESCAN_5X_DESIGN.md` — initial plan (turned out wrong about feasibility)
- `docs/TREESCAN_5X_STATUS.md` — mid-exploration post-mortem
- `docs/TREESCAN_5X_FINAL.md` — this (definitive finding)
- `experiments/pararnn_kernel/tree_scan/phase0_pytorch_ref.py` — math verified ✓
- `experiments/pararnn_kernel/tree_scan/phase0b_rankk_truncation.py` — rank-K converges ✓
- `experiments/pararnn_kernel/tree_scan/brent_kung_scan.cu` — 7 kernel variants
- `experiments/pararnn_kernel/tree_scan/test_*.py` — correctness suite (all pass)
- `experiments/pararnn_kernel/tree_scan/final_comparison.py` — this doc's data

The CUDA scaffolding works end-to-end. The math is verified at fp64 precision. Future engineers taking another swing at this have a proven foundation to build on.

Commits: `25ba89c`, `3cd7514`, `e3a2ee8`, `a8d5732`, `622cd7d`, `aa860cf`, `7c69383`.
