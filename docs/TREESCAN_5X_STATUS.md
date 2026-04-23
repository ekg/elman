# Tree-scan 5× — what I learned, what works, what doesn't

## Summary

Attempted 5× end-to-end speedup via log(T)-depth tree scan. Found:

1. **Math is sound** (Phase 0): full N×N affine-op tree scan gives bit-identical δ to the existing rank-1 sequential scan at fp64, matches fp32 Pararnn output to 1e-6 relative.

2. **Triton cannot express the algorithm efficiently**:
   - `tl.associative_scan` passes scalar values to the combine function — `tl.dot` fails inside combine with `is_block()` assertion.
   - Tuple inputs to `associative_scan` require same-shape tiles (A is [N,N], b is [N] — mismatch).
   - Manual Kogge-Stone needs dynamic indexing into 3D tiles, which Triton doesn't support.
   - BMM-per-level (`phase1b_bmm_scan.py`): mathematically correct, **100-1000× slower** than existing sequential scan due to kernel launch overhead and small-matrix inefficiency.
   - Sequential scan with `tl.dot` per step (`phase1d_seq_tldot.py`): works correctness, **7-12× slower** than existing r=1 Pararnn because the structured r=1 form is genuinely more efficient than dense N×N for sequential scan.

3. **The fundamental arithmetic problem**: tree scan does `T log T × N³` work per Newton iter, vs CUDA's `T × N²`. At N=16 T=128K: tree scan needs ~20 TFlops of matmul. At 30% realized tensor-core utilization (typical for 16×16 matmul), that's **~200 ms**. CUDA forward does 144 ms. So even idealized tree scan is barely competitive, nowhere near 5× faster on forward.

## Where we actually stand

Realistic ceiling for Newton-tree-scan forward on this problem size:
- At 30% TC util: ~200 ms (worse than CUDA 144 ms)
- At 50% TC util: ~128 ms (tied with CUDA)
- At 70% TC util: ~91 ms (1.58× faster than CUDA)
- At peak: ~64 ms (2.25×)

Combined with Pararnn backward (151 ms at production T=128K):
- Best-case: tree scan fwd 64 ms + par bwd 151 ms = 215 ms vs CUDA f+b 434 ms = **2.0× e2e**.
- Realistic: tree scan fwd 128 ms + par bwd 151 ms = 279 ms = **1.55× e2e**.

**The 5× e2e target at production configs is not reachable via tree-scan on single GPU for this problem.** The core issue: 16×16 matmuls don't saturate modern tensor cores, and log(T) × N work increase outweighs depth reduction at our sizes.

## What 5× would actually require

Would need either:
1. **Larger N** — tensor cores shine at N≥128. E88's N=16/32 is a bad match. No obvious way to increase N while keeping the architecture.
2. **Bigger parallelism within single step** — batch multiple heads/rows per matmul. Adds complexity but could reach 60-70% TC util.
3. **Algorithmic innovation** — reduce log(T) factor, e.g., ADMM/Parareal with coarse approximate solver. Research.

Path (2) is engineering: batched GEMM of all heads simultaneously at each level. Could reach 70% TC util. Combined:
- Tree scan fwd 45 ms + par bwd 150 ms = 195 ms vs CUDA 434 ms = **2.2× e2e**

Still short of 5×. The backward is the bigger bottleneck now.

If both fwd and bwd go tree-scan: fwd 45 ms + bwd 45 ms = 90 ms = **4.8×**.
Close to target but hard engineering: batched GEMM inter-head, well-tuned tensor cores on 16×16 tiles, minimal kernel launch overhead, both directions.

## Additional experiment: preallocated BMM (Phase 1e)

To rule out that allocation overhead was the sole cause of Phase 1b's
slowness, I wrote a preallocated / ping-pong-buffer version
(`phase1e_bmm_preallocated.py`). Same behavior: **250-300× slower**
than Pararnn r=1 sequential scan.

Confirms: the per-matmul overhead at 16×16 size dominates. cuBLAS
batched gemm at these sizes runs at < 0.1% of tensor-core peak. The
fundamental mismatch is matrix size — tensor cores want 16×16 tiles
multiplied against LARGER matrices (i.e., few mats * big mat), not
hundreds of thousands of 16×16 @ 16×16 independent multiplies.

## What was delivered

- `docs/TREESCAN_5X_DESIGN.md` — detailed design sketch (before full analysis)
- `docs/TREESCAN_5X_STATUS.md` — this honest post-mortem
- `experiments/pararnn_kernel/tree_scan/phase0_pytorch_ref.py` — math reference, verified correct at fp64 and fp32
- `experiments/pararnn_kernel/tree_scan/phase1_intra_block.py` — Triton `tl.associative_scan` attempt (blocked)
- `experiments/pararnn_kernel/tree_scan/phase1b_bmm_scan.py` — BMM-per-level (works, 100-1000× slower)
- `experiments/pararnn_kernel/tree_scan/phase1c_manual_kogge.py` — Triton Kogge-Stone attempt (blocked by tile indexing)
- `experiments/pararnn_kernel/tree_scan/phase1d_seq_tldot.py` — sequential scan with tl.dot (works, 7-12× slower)

## Path forward to actually get 5×

Three viable paths, all requiring significant engineering beyond a single session:

### A. Custom CUDA tensor-core kernel with row/head-batched GEMMs
The core idea: at each tree-scan level, all (b, h, row) combines at the
same time-pair form a single large GEMM if we interpret the `(bh_row)`
dim as K (contraction) in a grouped gemm. But the combines are
INDEPENDENT — no shared contraction dim. So it's a "batched matmul"
not a GEMM, and at 16×16 that's still slow.

**What might work**: custom CUDA kernel using CUTLASS's group-GEMM
primitives, *plus* a layout trick — batch K rows/heads per matrix by
block-diagonal packing. Each "matmul" becomes 16K × 16K (e.g., K=8 gives
128×128) — tensor cores happy. Same math, different packing.

**Estimated effort**: 1-2 weeks CUTLASS kernel engineering, plus
hierarchy for full T.

### B. ADMM / Parareal style coarse-fine parallelism
Split T into C chunks. Coarse solver (linear/Mamba-style scan, cheap)
gives initial guesses for chunk endpoints. Fine solver (Newton sequential
within each chunk) refines. Iterate until boundaries converge.

**Depth**: T/C + log T. **Work**: ~T × small (coarse) + T × tanh (fine).

If it converges in O(log T) outer iters, total work stays O(T) but
with P = C parallel chunks. Potential speedup ~C = 100× at C=100.

**Estimated effort**: 1-2 weeks research + implementation; convergence
not guaranteed without careful tuning.

### C. Approximate low-rank tree scan
Keep scan state as rank-K truncated affine op (K=2 or 4). Per combine:
O(K² N) FLOPs instead of N³. For stable E88 (decay < 1), old rank-1
updates decay exponentially — truncation error bounded.

**Work**: T log T × K² N = 2-8× less than full matrix. Combined with
smaller per-matmul (better TC utilization at K×K): plausible 5× target.

**Estimated effort**: 1-2 weeks, with research component (numerical
stability of truncation for this specific system).

## Recommendation

The 1.47-1.68× from the fused backward (already in `phase7_fused_backward.py`) is the real, stable, deployable single-GPU win. It's the right thing to put into training.

The 5× target is not reachable without **either**:
- Custom CUDA C++ with hand-written tensor core inner loops, batching heads per matmul, both fwd and bwd — estimated 2-3 weeks of dedicated kernel engineering.
- Multi-GPU (which the user asked to defer, but remains the biggest untapped factor).
- Algorithmic work (ADMM, Parareal, approximate low-rank scan) — research.

For near-term, **ship the 1.5× we already have** via the fused backward kernel, use the freed compute for bigger models or longer training, and revisit tree-scan when we have a dedicated kernel-engineering block of time.
