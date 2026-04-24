# Quasi-DEER for E88 — prototype report

**Date:** 2026-04-23
**Scope:** Evaluate quasi-DEER (Gonzalez et al. 2024, diagonal-Jacobian Newton)
as a parallel-scan alternative to the existing sequential Triton forward
for E88 nonlinear recurrence.

**Verdict: NO-GO.** Quasi-DEER converges correctly at realistic E88 input
distributions, but at production shapes (H=141 N=16) it is **~15× slower
end-to-end** than the existing sequential forward, even at T=32K. The
diagonal approximation has a fundamental memory-traffic disadvantage that
no kernel-engineering can close.

Files produced (all in `experiments/pararnn_kernel/tree_scan/`):
- `quasi_deer_ref.py` — PyTorch reference + sensitivity study.
- `quasi_deer_triton.py` — Triton kernel using `tl.associative_scan` with
  scalar diagonal combine. Correctness verified to fp32 precision.
- `quasi_deer_bench.py` — end-to-end benchmark vs sequential baseline.
- `qd_profile.py` — per-iteration cost breakdown.
- `qd_warmstart.py` — can a linear warm start reduce iter count? (a bit)
- `qd_longT.py` — find crossover T where pure scan beats sequential per iter.

## What is quasi-DEER for E88

E88 step: `S_{t+1}[p,q] = tanh(decay_t * S_t[p,q] + (V_t[p] - <S_t[p,:],K_t>) * K_t[q])`

Per-row-p, per-col-q, the Jacobian w.r.t. `S_t[p,:]` is
```
  A[t,p,:,:] = diag(D[t,p,:])  -  u[t,p,:] v[t,:]^T
```
where `D[t,p,c] = decay[t]*(1 - tanh(pre[t,p,c])^2)`, `u[t,p,c] =
(1-tanh²)*K[t,c]`, `v[t,c] = K[t,c]`.

**Quasi-DEER drops the `-u v^T` rank-1 term**, keeping only the diagonal
`diag(D)`. The Newton fixed-point system per (p,c) becomes a pure scalar
linear recurrence:
```
  x[t] = D[t,p,c] * x[t-1] + b[t,p,c]       b = -residual
```
which is a classical prefix-sum with the combine rule
`(D_L,b_L) ∘ (D_R,b_R) = (D_L*D_R, D_R*b_L + b_R)`. Triton's
`tl.associative_scan` applies directly (scalar combine, no `tl.dot`).

## Correctness

**All tests pass at fp32.**

Sequential vs parallel scan (fp64, identical math): `max|diff| = 2e-16`
(machine epsilon), every shape tested.

Triton scan vs PyTorch diag scan: `max|diff| / max|ref| ≤ 1e-7` at fp32,
at shapes up to B=1 H=16 T=16384 N=32.

Triton quasi-DEER Newton → sequential E88 trajectory:
| B | H | T | N | iters | residual | max\|tri−seq\|/max\|seq\| |
|---|---|---|---|-------|----------|---------------------------|
| 1 | 4  | 128  | 16 | 11 | 2.4e-5 | 4.1e-5 |
| 1 | 16 | 512  | 16 | 12 | 6.9e-5 | 8.8e-5 |
| 1 | 16 | 4096 | 16 | 13 | 9.1e-5 | 1.3e-4 |
| 1 |  4 | 1024 | 32 |  9 | 6.9e-5 | 1.3e-4 |

**bf16 limitation:** bf16 storage (compute-in-fp32 inside kernel) stalls
at residual ~3e-3 due to bf16 round-off. fp32 storage needed for
tight-precision Newton.

## Convergence (iteration count)

**Iteration count is nearly T-independent** and depends mostly on the
*magnitude of the rank-1 term we're dropping*. This is a real win for
quasi-DEER — consistent iter count across context lengths.

### Production-like (L2-normalized K, sigmoid(0.5)~0.62 decay, V~N(0, 0.3))

| H | T | N | iters to 1e-4 |
|---|---|---|---|
| 16 | 512   | 16 | 12 |
| 16 | 4096  | 16 | 13 |
| 4  | 16384 | 16 | 13 |
| 4  | 65536 | 16 | 13 |
| 16 | 512   | 32 |  9 |
| 16 | 4096  | 32 | 10 |
| 16 | 16384 | 32 | ~10 |

Convergence rate per iter ≈ 0.4-0.45 (linear, not quadratic). To reach
1e-6 residual ≈ 26 iters; 1e-3 ≈ 9 iters.

### Bad regimes (quasi-DEER diverges)

**Quasi-DEER DIVERGES when the dropped rank-1 term is not small.** Tested
with un-normalized K:
- K~N(0,0.5): residual EXPLODES from 1.0 → 3.6 → ... (diverges)
- K~N(0,0.8): diverges
- decay≈sigmoid(+2)≈0.88: diverges (stiff recurrence)

Production E88 uses L2-normalized K (so `|K[c]| ≈ 1/sqrt(N) ≈ 0.25`
at N=16) and moderate decay, placing it in the convergent regime.
**Practical training would need a safeguard** (e.g., fall back to
sequential if residual increases for 3 iters).

### Comparison to full-Jacobian Newton

Same setup (H=16 T=1024 N=16 production-like):

| Method | iters to 1e-4 | iters to 1e-7 | combine cost per step |
|--------|---|---|---|
| Full-Jacobian Newton | 3 | 4 (quadratic conv.) | O(N³) full matmul |
| Quasi-DEER (diag)    | 12 | 26 (linear conv.)   | O(N) elementwise |

Paper's claim (2× more iters) underestimates E88 — we see **3-4× more
iters** at equal precision. Still, per-combine work is `N²` cheaper, so
algorithmically quasi-DEER does less total work — in theory.

## Benchmark results

**All numbers: GPU 0 = single H100 48GB, fp32 (quasi-DEER), seed=0,
L2-normalized K, v_scale=0.3, sigmoid(0.5) decay, 10 repeats after warmup.**

### Production E88 shapes

| H | T | N | seq fwd (ms) | qd Newton (ms) | single qd scan (ms) | iters | speedup |
|---|---|---|---|---|---|---|---|
| 141 | 1024  | 16 | 0.47  | 70.0   | 0.51 | 13 | **0.01× (LOSE)** |
| 141 | 4096  | 16 | 1.85  | 279.8  | 2.04 | 13 | **0.01× (LOSE)** |
| 141 | 16384 | 16 | 7.38  | 1283   | 8.15 | 15 | **0.01× (LOSE)** |
| 83  | 1024  | 32 | 0.60  | 125    | 1.21 | 10 | **0.00× (LOSE)** |
| 83  | 4096  | 32 | 2.36  | 500    | 4.82 | 10 | **0.00× (LOSE)** |
| 32  | 65536 | 16 | 26.33 | 1166   | 7.51 | 15 | **0.02× (LOSE)** |

### Per-iteration cost breakdown

PyTorch ingredient build (residual + diag Jacobian) dominates per iter:

| H | T | N | ingredients (ms) | scan (ms) | S_var update (ms) | scan fraction |
|---|---|---|---|---|---|---|
| 141 | 1024  | 16 |  4.37 | 0.52 | 0.53 | 10% |
| 141 | 4096  | 16 | 17.55 | 2.04 | 2.11 |  9% |
| 141 | 16384 | 16 | 69.98 | 8.17 | 8.21 |  9% |
|  83 | 4096  | 32 | 40.73 | 4.83 | 4.85 | 10% |
|  32 | 65536 | 16 | 63.37 | 7.51 | 7.44 | 10% |

If we **fused** ingredient-build into the scan kernel, best case per iter
≈ 2-8ms × iters:

### Idealized fused-kernel estimate (assumes ingredients = free)

| H | T | N | iters × scan_ms | seq_ms | speedup (ideal) |
|---|---|---|---|---|---|
| 141 | 16384 | 16 | 15 × 8.17 = 122  | 7.38  | 0.06× |
| 32  | 65536 | 16 | 15 × 7.51 = 113  | 26.33 | 0.23× |
| 8   | 131K  | 16 | 15 × 12 = 180    | 49.5  | 0.27× |

**Even with a perfectly fused kernel, quasi-DEER is 4-15× slower than
sequential.** The bottleneck is fundamental: each scan iter must touch
the full `[B,H,T,N,N]` Jacobian D and residual b (N = 16× more memory
per position than sequential's K/V/decay).

## Scan/seq per-iteration ratio (parallelism regime study)

`scan_ms / seq_ms` at T=16K-131K:

| H | N | T | scan/seq ratio |
|---|---|---|---|
| 141 | 16 | 16K-32K | **1.2×** (scan slower) |
| 32  | 16 | 16K-262K | 0.31× (scan faster) |
| 8   | 16 | 65K-262K | 0.25× |
| 1   | 16 | 16K-524K | 0.25-0.40× |
| 141 | 8  | 16K | 0.35× |
| 32  | 8  | 16K | **0.12×** |

**Key asymmetry:** at production H=141, the sequential kernel already
saturates the GPU (141 programs × N=16 per timestep). The log-depth scan
has nothing more to parallelize, and pays N× more memory traffic.

To make quasi-DEER competitive end-to-end (iters × scan_ms < seq_ms with
13 iters), we'd need scan/seq < 0.08 — we measured best-case 0.12.

## Why quasi-DEER loses for E88

1. **Sequential forward is memory-bandwidth efficient** — it reads only
   K[B,H,T,N], V[B,H,T,N], decay[B,H,T]. Quasi-DEER reads D[B,H,T,N,N],
   b[B,H,T,N,N]. That's **N× more HBM per timestep**.

2. **Production H=141 saturates the GPU.** The sequential kernel has 141
   fully-parallel programs. Log-depth T scan gives no extra exploitable
   parallelism since the chain critical path is already hidden by H
   parallelism.

3. **Linear (not quadratic) convergence of quasi-DEER.** 12-16 iters for
   1e-4 residual, 26 for 1e-6. Full Newton reaches 1e-7 in 4 iters.

4. **Divergence risk.** Quasi-DEER diverges when `|K|` or `decay` grows
   beyond the tested production range. Real E88 uses L2-normalized K, so
   this stays safe in practice, but a training-time safeguard would be
   needed.

## Where quasi-DEER COULD win

- **Inference with very low B·H and long T** (B·H=1-4, T>=128K). Here
  sequential has no H-parallelism to exploit, and scan's log-depth
  becomes the dominant factor. But this regime doesn't match production
  training (H=83-141).
- **Smaller state (N=8)** gives scan/seq = 0.12× at H=32. Still not enough
  to overcome 13-23 iters.

## Recommendation

**Do NOT integrate quasi-DEER into PararnnHybridE88V2.** The existing
sequential Triton forward (`pararnn_seq_fwd_v2`) is near-optimal at
production shapes: it's fully H-parallel, memory-bandwidth-efficient,
and single-pass. Quasi-DEER's log-depth advantage is entirely swamped by
(a) 13-16× more iterations than full Newton and (b) N× more memory
traffic per iteration.

**If revisiting:** the promising direction is NOT quasi-DEER but
**chunked Parareal / ADMM-style coarse-fine splits** — use a cheap
linear/diag coarse solver for T-parallelism, then refine with a few
sequential sweeps on windows. The quasi-DEER scan could serve as the
coarse solver in such a scheme, but the overall speedup bound is still
set by the fine solver + communication.

## Interesting observations

- **Iteration count is T-independent in the convergent regime.** Whether
  T=1K or T=65K, ~12-13 iters to 1e-4 at N=16. This is qualitatively the
  best property of quasi-DEER, consistent with the Gonzalez et al. paper's
  claim of T-independent iter count.
- **Iters depend on N more than T.** N=8 → 23 iters, N=16 → 13, N=32 → 10.
  Larger N → smaller `|K[c]|` → smaller rank-1 term → fewer iters needed
  to correct for the dropped term.
- **Divergence is a real risk.** L2-normalized K is essential.
- **bf16 precision floor ≈ 5e-3 relative.** Probably fine for training
  but wouldn't support tight-precision Newton.
- **"Linear warm-start" helps modestly.** Initializing S_var with the
  linear recurrence (no tanh) gives 10 iters instead of 13. Not enough
  to change the verdict.

## Data-points worth preserving

- Sequential Triton forward `pararnn_seq_fwd_v2` at H=141 T=16K N=16 fp32:
  **7.4 ms** (best in-house kernel).
- Quasi-DEER Triton scan alone at same shape: **8.2 ms per iter** — not
  much slower than full sequential forward. The pure-scan primitive is
  healthy; the approach fails because 13 iters are needed.
- `tl.associative_scan` with tuple state `(D, b)` and a 2-input/2-output
  `@triton.jit` combine function **works correctly** (not just scalar —
  we're passing BLOCK_T×NN tuples). Useful precedent.
