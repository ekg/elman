# Custom Parallel-Scan Kernel for E88 — Implementation Plan

**Purpose**: parallelize E88's nonlinear sequential recurrence along the time
axis via Newton's method + parallel scan, exploiting the diag+rank-1
structure of E88's per-row Jacobian.

**Scope**: mathematical correctness first, performance second. Each phase
locks in its ground truth from the preceding phase. We never move forward
without verified numerical match.

---

## Ground truth: the existing sequential implementation

Our baseline for everything is E88's existing sequential implementation:
- CUDA kernel: `elman/cuda/lib/e88_fla_hybrid_gpu.cu.cc`
- PyTorch fallback: the vectorized loop in `elman/models/e88_fla_hybrid.py`

These agree with each other within bf16 tolerance. Call the result of
running them on input `X` and initial state `S_0` the **reference
trajectory** `{S_1, ..., S_T}`.

Every phase below must, at the end of its scan, reproduce this reference
trajectory within a measured tolerance.

---

## Mathematical framing

### The recurrence per row `i` of the matrix state

```
S_t[i, :] = tanh(decay_t · S_{t-1}[i, :] + delta_t[i] · k_t)
where  delta_t[i] = v_t[i] − S_{t-1}[i, :] · k_t
```

Row `i` evolves independently given the shared inputs `(k_t, v_t, decay_t)`.

### Per-row Jacobian

```
J_t[i] = ∂S_t[i, :] / ∂S_{t-1}[i, :]
       = diag(decay_t · tanh'(pre_t[i, :]))  -  (tanh'(pre_t[i, :]) ⊙ k_t) ⊗ k_t
       = D_t[i] − u_t[i] · k_t^T
```

where
- `D_t[i] ∈ R^n` is the diagonal vector
- `u_t[i] = tanh'(pre_t[i, :]) ⊙ k_t ∈ R^n`
- `k_t ∈ R^n` is the input key at time `t`

**Each row's Jacobian is diagonal-plus-rank-1.**

### Parallel Newton scan

Newton iteration on the simultaneous solve of all timesteps' states:
given current guess `{S_t}`, residuals `r_t = S_t − f(S_{t-1}, x_t)`, we
update via `δ_t = Σ_{s ≤ t} (Π_{u=s+1..t} J_u) · r_s` — a prefix product
over Jacobians, then a scan-weighted sum over residuals.

The prefix products are the computational core. Two structured matrices
combine as:

```
(D1 − u1 k1^T)(D2 − u2 k2^T)
  = D1 D2
    + (k1^T u2) · u1 k2^T         (scalar coeff)
    − (D1 u2) k2^T                 (rank-1)
    − u1 (D2 k1)^T                 (rank-1)
```

So the product of two diag+rank-1 matrices is **diagonal + rank-3**. More
generally, a product of m such matrices has rank ≤ m in its off-diagonal
part. We'll cap this at some rank `r` via truncated SVD (or Jacobi
rotations) at each combine step.

### Rank budget question

Critical unknown: how small can `r` be while preserving Newton convergence?
This is what Phase 2 is designed to measure empirically.

---

## Phase 1 — Pure-Python reference (naive, unstructured)

**Goal**: a mathematically unambiguous parallel-scan implementation, using
no structure exploitation. Dense n×n matrices throughout. Slow, correct.

**Steps**:
1. Take input `(k, v, decay, S_0)`, compute the reference trajectory via
   direct sequential recurrence (our existing impl).
2. Initialize guess: `S_t = 0` for all t (or `S_t = S_0`).
3. Newton loop:
   a. Compute residuals `r_t = S_t − f(S_{t-1}, x_t)` for all t in parallel
   b. Compute Jacobians `J_t = ∂f(S, x_t)/∂S |_{S=S_{t-1}}` via autograd —
      full n×n dense matrices
   c. Form prefix products `P_t = J_t · P_{t-1}` (sequential, but this is
      what the parallel scan will replace later)
   d. Compute update: `δ_t = Σ_{s ≤ t} (Π_{u=s+1..t} J_u) · (-r_s)` via
      prefix-sum-of-Jacobian-weighted-residuals
   e. Update `S_t ← S_t + δ_t`
   f. If `max |r| < tol`: break
4. Assert final `{S_t}` matches reference trajectory within
   `seq_len × ε_fp32`.

**Test**: sweep `(n, T)` in `{(4, 32), (8, 128), (16, 256), (32, 512)}`.
All must pass with `max |S_parallel − S_reference| < T × 1e-6`.

**Deliverable**: `experiments/pararnn_kernel/phase1_reference.py`

---

## Phase 2 — Structured propagation with rank truncation

**Goal**: represent prefix products as `(D, U, V)` where `D` is a diagonal
(length n), `U, V` are n × r, and enforce rank-r by SVD truncation at each
combine.

**Steps**:
1. Define combine op `(D1, U1, V1) ⊕ (D2, U2, V2) → (D3, U3, V3)`:
   - Compute full combined matrix: `M = D1·D2 + u1·k2 − D1·U2·V2^T − U1·V1^T·D2 + …`
   - Actually form the structured update: the diagonal part stays diagonal;
     the rest is a rank-≤(r1+r2+constant) matrix. Truncate to rank r via SVD.
2. Prefix-scan the combine across time.
3. Back out states `S_t` using the structured prefix products.

**Test**: for each `(n, T)` in the Phase 1 sweep, and for `r ∈ {2, 4, 8, 16}`:
measure `max |S_phase2(r) − S_phase1|`. Plot error vs `r`. Pick smallest `r`
where error is within Newton convergence tolerance (same order as Phase 1
error).

**Deliverable**: `experiments/pararnn_kernel/phase2_structured.py`

**Go/No-Go gate**: if error at `r=16` is still larger than Phase 1's
expected float32 error, the structured approximation is fundamentally
insufficient — fall back to dense-block kernel (Phase 3 alternative).

---

## Phase 3 — Triton kernel for structured combine (correctness-only)

**Goal**: implement Phase 2's associative combine in Triton, running on
GPU. Single head, single batch, single row of S. No perf target.

**Design**:
- One Triton program per row `i` of S
- State in shared memory: `(D, U, V)` triple, size n + 2·n·r
- Hillis–Steele scan across time dimension
- SVD at each combine — in Triton, implement via a few Jacobi rotations
  (QR-like, preserves the structured form, no full SVD needed)
- bfloat16 storage, fp32 accumulation

**Tests**:
1. Unit test: one combine of two random `(D, U, V)` triples, result
   matches Phase 2's Python reference within bf16 rounding
2. Short-sequence scan test: T=64, n=16. Output states match Phase 1
   (dense reference) within `seq_len × ε_bf16 ≈ 2e-3`
3. Full-sequence scan test: T=512, n=32. Same tolerance.

**Deliverable**: `experiments/pararnn_kernel/phase3_triton.py`

---

## Phase 4 — Multi-batch, multi-head, E88 shape

**Goal**: extend Phase 3 to process E88's full tensor shape
`[B, H, T, n, n]` by launching one program per `(batch, head, row)`.

**Tests**:
1. Forward output of the full E88 layer using Phase 4's scan matches
   `elman/models/e88_fla_hybrid.py`'s forward within bf16 tolerance for
   B=1, H=112, T=512, n=32
2. Same for T=2048 and T=8192 (stress test for scan depth)

**Deliverable**: `experiments/pararnn_kernel/phase4_e88_layer.py`

---

## Phase 5 — Backward pass

**Goal**: gradient flow through the scan. Two options:
(a) Re-run the scan in reverse on transposed Jacobians (ParaRNN's approach)
(b) Save forward prefix products; reverse-scan analytically

**Tests**:
1. Finite-difference gradient check at small n, short T
2. Agreement with `e88_fla_hybrid.py`'s CUDA backward on matched inputs
   within bf16 tolerance, full E88 shape

**Deliverable**: `experiments/pararnn_kernel/phase5_backward.py`

---

## Phase 6 — Integration + performance

**Goal**: ship as an autograd function that plugs into the existing E88
layer behind a feature flag.

**Perf targets**:
- Forward at T=512: at least parity with sequential
- Forward at T≥2048: ≥ 2× speedup
- Forward at T≥8192: ≥ 4× speedup
- Backward: similar ratios

**Tests**:
- E88 layer forward/backward correctness match across all supported T
- Training-step numerical parity over 10 steps: loss trajectory matches
  sequential within ~1e-4 relative

**Deliverable**: a runtime flag `--use_pararnn_kernel` on `train.py` that
swaps in the parallel path.

---

## Test infrastructure (shared across phases)

Each phase lives in `experiments/pararnn_kernel/` and includes:
1. `reference.py` for that phase — the canonical implementation
2. `test_<phase>.py` — pytest-style test file running the numerical checks
3. The reference trajectory is always the output of E88's sequential
   impl on the same inputs

A single `experiments/pararnn_kernel/run_all.py` runs every phase's tests
end-to-end. CI-able.

---

## Open questions to track

- Is `r=4` sufficient for Newton convergence? (Phase 2 answers this.)
- Does Triton's compile time scale to n=32 better than CUDA C++? (Phase 3
  answers this.)
- What's the wall-clock crossover point where parallel beats sequential
  (depends on kernel launch overhead + reduction depth)? (Phase 6.)

---

## Status (updated as we go)

- [x] Plan documented
- [x] **Phase 1 — Pure-Python reference** (2026-04-22):
  All sweep points (T=32..512, n=4..32) converge in 5–10 Newton iters;
  max|sequential − scan| ≤ 4.4e-12 at float64. Forward-substitution
  solve works cleanly.
- [x] **Phase 2 — Structured + rank truncation** (2026-04-22):
  **r=1 is sufficient** — structured scan matches Phase 1 dense to
  machine epsilon (< 6e-16 at float64) across all tested shapes and
  seeds. The low-rank approximation is essentially lossless because
  each step's Jacobian IS rank-1 in its off-diagonal part, and
  combined products retain a dominant rank-1 mode. At extreme input
  amplitudes (not in the realistic E88 regime) Newton itself fails to
  converge, but that's not a truncation issue.

  **Consequence**: the kernel simplifies dramatically. State per
  prefix is 4n scalars = 256 B at n=32 bf16. Combine needs no SVD —
  just a few vec/mat products. Fits in registers with huge occupancy
  headroom.
- [ ] Phase 3 — Triton kernel, single head (r=1 fixed, no SVD)
- [ ] Phase 4 — Multi-head, multi-batch
- [ ] Phase 5 — Backward pass
- [ ] Phase 6 — Integration + benchmark
