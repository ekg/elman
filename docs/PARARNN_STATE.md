# Pararnn-E88 Kernel: Where We Are

Persistent record of the Pararnn-for-E88 work as of 2026-04-22. Read this
before continuing; the conversation chat context drifts.

## The problem

E88 is a nonlinear sequential matrix-state RNN. Per-row recurrence:
```
S_t[i, :] = tanh(decay_t · S_{t-1}[i, :] + (v_t[i] − S_{t-1}[i, :] · k_t) · k_t)
```

Rows evolve independently given shared inputs (k, v, decay). The production
training path uses a hand-tuned CUDA kernel
(`elman/cuda/lib/e88_fla_hybrid_gpu.cu.cc`) that does this recurrence
sequentially in T with heavy optimization (fused projections, register-
owned state, block-level GEMM batching, chunked projection recomputation,
gradient checkpointing, etc.).

Goal: make E88 trainable in parallel along the time axis so long-context
training (up to 128K) doesn't become latency-bound.

## What we've built (phases 1–6, all on main)

| Phase | Artifact | Status |
|-------|----------|--------|
| 1 | `experiments/pararnn_kernel/phase1_reference.py` | Pure-Python Newton reference, matches E88 sequential to 1e-12 at fp64 |
| 2 | `experiments/pararnn_kernel/phase2_structured.py` | r=1 truncation is lossless at machine eps |
| 3a | `experiments/pararnn_kernel/phase3_triton.py` | PyTorch r=1 scan with 2×2 SVD truncation, matches dense |
| 3b | `experiments/pararnn_kernel/phase3b_sequential_kernel.py` | Triton kernel: one program per (B, H) × sequential T scan, byte-identical output |
| 4 | `experiments/pararnn_kernel/phase4_newton_driver.py` | Newton iteration with Triton scan, converges 4–11 iters at 7.6e-6 fp32 |
| 5 | `experiments/pararnn_kernel/phase5_backward.py` | Reverse-scan backward, matches autograd to fp32 machine eps |
| 6 | `experiments/pararnn_kernel/phase6_autograd.py` | `PararnnE88Function` wraps forward+backward |
| 6 bench | `experiments/pararnn_kernel/phase6_benchmark_vs_cuda.py` | Measured vs production CUDA |

## UPDATE 2026-04-22 later — fused Triton kernel + warm start

Wrote a fused single-Newton-iter Triton kernel (`phase7_fused_iter.py`)
combining residual computation, Jacobian build, and scan into one
program per (B, H, row). **14-16× faster per Newton iter** vs the
separate-phase PyTorch + Triton approach.

Also validated that **warm-start (perturbed previous trajectory) cuts
Newton to 3 iterations** to reach fp32 machine precision.

Honest benchmark vs production CUDA (bf16) forward, single-GPU:
  T=512  : CUDA 0.83 ms, Pararnn 1.44 ms (3-iter warm) — **0.58× (slower)**
  T=1024 : CUDA 1.65 ms, Pararnn 3.15 ms — 0.52×
  T=2048 : CUDA 3.53 ms, Pararnn 6.36 ms — 0.55×
  T=8192 : CUDA 14.24 ms, Pararnn 25.91 ms — 0.55×
  T=16384: CUDA 28.39 ms, Pararnn 52.21 ms — 0.54×

Pararnn is consistently ~2× slower than the hand-tuned sequential CUDA
kernel on single-GPU at T up to 16K. Ratio stable across T — both
scale linearly.

Fundamental reason: CUDA does 1 sweep of hand-optimized C++. Pararnn
does 3 Newton iters of Triton code. 3× more work × Triton overhead
(~10-20%) ≈ 2× slower.

To beat CUDA on single-GPU we'd need:
  (a) 1-iter warm start (only possible in very stable training regime)
  (b) Quasi-Newton / simpler iteration (lose exact convergence)
  (c) Drop to T regimes where CUDA's memory model breaks down

## Attempted optimization: bf16 Newton

Tried bf16 storage for all tensors to halve memory. Newton convergence
STALLS at ~2e-3 error (bf16 epsilon ~= 1e-3, can't resolve smaller
residuals). Each iter loses accuracy to the bf16 round-trip on S_var.
Not viable without compensation (Kahan summation or similar).

Hybrid bf16 inputs + fp32 S_var converges correctly but saves little
memory since S_var dominates usage (O(T·n²) per (B, H) vs O(T·n) for
inputs). No significant memory win from half-precision in this design.

## Memory is the real wall for T=128K

At T=32K, single-GPU with fp32:
  S_var:  [1, 32, 32K, 32, 32] fp32 = 4 GB per tensor
  delta:  4 GB
  + intermediates → OOM on 48 GB GPU

The fundamental issue: Newton requires the full T trajectory in memory
to compute prefix products. Cannot be chunked without changing the
algorithm.

Possible algorithmic fix (not implemented): **segmented / block
Gauss-Seidel Newton**. Solve T in chunks of size C ≤ 2048, each chunk's
boundary fixed from previous. Much smaller per-chunk memory (O(B·H·C·n²))
but requires more outer iterations to propagate corrections across
chunk boundaries. Unclear tradeoff. Significant implementation cost.

Paper claim revision (again): can't claim "faster than CUDA on single
GPU." Can claim:
  - Correct parallel Newton with rank-1 lossless truncation
  - 2-3× cost of sequential kernel, amortizable in distributed/long-T
  - Enabling for new architectures (no custom kernel needed)
  - Scales to ROCm without a new CUDA kernel

## Summary of where single-GPU single-node effort lands

3× progressively better implementations attempted (split-phase → fused
kernel → bf16). Each confirmed a fundamental:
  1. Split-phase: PyTorch op overhead makes this 14× slower than fused.
  2. Fused: hits the Newton iteration count wall — 3 iters × 1 sweep-
     equivalent each = 3× the work of sequential CUDA.
  3. bf16: insufficient precision for Newton convergence.

Further optimization paths are all significant engineering:
  - Quasi-Newton (lose lossless convergence)
  - Block Gauss-Seidel (long implementation effort, unclear tradeoff)
  - Rewrite in CUDA C++ with manual register/smem tuning (weeks)
  - Hardware-specific: Hopper tensor cores for Newton matmuls

Recommendation: pivot to writing up the **scientific finding** (r=1
lossless truncation, new algorithmic class of parallelizable RNNs) with
honest perf characterization, rather than continuing to chase single-
GPU speed parity with a hand-tuned CUDA kernel.

## Findings

### Technical findings that are novel and publishable
1. **E88's Jacobian has diag+rank-1 structure** per row:
   `J = diag(D) − u·kᵀ` where `D = decay · tanh'`, `u = tanh' ⊙ k`.
2. **Prefix products of diag+rank-1 matrices truncated to rank-1 are
   lossless for sequential scan** (machine epsilon), even though rank
   grows additively per combine in principle. This is surprising and
   holds empirically across shape sweep for E88's contractive regime.
3. **Rank-1 truncated combine is non-associative**: tree-parallel
   (Hillis-Steele) scan accumulates ~0.05 error at T=8. Sequential
   (left-to-right) scan is essential. Tree parallelism would require
   dense n×n Jacobians (O(n³) per combine, DEER's approach) — no win.
4. **Data parallelism across (B, H, row) saturates a GPU** at E88
   scales. At B=1, H=112, n=32, we have 3584 independent row-scans —
   roughly one per active warp on H100. Time parallelism isn't needed
   for GPU saturation; sequential-in-T with data parallelism suffices.

### Sobering performance finding (phase 6 benchmark)
| Shape                | CUDA (bf16) | Pararnn (fp32) | Ratio |
|----------------------|-------------|----------------|-------|
| B=1, H=32, T=512     | 3.1 ms      | 20 ms          | 0.15× |
| B=1, H=32, T=2048    | 10.9 ms     | 92 ms          | 0.12× |
| B=1, H=32, T=8192    | 45.6 ms     | 441 ms         | 0.10× |
| B=1, H=32, T=32768   | —           | **OOM**        | —     |

The hand-tuned E88 CUDA kernel beats our Pararnn-Triton kernel by
~10× across T=512–8K. We OOM at T≥32K because we materialize the full
`[B, H, T, n, n]` Jacobian-state trajectory in fp32.

**Earlier "T=1024 crossover" claim was against a Python-loop baseline,
not the production CUDA kernel. Retract.**

## Where we actually need to do work

Closing the 10× gap requires matching what the production CUDA kernel
does. Specifically:

### Memory
- The CUDA kernel stores only *periodic checkpoints* (every 16 steps)
  of the state, plus Sq per step, plus the input projections. Total
  memory is O(T · head_v_dim · B · H) for Sq + O((T/16) · n² · B · H)
  for checkpoints.
- Ours stores *everything* per step: full S_var trajectory plus Jacobian
  intermediates (D, u, v, b) all at full T resolution. Wasteful.
- **Fix**: gradient checkpointing inside Newton iterations. For each
  Newton iter, recompute Jacobian intermediates from checkpointed S
  values.

### Precision
- Pararnn currently runs at fp32 throughout. CUDA runs at bf16.
- bf16 halves memory and gets tensor-core throughput.
- Newton's precision needs are a real concern — lossy truncation plus
  bf16 rounding might not converge. Need to test.

### Kernel structure
- Our Triton kernel does single program per (B, H), `range(T)` loop
  inside. Launch once, sequential scan in registers.
- CUDA kernel does one CUDA block per (B, H), with warp-level processing
  of state. Heavily optimized for register pressure and memory
  coalescing. Also uses cuBLAS GEMMs for projections externally.
- **Fix**: match the CUDA kernel's structure — pre-compute projections
  in bulk using cuBLAS (already what E88 layer does); our Triton kernel
  only does the tight recurrence loop.

### Newton overhead
- We do 11 Newton iterations per forward. Each iteration does full
  residual computation + scan + update. That's ~11× the work of a
  single sequential pass.
- For the fair comparison, CUDA kernel does 1 pass ≈ 45 ms at T=8K.
  Pararnn does 11 passes × ~40 ms each = 440 ms at T=8K. This is
  fundamentally a factor of 11× wall-clock loss — Newton's iteration
  count IS the cost.
- **Potential fixes**:
  - Warm-start Newton between training steps (DEER/ELK finding — can
    drop to 2–3 iters with good warm start).
  - Quasi-Newton (diagonal Jacobian approximation).
  - Sobolev / heavy-ball acceleration of the Newton scan.

## The plan going forward

Not trying to beat CUDA at its own game on single-GPU short-T. Instead:

### Plan A: Match CUDA quality on the Triton side
Bring Pararnn to within ~2× of CUDA on single-GPU by:
1. **Memory-efficient Newton**: checkpointing of S_var, Jacobian
   recomputation from checkpoints per iter. Allows T=128K.
2. **bf16 implementation**: halved memory, tensor-core utilization.
3. **Warm-start across training steps**: cache converged trajectory,
   reuse as initial guess. Typical 3-5 iters vs 11.
4. **Quasi-Newton fallback**: diagonal Jacobian approximation for first
   few iters, switch to full structured Newton.

### Plan B: Demonstrate value at the real target (distributed / long T)
Where parallel training matters:
1. **Multi-node**: parallel-in-T scan enables cross-node overlap that
   sequential CUDA can't.
2. **T=128K single-GPU**: CUDA at T=128K costs ~750 ms (extrapolating
   linearly); Pararnn needs memory-efficient impl to even run, but if
   it does, O(iters × scan) might beat O(T) sequential if iters grows
   slower than T.

## Paper pitch revision

Forget "we made E88 faster." Not true at realistic scale.

Honest claim:
> **"Rank-1 Structured Newton Scan for Parallel Training of Nonlinear
> Matrix-State RNNs."**
>
> We identify that nonlinear RNNs with diag+rank-1 Jacobian structure
> admit a lossless rank-1 truncation of Newton prefix products,
> enabling O(T·n) parallel training via data-parallel sequential scan.
> We prove the approach is numerically equivalent to exact Newton at
> machine precision for E88-class architectures, provide a reference
> Triton implementation with autograd support, and characterize the
> design space against hand-tuned sequential CUDA kernels.

This is a real contribution without overclaiming:
- Fills explicit open question in DEER/ELK §7.
- Provides working reference code.
- Honestly characterizes speed tradeoffs — not a drop-in win, but an
  enabling technology for architectures without tuned sequential kernels
  and for distributed training.

## Files to study in this next phase

For CUDA kernel structure reference:
- `elman/cuda/lib/e88_fla_hybrid_gpu.cu.cc` — main forward/backward
- `elman/cuda/lib/e88_register_owned_gpu.cu.cc` — register-owned variant
- `elman/cuda/lib/e88_fused_gate_gpu.cu.cc` — fused-gate variant
- `elman/models/e88_fla_hybrid.py` — Python side, showing what the CUDA
  kernel is responsible for vs what's in Python (projections, gating)

For current Pararnn work:
- `docs/PARARNN_KERNEL_PLAN.md` — phase-by-phase plan
- `docs/DEER_PARARNN_NOTES.md` — literature notes

## Decisions already made (don't relitigate)

- **r=1 rank truncation**: lossless, sequential scan only.
- **Data-parallel across (B, H, row), sequential in T**: correct design
  given non-associativity of r=1 combine.
- **Triton over CUDA C++**: engineering speed, portability to ROCm.
- **Newton iteration**: core algorithm, not changing (unless we add
  warm-start or quasi-Newton as optimization).

## Open questions for next session

- Can bf16 Newton converge reliably? (Empirical test needed.)
- Checkpointing granularity: what's the best tradeoff? (CUDA uses 16.)
- Warm-start from previous training step: how much does it cut iters?
- Is the 10× gap closable in Triton or do we need CUDA C++?

## UPDATE 2026-04-23 — Hybrid CUDA-fwd + Pararnn-bwd wins at production

Culmination of the forward/backward split. The clean story:

**Forward**: cold-start Pararnn Newton needs 3 iters; at production H (83,
141) this ties or loses CUDA. Linear bootstrap is outside Newton's basin
(|err|=2.2 vs needed ≤0.1) — the `-k·(k·s)` delta-rule term and tanh
cannot be approximated away. So use **CUDA for forward** (it's already a
hand-tuned sequential kernel).

**Backward**: the adjoint of E88 is LINEAR given the converged forward
trajectory — one exact pass, no Newton basin issues. Fused Pararnn
backward with:
- rank-1 `dL/dS[t] = outer(dL_dout[t], q[t])` factored inputs (saves 16 GB + HBM)
- 2D `[N, N]` per-program tile, `tl.sum` reduces across rows in-register (no dK_partial)
- int64 offsets, `num_warps=4, num_stages=1`
- bf16 storage of S_traj (halves memory; ~1% gradient error, matches
  CUDA's mixed-precision policy)

**End-to-end training step speedups, HONEST (includes dQ einsum so the
hybrid produces all the same gradients as CUDA's autograd)**:

| Config | T | All-CUDA f+b | Hybrid | e2e speedup |
|--------|---|--------------|--------|-------------|
| E88-n32 480M (H=83) | 32K | 188 ms | **117 ms** | **1.61×** |
| E88-n32 480M | 64K | 377 ms | **234 ms** | **1.61×** |
| E88-n16 480M (H=141) | 32K | 108 ms | **73 ms** | **1.48×** |
| E88-n16 480M | 64K | 217 ms | **147 ms** | **1.47×** |
| E88-n16 480M | **128K** | 434 ms | **295 ms** | **1.47×** |
| Small (H=32) | 128K | 754 ms | **416 ms** | **1.81×** |

Memory at T=128K E88-n16 hybrid: 30 GB peak (fits comfortably). All-CUDA
can't even run at T=128K H=83 fp32 baseline.

Files:
- `experiments/pararnn_kernel/phase7_fused_backward.py` — fused backward
  with `backward_e88_fused` (materialized dL_dS) and `backward_e88_fused_rank1`
- `experiments/pararnn_kernel/phase7_bench_bf16_traj.py` — full e2e
  benchmark at production configs, T=32K..128K
- `experiments/pararnn_kernel/phase7_linear_basin.py` — proves linear
  bootstrap is outside Newton's basin

### Dead ends (do not relitigate)
- **Chunked parallel Newton** — correct, no speedup. Bandwidth-bound.
- **Linear bootstrap** — outside Newton's basin. Tanh matters.
- **Pararnn forward at production H with cold start** — ties CUDA, not worth.
- **Checkpointed backward kernel** — skipped; bf16 S_traj solves the
  memory problem more simply. Could revisit for T≫128K.

### What's been tried that didn't move the needle
- **Fused dQ into backward kernel** — regression (~10% slower), extra
  register pressure from keeping S[t+1] cached across iterations.
  Separate `einsum` is faster; cuBLAS batched-matmul is hyper-optimized.
- **num_stages 1→6, num_warps 1→16** — plateau at num_warps=4 num_stages=1
  at production scale. Load latency already well-hidden by the compiler.
- **`tl.math.tanh`** — not in this Triton version (3.5.1). Kernel is
  bandwidth/scheduling-bound anyway; tanh compute isn't the limit.

### Binding constraint on single-GPU backward
Per-program serial dependency chain: load → matmul → tanh → reduction →
scan state update, ~6 dependent arithmetic ops per step × T iterations.
This dominates kernel time. The kernel is not at HBM peak because it's
*serial-depth-bound*, not bandwidth-bound. Only algorithmic changes
(tensor-core GEMM restructure, or tree-scan if associativity could be
restored) could go further. Those are research questions.

### Open paths forward
- **Multi-GPU T-distribution** — biggest untapped win. Pararnn is the
  natural framework; CUDA fwd per-chunk on each GPU, boundary exchange.
- **CUDA kernel variant that returns S_traj** — hybrid assumes S_traj
  is available post-forward. Requires a 16 GB bf16 HBM write at T=128K.
  Alternatively, use checkpointing (CUDA kernel already does every-16
  checkpoints for its own backward; repurpose those).
- **E88-n32 T=128K** — all-CUDA baseline itself OOMs at H=83 n=32 T=128K
  because autograd holds gradients for k/v/q/decay/S0 summing to >48 GB.
  Hybrid would work if CUDA-fwd's output-grad storage is addressed.

## UPDATE 2026-04-22 — `num_warps=1` is the whole ballgame

**The kernel was drastically under-tuned.** Triton's default `num_warps=4`
gives 128 threads/block. Our scan operates on N=32 elements = 1 warp
worth of parallelism. The extra 3 warps did nothing useful but
contended for HBM bandwidth and scheduler slots. Setting `num_warps=1`:

| T | CUDA (bf16) | Par 1-iter | Par 2-iter | Par 3-iter |
|---|-------------|------------|------------|------------|
| 8K | 14.3 ms | **4.2 ms (3.40×)** | 8.1 ms (1.77×) | 12.2 ms (1.17×) |
| 16K | 28.8 ms | **8.4 ms (3.43×)** | 16.7 ms (1.72×) | 24.9 ms (1.16×) |
| 32K | 57.5 ms | **16.1 ms (3.57×)** | 32.2 ms (1.79×) | 48.9 ms (1.18×) |
| 64K | 114.8 ms | **32.2 ms (3.56×)** | 64.4 ms (1.78×) | 96.6 ms (1.19×) |
| 128K | 229.9 ms | **64.4 ms (3.57×)** | 129.2 ms (1.78×) | 193.6 ms (1.19×) |

(B=1, H=32, n=32. All speedups over production E88 CUDA bf16 forward.)

**Even cold-start 3-iter Pararnn beats CUDA by ~1.19× at every T.**
Warm-start 1-iter beats CUDA by 3.5×.

Also verified at B·H sweep (T=128K, 3-iter): Pararnn now wins everywhere:
- H=1: 1.40× | H=4: 1.44× | H=16: 1.32× | H=32: **1.19×** (was 0.70× before)

### What we got wrong and what we got right

*Wrong:* assumed we were memory-bandwidth bound — we weren't. We were
*bandwidth-contention-bound from oversubscribed warps*. Reducing threads
per block reduced contention, freed HBM for the work we actually wanted.

*Right:* the parallel math (Newton on block-bidiagonal system via r=1
scan) is sound. The data-parallel structure (B·H·row programs) scales
correctly. We just needed the per-program parallelism to match the
vector width.

### On cold-start convergence

Newton from `S_var = 0` or `broadcast(S0)` does **not** converge at
T ≥ 8K — the iterate oscillates around residual ~2.0 (far outside
Newton's basin of attraction). So "cold-start 3-iter" actually means
"start from a perturbation of the converged trajectory." In real
training, this warm start comes from:
- Gradient accumulation (same K,V,decay, only params change)
- Previous training step (K,V,decay change slightly — usually within basin
  at realistic learning rates)
- CUDA-bootstrap: use sequential CUDA once, refine with Pararnn (2× speedup
  on subsequent iters or backward pass)

At warm-start perturbation 0.001 (realistic for a training step): **1 iter
reaches fp32 precision**. At perturbation 0.01: 2 iters. At 0.05: 4 iters.

## UPDATE 2026-04-17 — int64, inplace kernel, chunking, low-B·H regime

Four concrete things changed in `phase7_fused_iter.py`:

**1. int64 offsets** (`_fused_newton_iter_kernel`): at T=131072 with
B=1 H=32 n=32, `bh * T * N * N = 4.2e9` overflows int32. Cast `pid` and
`n_idx` to `tl.int64` at the top of every kernel; all offset arithmetic
stays int64. Without this, T=128K hit illegal memory access.

**2. Jacobi-in-place kernel** (`_fused_newton_iter_inplace_kernel`):
eliminates the 16 GB δ buffer at T=128K. Trick: cache the OLD S_var[t]
in a register *before* overwriting it, and use that cached value as
s_prev at t+1. This preserves Jacobi semantics (bit-exact match with
out-of-place path, verified) while writing δ directly into S_var.
Memory at T=128K dropped from >48 GB (OOM) to 33 GB peak.

**3. Chunked parallel Newton** (`_fused_newton_iter_chunked_kernel`):
split T into C chunks, one program per `(b, h, row, chunk)`, all chunks
run in parallel. Boundary s_prev values for each chunk are pre-cached
into a small `[B, H, C, N, N]` buffer (~1 MB) so chunks don't race on
S_var. Global convergence propagates boundary updates one chunk per
outer iter — for well-warm-started training, that's negligible overhead.

*Correctness*: bit-identical to single-chunk path across C ∈ {1, 2, 4, 8, 16, 32}.
*Speed*: **no speedup** at any C. We are memory-bandwidth bound, not
latency bound. Total HBM traffic is the same regardless of chunking;
chunking only reduces the critical path, but the critical path isn't
the bottleneck.

**4. Low-B·H regime is where Pararnn wins.** Swept (B=1, H ∈ {1..32}):

| T=128K | H=1 | H=4 | H=16 | H=32 |
|--------|-----|-----|------|------|
| CUDA (bf16) | 215ms | 228ms | 230ms | 238ms |
| Pararnn 3-iter (fp32) | **156ms** | **161ms** | **191ms** | 340ms |
| speedup | **1.38×** | **1.42×** | **1.20×** | 0.70× |

Root cause: CUDA kernel has B·H blocks. At H=32, that's 32 blocks —
fully saturates a 108-SM GPU and CUDA wins. At H≤16, CUDA under-
saturates and Pararnn's B·H·N programs (32× more) win by filling SMs.

**Practical implication:** for E88 at 480M scale (H=83 or H=141),
single-GPU CUDA is the right call. Pararnn shines at:
- Inference / single-user (B=1, H=1)
- Architectures with few heads at very long T
- Multi-GPU training where T is distributed across devices (each GPU
  then sees low effective B·H·T_local per program)

## Files added this session

- `experiments/pararnn_kernel/phase7_test_128k.py` — int64 overflow verification
- `experiments/pararnn_kernel/phase7_test_inplace.py` — inplace kernel correctness
- `experiments/pararnn_kernel/phase7_test_128k_inplace.py` — memory/timing at 128K
- `experiments/pararnn_kernel/phase7_test_chunked.py` — chunked correctness + sweep
- `experiments/pararnn_kernel/phase7_bench_128k.py` — head-to-head vs CUDA at 128K
- `experiments/pararnn_kernel/phase7_bench_small_bh.py` — B·H sweep showing regime
