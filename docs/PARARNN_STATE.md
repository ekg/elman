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
