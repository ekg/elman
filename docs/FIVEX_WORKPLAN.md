# Work plan: push from 1.98× → ~5× on single-GPU training step

## Current baseline (committed, measured, bit-exact correct)

**Hybrid training step at production E88 configs:**

| Config | All-CUDA f+b | Hybrid (ADMM fwd + Pararnn-fused bwd) | Speedup |
|---|---|---|---|
| E88-n16 480M T=65K | 217 ms | 110 ms | **1.98×** |
| E88-n32 480M T=65K | 377 ms | 156 ms | **2.41×** |

This is the number to beat. Every phase below has a measurable delta against it.

## Framing

Four orthogonal knobs, each pulls on a different constraint:

| Knob | What it attacks | Rough e2e headroom | Risk | Cost |
|------|-----------------|--------------------|------|------|
| (A) **Warm-start ADMM** from previous training step | Collapse 2 outer iters → 1 | ~1.4× forward, ~1.2× e2e | Low (math works, just plumbing) | 2-4 hr |
| (B) **Coarse solver** for cold-start | Ensure 1-iter even on fresh state | Same ceiling as (A) but no warm-start dependency | Moderate (coarse accuracy Q) | 4-8 hr |
| (C) **Faster backward** (CUDA rewrite w/ prefetch) | Break latency-bound backward | ~1.3× e2e | Low-moderate (pure kernel work) | 1-2 days |
| (D) Production model config (n=16 vs n=32) | Not a speedup per se — which model to ship | n/a (informational) | None | 1 hr |

**Stacking**: (A) and (B) are alternatives addressing the same thing (warm-start vs cold-start); they both aim at the same 1 iter target. (C) is orthogonal to both. (D) informs which config to optimize hardest for.

Multiplicative estimate if all land: 1.98× × 1.4× × 1.3× ≈ **3.6× at E88-n16 production**, 2.41× × 1.4× × 1.3× ≈ **4.4× at E88-n32**. That's a plausible path to the 5× target but requires all three to work.

## Proposed execution order

Principle: **validate independently first, compose after**. Ship a measured improvement at each phase.

### Phase 0 — lock in the baseline (30 min)
- Re-measure hybrid at full production T range (4K, 16K, 32K, 64K, 128K) for both n=16 and n=32
- Single doc page with the numbers we're beating
- Establish the dQ-included "honest" e2e measurement as the reference benchmark
- **Deliverable**: `docs/FIVEX_BASELINE.md` with a single table

### Phase 1 — warm-start ADMM (highest ROI per hour)
- **Hypothesis**: if boundaries from previous training step are within the convergence basin, ADMM converges in 1 outer iter instead of 2
- Synthetic test: take converged boundaries, perturb them by realistic-step-size noise (~1e-2), measure outer iters to re-converge
- If single iter suffices, integrate into `admm_cuda_forward` as an optional `init_boundaries` argument
- **Deliverable**: measured 1-iter convergence test + new API accepting warm boundaries
- **Expected wins** at T=65K: 
  - E88-n16: 110ms → ~85ms = **2.55× e2e**
  - E88-n32: 156ms → ~115ms = **3.28× e2e**
- Risk: batch-to-batch variance might be too large for 1-iter convergence; if so, fall back to 2 iters (no regression)

### Phase 2 — coarse solver for cold-start
- **Hypothesis**: a cheap O(log T) coarse solver (e.g., linear E88 with `decay·S[t-1] + outer(V, K)` — no tanh) gives boundaries close enough for 1-iter Newton convergence even from fresh state
- Implement cheap coarse solver (PyTorch first for correctness, CUDA if needed for speed)
- Combine: run coarse solver → feed boundaries to ADMM → outer iter
- **Deliverable**: e2e benchmark showing 1-iter convergence from cold start
- **Expected wins**: same as Phase 1 but for fresh batches (no warm-start required)
- Risk: coarse approximation error might exceed basin; may need 1.5-iter equivalent
- If Phase 1 works and we're always warm-started in real training, Phase 2 is nice-to-have, not essential

### Phase 3 — faster backward (CUDA rewrite)
- **Hypothesis**: current Triton fused backward is latency-bound on dependent load chain (same issue as forward HBM analysis showed). CUDA version with `cp.async` prefetching can hide it.
- Profile current backward kernel: is it compute/latency/bandwidth bound? (Run microbench, count cycles per step)
- Write CUDA C++ fused backward equivalent to `phase7_fused_backward.py`
- Add async prefetching for next-step loads
- **Deliverable**: CUDA backward kernel with correctness PASS vs Triton version, 1.5-2× speedup
- **Expected win**: backward 75ms → 40ms at T=65K H=141, e2e gains another 1.25×
- Risk: kernel engineering always has surprises; could stall if the backward is actually memory-bandwidth bound (unlikely given current 2× gap)

### Phase 4 — composition + production config selection
- Combine all above into one integrated pipeline
- Re-measure at production configs for both n=16 and n=32
- Identify final best config per shape
- **Deliverable**: final e2e speedup numbers, commit, state doc

### Phase 5 — training-loop integration
- Wire the hybrid into `train.py` as a drop-in replacement for CUDA forward/backward
- Verify no regression in training loss curve on a short run
- Measure wallclock training tokens/sec vs baseline
- **Deliverable**: a training run with the new pipeline, loss matches baseline within noise, wall-clock speedup matches microbenchmark

### Phase 6 — stretch: push individual components further
Only if target 5× not reached. Concrete options:
- ADMM with 4+ outer iters to get P-level speedup past the GPU-saturation cap (requires super-quick boundary propagation)
- Reduce Pararnn backward memory footprint further (enable bigger P in ADMM)
- Mixed-precision tricks (bf16 scan state, fp32 reductions)

## Measurement discipline

Every phase has two measurements:
1. **Microbenchmark**: isolated kernel/operation, T sweep, H sweep, correctness vs gold-standard
2. **E2e**: full training step with all gradients, compared to all-CUDA baseline

Both measurements committed. No phase closes without bit-exact or documented-tolerance correctness.

## Shipping criteria

At the end of each phase, commit code + update `docs/FIVEX_BASELINE.md` with the new numbers. Don't claim improvements without measurement.

## What I am NOT planning to do

- More tree-scan variants (shown not viable for E88 shape)
- Custom tensor-core primitives (existing WMMA sufficient when applicable)
- Multi-GPU work (user deferred)
- Algorithmic changes to E88 itself (change of model, not engineering)

## Request for approval

Proceed with Phase 0 → 1 → 2 → 3 → 4 → 5 in that order?

If you want to prioritize differently (e.g., do Phase 3 first since it's pure kernel work, or skip Phase 2 if Phase 1 is guaranteed in real training), say so here and I'll adjust.
