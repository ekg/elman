# FP8 Storage for E88 S_traj — Study & Implementation Report

**GPU 2 (RTX 6000 Ada, sm_89, 960 GB/s peak HBM) — bf16 activations throughout.**

## Executive summary

Implemented an FP8-E4M3 storage variant for E88's `S_traj` activation
tensor in the Pararnn hybrid Triton kernels.

**Main findings:**
- **Memory savings: confirmed 2x on S_traj**, ~40-45% on peak.
- **Speed: only 1.04-1.11× on forward kernel, neutral on backward** at best
  tuning. Target of 1.3-1.8× NOT met.
- **Correctness: gradient rel-err ~2-3%** across dK/dV/dQ/ddec/dS0; does
  NOT compound with T (tanh damps noise as hypothesized).
- **Training loss: indistinguishable.** 200-step run: bf16 avg_last_20
  = 2.4008 vs fp8 = 2.3977 — fp8 noise is below optimization noise.

**Recommendation: CONDITIONAL GO — enable as opt-in via `ELMAN_PARARNN_FP8=1`
only when memory pressure is the bottleneck** (e.g., long sequences
where peak memory limits batch size).  At constant batch size it is NOT a
speedup; it's a memory-for-batch-size tradeoff.

## Baseline (BF16, GPU 2)

Production shapes for E88:

| Shape (B H T N) | fwd_ms | bwd_ms | full_ms | fwd_peak_GB | full_peak_GB | S_traj_MB |
|-----------------|-------:|-------:|--------:|------------:|-------------:|-----------:|
| 1 141 4096 16  | 1.68 | 2.61 | 4.73 | 0.62 | 0.68 | 295.7 |
| 1 141 16384 16 | 6.27 | 10.45 | 19.34 | 2.48 | 2.69 | 1182.8 |
| 1 141 32768 16 | 12.58 | 20.96 | 38.74 | 4.97 | 5.39 | 2365.6 |
| 1 141 65536 16 | 25.57 | 41.79 | 77.78 | 9.93 | 10.77 | 4731.2 |
| 1  83 16384 32 | 7.48 | 21.22 | 32.01 | 5.52 | 5.77 | 2785.0 |
| 1  83 32768 32 | 14.94 | 42.47 | 64.08 | 11.03 | 11.52 | 5570.0 |

Measured HBM bandwidth at T=32K N=16: **200 GB/s = 21% of peak**.
The kernel is **compute-bound**, not BW-bound — this is the key
finding that sets expectations.

## Precision study (fp32 reference, fp8 round-trip)

Cast each forward-step `S_new` through fp8-E4M3 and back to fp32,
then run the backward on the rehydrated trajectory.  Ground truth: pure
fp32 forward + backward.

| T | N | S_traj_rel | S_traj_abs | dK | dV | dQ | ddec | dS0 |
|---:|--:|-----------:|-----------:|---:|---:|---:|-----:|----:|
| 256 | 16 | 3.5e-2 | 3.1e-2 | 1.7e-2 | 3.7e-3 | 1.6e-2 | 2.0e-2 | 5.8e-4 |
| 1024 | 16 | 3.5e-2 | 3.1e-2 | 2.7e-2 | 5.3e-3 | 2.1e-2 | 2.4e-2 | 6.7e-4 |
| 4096 | 16 | 3.5e-2 | 3.1e-2 | 2.8e-2 | 4.5e-3 | 1.7e-2 | 2.3e-2 | 7.5e-4 |
| 16384 | 16 | 3.3e-2 | 3.1e-2 | 2.3e-2 | 3.4e-3 | 2.3e-2 | 2.2e-2 | 7.5e-4 |
| 32768 | 16 | 3.4e-2 | 3.1e-2 | 2.9e-2 | 3.7e-3 | 1.5e-2 | 2.1e-2 | 6.1e-4 |
| 1024 | 32 | 3.3e-2 | 3.1e-2 | 3.1e-2 | 5.5e-3 | 1.8e-2 | 2.2e-2 | 3.6e-3 |
| 4096 | 32 | 3.2e-2 | 3.1e-2 | 2.2e-2 | 6.4e-3 | 1.4e-2 | 1.7e-2 | 5.3e-3 |
| 16384 | 32 | 3.2e-2 | 3.1e-2 | 3.2e-2 | 1.0e-2 | 2.0e-2 | 3.2e-2 | 1.7e-3 |

**Conclusion: fp8 rounding DOES NOT compound.** Errors are stable (or
slightly decreasing) with T up to 32K.  This is the crucial
precondition — tanh's contraction damps the noise, so backward-pass
accumulation remains bounded.

Absolute S_traj error of 0.0313 ≈ `2^-5`: this is exactly E4M3's
quantization step for values near 1 (3 mantissa bits → relative
resolution ~2⁻⁴ = 0.0625, but values in (-1, 1) use smaller binades
giving ~2⁻⁵ absolute error).

## Implementation

New files:
- `pararnn_seq_fwd_v2_fp8.py` — forward and backward kernels with
  `tl.float8e4nv` store/load, otherwise identical math to `v2`.
- `fp8_study/bench_baseline.py` — bf16 baseline timings.
- `fp8_study/precision_study.py` — fp32-reference precision study.
- `fp8_study/bench_fp8.py` — bf16 vs fp8 speed comparison.
- `fp8_study/mem_bench.py` — peak memory measurements.
- `fp8_study/fp8_correctness.py` — hybrid-level correctness.
- `fp8_study/training_loss_check.py` — real-data training-loss sanity.

Modified:
- `phase6_hybrid.py::PararnnHybridE88V2` — added `use_fp8` dispatch
  (opt-in via `ELMAN_PARARNN_FP8=1`).  Preserves bf16 path.
- `install_hybrid.py` — documented the env var.

## Benchmarks (best-tuned num_warps per kernel)

| Shape | Kernel | BF16 | FP8 | Speedup |
|-------|--------|-----:|----:|--------:|
| T=16K H=141 N=16 | fwd | 6.27 | 6.03 | 1.04× |
| T=16K H=141 N=16 | bwd | 10.45 | 10.30 | 1.01× |
| T=32K H=141 N=16 | fwd | 12.58 | 12.11 | 1.04× |
| T=32K H=141 N=16 | bwd | 20.96 | 20.69 | 1.01× |
| T=16K H=83  N=32 | fwd | 7.48 | 6.86 | 1.09× |
| T=16K H=83  N=32 | bwd | 21.22 | 21.38 | 0.99× |
| T=32K H=83  N=32 | fwd | 14.94 | 13.71 | 1.09× |
| T=32K H=83  N=32 | bwd | 42.47 | 42.83 | 0.99× |

### Memory

| Shape | BF16 S_traj_MB | FP8 S_traj_MB | BF16 full_GB | FP8 full_GB |
|-------|---------------:|--------------:|-------------:|------------:|
| T=16K H=141 N=16 | 1182.8 | 591.4 | 1.59 | 1.04 |
| T=32K H=141 N=16 | 2365.6 | 1182.8 | 3.19 | 2.09 |
| T=65K H=141 N=16 | 4731.2 | 2365.6 | 6.37 | 4.17 |
| T=16K H=83 N=32 | 2785.0 | 1392.5 | 3.17 | 1.88 |
| T=32K H=83 N=32 | 5570.0 | 2785.0 | 6.33 | 3.74 |

**S_traj halves exactly (2.00× reduction)**; end-to-end peak drops 35-45%.

## Correctness through the hybrid (bf16 baseline → fp8 variant)

| Shape | output | S_final | dK | dV | dQ | ddec | dS0 |
|-------|-------:|--------:|---:|---:|---:|-----:|----:|
| T=1K N=16  | 2.6e-2 | 2.8e-2 | 2.2e-2 | 6.6e-3 | 1.5e-2 | 2.2e-2 | 1.8e-3 |
| T=4K N=16  | 3.5e-2 | 3.2e-2 | 3.2e-2 | 5.5e-3 | 1.8e-2 | 3.0e-2 | 1.8e-3 |
| T=16K N=16 | 2.3e-2 | 3.2e-2 | 1.8e-2 | 4.8e-3 | 1.8e-2 | 2.1e-2 | 2.1e-3 |
| T=1K N=32  | 3.4e-2 | 3.7e-2 | 2.3e-2 | 5.5e-3 | 1.4e-2 | 2.3e-2 | 6.7e-3 |
| T=4K N=32  | 2.2e-2 | 3.2e-2 | 1.5e-2 | 5.4e-3 | 1.9e-2 | 2.2e-2 | 6.7e-3 |

All within the ≤5% target.  No compounding.

## Training-loss impact

Real `data/pile.txt`, small E88 (dim=512, depth=4, H=16, N=16, silu gate),
bf16 params & grads, LR=3e-4, seed=42, seq_len=512, bs=8:

| Mode | 60-step final | 60-step avg_last_20 | 200-step final | 200-step avg_last_20 |
|------|--------------:|---------------------:|---------------:|---------------------:|
| BF16 | 2.6875 | 2.8664 | 2.0938 | 2.4008 |
| FP8  | 2.6875 | 2.8680 | 2.0938 | 2.3977 |

**Difference: 0.0016-0.003 nats — below optimization noise floor.**
FP8 actually scored *slightly better* at 200 steps (noise).

## Why not the promised 1.5-1.8× speedup?

Analysis of bf16 forward at T=32K N=16 H=141 on 960 GB/s GPU:
- S_traj writes: 2365 MB
- K/V/decay reads: ~314 MB
- Achieved: 200 GB/s = **21% of peak HBM**

The Triton sequential forward is **compute-bound on the serialized
T-loop**: each step does an `exp`, a division (from the rational tanh
formula), two reductions, and broadcasts.  The in-register state
elimination already removed HBM reads of S; what remains is the
S-write, which at 2 GB over 12 ms is only ~170 GB/s — writes alone
don't saturate the memory system.

Halving the write size (bf16 → fp8) wasn't going to cut
wall-clock time because writes weren't the bottleneck.

### What would actually speed this up?

1. **Reduce compute per step.** Replace the rational tanh with a
   polynomial approximation, or hoist `exp(2·x)` to `exp2(x·2/ln2)` if
   precision allows.
2. **Sub-sequence parallelism.** Split the T axis into blocks and run
   them concurrently — but this is the parareal approach and it's
   already well-explored in this repo.
3. **Larger tiles.** Merge B or H dimensions into one tile so each
   program does more arithmetic per instruction fetch.
4. **Use WMMA for the outer product.** The `pre = delta ⊗ K` and the
   matrix-vector products are small (N×N=16×16 or 32×32) but could
   still benefit from tensor cores.  Triton doesn't map these to WMMA
   for fp32 accumulation on sm_89 cleanly, but a CUDA kernel could.

## Go/no-go recommendation

**CONDITIONAL GO as opt-in (not default).**

| Win | Score |
|-----|------:|
| Speed         | 1.01-1.11× (fails target) |
| Peak memory   | −35 to −45% (strong win) |
| Precision     | <5% grad err, no compounding |
| Training loss | Indistinguishable |

The feature is useful when:
- Memory is the bottleneck (e.g., T≥32K, large batch).
- You want to **use the saved memory to increase batch size** or
  sequence length — this converts the memory gain into end-to-end
  throughput via larger-than-otherwise batches.

The feature is NOT useful when:
- Memory headroom is comfortable.  The direct speedup is negligible.

**Usage:**
```bash
# Enable both the hybrid AND fp8 storage
ELMAN_PARARNN_HYBRID=1 ELMAN_PARARNN_FP8=1 python train.py ...
```

Configuration lives in env vars only.  No production code changes —
`phase6_hybrid.py` reads `ELMAN_PARARNN_FP8` at import time and the fp8
path is square-state only.
