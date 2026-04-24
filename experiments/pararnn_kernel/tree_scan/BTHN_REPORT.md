# [B,T,H,N]-Native Triton Kernels (V3) — Report

## Summary

Wrote stride-parametrized Triton forward + backward kernels (V3) that accept
training tensors directly in `[B, T, H, N]` layout (production E88 layout),
eliminating 10+ `.contiguous()` permute copies per kernel invocation that the
V2 path does internally.

**Headline result:** V3 achieves **1.66x–1.93x speedup over V2** at the
training-relevant shape `B=16, T=512`, and crucially, V3 now beats raw CUDA
by 1.28x–1.41x at that shape (V2 was actually slower than CUDA at
`B=16 T=512 N=32` — V3 fixes that regression).

## Permute overhead baseline (`bench_permute_baseline.py`)

| Shape                    | adapter ms | internal ms | output ms | total permutes | full patched ms | permute fraction |
|--------------------------|------------|-------------|-----------|----------------|-----------------|------------------|
| B=1  T=32K H=141 N=16    |   0.007    |   2.517     |   0.350   |   2.874        |   18.57         |  **15.5%**       |
| B=16 T=512 H=141 N=16    |   0.348    |   0.264     |   0.122   |   0.735        |    2.23         |  **33.0%**       |
| B=1  T=32K H=83  N=32    |   0.006    |   1.810     |   0.401   |   2.217        |   24.77         |  **9.0%**        |
| B=16 T=512 H=83  N=32    |   0.406    |   0.313     |   0.154   |   0.873        |    4.66         |  **18.7%**       |

At the production training shape (B=16 T=512), **33% of the patched-hybrid
runtime was permutes**. These copies were 1-3 ms each at long-T shapes.

## Correctness

### V3 kernel vs V2 (same inputs, in-kernel)
Zero error at fp32 (S_traj exact, Sq at round-off 1e-7), matches expected bf16
noise bound.

### V3 hybrid (with fused silu gate) vs V2 hybrid — end-to-end autograd
Comparing full fwd+bwd (including grad propagation through the gate) at
production shapes:

| dtype | Range of max rel error (across dK/dV/dQ/dg/ddec/dS0/output/S_final) |
|-------|--------------------------------------------------------------------|
| fp32  | 1e-7 to 3e-7  (numerical round-off only)                           |
| bf16  | 1e-3 to 8e-3  (bf16 summation-order noise; same as V2 vs CUDA)     |

### V3 installed vs raw CUDA (end-to-end through `E88OptimizedCUDAFunction.apply`)
At B=16 T=512 and similar:
- All gradients agree within **5e-3 to 8e-3 relative error** in bf16.
- This is the same level of agreement V2 hybrid has with the CUDA kernel
  (bf16 accumulation noise from slightly different summation orders).

**Verdict: V3 is a clean drop-in replacement for V2.**

## Benchmark — V3 vs V2 (fwd+bwd w/ gate, bf16)

From `phase7_v3_full_bench.py`:

| Shape                    | CUDA ms | V2 ms  | V3 ms  | V3/V2   | V3/CUDA |
|--------------------------|---------|--------|--------|---------|---------|
| B=16 T=512  H=141 N=16   |   3.57  |   4.65 |   2.53 | **1.84x** |   1.41x |
| B=16 T=512  H=83  N=32   |   6.43  |   8.47 |   5.03 | **1.68x** |   1.28x |
| B=1  T=32K  H=141 N=16   | 110.18  |  43.21 |  35.93 |   1.20x |   3.07x |
| B=1  T=32K  H=83  N=32   | 139.58  |  69.91 |  66.28 |   1.05x |   2.11x |
| B=8  T=1024 H=141 N=16   |   4.42  |   5.26 |   2.72 | **1.93x** |   1.62x |
| B=4  T=2048 H=141 N=16   |   7.23  |   5.62 |   3.40 |   1.66x |   2.13x |

**Key observations:**

1. **At short T and large B (training): 1.66x-1.93x speedup.** The permutes
   are a larger fraction of the total time at short T because the Triton
   kernel itself is fast.

2. **At long T (32K): 1.05x-1.20x speedup.** Smaller fraction because the
   sequential scan dominates.

3. **V3 beats CUDA at B=16 T=512.** Previously V2 was actually slower than
   CUDA at N=32 (8.47 ms vs 6.43 ms) — **V2 was a regression at training
   scale**. V3 fixes this: 5.03 ms vs 6.43 ms = 1.28x faster.

## Files added

All new, production path untouched:

- `pararnn_seq_fwd_v3_bthn.py` — stride-parametrized forward; emits
  `S_traj[B,H,T,N,N]` + `Sq[B,T,H,N]` directly.
- `pararnn_seq_bwd_v3_bthn.py` — stride-parametrized backward; consumes
  `dL_dout[B,T,H,N]`, emits `dK/dV/dQ[B,T,H,N]`, `ddec[B,T,H]`, `dS0[B,H,N,N]`.
- `phase6_hybrid_v3.py` — `PararnnHybridE88V3` autograd.Function +
  `hybrid_v3_with_fused_gate` wrapper.
- `install_hybrid_v3.py` — monkey-patch installer; replaces
  `E88OptimizedCUDAFunction.apply` with V3 (no permutes).
- `phase7_v3_test.py` — correctness (fp32+bf16) and microbench
  (hybrid-level V3 vs V2).
- `phase7_v3_full_bench.py` — end-to-end V3 vs V2 vs CUDA via the
  installed apply path.
- `phase7_v3_installed_test.py` — verifies that the installed V3 patch
  produces correct gradients when called through `E88OptimizedCUDAFunction.apply`.
- `bench_permute_baseline.py` — measurement of the permute overhead V3
  eliminates.

## Go/No-Go: make V3 the default

**Go.** Specifically:

1. **For the fused-gate + square-state path (the training default):** V3 is
   strictly better than V2 and beats raw CUDA. The code path is covered by
   correctness tests at fp32 and bf16 precision. Layout: V3 is the
   direct-[B,T,H,N] drop-in that the E88 training code actually wants.

2. **Flag to enable:** `ELMAN_PARARNN_HYBRID_V3=1` → call `install_hybrid_v3.install()`.
   This is orthogonal to the V2 `ELMAN_PARARNN_HYBRID=1` flag; if a user sets
   V3, we patch only `E88OptimizedCUDAFunction` (the main training entry
   point). The older non-optimized CUDA paths aren't touched, so V2-based
   codepaths remain available for comparison / fallback.

3. **Constraints preserved:** V3 only dispatches for square state (Ns == Hv),
   gated (apply_gate=True). Rectangular state or no-gate cases fall through
   to the original CUDA kernel — same dispatch logic as V2.

4. **Not yet done:** The unfused and fused-gate-through-[T,B,H,N] code paths
   (E88FLAHybridCUDAFunction, E88FusedGateCUDAFunction) still use V2 via
   `install_hybrid.py`. These are not the training path
   (USE_OPTIMIZED_KERNELS=True is the default and all training uses
   E88OptimizedCUDAFunction). Future work: patch those too if we want to
   stop maintaining V2 entirely.

## Training impact estimate

At batch_size=16, T=512, E88 n16 (H=141), there are 17 layers per 480M
model, so roughly 17 forward+backward kernel calls per step.

- V2 per-call: 4.65 ms → ~79 ms/step just in the recurrence kernels
- V3 per-call: 2.53 ms → ~43 ms/step (saves **36 ms/step**)
- CUDA per-call: 3.57 ms → ~61 ms/step

If the training step is ~100 ms total, V3 saves roughly **25-30% of
end-to-end step time** at training shape.
