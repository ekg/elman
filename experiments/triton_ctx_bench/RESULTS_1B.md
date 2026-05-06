# Triton vs CUDA at production 1.27B scale (H=386)

**Setup.** E88 dim=1408, depth=14, **n_heads=386**, n_state=32 (~1.27B params). 5 min Pile training, sf-AdamW lr=1.054e-3 bf16. `--gradient_checkpointing` for both backends.

## Results

| ctx (T) | bs | backend | tok/s | last-100 loss | peak GB | notes |
|---|---|---|---|---|---|---|
| 512   | 8 | CUDA   | **4746** | 2.62 | 11.8 | |
| 512   | 8 | Triton | 3413 | 2.90 | 17.8 | 28% slower, more memory |
| 512   | 4 | Triton | 3201 | 2.98 | 13.5 | smaller bs same story |
| 512   | 8 | Triton (NO ckpt) | — | — | — | OOM |
| 4K    | 2 | CUDA   | **4727** | 3.15 | 12.4 | |
| 4K    | 2 | Triton | — | — | — | **OOM crash** |
| 4K    | 1 | Triton | 3491 | 3.56 | 17.5 | works at bs=1, still slower |
| 16K   | 1 | CUDA   | **4441** | 3.33 | 15.3 | |
| 32K   | 1 | CUDA   | **4385** | 4.70 | 22.6 | |

**At production 1.27B H=386, Triton is 1.3-1.4× slower than CUDA and uses ~30-50% more memory.**

This is the opposite of the 12M-scale result. The 12M bench (H=16) showed Triton winning 1.04-1.99× across contexts.

## Why the regression at H=386

The current Triton kernel autotune picks BLOCK_H=4 for H≥256. At H=386 that's ~97 programs per batch — too few for good SM occupancy on a ~100-SM device. CUDA reg-own launches B×H = 3088 single-warp programs at T=512 bs=8, which saturates much better.

**The Triton kernel needs work before it's a drop-in for production:**

1. **Higher BLOCK_H candidates** at large H. BLOCK_H=8 or 16 with appropriate num_warps may improve occupancy. SRAM budget: BLOCK_H × N × V × 4 bytes for fp32 state tile = 32KB at BLOCK_H=8 N=V=32 — fits in 48 KB shared.

2. **Sparse forward checkpointing** (every 16 steps like CUDA). Currently the Triton forward stores per-step S history; at H=386 N=V=32 this is far heavier than CUDA's compact checkpoint. Plus it forces the existing `--gradient_checkpointing` to be on, which costs throughput.

3. **Backward kernel re-tuning at H=386**. The backward currently uses a fixed `block_h=4 num_warps=4` for H≥256 without per-shape autotune.

## Implication for the 1.27B campaign

The kernel-swap path that looked clean at 12M is not yet ready at production scale. Two paths forward:

- **A:** Stay on CUDA for the 1.27B 32K/64K runs. Triton port is still valuable for ROCm portability and for smaller-scale work (sub-100M), but defer for the production campaign.
- **B:** Spend the kernel-engineering time first (estimated 1-3 days for sparse-ckpt + per-H autotune), then re-bench, then decide.

The 12M-scale wins remain real and the parity story is solid — the kernel just hasn't been tuned for the H~400 regime that production E88 lives in.

## Update (May 6, after kernel work)

Two kernel fixes landed:
1. **BLOCK_H autotune at high H**: empirical sweep at H=386 N=V=32 showed BLOCK_H=1 num_warps=2 is the fastest config — 2× kernel-only speedup vs the previous default. Committed.
2. **int64 offset arithmetic**: T=4K bs=2 H=386 was crashing with illegal memory access due to int32 stride×timestep overflow at >2B-element tensors. Fixed by casting `t` and `t+1` to int64 in the kernel time loops. Now T=4K bs=2 trains cleanly at ~3700 tok/s. Parity tests still pass (fp32 max_rel ~1e-7, bf16 ~5e-3).

End-to-end 1.27B T=512 bs=8 with both fixes: **3578 tok/s** vs CUDA's 4746 (Triton 25% slower, was 28% slower before). Kernel-only 2× wins don't fully translate because at depth=14 the S_ckpt allocation overhead per layer call dominates. **Sparse forward checkpointing** (save S every K=16 steps, recompute in backward) would shrink S_ckpt 16×, and is the remaining work needed to make Triton beat CUDA at production scale.

Recommendation: stay on CUDA for the 1.27B 32K/64K campaign for now; revisit Triton after the sparse-ckpt rewrite (~1-2 hours focused work).

## Update (May 6, after sparse-checkpoint forward+backward landed)

The Triton forward kernel now stores S only every CKPT_INTERVAL=16 steps
(matching CUDA register-owned). The backward kernel does
forward-replay through each segment from the saved checkpoint, mirroring
the CUDA reg-own segment_cache pattern.

Re-bench at production E88 1.27B (3 min Pile, sf-AdamW, bf16, --gradient_checkpointing):

| ctx | bs | backend                | tok/s | peak GB | notes |
|---|---|---|---|---|---|
| 512 | 8 | CUDA                       | 4501 | 11.8 | reference |
| 512 | 8 | Triton dense-ckpt (prior)  | 3578 | 17.7 | from above |
| 512 | 8 | **Triton sparse-ckpt**     | **3728** | **12.0** | **memory parity reached** |
| 4K  | 2 | CUDA                       | 4456 | 12.4 | reference |
| 4K  | 2 | Triton dense-ckpt          | OOM  | —    | from above |
| 4K  | 2 | **Triton sparse-ckpt**     | **3715** | **14.7** | **runs cleanly, 11 GB cheaper than prior bs=1 dense** |

(tok/s above are at ~step 130 / 60; reported as the steady-state value
once the warmup smoothing has settled.)

**Key wins:**
- T=512 bs=8 peak memory: **17.7 → 12.0 GB** (32% reduction; now within
  ~2% of CUDA's 11.8 GB).
- T=4K bs=2: dense-ckpt OOM'd; sparse-ckpt fits at **14.7 GB peak**.
- Throughput improved ~4% at T=512 (3578 → 3728). The remaining 17%
  gap to CUDA at T=512 is from the per-step kernel work itself, not
  from S_ckpt allocation any more.

**Implication.** Triton is now memory-competitive at production scale,
which unlocks larger batch sizes and longer contexts that were
previously OOM. CUDA still wins on throughput by 17-20% at this shape.
For the 1.27B 32K/64K campaign:
- A 17% throughput penalty is tolerable if other factors favor Triton
  (e.g., ROCm portability, kernel iteration speed). At the same memory
  budget, CUDA still gets you more tokens/wall-clock.
- The path to closing the throughput gap further is on the per-step
  kernel work (e.g. registerizing the state tile à la CUDA reg-own;
  this Triton kernel still keeps S in shared memory which isn't free).

Parity preserved end-to-end: forward parity tests pass at fp32 (~1e-7
max_rel) and bf16 (~5e-3); backward parity passes at fp32 (~5e-5 abs)
and bf16 (~5e-2 abs); E88-layer use_triton smoke test passes at bf16.
