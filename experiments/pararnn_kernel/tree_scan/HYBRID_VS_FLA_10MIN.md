# E88 hybrid vs CUDA vs FLA-GDN — 10-min training on data/pile.txt

**GPU**: RTX 6000 Ada (48 GB)
**Setup**: bf16, seed=42, `data/pile.txt` byte-level, AdamW schedulefree

## TL;DR

Our Phase 7 hybrid kernel shows a 2.65-2.78× speedup on the
**backward kernel in isolation** (B=1, T=16K+, no autograd, no grad-ckpt).
That speedup does **not** translate to training-pipeline throughput:

- **Typical training (chunk=512, B=16)**: hybrid is *5-11% slower* than
  CUDA — autograd/layout overhead exceeds bwd savings at high B.
- **Long-seq training (T=32K, grad-ckpt)**: hybrid and CUDA *tied*.
  Gradient checkpointing reruns the forward during backward, diluting
  the backward-only speedup.
- **FLA-GDN wins** on both throughput (3.2× at T=32K!) and loss in
  conventional setups.
- **E88 beats FLA-GDN at T=128K with small models** (rare regime where
  FLA-GDN's parallel scan has too few heads to saturate).

## Short-seq production config (chunk=512, B=16, 439M params, 10 min)

| Model | Config | steps | last-100 loss | tok/s | 24 h tokens |
|-------|--------|-------|---------------|-------|-------------|
| **FLA-GDN** | d=17 exp=2 h=24 dim=1920 | 1613 | **1.6313** | **22.3K** | **1.93B** |
| E88-n32 CUDA | d=17 h=83 n=32 dim=1920 | 1432 | 1.6660 | 19.8K | 1.71B |
| E88-n32 HYBRID | same | 1351 | 1.6670 | 18.6K (94% CUDA) | 1.61B |
| E88-n16 CUDA | d=25 h=141 n=16 dim=1536 | 1351 | 1.6628 | 18.7K | 1.62B |
| E88-n16 HYBRID | same | 1206 | 1.6909 | 16.6K (89% CUDA) | 1.43B |

**Observation**: at typical pretraining config, FLA-GDN wins throughput
(~15%) AND loss (0.035 nats vs E88).  Hybrid is the slowest because
at B=16, the CUDA kernel already saturates the GPU and the Triton
hybrid's autograd.Function wrapping adds overhead.

## Long-seq T=32K (B=1, grad_ckpt, 439M params, 10 min)

| Model | steps | loss | tok/s | 24 h tokens |
|-------|-------|------|-------|-------------|
| FLA-GDN d=17 h=24 | ~170 | ~1.6 | **18.2K** | **1.57B** |
| E88-n16 CUDA | 103 | 2.55 | 5.66K | 489M |
| E88-n16 HYBRID | ~103 (proj.) | ~2.1 | 5.62K | 486M |

**FLA-GDN is 3.2× faster at T=32K** because its associative scan is
parallel-in-time — O(log T) depth.  E88's tanh recurrence is
fundamentally sequential, so per-step cost × T dominates.

The hybrid/CUDA parity here is from gradient_checkpointing: the forward
runs twice (once in no-grad eval, once rebuilt during bwd) for both
implementations, diluting our backward-only speedup.

## Long-seq T=128K (B=1, grad_ckpt, small 8M params, 10 min)

Large E88 (439M params) **OOMs** at T=128K on 48 GB even with grad-ckpt.
With a smaller model (dim=512, d=6, H=32):

| Model | steps (partial) | tok/s | 24 h tokens |
|-------|-----------------|-------|-------------|
| E88 CUDA | ~44 | 29.3K | 2.53B |
| E88 HYBRID | ~44 | 29.4K | 2.54B (tied) |
| FLA-GDN d=17 h=24 | ~39 | 17.0K | **1.47B (slower)** |

**E88 is ~1.7× faster than FLA-GDN at T=128K with small H**.
FLA-GDN's parallel-scan advantage dilutes when there are few heads;
E88's sequential scan at small state is already fully utilizing the GPU.

## Why the kernel-level 2.65× doesn't translate

1. **Gradient checkpointing doubles forward cost.**  Our hybrid saved
   backward-only, but grad-ckpt makes forward run twice (once eval,
   once rebuild during backward).  Savings on the backward segment are
   diluted proportionally.

2. **At B=16 (typical training), CUDA is saturated.**  Our hybrid's
   2.65× was for B=1 low-parallelism regime.  B=16 provides 16× more
   work per kernel call, filling the GPU regardless of kernel quality.

3. **Autograd / layout / optimizer overhead** is significant relative
   to the kernel time in the training loop.  A 2.65× kernel speedup
   becomes ~1% of wall clock if the kernel is only 1% of wall clock.

4. **Per-layer dispatch overhead** — a 25-layer model calls the kernel
   25× per step.  Any per-call overhead in the Triton path (jit check,
   tensor permutes to/from Pararnn convention) adds up.

## What the hybrid is actually useful for

- Scientific kernel benchmarking (B=1, pure backward, no checkpointing):
  our 2.65-2.78× result stands there and is correctness-verified.
- Research on different E88 variants where the backward kernel itself
  is the bottleneck (e.g., gradient analysis, Jacobian studies).
- **Not** useful for standard training acceleration on current configs.

## Recommendation

Keep the hybrid kernel as a research artifact.  For production E88
training, use the CUDA kernel.  For better training throughput at any
sequence length, FLA-GDN is the better architecture choice based on
both speed and loss at conventional chunk sizes.
