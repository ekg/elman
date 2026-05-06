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
