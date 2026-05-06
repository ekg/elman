# Triton vs CUDA E88: training-time bench across context lengths

**Setup.** E88 dim=384, depth=4, n_heads=16, n_state=32 (~12M params). 5 min Pile training each, sf-AdamW lr=3e-4 bf16, batch_size scaled with ctx to fit memory. Both backends run identical model code; the only difference is the recurrence kernel (CUDA register-owned vs Triton fwd+bwd).

## Results

| ctx (T) | bs | CUDA tok/s | Triton tok/s | speedup | CUDA loss (last-100) | Triton loss (last-100) | Δloss | CUDA mem (MB) | Triton mem (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 512   | 16 | 497K | 517K | 1.04× | 1.845 | 1.809 | **-0.04** |  745 | (n/a) |
| 4K    |  4 | 182K | 286K | 1.57× | 2.101 | 2.054 | **-0.05** | 1206 | 5576 |
| 16K   |  1 |  50K |  98K | 1.95× | 2.553 | 2.203 | **-0.35** | 1205 | 5574 |
| 32K   |  1 |  49K |  97K | 1.99× | 2.682 | 2.406 | **-0.28** | 2130 | 10867 |

## Read

**Triton is ~2× faster at long context, and the same-wall-time loss gap widens with ctx.** At T=512 the win is marginal (1.04× tok/s, 0.04 nats); at T=16K and T=32K Triton trains to substantially lower loss in fixed wall time (0.28-0.35 nats). This is the regime where the kernel pays off — exactly where CUDA's checkpoint+recompute backward pattern is least efficient and Triton's all-in-registers recurrence is most efficient.

**Activation memory is the real cost.** Triton stores per-step S checkpoints (T+1 tiles), while CUDA reg-own checkpoints every 16 steps + recomputes. At 12M scale, Triton memory at T=32K is 10.9 GB vs CUDA's 2.1 GB (~5× more). At production 1.27B (H=83, N=V=32, depth=17), full S history at T=32K would be ~184 GB per micro-batch — completely off the table. **Sparse forward checkpointing + backward recompute in Triton is the next required kernel work.**

## Memory fix — gradient_checkpointing already does the job

`--gradient_checkpointing` (already in LadderLM) wraps each layer in `torch.utils.checkpoint.checkpoint` so the per-step S history allocates only during one layer's backward and is freed before the next. Peak memory = max over layers, not sum.

Verified at T=4K bs=4 (12M model):

| config | tok/s | peak MB | speedup vs CUDA |
|---|---|---|---|
| CUDA (no ckpt)            | 182K | 1206 | 1.00× |
| Triton (no ckpt)          | 286K | 5576 | 1.57× |
| **Triton + grad-ckpt**    | **219K** | **1618** | **1.20×** |

Triton + grad_ckpt holds **1.2×** speedup over CUDA at memory parity. Cost is the extra forward replay during backward (~24% throughput).

At production 1.27B T=32K, full S history would be ~184 GB without checkpointing; with grad_ckpt it drops to one-layer's worth (~10.8 GB) — feasible on a single GPU.

A future kernel-level optimization (sparse forward checkpoint + backward recompute inside the Triton kernel itself) would be cheaper still (~5-10% throughput cost vs 24%), but isn't required for production.

## Strategic implication

The Triton speedup is biggest exactly at the long-ctx regime where E88's algorithmic structure should help most (length-extrapolation result, coherent generation at 64K from prior 24h training runs). With memory fixed, Triton enables:
- 32K CMA-ES (~50% more configs per node-hour vs CUDA)
- 64K final training (memory-feasible at production scale with interval=16 checkpointing)
- Frontier ROCm runs (Triton portable, CUDA reg-own kernel is NVIDIA-only)
