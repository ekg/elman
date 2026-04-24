# E88 Per-Kernel-Call Overhead Investigation

**Date:** 2026-04-23
**Config:** B=1, T=32768, H=141, N=16, depth=25, dim=1536 (E88-n16 production with gradient checkpointing)
**GPU:** NVIDIA RTX 6000 Ada Generation (CUDA 12.8, PyTorch 2.9.1)
**Code commit:** main @ 15de503

## TL;DR

**The "37 ms per-kernel-call overhead" described in the task premise is not real.**  It was computed by comparing the **CUDA-Opt kernel's** per-call time (51.4 ms) against the **Triton hybrid kernel's** microbench (14 ms). Those are *two different kernels* — the gap is kernel speed, not dispatch overhead.

When we anchor on the *same* path (hybrid) and instrument every sub-step, the per-call overhead above the Triton kernel is **~5 ms**, not 37 ms. torch.compile and CUDA graphs provide **essentially zero** additional speedup at this config. The current hybrid path already delivers **~11K tok/s** (vs. 9.3 K claimed). The proposed ">13 K tok/s target from overhead removal" is therefore unachievable by overhead removal alone — the Triton kernel itself (~15 ms/call × 50 calls = 750 ms/step) sets the floor, independent of wrapper cost.

## 1. Reproduced Baseline Measurements

Command:
```
CUDA_VISIBLE_DEVICES=1 ELMAN_PARARNN_HYBRID=1 python experiments/pararnn_kernel/tree_scan/time_fwd_and_bwd.py
```

| Path | Step time | fwd ms/call | bwd ms/call | Non-kernel | tok/s |
|---|---|---|---|---|---|
| CUDA-Opt (baseline, no hybrid) | 5674 ms | 50.9 ms | 85.5 ms | 992 ms (17%) | 5 776 |
| **Hybrid (ELMAN_PARARNN_HYBRID=1)** | **2987 ms** | **21.8 ms** | **26.9 ms** | 1 224 ms (41 %) | **10 971** |

Hybrid already delivers **1.90× speedup** over the CUDA-Opt path. The ~37 ms gap that the task premise describes *is* this kernel-speed gap (51.4 − 14 ≈ 37 ms), **not** overhead above a Triton launch.

## 2. Breakdown of the Hybrid 21.8 ms/call

Script: `experiments/pararnn_kernel/tree_scan/breakdown_overhead2.py`
All numbers are medians of 20 runs at B=1 T=32 768 H=141 N=16, bf16 on RTX 6000 Ada.

| Stage | Median ms | Cumulative |
|---|---|---|
| Triton kernel alone (no permutes, no grad, no gate) | 15.62 | 15.62 |
| + `PararnnHybridE88V2.apply` (5 permutes in TB layout) | 17.80 | +2.18 (permutes) |
| + silu(g) * Sq (F.silu + elementwise mul) | 18.68 | +0.88 (gate) |
| + `E88OptimizedCUDAFunction.apply` (BT↔TB transposes) | 18.68 | ≈0 (ping-pong cancels) |
| + autograd grad-tracking (saves_for_backward, ctx.dims) | 20.12 | +1.44 (autograd) |

Training-path observed: 21.80 ms/call, which matches the 20.1 ms grad-on measurement plus ~1.5 ms scattered launch/GIL noise per call in a deeper model.

**Bottom line:** of the 21.8 ms/call in training, ~15.6 ms is the Triton kernel itself and **~6 ms is everything else combined** (permutes 2.2, gate 0.9, autograd 1.4, driver/GIL ~1.5).

### Sub-stage detail (script: `breakdown_overhead.py`)

| Stage | ms |
|---|---|
| BT→TB transpose (5 tensors) | 0.015 |
| TB→Pararnn [B,H,T,N] permute (5 tensors) | **2.519** |
| Triton forward kernel | 14.942 |
| S_final transpose | 0.031 |
| Output permute [B,H,T,M]→[T,B,H,M] | 0.382 |
| F.silu(g) * output | 0.863 |
| output TB→BT | 0.009 |
| **Sum** | **18.762** |

The single biggest non-kernel cost is the **Pararnn permute at 2.5 ms/call** — mostly `aten::copy_` to materialise a contiguous [B,H,T,N] tensor.

### Training-step kineto profile

```
PararnnHybridE88V2 fwd:        841 ms self-CUDA (50 calls, 16.8 ms avg kernel)
PararnnHybridE88V2Backward:    617 ms self-CUDA (25 calls, 24.7 ms avg kernel)
aten::mm (MmBackward0 etc.):   650-1100 ms (linear projections & their bwd)
aten::contiguous (525 calls):  192 ms
aten::copy_ (751 calls):       220 ms
Optimizer.step (foreach):      439 ms
Command Buffer Full (cudaLaunchKernel):  414 ms (driver overhead, 2906 launches)
```

Non-kernel budget = 1 224 ms/step. It is dominated by:
* linear-projection GEMMs (280 ms forward + ~600 ms backward) — intrinsic work, can't eliminate
* optimizer step (439 ms, already using `_foreach_*` ops)
* aten::copy_ / contiguous (~220 ms) — includes the same permutes counted above
* driver kernel-launch overhead (~414 ms across 2 906 launches)

## 3. torch.compile results

Script: `experiments/pararnn_kernel/tree_scan/try_torch_compile.py`

| Mode | Step time | vs. baseline | Notes |
|---|---|---|---|
| no compile (baseline) | 2979 ms | 1.00× | |
| `mode='default'` | 2990 ms | 1.00× | no meaningful change |
| `mode='reduce-overhead'` | 3179 ms | **0.94×** | slower — CUDA graph replay interacting badly with custom autograd.Function + grad-ckpt |
| `mode='max-autotune'` | 3021 ms | 0.99× | no meaningful change |

Dynamo warnings during compilation:
- Custom pybind11 kernels (`hasty_pytorch_lib.e88_coalesced_forward` etc.) can't be traced; Dynamo falls back to eager for those subgraphs.
- Recompile-limit hits (8+) on `rms_norm_fn` in mamba_ssm's triton layernorm — each of the 25 layers recompiles because `prenorm`/`residual` state is captured as a guard.

**Conclusion:** torch.compile gives no speedup here, because (a) the custom autograd functions break graph tracing, (b) the Triton kernels aren't inductor-generated so there's nothing to fuse, and (c) `reduce-overhead` (which would use CUDA graphs) is *slower* because of the dynamic residual state in gradient checkpointing.

## 4. CUDA graph results

Scripts: `try_cuda_graphs.py`, `cuda_graph_minimal.py`, `cuda_graph_batch.py`

### Full-step capture: **OOM**
Capturing the full 25-layer LadderLM forward+backward with grad-ckpt disabled requires > 47 GiB on a 48 GiB card (the graph needs a static memory pool, and without grad-ckpt activations explode at T=32K).

### Per-kernel capture: **no speedup**
Replacing just one hybrid call with a CUDA graph:

| Config | Baseline (no graph) | Graph replay | Speedup |
|---|---|---|---|
| B=1 T=4K H=141 N=16 | 2.080 ms | 2.077 ms | 1.00× |
| B=1 T=8K H=141 N=16 | 4.504 ms | 4.505 ms | 1.00× |
| B=1 T=16K H=141 N=16 | 9.282 ms | 9.290 ms | 1.00× |
| B=1 T=32K H=141 N=16 | 18.675 ms | 18.684 ms | 1.00× |
| B=16 T=2K H=141 N=16 | 8.289 ms | 8.687 ms | 0.95× |
| B=32 T=1K H=141 N=16 | 7.923 ms | 8.409 ms | 0.94× |

CUDA graphs work (no errors), but provide no speedup (sometimes a small slowdown) because the Triton kernel itself already dominates total time — there is insufficient CPU-side dispatch overhead to hide.

## 5. Other Explorations

### Direct [B,T,H,N] → [B,H,T,N] hybrid (skip intermediate [T,B,H,N])
Script: `direct_bt_hybrid.py`.  Saves two transposes; measured speedup = **1.01×**.  Not worth the code duplication.

### Fuse silu(g) into the Triton kernel output
Script: `kernel_with_fused_gate.py`. Potential saving: 0.4–0.8 ms / 39 ms = **1–2 %**.
NaN correctness issues during testing (un-root-caused); not worth pursuing for such small gains.

## 6. End-to-end throughput achieved

| Configuration | Step time | tok/s |
|---|---|---|
| CUDA-Opt (pre-hybrid baseline) | 5674 ms | 5 776 |
| **Hybrid (current best)** | **2987 ms** | **10 971** |
| Hybrid + torch.compile default | 2990 ms | 10 960 |
| Hybrid + torch.compile reduce-overhead | 3179 ms | 10 308 (slower) |

Target from problem statement (> 13 K tok/s) is **not achieved**. The floor imposed by the Triton kernel alone is:
```
15 ms/call × 50 fwd-calls + 25 ms/call × 25 bwd-calls = 1375 ms/step (kernel-only)
```
Any number above `1375 + projection_gemm_time + optimizer_time` is achievable — current 2987 ms has ~1600 ms of non-kernel work (of which ~900 ms are intrinsic GEMMs). No amount of compile-level fusion will remove that.

## 7. Concrete next-step recommendations

Ranked by expected speedup:

1. **Faster Triton kernel is the only big lever** (≥ 50 % of step time).
   - Current 15 ms/call for fwd and 25 ms/call for bwd. At 141 heads × 32 768 timesteps × 16² state, arithmetic intensity is very low; likely HBM-bound.
   - Attempt 1: SM occupancy — the current grid is `B*H = 141`. RTX 6000 Ada has 142 SMs. At B=1 we get ≈1 head/SM which is correct, but if B=4 the grid goes to 564, oversubscribing. Consider block-splitting one head across multiple SMs using T-chunking with a serial merge (like in pararnn_bwd_rect chunked versions).
   - Attempt 2: Fuse L2-norm of k,q into the kernel (currently done as separate triton_l2_norm calls in LadderLM).
   - Attempt 3: Switch n_state=16 to a purely register-resident path (16×16 = 256 fp32 = 1 KB per head) with explicit unrolling.

2. **Eliminate the 220 ms `aten::copy_` / contiguous** in non-kernel time.
   - Most of these come from `.transpose().contiguous()` on q/k/v projections before the kernel call.
   - Option: modify kernel to accept `[B, T, H, N]` strides directly (skip contiguous). Would need per-call Triton tuning but would save ~150 ms/step (5 %).
   - The DirectBT experiment showed this alone saves only ~0.1 ms per layer call in training, so the savings are mostly outside the kernel entry.

3. **Reduce launch count** via wider kernels rather than CUDA graphs.
   - 2 906 kernel launches per step.  grad-ckpt forces refwd during backward, roughly doubling.
   - Combining the fwd+bwd Triton kernels into one persistent kernel is hard but could remove ~200 ms.

4. **Do NOT pursue**: torch.compile, CUDA graphs, or direct-BT layout — each was tested and shown to give < 2 % improvement for this config.

## 8. Reproducing these results

```bash
# Baseline timing (as specified by task)
CUDA_VISIBLE_DEVICES=1 ELMAN_PARARNN_HYBRID=1 \
  python experiments/pararnn_kernel/tree_scan/time_fwd_and_bwd.py

# Stage-by-stage overhead breakdown
CUDA_VISIBLE_DEVICES=1 \
  python experiments/pararnn_kernel/tree_scan/breakdown_overhead.py
CUDA_VISIBLE_DEVICES=1 \
  python experiments/pararnn_kernel/tree_scan/breakdown_overhead2.py
CUDA_VISIBLE_DEVICES=1 \
  python experiments/pararnn_kernel/tree_scan/bwd_breakdown.py

# torch.compile attempts
CUDA_VISIBLE_DEVICES=1 \
  python experiments/pararnn_kernel/tree_scan/try_torch_compile.py

# CUDA graphs
CUDA_VISIBLE_DEVICES=1 \
  python experiments/pararnn_kernel/tree_scan/cuda_graph_minimal.py
CUDA_VISIBLE_DEVICES=1 \
  python experiments/pararnn_kernel/tree_scan/cuda_graph_batch.py

# Direct BT-layout hybrid (skip TB intermediate)
CUDA_VISIBLE_DEVICES=1 \
  python experiments/pararnn_kernel/tree_scan/direct_bt_hybrid.py

# Fused-gate Triton kernel (+ NaN bug)
CUDA_VISIBLE_DEVICES=1 \
  python experiments/pararnn_kernel/tree_scan/kernel_with_fused_gate.py
```

All scripts use GPU 1 exclusively (CUDA_VISIBLE_DEVICES=1).

## 9. Files created (all inside experiments/pararnn_kernel/tree_scan/)

- `breakdown_overhead.py` — per-stage timing of hybrid_with_fused_gate
- `breakdown_overhead2.py` — kernel-only vs. apply vs. gate vs. grad-tracking
- `bwd_breakdown.py` — fwd+bwd combined timing of hybrid
- `try_torch_compile.py` — torch.compile in default/reduce-overhead/max-autotune
- `try_cuda_graphs.py` — full-step CUDA graph attempt (OOM)
- `cuda_graph_minimal.py` — single-layer CUDA graph (works, no speedup)
- `cuda_graph_batch.py` — CUDA graphs at larger batch (still no speedup)
- `direct_bt_hybrid.py` — `[B,T,H,N]` native hybrid (correctness passes, ~1 % speedup)
- `kernel_with_fused_gate.py` — Triton kernel with silu gate fused in (1 % savings, has NaN bug, not integrated)
