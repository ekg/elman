# Mamba2 Configurable Parameters - Complete Analysis

**Task:** Scale Mamba2 to chunk_size=2048
**Date:** 2026-02-20
**Researcher:** Claude (via scale-mamba2-2048 workgraph task)

---

## Executive Summary

Mamba2 has **25 configurable parameters** across 4 categories:
1. **Model architecture** (8 params) - control model structure and capacity
2. **Training hyperparameters** (10 params) - learning dynamics and optimization
3. **Advanced SSM parameters** (7 params) - low-level state-space model tuning
4. **System/performance** (5 params) - compilation, precision, parallelism

**Critical finding for scaling:** `chunk_size` was NOT in the CMA-ES search space. Added support in `cmaes_search_v2.py` to enable sequence length scaling experiments.

---

## 1. Model Architecture Parameters

These define the model structure and parameter count.

| Parameter | Type | Default | Valid Range/Values | In cmaes_search.py? | Notes |
|-----------|------|---------|-------------------|---------------------|-------|
| `vocab_size` | int | 256 | Any positive int | Fixed at 256 | Byte-level tokenization |
| `dim` (d_model) | int | 512 | 1024-3072, 128-aligned | ✅ YES (search space) | Main model dimension |
| `depth` | int | 12 | 10-40 | ✅ YES (search space) | Number of Mamba2 layers |
| `d_state` | int | 64 | 64-256, 16-aligned | ✅ YES (search space) | SSM state dimension |
| `d_conv` | int | 4 | 2-8 | ❌ NO (fixed at 4) | Causal conv kernel size |
| `expand` | int | 2 | 1-3 | ✅ YES (search space) | d_inner = dim × expand |
| `headdim` | int | 64 | 32, 64, 128 (power of 2) | ❌ NO (fixed at 64) | Per-head dimension |
| `dropout` | float | 0.0 | 0.0-0.3 | ❌ NO (fixed at 0.0) | Not used in research runs |

**Constraint:** `dim * expand` must be divisible by `headdim` (checked by mamba_ssm)

**Search space coverage:** 4/8 parameters are searchable (dim, depth, d_state, expand)

---

## 2. Training Hyperparameters

These control the training dynamics and optimization.

| Parameter | Type | Default | Valid Range/Values | In cmaes_search.py? | Notes |
|-----------|------|---------|-------------------|---------------------|-------|
| `batch_size` | int | 16 | 4-32 | ⚠️ AUTO-SCALED | Scales with params & chunk_size |
| `chunk_size` | int | 512 | 128-4096 | **✅ ADDED (v2.1)** | Sequence length - **NEW!** |
| `lr` | float | 1e-4 | 1e-4 to 3e-3 (log scale) | ✅ YES (search space) | Learning rate |
| `weight_decay` | float | 0.01 | 0.0-0.1 | ❌ NO (fixed at 0.01) | AdamW L2 regularization |
| `grad_clip` | float | 1.0 | 0.0-5.0 | ❌ NO (fixed at 1.0) | Gradient norm clipping |
| `train_minutes` | float | None | Any positive float | ✅ YES (CLI arg) | Time-based training |
| `warmup_steps` | int | 0 | 0-5000 | ❌ NO (schedulefree optimizer) | LR warmup steps |
| `optimizer` | str | 'schedulefree' | adamw/schedulefree | ✅ YES (fixed schedulefree) | Optimizer type |
| `seed` | int | 42 | Any int | ✅ YES (fixed at 42) | Random seed |
| `bf16` | bool | False | True/False | ✅ YES (always True) | BFloat16 training |

**Key update:** Added `--chunk_size` parameter to enable sequence length scaling experiments.

**Batch size auto-scaling logic (NEW):**
```python
seq_scale = chunk_size / 512.0  # Scaling factor
if params > 600M:
    batch_size = max(4, int(8 / seq_scale))
elif params > 400M:
    batch_size = max(4, int(16 / seq_scale))
else:
    batch_size = max(4, int(32 / seq_scale))
```

At chunk_size=2048 (4x longer), batch_size drops to **4** for 480M models (from 16).

---

## 3. Advanced SSM Parameters (mamba_ssm.Mamba2)

Low-level Mamba2 parameters - NOT currently exposed in wrapper.

| Parameter | Type | Default | Description | Exposable? |
|-----------|------|---------|-------------|------------|
| `ngroups` | int | 1 | Number of groups for grouped conv | Yes |
| `A_init_range` | tuple | (1, 16) | Range for SSM A matrix init | Rarely tuned |
| `D_has_hdim` | bool | False | Whether D has head dimension | Architecture choice |
| `rmsnorm` | bool | True | Use RMSNorm vs LayerNorm | Could be useful |
| `norm_before_gate` | bool | False | Normalization order | Could be useful |
| `dt_min` | float | 0.001 | Minimum delta timestep | SSM dynamics |
| `dt_max` | float | 0.1 | Maximum delta timestep | SSM dynamics |
| `dt_init_floor` | float | 0.0001 | Min dt initialization | SSM dynamics |
| `dt_limit` | tuple | (0.0, inf) | Timestep limits | SSM dynamics |
| `bias` | bool | False | Use bias in projections | Memory/speed tradeoff |
| `conv_bias` | bool | True | Use bias in convolution | Usually kept True |
| `use_mem_eff_path` | bool | True | Memory-efficient attention | Keep True |
| `sequence_parallel` | bool | True | Enable sequence parallelism | Multi-GPU only |

**Recommendation:** Keep defaults. These are rarely tuned in practice.

---

## 4. System/Performance Parameters

These control compilation and execution.

| Parameter | Type | Default | Valid Values | In cmaes_search.py? | Impact |
|-----------|------|---------|--------------|---------------------|--------|
| `compile` | bool | False | True/False | ✅ YES (--compile flag) | +17% throughput |
| `compile_mode` | str | 'max-autotune' | default/reduce-overhead/max-autotune | ✅ YES | Optimization level |
| `device` | str | 'cuda' | cuda/cpu | ✅ YES (always cuda) | Device placement |
| `dtype` | str | 'bfloat16' | float32/bfloat16/float16 | ✅ YES (bf16) | Precision |
| `tbptt` | bool | False | True/False | ❌ NO (not used) | Truncated BPTT |

**Recommendation:** Always use `--compile --compile_mode max-autotune` for +17% speed.

---

## CMA-ES Search Space Summary

**Currently searched (5D):**
1. `dim` - Model dimension (1024-3072, 128-aligned)
2. `depth` - Number of layers (10-40)
3. `d_state` - SSM state dimension (64-256, 16-aligned)
4. `expand` - Expansion factor (1-3)
5. `lr` - Learning rate (1e-4 to 3e-3, log scale)

**Fixed/auto-scaled:**
- `chunk_size` - **NOW CONFIGURABLE** (was 512, can be 2048)
- `batch_size` - Auto-scaled based on params and chunk_size
- `headdim` - Fixed at 64
- `d_conv` - Fixed at 4
- `optimizer` - Fixed at schedulefree
- `bf16` - Always True
- `seed` - Fixed at 42

**Not exposed (advanced):**
- All SSM dynamics parameters (dt_min, dt_max, etc.)
- Normalization choices (rmsnorm, norm_before_gate)
- Parallelism settings (sequence_parallel)

---

## Changes Made to cmaes_search_v2.py

**Added chunk_size parameter support:**

1. **Global variable** (line 62):
   ```python
   CHUNK_SIZE = 512  # Default, overridden by --chunk_size arg
   ```

2. **Argument parser** (line 1011):
   ```python
   parser.add_argument('--chunk_size', type=int, default=512,
                       help='Sequence chunk size (default: 512, for scaling: 1024, 2048)')
   ```

3. **Build command function** (line 387-392):
   ```python
   # Adjust batch size based on model size AND sequence length
   seq_scale = CHUNK_SIZE / 512.0  # Scaling factor relative to baseline 512

   if actual_params > 600_000_000:
       batch_size = max(4, int(8 / seq_scale))
   elif actual_params > 400_000_000:
       batch_size = max(4, int(16 / seq_scale))
   else:
       batch_size = max(4, int(32 / seq_scale))
   ```

4. **Main function** (line 1022):
   ```python
   global COMPILE_ENABLED, COMPILE_MODE, CHUNK_SIZE
   CHUNK_SIZE = args.chunk_size
   ```

---

## Experiment: Mamba2 @ chunk_size=2048

**Command:**
```bash
CUDA_VISIBLE_DEVICES=2 python -u cmaes_search_v2.py \
  --model mamba2 --phase both --train_minutes 10 --gpus 2 \
  --params 480M --output benchmark_results/cmaes_v10_compiled/scale_2048 \
  --lhs_samples 16 --compile --chunk_size 2048
```

**Expected behavior:**
- Batch size auto-scaled: 16 → **4** (4x sequence length)
- Memory usage: 4x higher due to parallel scan materialization
- Throughput: Lower tokens/s but same batches/s
- May encounter OOM for very large configs

**Monitoring:**
```bash
tail -f benchmark_results/cmaes_scale_2048_mamba2.log
```

**Results location:**
- Log: `benchmark_results/cmaes_scale_2048_mamba2.log`
- Results: `benchmark_results/cmaes_v10_compiled/scale_2048/mamba2_*/results.json`

---

## Future Work

**Potential additions to search space:**
1. `d_conv` - Causal conv kernel size (currently fixed at 4)
2. `headdim` - Per-head dimension (32, 64, 128)
3. `weight_decay` - L2 regularization (currently 0.01)
4. `rmsnorm` vs LayerNorm - Normalization type
5. `norm_before_gate` - Gating order

**Sequence scaling study:**
- Test chunk_size = 512, 1024, 2048, 4096
- Compare memory usage and throughput
- Find OOM boundaries for different model sizes
- E88 should scale better (sequential RNN, no scan materialization)

**Variable batch/sequence in CMA-ES:**
- Add batch_size and chunk_size as searchable params
- Handle OOM gracefully (worst fitness score)
- Explore pareto frontier: loss vs memory vs throughput

---

## Conclusion

Mamba2 has a rich parameter space, but CMA-ES focuses on the 5 most impactful dimensions: dim, depth, d_state, expand, lr. The addition of `--chunk_size` enables critical sequence length scaling experiments.

**Key insight:** The parallel scan architecture means memory scales linearly with sequence length. At chunk_size=2048, batch_size must drop from 16 → 4 to fit in GPU memory. This is a fundamental limitation of SSM models vs sequential RNNs like E88.

**Status:** Experiment running on GPU 2. Results expected in ~3 hours (16 LHS + CMA-ES refinement).
