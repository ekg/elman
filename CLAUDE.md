# Elman Project Guidelines

## Critical Design Principles

1. **NO PARALLEL SCAN** - We are testing nonlinear recurrent systems. Parallel scan only works for linear systems. All recurrence must be sequential.

2. **ALL models must use cuBLAS GEMMs** - This is the key to performance. Element-wise diagonal operations are memory-bound and slow. Follow e0's pattern:
   - Pre-compute W_x @ x for ALL timesteps (one big GEMM)
   - Per-timestep W_h @ h_prev (GEMM)

3. **Same number of layers across models** - For fair comparison, models should have similar depth. If a model has fewer params per layer, increase d_inner, don't increase depth.

## E2 Architecture Fix Needed

The current e2 (SlotElman) uses diagonal decay per slot with NO GEMMs - this is wrong:
- Current: h_t[:,s] = decay[:,s] * h_{t-1}[:,s] + B[:,s] * x (element-wise, O(d*n_slots))
- Correct: Each slot should have full W_h matmul like e0

Correct e2 design should be:
- n_slots independent Elman cells, each with W_x, W_h GEMMs
- Batch slots together for efficient GEMM: [B*n_slots, d] @ W_h
- Combine slots for output

This gives same depth as e0 with n_slots more memory capacity.

## Model Implementation Rules

1. **All new models MUST be implemented in CUDA first.** The CUDA kernel is the primary implementation.

2. **PyTorch fallbacks are optional** and only for:
   - Validation and correctness testing against CUDA
   - CPU-only debugging
   - Never for production benchmarking

3. **CUDA kernels go in** `elman/cuda/lib/` with naming convention `{model_name}_gpu.cu.cc`

4. **Python bindings go in** `elman/cuda/pytorch/elman_ladder.cc`

5. **Header declarations go in** `elman/cuda/lib/hasty/elman_ladder.h`

## Data Handling Rules

1. **NEVER pretokenize data** - No .npy token files, no preprocessing pipelines
2. **Use memory-mapped files** - mmap the raw text files directly
3. **Byte-level training preferred** - vocab_size=256, raw bytes as tokens
4. **Dynamic loading** - Sample random positions from mmap at training time:
   ```python
   with open('data/pile.txt', 'rb') as f:
       mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
   pos = np.random.randint(0, len(mm) - seq_len - 1, size=batch_size)
   for i, p in enumerate(pos):
       buf[i] = np.frombuffer(mm[p:p+seq_len+1], dtype=np.uint8)
   ```
5. **Data files**: `data/pile.txt` (has 0x1e document delimiters)

## Benchmarking Rules

1. Always use the same batch size across models for fair comparison
2. Use byte-level data (vocab_size=256) for benchmarks
3. Report both throughput (tok/s) and loss
4. Test at multiple batch sizes to understand scaling behavior
5. Use Last-100-step averaged loss for fair comparison (not instantaneous)

## Current Model Hierarchy

- `pure`: No output gating (h only)
- `x_gated`: X-only gating (h * silu(x + b)) - **BEST SIMPLE ELMAN**
- `diagonal`: Linear diagonal recurrence + x-gating - **WORSE LOSS** (no tanh hurts)
- `0` (Level 0): H+X gating (h * silu(h + x + b))
- Higher levels: Various selective mechanisms

## Benchmark Results (batch=32, 500 steps, 50M params)

| Model | Loss | Tok/s |
|-------|------|-------|
| Mamba2 | 1.95 | 100k |
| X-Gated | 2.03 | 35k |
| Level 0 | 2.05 | 34k |
| Diagonal | 2.52 | 39k | ← tanh is important!

**Key finding**: Linear recurrence (no tanh) loses too much expressivity. The x-gate alone cannot replace tanh as the nonlinearity.
