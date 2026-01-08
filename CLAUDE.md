# Elman RNN Research Guidelines

## Current Best Model

**E1 (Gated Elman) d1280×6** - 1.43 loss, 254K tok/s at 50M params
- Beats Mamba2 (1.53 loss, 101K tok/s) by 3× throughput
- Sweet spot: depth=6, wider is better up to a point

## Model Variants

| Model | Description | Status |
|-------|-------------|--------|
| E0 | Stock Elman: h = tanh(W_h @ h + W_x @ x) | Baseline |
| E1 | + h+x selective gating | **Best** |
| E5 | Low-rank: U @ V @ h instead of W @ h | Slower |

## Critical Design Principles

1. **NO PARALLEL SCAN** - We test nonlinear recurrence. Sequential only.

2. **ALL models must use cuBLAS GEMMs** - Element-wise ops are memory-bound.
   - Pre-compute W_x @ x for ALL timesteps (one big GEMM)
   - Per-timestep W_h @ h_prev (GEMM)

3. **Wider + shallower wins** - But not too shallow (depth=6 optimal)

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

## Latest Benchmark Results (50M params, 10 min training, Last-100 avg)

| Model | Loss | Throughput |
|-------|------|------------|
| E1 d1280×6 | 1.43 | 254K tok/s |
| E1 d1024×10 | 1.45 | 214K tok/s |
| Mamba2 | 1.53 | 101K tok/s |
| E5 d1536 r270 | 1.61 | 81K tok/s |

**Key finding**: E1's throughput advantage (3×) dominates. Low-rank (E5) is theoretically interesting but slower in practice.

## Activation Function Comparison (5 min training, d1280×6)

| Activation | Loss | Throughput |
|------------|------|------------|
| tanh (E1) | 1.49 | 139.8K tok/s |
| softsign (E15) | 1.53 | 138.6K tok/s |

**Key finding**: tanh beats softsign by ~0.04 nats. Modern GPUs have optimized tanh implementations that make it competitive despite the exp() cost. Stick with tanh for E1.
