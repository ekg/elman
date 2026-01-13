# Elman RNN Research Guidelines

## Current Best Model

**E42 (Linear Tied Self-Gated) d1536×6** - 1.59 loss, 137K tok/s at 43M params
- Linear recurrence (no tanh) + tied weights (W_x = W_h) + self-gating
- Beats E33 baseline (1.62 loss, 116K tok/s) on both metrics
- 25% fewer params than E33 at equal quality, or better quality at equal params

**Architecture:**
```python
h_t = W @ x_t + W @ h_{t-1} + b    # Linear recurrence, tied weights
output = h_t * silu(h_t)            # Self-gating (only nonlinearity)
```

**Key optimizations:**
- Batched GEMM: pre-compute W @ x for all timesteps (E37v2 lesson)
- Spectral normalization for stability (linear recurrence needs ||W|| < 1)

## Model Variants

| Model | Description | Status |
|-------|-------------|--------|
| E0 | Stock Elman: h = tanh(W_h @ h + W_x @ x) | Baseline |
| E1 | + h+x selective gating | Fast |
| E33 | Self-gating: output = h * silu(h) | Good |
| E36 | Linear recurrence (no tanh) | Better loss |
| E37v2 | Tied weights + batched GEMM | Efficient |
| E42 | E36 + E37v2 (linear + tied + batched) | **Best** |

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

## Experiment Execution Rules

1. **RUN EXPERIMENTS IN PARALLEL** - When comparing multiple models/configs, launch them simultaneously on separate GPUs. Never run sequentially when parallelism is possible.
2. Use bash job scheduling pattern:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train.py --config a 2>&1 | tee /tmp/exp_a.log &
   CUDA_VISIBLE_DEVICES=1 python train.py --config b 2>&1 | tee /tmp/exp_b.log &
   CUDA_VISIBLE_DEVICES=2 python train.py --config c 2>&1 | tee /tmp/exp_c.log &
   wait  # Wait for all to complete
   ```
3. Up to 8 parallel jobs on 8 GPUs - assign each experiment to a different CUDA_VISIBLE_DEVICES
4. Always log outputs to files for later analysis

## Benchmarking Rules

1. **CRITICAL: ONE MODEL PER GPU** - Never run multiple models on the same GPU. They interfere with each other and produce unreliable results. Always use separate GPUs for each model being compared.
2. Always use the same batch size across models for fair comparison
3. Use byte-level data (vocab_size=256) for benchmarks
4. Report both throughput (tok/s) and loss
5. Test at multiple batch sizes to understand scaling behavior
6. Use Last-100-step averaged loss for fair comparison (not instantaneous)
7. Use the same random seed for data loading across all models for fair comparison

## CRITICAL: Testing Models with LadderLM

**ALWAYS use `LadderLM` from `elman.models` for benchmarking E-series models.** Do NOT write custom LM wrappers.

LadderLM includes:
- Mamba2's fused add+norm (from mamba_ssm.ops.triton.layer_norm)
- Proper prenorm + residual stream pattern
- Tied embeddings

Wrong (loses ~1 nat!):
```python
# DON'T DO THIS - missing fused norm, wrong residual pattern
class BadE1LM(nn.Module):
    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            out, _ = layer(h)
            h = h + out  # WRONG residual pattern
        return self.head(self.norm(h))
```

Correct:
```python
from elman.models import LadderLM
model = LadderLM(vocab_size=256, dim=1024, depth=6, level=1)
loss = model(x, return_loss=True)
```

For custom cells (E29a, E29c, etc.) that aren't in LadderLM:
- Copy the exact forward pattern from LadderLM (fused_add_norm + residual stream)
- Or add the cell to `ladder_lm.py:get_ladder_level()`

## Latest Benchmark Results (10 min training, Last-100 avg, seed=42)

### E42 vs Baselines (matched ~40M params, depth=6)

| Model | Config | Params | Loss | Throughput |
|-------|--------|--------|------|------------|
| **E42** | d1536×6 | 42.9M | **1.59** | **137K tok/s** |
| E33 | d1280×6 | 39.7M | 1.62 | 116K tok/s |
| E36 | d1280×6 | 39.7M | 1.63 | 138K tok/s |
| E37v2 | d1280×6 | 29.8M | 1.58 | 121K tok/s |

### E33-E41 Simplification Experiments (d1280×6)

| Model | Loss | Throughput | Description |
|-------|------|------------|-------------|
| E33 | 1.665 | 140K | Self-gate baseline |
| E34 | 1.769 | 253K | Diagonal W_h (+80% speed) |
| E36 | 1.630 | 138K | Linear recurrence (best loss) |
| E37v2 | 1.576 | 121K | Tied weights + batched GEMM |
| E39 | 1.667 | 145K | No bias (fastest) |

**Key findings:**
- E42 combines E36 (linear) + E37v2 (tied + batched) for best overall
- Linear recurrence removes tanh, improves gradient flow
- Tied weights reduce params, batched GEMM recovers speed
- Self-gating (h * silu(h)) provides sufficient nonlinearity
