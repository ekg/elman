# Elman RNN Research Guidelines

This repo is a study into the computational expressiveness of extremely minimal recurrent neural network architectures.
To engage in this study, we implement these kernels in CUDA, highly optimized CUDA kernels that are at the maximum level of efficiency we expect to be able to get out of our hardware.
This avoids basic contention about implementation details. It basically renders all models equivalent in terms of their comparison also to existing systems like Mamba 2, which we're profiling, GRU systems, LSTM as well.
This work is, however, quite laborious, and so in the context of it, I do suggest that the agent remembers to run background sub-agents to implement custom CUDA kernels and the particular variants which we're developing in the process of this research.
Each variant is typically labeled as E something. So we have E from zero is kind of a standard eleman network. I don't know if it's actually exactly standard to be perfectly honest, but it was the first implementation we came in with.
After this, we've made some modifications generating E1, and that for a long time was the best performing system. It still may be actually, even in tests that we've done on scaling to a billion parameters. There's something wrong with its scaling performance...
And that in a nutshell is a lot of the kind of work that's been happening. It's been a slow process to try to regularize, organize all the effort to develop the standard protocol for how to implement the CUDA kernels to... Basically the standard protocol involves cross-checking the Python and CUDA implementation in forward and backward passes on the same data to verify no difference in output.
We want to make sure that we remember to do this, and this process is very involved. So we basically need to be sure to run subagents in the background to do this. And in fact, for really a lot of effort, you want to be using salvations. I think it helps you save your focus and not get confused or distracted and allows a lot of work to be done at the same time. It also can simplify your interface with the system.

## Current Best: E88 at 480M Scale (Jan 26, 2026)

### Benchmark Results (Matched Params, Same Seed)

All models tested at ~480M params, seed=42, 10 min training, batch_size=16, chunk_size=512, lr=3e-4, bf16.

| Model | Params | Loss | Steps | Tok/s | Notes |
|-------|--------|------|-------|-------|-------|
| **E88** | 480M | **1.37** | 626 | 8.3K | Nonlinear matrix state (h=104, n=32) |
| **Mamba2** | 475M | **1.37** | 1541 | 20.8K | SSM baseline |
| FLA-GDN | 474M | 1.39 | 1433 | 19.1K | ICLR 2025 baseline |
| E88 d=24 | 479M | 1.56 | 850 | 11.0K | Too shallow |
| E88 d=20 | 478M | 1.62 | 720 | 9.3K | Too shallow |
| E88 d=40 | 478M | 1.78 | 450 | 6.0K | Too deep |
| E88 d=36 | 481M | 1.85 | 510 | 6.5K | Too deep |

**Key findings:**
- E88 matches Mamba2 loss (1.37 vs 1.37) and beats FLA-GDN (1.37 vs 1.39)
- E88 learns **2.4x faster per step** (1.37 in 626 steps vs 1541 for Mamba2)
- E88 is ~2.5x slower throughput due to sequential recurrence (no parallel scan)
- Depth 28-32 is optimal for E88 at 480M scale (clear U-shape in loss vs depth)
- n_state=32 with SiLU gating is optimal

### State Size Comparison (Critical Finding)

E88 achieves competitive loss with **dramatically less state memory**:

| Model | Config | State Structure | State/Layer | vs E88 |
|-------|--------|-----------------|-------------|--------|
| **E88** | h=104, n=32 | 104 × (32×32) matrices | **106,496** | 1x |
| **Mamba2** | d=1664, exp=2 | 3328 × 64 SSM | **213,248** | 2x |
| **FLA-GDN** | d=1984, exp=2 | ~16 × (124×248) | **~492,032** | 4.6x |

**State Architecture Details:**

```
E88:     104 heads × 32×32 NONLINEAR matrix state (tanh applied)
         Update: S = tanh(decay * S + outer(delta, k))

Mamba2:  3328 × 64 diagonal SSM (LINEAR, parallel scan)
         Update: h = A*h + B*x

FLA-GDN: 16 heads × 124×248 LINEAR attention matrix
         Update: S = decay*S + outer(k, v)
```

**Why this matters:** E88's 104 small (32×32) nonlinear matrix heads provide high expressivity through **parallel specialized memories** rather than through raw state size. Each head learns independent key-value associations.

### Optimal E88 Configuration (480M)

```
dim=896, depth=32, n_heads=104, n_state=32, use_gate=1, gate_activation=silu
```

Or equivalently at depth=28:
```
dim=1024, depth=28, n_heads=104, n_state=32, use_gate=1, gate_activation=silu
```

### Reproducing the Benchmark

```bash
# E88 at 480M (optimal config)
CUDA_VISIBLE_DEVICES=0 python train.py --level E88 \
  --dim 896 --depth 32 --n_heads 104 --n_state 32 \
  --use_gate 1 --gate_activation silu \
  --data data/pile.txt --batch_size 16 --chunk_size 512 \
  --lr 3e-4 --seed 42 --bf16 --train_minutes 10

# Mamba2 at 480M
CUDA_VISIBLE_DEVICES=1 python train.py --level mamba2 \
  --dim 1664 --depth 28 \
  --data data/pile.txt --batch_size 16 --chunk_size 512 \
  --lr 3e-4 --seed 42 --bf16 --train_minutes 10

# FLA-GDN at 480M
CUDA_VISIBLE_DEVICES=2 python train.py --level fla-gdn \
  --dim 1984 --depth 24 \
  --data data/pile.txt --batch_size 16 --chunk_size 512 \
  --lr 3e-4 --seed 42 --bf16 --train_minutes 10
```

### n_state Scaling (480M, SiLU gate)

| n_state | dim | depth | n_heads | Loss | Tok/s |
|---------|-----|-------|---------|------|-------|
| 16 | 2432 | 32 | 76 | 1.49 | 15.9K |
| **32** | 896 | 32 | 104 | **1.31** | 8.1K |
| 48 | 768 | 24 | 108 | 1.81 | 5.4K |

n_state=32 is optimal. Smaller (16) or larger (48) performs worse.

### Performance Gap Analysis

E88 is 2x slower than Mamba2/FLA-GDN. Next step: **profile and optimize CUDA kernel**.

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

## CRITICAL: CUDA-First Development

**EVERYTHING is implemented in CUDA. Python is ONLY for mathematical verification.**

Python implementations are 100-300x slower than optimized CUDA kernels due to:
- Sequential Python loops over timesteps
- No GPU parallelization
- High memory bandwidth from frequent small tensor operations

**The standard workflow is:**
1. Design the algorithm mathematically
2. Write the CUDA kernel (copy existing similar kernel, modify incrementally)
3. Write a minimal Python reference for gradient verification
4. Cross-check Python and CUDA outputs on the same data
5. Only benchmark the CUDA version

**When copying kernels for modification:**
- NEVER rewrite from scratch - copy an existing working kernel
- Test after every small modification
- Use the same memory layout patterns as the source kernel

## Model Implementation Rules

1. **All new models MUST be implemented in CUDA first.** The CUDA kernel is the primary implementation.

2. **PyTorch fallbacks are optional** and only for:
   - Validation and correctness testing against CUDA
   - CPU-only debugging
   - Never for production benchmarking

3. **CUDA kernels go in** `elman/cuda/lib/` with naming convention `{model_name}_gpu.cu.cc`

4. **Python bindings go in** `elman/cuda/pytorch/elman_ladder.cc`

5. **Header declarations go in** `elman/cuda/lib/hasty/elman_ladder.h`

6. **Level naming convention**: Use `E75h2`, `E75h4`, etc. (with the "E" prefix) for all variant names in ladder_lm.py

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

1. **USE THE JOB SCHEDULER** - For benchmarks, always use `sched.py` to ensure one job per GPU:
   ```bash
   python gen_jobs.py e75_100m --minutes 10 > jobs.txt
   python sched.py jobs.txt
   python extract_results.py sched_logs/TIMESTAMP/
   ```

2. **NEVER launch ad-hoc parallel jobs** - The old bash pattern caused GPU contention issues.

3. Up to 8 parallel jobs on 8 GPUs - scheduler handles assignment automatically

4. **CHECK GPU AVAILABILITY BEFORE ASSIGNING** - Before launching experiments, check which GPUs are free:
   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
   ```
   Avoid GPUs with significant memory usage or high utilization - they may be in use by other users. When a user specifies to stay off certain GPUs, respect that.

5. Always extract results with `extract_results.py` for consistent last-100 avg loss

## Benchmarking Rules

1. **CRITICAL: ONE MODEL PER GPU** - Never run multiple models on the same GPU. They interfere with each other and produce unreliable results. Always use separate GPUs for each model being compared.
2. Always use the same batch size across models for fair comparison
3. Use byte-level data (vocab_size=256) for benchmarks
4. Report both throughput (tok/s) and loss
5. Test at multiple batch sizes to understand scaling behavior
6. Use Last-100-step averaged loss for fair comparison (not instantaneous)
7. Use the same random seed for data loading across all models for fair comparison

## GPU Job Scheduler (Preferred Benchmarking Method)

**ALWAYS use the job scheduler for benchmarks.** This ensures one job per GPU, no contention, reproducible results.

### Quick Start
```bash
# 1. Generate jobs file
python gen_jobs.py e75_100m --minutes 10 > jobs.txt

# 2. Run jobs (one per GPU, automatic scheduling)
python sched.py jobs.txt

# 3. Extract results (computes last-100 avg loss)
python extract_results.py sched_logs/TIMESTAMP/
```

### Available Tools

| Tool | Purpose |
|------|---------|
| `gen_jobs.py` | Generate benchmark job files |
| `sched.py` | Run jobs one-per-GPU |
| `extract_results.py` | Extract last-100 avg loss from logs |
| `calc_dim.py` | Calculate 128-aligned dims for target params |

### Available Benchmarks
```bash
python gen_jobs.py --list
# e75_100m        - E75 Multi-Head 100M param comparison (7 models)
# e75_extended    - Extended E75 scan (10 models)
# baselines       - Baseline models (4 models)
# quick_test      - Quick 2-model test
```

### Dimension Calculator
```bash
# Show all standard 100M configs
python calc_dim.py --standard

# Calculate for specific model
python calc_dim.py --model E75h4n32 --params 100M --depth 20
```

### Standard 100M Configurations
| Model | Dim | Depth | Extra | Params |
|-------|-----|-------|-------|--------|
| mamba2 | 896 | 20 | expand=2 | 101M |
| fla-gdn | 768 | 20 | expansion=2.0 | 95M |
| E75h4n16 | 2048 | 20 | H=4, n=16, exp=1.0 | 98M |
| E75h4n24 | 2048 | 20 | H=4, n=24, exp=1.0 | 104M |
| E75h4n32 | 1920 | 20 | H=4, n=32, exp=1.0 | 99M |
| E75h8n16 | 1920 | 20 | H=8, n=16, exp=1.0 | 99M |
| E75h8n24 | 1792 | 20 | H=8, n=24, exp=1.0 | 99M |

### CRITICAL: E75 n_state Constraints
**n_state MUST be a multiple of 8** for numerical stability. Valid values: 16, 24, 32, 40, 48, etc.
Values like 20, 28 cause NaN during training.

## Configuration Search: CMA-ES

We use **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** for architecture/hyperparameter search.

### Why CMA-ES?

- Gradient-free optimization for discrete/mixed parameters (n_heads, n_state, depth)
- Handles non-convex fitness landscapes (loss vs config is highly irregular)
- Runs population in parallel across GPUs
- Found optimal E88 config: h=104, n=32 (104 heads is unexpected!)

### Running CMA-ES Search

```bash
pip install cma

# E88 search (most likely to find improvements)
python cmaes_search.py --model e88 --generations 50 --train_minutes 2 --gpus 0,1,2,3

# FLA-GDN search
python cmaes_search.py --model fla-gdn --generations 30 --train_minutes 2 --gpus 0,1,2,3

# Mamba2 search (already well-optimized)
python cmaes_search.py --model mamba2 --generations 20 --train_minutes 2 --gpus 0,1,2,3
```

### CMA-ES Search Space

| Model | Parameters Searched |
|-------|---------------------|
| E88 | n_heads (64-160), n_state (16/32/48/64), depth (20-40) |
| FLA-GDN | expansion (1-3), depth (16-40), n_heads (8-32) |
| Mamba2 | d_state (64-256), expand (1-3), depth (16-40) |
| Transformer | n_heads (8-32), expansion (2-6), depth (12-36) |
| MinGRU/MinLSTM | expansion (1-4), depth (12-40) |
| GRU/LSTM | expansion (1-3), depth (12-48) |

**Note:** LR is fixed at 3e-4 for all models to ensure fair comparison.

### CMA-ES Study Results (Jan 27, 2026)

Full 480M-scale search across 8 models. See `docs/CMAES_480M_STUDY.md` for details.

| Model | CMA-ES Best Loss | Optimal Config Found |
|-------|------------------|----------------------|
| **Mamba2** | **1.2713** | d_state=96, expand=2, depth=25, dim=1792 |
| **FLA-GDN** | **1.2727** | expansion=2, depth=17, n_heads=24, dim=1920 |
| **E88** | 1.37* | n_heads=104, n_state=32, depth=32 (*prior benchmark) |
| Transformer | 1.5054 | n_heads=8, expansion=4, depth=13, dim=1536 |
| MinGRU | 1.5281 | expansion=1, depth=14, dim=2944 |
| MinLSTM | 1.5608 | expansion=1, depth=31, dim=1792 |
| GRU | Failed | Training instability at 480M scale |
| LSTM | Failed | Training instability at 480M scale |

**Key Findings from CMA-ES:**
- Mamba2 optimal d_state=96 (vs default 64) - larger state helps
- FLA-GDN optimal depth=17 (vs default 24) - shallower is better
- MinGRU/MinLSTM both prefer expansion=1 (wider networks, no expansion)
- CUDA GRU/LSTM have training instability at 480M scale (needs investigation)

## E88 Configuration Reference

E88 combines FLA-GDN's Mamba2-style decay with nonlinear (tanh) matrix state.

### Optimal E88 Parameters (480M scale)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_state` | **32** | Critical - 16 or 48 are worse |
| `use_gate` | **1** | SiLU gating helps at 480M scale |
| `gate_activation` | **silu** | |
| `expansion` | **1.0** | Square state matrix |
| `depth` | **28-32** | U-shaped loss curve, these are optimal |
| `use_conv` | 0 | Not needed |
| `use_output_norm` | 0 | Not needed |

### CUDA Kernel n_state Support

E88 CUDA kernel supports: **4, 8, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 80, 88, 96, 128**

Large n_state (>=80) uses global memory fallback (slower).

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

## 100M Parameter Benchmark (Reference: Jan 14, 2026)

### Running the Benchmark

```bash
# Run all models (uses all available GPUs automatically)
python run_100m_benchmark.py

# Results saved to: benchmark_results/100m_10min/
# - Logs: benchmark_results/100m_10min/{model}.log
# - Checkpoints: benchmark_results/100m_10min/{model}/level{model}_100m_{timestamp}/
# - Configs: benchmark_results/100m_10min/configs.json
```

### Benchmark Configuration

- **Data**: `data/pile.txt` (byte-level, vocab_size=256)
- **Training time**: 10 minutes per model
- **Batch size**: 32, Chunk size: 512
- **Learning rate**: 3e-4, Warmup: 1000 steps
- **Target params**: ~100M (dim varies by model architecture)
- **Depth**: 20 layers for all models
- **Seed**: 42

### CRITICAL: Model Dimension Requirements

**All benchmark configurations MUST follow these rules:**

1. **Dimensions must be 128-aligned** - All model dims must be divisible by 128 (e.g., 768, 896, 1024, 1152, 1280, etc.)
2. **Expansion factor = 2.0** - All models use d_inner = dim × 2 (except where architecture-specific)
3. **Target ~100M params** - Adjust dim to hit approximately 100M parameters
4. **Depth = 20 layers** - Standard depth for fair comparison

**Reference configs for ~100M params at depth=20:**
```
# E74 with expansion=1.0 (wider dims, no internal expansion)
E74 Full/Diag:    dim=2048-2176, expansion=1.0, n_state=32-96

# E74 with expansion=2.0 (narrower dims, 2x d_inner)
E74 Full/Diag:    dim=1408-1536, expansion=2.0, n_state=32-96

# Baselines (all use expansion=2.0)
E1/E42 style:     dim=640,  expansion=2.0
Mamba2:           dim=896,  depth=20, expand=2
FLA-GDN:          dim=768,  expansion=2.0
CUDA GRU/LSTM:    dim=384,  expansion=2.0
```

**Testing both expansion factors:** For E74 models, test both expansion=1.0 and expansion=2.0
to determine which performs better. expansion=2.0 gives wider d_inner at cost of smaller dims.

**Why 128-aligned?** GPU tensor cores and memory access patterns are optimized for multiples of 128. Non-aligned dims waste compute and memory bandwidth.

### Extracting Results

**If logs are overwritten, get results from checkpoint filenames:**
```bash
for dir in benchmark_results/100m_10min/*/level*; do
  model=$(echo $dir | sed 's|.*/\([^/]*\)/level.*|\1|')
  latest=$(ls -t $dir/checkpoint_*.pt 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    loss=$(basename $latest | sed 's/.*loss_\([0-9.]*\)\.pt/\1/')
    steps=$(basename $latest | sed 's/.*step_0*\([0-9]*\)_.*/\1/')
    echo "$model: steps=$steps loss=$loss"
  fi
done | sort -t'=' -k3 -n
```

### Reference Results (Jan 14, 2026 run)

| Model | Steps | Final Loss | Dim | Notes |
|-------|-------|------------|-----|-------|
| **fla-gdn** | 2228 | **1.16** | 768 | FLA GatedDeltaNet (ICLR 2025) |
| **mamba2** | 2200 | **1.23** | 896 | Mamba2 SSM baseline |
| e68 | 3047 | 1.44 | 640 | Self-gating h-dependence |
| e67 | 2971 | 1.52 | 640 | H-gated alpha |
| e1 | 1657 | 1.58 | 640 | Gated Elman baseline |
| e42 | 1838 | 1.59 | 768 | Linear tied self-gate |
| e56 | 1424 | 1.63 | 640 | Concat Elman |
| e65 | 3233 | 1.63 | 640 | Diagonal H |
| e64 | 3122 | 1.69 | 640 | Additive H |
| llama | 1690 | 1.69 | 640 | Transformer baseline |
| e61 | 2964 | 1.76 | 640 | Decay gated |
| e63 | 2154 | 1.77 | 512 | Nonlinear delta |
| e62 | 2960 | 1.78 | 640 | Selective write |
| e66 | 1021 | 1.85 | 640 | Low-rank H |

### Matrix State Models (E70-E73)

Matrix state models have O(n²) state size and require different dims to hit 100M params:

| Model | Dim | n_state | State Size | Notes |
|-------|-----|---------|------------|-------|
| E70 | 1408 | 96 | 96×96 | Linear matrix update |
| E71 | 1408 | 96 | 96×96 | S-dependent gating |
| E72 | 1408 | 96 | 96×96 | Memory-gated value |
| E73 | 1408 | 96 | 96×96 | Nonlinear delta rule |
| E73cp | 1408 | 96 | 96×96 | E73 with gradient checkpointing |

**E73cp Checkpointing**: 2.5x memory reduction (7.8 GB vs 19.7 GB) with ~6% throughput overhead.

## E88 Files

- `elman/cuda/lib/e88_fla_hybrid_gpu.cu.cc` - CUDA forward/backward kernel
- `elman/models/e88_fla_hybrid.py` - Python model definition
