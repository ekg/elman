# Standard Testing Protocol

## Benchmark Configuration

All model comparisons use this standard protocol:

| Parameter | Value |
|-----------|-------|
| Data | `data/pile.txt` (1.3TB, byte-level with 0x1e doc delimiters) |
| Vocab | 256 (byte-level) |
| Batch size | 256 |
| Sequence length | 512 |
| Steps | 1000 |
| Optimizer | AdamWScheduleFree (lr=3e-4, weight_decay=0.1) |
| Precision | bfloat16 |
| Target params | ~50M |

## Metric: Last-100-Step Average Loss

We report **Last100** - the average loss over steps 901-1000.

This eliminates batch-level variance that can make instantaneous loss misleading.
Example: A model showing "1.39" final loss might have Last100=1.45, while another
showing "1.47" final might also have Last100=1.45.

## Running Experiments

Use `run_e5_e1_mamba2_parallel.py` as template:

```bash
python -u run_e5_e1_mamba2_parallel.py > results.log 2>&1 &
```

Each experiment runs on a separate GPU (CUDA_VISIBLE_DEVICES).
Results are saved to `benchmark_results/e5_e1_mamba2_compare/`.

## Reference Results (2026-01-05)

50M params, 1000 steps, batch=256:

| Model | Last100 | Params | tok/s |
|-------|---------|--------|-------|
| mamba2_50m | 1.4422 | 50.9M | ~104K |
| e5_d1536_r270 | 1.4463 | 50.3M | ~81K |
| e5_d2048_r200 | 1.4565 | 49.8M | ~87K |
| e5_d1024_r404 | 1.4620 | 50.0M | ~88K |
| e1_50m | 1.4742 | 49.7M | ~175K |
| e5_d768_r539 | 1.4815 | 49.9M | ~59K |

## E5 Parameter Calculation

E5 layer params: `6 * dim * rank + dim` (6 low-rank matrices + bias)
Total: `embed + depth * (layer_params + 2*dim) + 2*dim`

Where:
- embed = vocab_size * dim = 256 * dim
- 2*dim per layer for LayerNorm
- 2*dim for final norm

## Data Loading

For byte-level training on pile.txt:
1. Memory-map the file with `mmap`
2. Sample random positions: `np.random.randint(0, data_len - seq_len - 1, size=batch_size)`
3. Read contiguous chunks of `seq_len + 1` bytes
4. No tokenization needed - each byte IS a token

Note: pile.txt contains 0x1e document delimiters (~9M per 100MB), but random
sampling ignores these boundaries. This is acceptable for byte-level benchmarking.
