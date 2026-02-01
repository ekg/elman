# CMA-ES Experiment Handoff

**Date:** 2026-01-29
**Purpose:** Document CMA-ES hyperparameter search methodology for continuation

## Current Methodology

### Training Protocol (as of Jan 2026)

All CMA-ES experiments have used:
- **Training time:** 10 minutes per evaluation
- **Loss metric:** Last-100 step average (avoids noisy single-step values)
- **Data:** `data/pile.txt` (byte-level, vocab_size=256)
- **Batch size:** 8-16 depending on model memory
- **Chunk size:** 512 tokens
- **Learning rate:** 3e-4 (fixed across all models for fair comparison)
- **Precision:** bfloat16
- **Seed:** 42 for reproducibility

### CMA-ES Configuration

```python
# Population size: 8 (runs in parallel across GPUs)
# Generations: 15-30 depending on search
# Target params: 480M (±20M tolerance)
```

### Search Spaces by Model

| Model | Parameters Searched | Constraints |
|-------|---------------------|-------------|
| **E88** | n_heads ∈ [64,160], n_state ∈ {16,32,48,64}, depth ∈ [12,40] | n_state must be in CUDA kernel supported set |
| **E90** | n_heads ∈ [64,160], config_idx ∈ {0,1,2,3}, depth ∈ [10,30] | config_idx maps to (k_fast,k_slow) pairs |
| **Mamba2** | d_state ∈ [64,256], expand ∈ [1,3], depth ∈ [16,40] | |
| **FLA-GDN** | expansion ∈ [1,3], depth ∈ [16,40], n_heads ∈ [8,32] | |
| **Transformer** | n_heads ∈ [8,32], expansion ∈ [2,6], depth ∈ [12,36] | |
| **MinGRU/MinLSTM** | expansion ∈ [1,4], depth ∈ [12,40] | |
| **MoM-E88** | n_heads ∈ [32,128], top_k ∈ [4,32], n_state ∈ {32,64}, depth ∈ [8,24] | |

### Running CMA-ES

```bash
# Standard 10-minute search
python cmaes_search.py --model e88 --generations 30 --train_minutes 10 --gpus 0,1,2,3,4,5,6,7

# Results saved to: benchmark_results/cmaes_{model}_10min/
# - results.json: best config and full history
# - eval_{N}.log: training log for each evaluation
```

## Best Results (10-minute training, 480M scale)

| Model | Loss | Steps | Tok/s | Optimal Config |
|-------|------|-------|-------|----------------|
| **Mamba2** | 1.2713 | 3140 | 22.7K | d_state=96, expand=2, depth=25, dim=1792 |
| **FLA-GDN** | 1.2727 | 3110 | 22.3K | expansion=2, depth=17, n_heads=24, dim=1920 |
| **E88** | 1.3905 | 2050 | 14.0K | n_heads=98, n_state=32, depth=14, dim=2176 |
| **Transformer** | 1.5054 | 4820 | 34.1K | n_heads=8, expansion=4, depth=13, dim=1536 |
| **MinGRU** | 1.5281 | 4050 | 27.8K | expansion=1, depth=14, dim=2944 |
| **MinLSTM** | 1.5608 | 3190 | 21.3K | expansion=1, depth=31, dim=1792 |
| **MoM-E88** | 1.7620 | 740 | 20.4K | n_heads=40, top_k=8, n_state=64, depth=12 |
| **E90** | 1.7914 | 420 | 13.8K | n_heads=114, k_fast=8, k_slow=16, depth=13 |
| **GRU/LSTM** | FAILED | -- | -- | Training instability at 480M scale |

### Results Locations

```
benchmark_results/cmaes_mamba2_10min/mamba2_480M_15gen_20260127_031532/results.json
benchmark_results/cmaes_fla-gdn_10min/fla-gdn_480M_15gen_20260127_060247/results.json
benchmark_results/cmaes_e88_10min_v4/e88_480M_30gen_20260127_180658/results.json
benchmark_results/cmaes_transformer_10min/transformer_480M_15gen_20260126_171317/results.json
benchmark_results/cmaes_mingru_10min/mingru_480M_15gen_20260126_201433/results.json
benchmark_results/cmaes_minlstm_10min/minlstm_480M_15gen_20260126_235957/results.json
benchmark_results/cmaes_search/mom-e88_480M_30gen_20260128_171146/results.json
benchmark_results/cmaes_e90/e90_500M_20gen_20260129_140231/results.json
```

## Known Issues

1. **10 minutes is not convergence:** Models are still improving at 10 min cutoff. This introduces bias—faster models get more gradient steps.

2. **GRU/LSTM instability:** CUDA GRU/LSTM fail at 480M scale with gradient explosions. Needs investigation.

3. **E90 backward kernel bug:** k_slow=64 configs cause CUDA errors in backward pass. Excluded from search.

4. **MoM-E88 slow:** Only ~740 steps in 10 min due to routing overhead. May need longer training.

## Next Session: Convergence-Based Training

### Proposed Methodology

Instead of fixed 10-minute training, run until convergence:

**Convergence criterion:** Loss improvement < Δ nats over N iterations

Suggested parameters:
- Δ = 0.05 nats (or 0.1, 0.2 for faster experiments)
- N = 500 steps (or time-based: 5 minutes)
- Maximum training time: 60 minutes (safety cutoff)

### Implementation

```python
def is_converged(loss_history, window=500, threshold=0.05):
    """Check if loss has plateaued."""
    if len(loss_history) < window:
        return False
    recent = loss_history[-window:]
    improvement = recent[0] - recent[-1]  # positive if improving
    return improvement < threshold
```

### Key Changes Needed

1. **Modify train.py:** Add `--converge_threshold` and `--converge_window` args
2. **Modify cmaes_search.py:** Use convergence criterion instead of fixed time
3. **Longer maximum time:** 30-60 minutes vs current 10 minutes
4. **Report final converged loss:** Not last-100 avg at arbitrary cutoff

### Fair Comparison Considerations

- All models should use same convergence criterion
- Report both final loss AND training time to convergence
- Some models may converge faster (fewer steps) but to worse loss
- Consider compute-normalized comparisons (loss vs total FLOPs)

## Files to Know

- `cmaes_search.py`: Main CMA-ES search script
- `train.py`: Training script with time-based cutoff
- `calc_dim.py`: Calculate dimensions for target param count
- `docs/e88_math.typ`: Mathematical description of E88 (updated today)

## Documentation

- `docs/e88_math.typ` / `e88_math.pdf`: Full E88 math with CMA-ES results table
- PDF uploaded to: https://hypervolu.me/elman/e88_math.pdf
- CLAUDE.md: Project guidelines and benchmark standards
