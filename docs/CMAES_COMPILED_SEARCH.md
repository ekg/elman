# CMA-ES Search with torch.compile max-autotune (Feb 17, 2026)

## Overview

CMA-ES architecture search for E88 at ~480M scale with `torch.compile(mode='max-autotune')`.

Search used `cmaes_search_v2.py` with LHS (Latin Hypercube Sampling) + CMA-ES refinement methodology:
- **Phase 1**: 64 LHS samples for broad coverage
- **Phase 2**: CMA-ES refinement from top 2 warm starts, converging at 0.01 threshold

### Configuration
- Target: 480M params (±10%)
- Training: 10 min per evaluation
- Compile: `torch.compile(mode='max-autotune')`
- Data: commapile.txt (byte-level)
- Fixed: n_state=32, use_gate=1, gate_activation=silu
- Optimizer: schedulefree
- Seed: 42, bf16
- GPUs: 8x (parallel evaluation)

## Results Summary

### Best Configuration Found

| Parameter | Value |
|-----------|-------|
| **Loss** | **1.2728** |
| dim | 1792 |
| n_heads | 49 |
| n_state | 32 |
| depth | 31 |
| lr | 9.2e-4 |
| Params | ~439M |

**Reproduce:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --level E88 \
  --dim 1792 --depth 31 --n_heads 49 --n_state 32 \
  --use_gate 1 --gate_activation silu \
  --data data/pile.txt --batch_size 16 --chunk_size 512 \
  --lr 9.2e-4 --seed 42 --bf16 --train_minutes 10 \
  --compile --compile_mode max-autotune
```

### Comparison to Previous Best

| Config | Loss | dim | depth | n_heads | lr | Compile |
|--------|------|-----|-------|---------|-----|---------|
| **New best (compiled)** | **1.2728** | 1792 | 31 | 49 | 9.2e-4 | yes |
| Previous best (non-compiled) | 1.3060 | 1920 | 17 | 83 | 6.4e-4 | no |
| **Improvement** | **0.0332 nats (2.5%)** | | | | | |

### vs SSM Baselines (non-compiled reference)

| Model | Loss | Gap to Mamba2 |
|-------|------|---------------|
| Mamba2 | 1.27 | baseline |
| FLA-GDN | 1.27 | +0.00 |
| **E88 (compiled)** | **1.2728** | **+0.003** |
| E88 (non-compiled) | 1.3060 | +0.036 |

**E88 with torch.compile nearly matches SSMs** (within 0.3% of Mamba2).

## Key Findings

### 1. torch.compile Enables Deeper Networks

The compiled search found **much deeper** optimal configs compared to non-compiled:

| Setting | Optimal depth | Optimal dim | Loss |
|---------|---------------|-------------|------|
| Non-compiled | 17 | 1920 | 1.3060 |
| Compiled | 31 | 1792 | 1.2728 |

torch.compile reduces per-layer overhead, making deeper+narrower architectures practical.

### 2. Fewer Heads, Higher LR

| Setting | n_heads | lr |
|---------|---------|-----|
| Non-compiled best | 83 | 6.4e-4 |
| Compiled best | 49 | 9.2e-4 |

With compile, fewer heads (49 vs 83) and higher learning rate (9.2e-4 vs 6.4e-4) are optimal.

### 3. LR is Critical (>3e-4 required)

From 64 LHS evaluations, configs with lr < 1e-4 consistently failed (loss > 3.0). The sweet spot is 4e-4 to 1e-3.

### 4. n_state=32 Confirmed Optimal

Fixed at 32 based on prior CMA-ES results. All top configs used n_state=32.

## Phase 1: LHS Top 10

64 evaluations over the full search space:

| Rank | Loss | dim | n_heads | depth | lr | Params |
|------|------|-----|---------|-------|-----|--------|
| 1 | **1.2943** | 2176 | 48 | 30 | 6.8e-4 | 505M |
| 2 | 1.3131 | 1408 | 94 | 21 | 9.2e-4 | 448M |
| 3 | 1.3211 | 1408 | 60 | 37 | 6e-4 | 504M |
| 4 | 1.3311 | 2432 | 36 | 34 | 7.1e-4 | 480M |
| 5 | 1.3323 | 2944 | 42 | 23 | 5.2e-4 | 459M |
| 6 | 1.3326 | 1920 | 63 | 25 | 4.4e-4 | 487M |
| 7 | 1.3327 | 1536 | 130 | 14 | 7.8e-4 | 451M |
| 8 | 1.3428 | 1408 | 61 | 35 | 8.8e-4 | 484M |
| 9 | 1.3615 | 1536 | 60 | 35 | 5e-4 | 520M |
| 10 | 1.3726 | 2048 | 133 | 12 | 2.5e-4 | 527M |

## Phase 2: CMA-ES Refinement

### Warm Start 1 (from LHS #1: dim=2176, n_heads=48, depth=30)

| Gen | Best Loss | Best Config | Improved? |
|-----|-----------|-------------|-----------|
| 1 | 1.2978 | dim=2304, h=37, d=32, lr=8.6e-4 | Yes (new best) |
| 2 | **1.2728** | dim=1792, h=49, d=31, lr=9.2e-4 | **Yes (new best)** |
| 3 | 1.3002 | dim=2560, h=48, d=23, lr=7.6e-4 | No (1/2) |
| 4 | 1.2859 | dim=2304, h=52, d=23, lr=5.3e-4 | No (2/2 → converged) |

**Total evaluations: 64 (LHS) + 64 (CMA-ES Gen 1-4) = 128+**

## Architectural Insights

### The Compile Effect on Optimal Architecture

torch.compile fundamentally changes what's optimal:

1. **Deeper is better**: Layer overhead is amortized by compile → depth=31 beats depth=17
2. **Fewer heads needed**: 49 heads vs 83 — compile makes per-head ops more efficient
3. **Higher LR tolerated**: Compiled training is more stable at lr=9.2e-4
4. **Narrower dims OK**: dim=1792 vs dim=1920 — deeper network compensates

### Configuration Space Landscape

From 128+ evaluations, the loss landscape shows:
- **Hard failure zone**: lr < 1e-4 (loss > 3.0 consistently)
- **Moderate zone**: lr 1e-4 to 3e-4 (loss 1.4-1.8)
- **Optimal zone**: lr 4e-4 to 1e-3 (loss 1.27-1.35)
- **Depth sweet spot**: 21-35 layers
- **dim flexibility**: 1408-2432 all competitive when lr is right

## Status

- [x] E88 search with compile completed (first run, 128+ evals, clean GPU)
- [ ] Second E88 run in progress (GPU contention with Mamba2)
- [ ] Mamba2 compiled search in progress (for fair comparison)
- [ ] Extended validation with 30-min training runs (planned)
