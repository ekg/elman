# E88 500M Scale Study

**Date**: 2024-01-23
**Training Duration**: 30 minutes per config (10 min for overnight exploration)

## Summary

This study systematically explored E88 FLA-GDN Hybrid configurations at the 500M parameter scale to understand:
1. Optimal architectural choices (depth, n_heads, n_state)
2. Performance relative to Mamba2 and FLA-GDN baselines

## Key Findings

### E88 Does Not Match Baselines at 500M Scale

After 30 minutes of training on the Pile dataset:

| Rank | Model | Last-100 Loss | Config |
|------|-------|---------------|--------|
| 1 | **Mamba2** | **1.2224** | dim=1792, depth=24 |
| 2 | **FLA-GDN** | **1.2809** | dim=1664, depth=24, exp=2.0 |
| 3 | E88 #2 | 1.4908 | dim=3328, depth=24, h=32, n=48, ratio=0.46 |
| 4 | E88 Winner | 1.5069 | dim=2944, depth=20, h=44, n=48, ratio=0.72 |
| 5 | E88 #3 | 1.5120 | dim=2944, depth=20, h=52, n=40, ratio=0.71 |

**Gap**: E88 is ~0.21-0.27 behind Mamba2/FLA-GDN.

### Overnight Exploration Results (42 configs, 10 min each)

The overnight exploration tested 42 E88 configurations across 5 waves:
- Wave 1: Depth exploration (20-32 layers)
- Wave 2: Many heads, small state (h=32-200, n=16-24)
- Wave 3: Deep models (36-40 layers)
- Wave 4: Balance ratio exploration (0.5-2.0)
- Wave 5: New n_state values (36, 40, 44)

**Best by n_state** (10 min training):
| n_state | Best Loss |
|---------|-----------|
| 48 | 2.302 |
| 40 | 2.379 |
| 32 | 2.390 |
| 24 | 2.821 |
| 16 | 3.217 |

**Best by depth** (10 min training):
| Depth | Best Loss |
|-------|-----------|
| 20 | 2.302 |
| 24 | 2.357 |
| 28 | 2.604 |
| 32 | 2.611 |
| 36 | 2.941 |
| 40 | 3.108 |

### Architectural Insights

1. **Larger n_state is better**: n_state=48 consistently outperforms smaller values
2. **Shallower is better at 500M**: depth=20-24 beats deeper models
3. **Low balance ratios work**: ratio ~0.4-0.7 is optimal
4. **Training time matters**: 30 min shows much better loss than 10 min (1.49 vs 2.30)

## Best E88 Configuration at 500M

```
E88 d24_h32_n48 (ratio=0.46)
  dim: 3328
  depth: 24
  n_heads: 32
  n_state: 48
  expansion: 1.0
  use_gate: False
  params: ~500M
  Last-100 loss (30 min): 1.4908
```

## Why the Gap?

Several hypotheses for why E88 underperforms at 500M:

1. **Architecture limitations**: The nonlinear tanh state update may not scale as well as linear attention (FLA-GDN) or selective SSMs (Mamba2)

2. **Hyperparameter sensitivity**: E88 may require different learning rates, batch sizes, or other hyperparameters at this scale

3. **State matrix overhead**: E88's H × n² state per layer may be inefficient compared to Mamba2's linear state

4. **Training dynamics**: The "best" config from 10-min exploration may not be optimal for longer training

## Comparison with 100M Results

At 100M scale (from E88_EXPANSION_FINDINGS.md), E88 achieved:
- E88 (best): 1.40 loss
- FLA-GDN: 1.39 loss
- Gap: ~0.01

At 500M scale (this study):
- E88 (best): 1.49 loss
- FLA-GDN: 1.28 loss
- Gap: ~0.21

The gap widens significantly at larger scale, suggesting E88 has scaling limitations.

## Recommendations

1. **For production use at 500M**: Use Mamba2 or FLA-GDN instead of E88
2. **For further E88 research**:
   - Investigate why the architecture doesn't scale
   - Try longer training runs to see if the gap closes
   - Explore larger n_state values (64, 72) if CUDA kernel supports them
   - Consider hybrid architectures mixing E88 with attention

## Files

- Overnight exploration: `benchmark_results/overnight_e88/20260123_051430/`
- 30-min comparison: `benchmark_results/e88_500m_30min/20260123_123714/`
- Exploration script: `overnight_e88_exploration.py`
