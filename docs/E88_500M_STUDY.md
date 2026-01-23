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

## Linear vs Tanh: NOT the Problem

We tested whether the tanh nonlinearity was causing the scaling gap:

| Variant | Last-10 Loss (10 min) |
|---------|----------------------|
| E88 TANH | 1.686 |
| E88 LINEAR | 1.684 |

**Finding: Tanh is NOT the scaling problem.** Both achieve identical loss.

## SiLU Gating: Partial Improvement

FLA-GDN uses SiLU (swish) activation for its output gating, while E88 originally used sigmoid.

Gating comparison (30 min, 500M):
| Gating | Loss | Notes |
|--------|------|-------|
| SiLU gate | 1.4294 | FLA-GDN style: `output * silu(g)` |
| No gate | 1.4856 | Best E88 config from overnight |
| Sigmoid gate | worse | E88 original: `output * sigmoid(g)` |

**Finding: SiLU gating helps!** E88 with SiLU gating is 0.06 better than no gating. The difference is that FLA-GDN's gating is:
- `output = RMSNorm(x) * g * sigmoid(g)` = `output * silu(g)`

While E88's original was just:
- `output = RMSNorm(x) * sigmoid(g)`

The extra multiplication by `g` allows negative gates and better gradient flow.

**Updated comparison (30 min):**
| Rank | Model | Loss |
|------|-------|------|
| 1 | Mamba2 | 1.2752 |
| 2 | FLA-GDN | 1.3187 |
| 3 | E88 SiLU gate | 1.4294 |
| 4 | E88 no gate | 1.4856 |

**Gap after SiLU fix**: E88 is ~0.11 behind FLA-GDN (improved from ~0.21)

## Why the Remaining Gap?

With SiLU gating, E88 is closer to baselines but still ~0.11-0.15 behind. Remaining hypotheses:

1. **Projection bottleneck**: E88 projects dim→key_dim→dim, which may constrain capacity. FLA-GDN expands (dim→2*dim→dim) while E88 contracts.

2. **State matrix overhead**: E88's H × n² state per layer may be less parameter-efficient than FLA-GDN's d_state × 2d_inner or Mamba2's linear state.

3. **Gradient flow through tanh**: The tanh nonlinearity may still create gradient issues at scale, even if it doesn't directly hurt loss.

4. **Hyperparameter sensitivity**: E88 may require different learning rates or batch sizes at 500M scale.

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

1. **For E88 at 500M**: Use SiLU gating (`--use_gate 1 --gate_activation silu`)
2. **For production use at 500M**: Mamba2 or FLA-GDN still outperform E88 by ~0.11-0.15
3. **For further E88 research**:
   - Investigate the projection bottleneck (E88 contracts, FLA-GDN expands)
   - Try removing the projection entirely (direct state output)
   - Explore different normalization strategies
   - Test longer training runs to see if gap closes

## Files

- Overnight exploration: `benchmark_results/overnight_e88/20260123_051430/`
- 30-min comparison: `benchmark_results/e88_500m_30min/20260123_123714/`
- SiLU gating comparison: `benchmark_results/silu_baseline_30m/`
- Gate activation tests: `benchmark_results/gate_test/` and `benchmark_results/gate_test_10m/`
- Exploration script: `overnight_e88_exploration.py`
