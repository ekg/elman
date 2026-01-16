# 100M Parameter 10-Minute Benchmark Results

**Date:** 2026-01-16
**Config:** ~100M params, 10 minutes training, batch=32, chunk=512, lr=3e-4, bf16
**Data:** Pile dataset (byte-level, vocab=256)

## Main Results

### Best Performing Models

| Model | Params | Loss | Throughput | Description |
|-------|--------|------|------------|-------------|
| **Mamba2** | 102M | **1.269** | 78,657 tok/s | State-of-art SSM baseline |
| **E61** | 98M | **1.351** | 87,979 tok/s | Decay-gated Elman |
| **E67** | 98M | 1.362 | 83,644 tok/s | H-gated Elman |
| **E68** | 98M | 1.366 | 88,117 tok/s | Self-gating Elman |
| **E62** | 98M | 1.372 | 80,675 tok/s | Selective write Elman |
| E65 | 98M | 1.409 | 90,517 tok/s | Diagonal H Elman |
| E63 | 84M | 1.440 | 53,195 tok/s | Nonlinear delta |
| E56 | 115M | 1.450 | 40,942 tok/s | Concat Elman |
| E1 | 115M | 1.466 | 45,481 tok/s | Classic Elman + gating |
| E42 | 95M | 1.485 | 45,610 tok/s | Linear tied self-gated |
| E64 | 98M | 1.492 | 87,015 tok/s | Additive H |
| E66 | 115M | 1.568 | 23,354 tok/s | Lowrank H |
| Llama | 131M | 1.886 | 71,301 tok/s | Transformer baseline |

### E74 Full Matrix (Delta Rule Associative Memory)

| Variant | Params | Loss | Throughput | Notes |
|---------|--------|------|------------|-------|
| E74 n=16 | 98M | 1.815 | 99,225 tok/s | Best speed/quality |
| E74 n=32 | 102M | 1.667 | 78,497 tok/s | Best quality |
| E74 n=48 | 105M | 1.667 | 51,590 tok/s | Same as n=32 |
| E74 n=64 | 109M | 1.689 | 30,206 tok/s | Diminishing returns |

### E74 Diagonal (Simple Delta Rule)

| Variant | Params | Loss | Throughput | Notes |
|---------|--------|------|------------|-------|
| E74-d-tiedkvq-t | 104M | 2.470 | 366K tok/s | Fastest, tanh |
| E74-d-noz-t | 99M | 2.573 | 388K tok/s | No z projection |
| E74-d-full-t | 104M | 2.584 | 374K tok/s | Full projections |
| E74-s-tiedkvq-t | 104M | 2.495 | 379K tok/s | Simple update |

Note: E74 diagonal with linear nonlinearity (no tanh) produces nan loss.

### Failed/Unstable Models

| Model | Issue |
|-------|-------|
| E70 | Very slow (3.5K tok/s), poor loss (3.27) |
| E71 | NaN loss |
| E72 | NaN loss |
| E73 | Slow (15K tok/s), high loss (2.22) |
| E63m | NaN loss |
| FLA-GDN | Too large (284M), poor loss (2.07) |

## Key Findings

1. **Best overall: Mamba2** - 1.269 loss at 78K tok/s is hard to beat
2. **Best Elman: E61** - 1.351 loss at 88K tok/s, competitive with Mamba2
3. **E61-E68 family** - All achieve ~1.35-1.41 loss, very consistent
4. **E74 full matrix** - Interesting but needs more work to match E61
5. **E74 diagonal** - Very fast (366K+ tok/s) but worse loss (~2.5)
6. **Transformers (Llama)** - 1.886 loss, much worse than RNNs at this scale

## Architecture Insights

### What Works
- Decay gating (E61)
- Self-gating: `out = h * silu(h)` (E68)
- Selective write mechanisms (E62)
- Simple diagonal state with proper gating

### What Doesn't Work
- Matrix state without checkpointing (OOM)
- Linear recurrence without tanh (unstable)
- Lowrank H projections (E66 - too slow)
- Complex delta rules without proper initialization

## Throughput vs Quality Tradeoff

```
Loss    Model           Throughput
1.27    Mamba2          78K tok/s   ████████
1.35    E61             88K tok/s   █████████
1.37    E62             81K tok/s   ████████
1.37    E68             88K tok/s   █████████
1.41    E65             91K tok/s   █████████
1.67    E74-n32         78K tok/s   ████████
1.81    E74-n16         99K tok/s   ██████████
2.47    E74-diag        366K tok/s  ████████████████████████████████████
```

## Recommendations

1. **For quality**: Use Mamba2 or E61
2. **For speed**: Use E74 diagonal (2.5x faster than Mamba2)
3. **For balance**: E74 n=16 offers good speed/quality tradeoff
4. **For research**: E61-E68 family is promising for further optimization
