# E88 Dimension Sweep Results

**Date**: January 22, 2026
**Goal**: Find optimal width/depth trade-off at ~500M params

## Results

| Config | dim | depth | n_heads | Loss | Tok/s | Params |
|--------|-----|-------|---------|------|-------|--------|
| **dim1792** | 1792 | 38 | 56 | **1.44** | 10,765 | ~493M |
| dim2560 | 2560 | 18 | 80 | 1.47 | 12,341 | ~477M |
| dim2304 | 2304 | 22 | 72 | 1.47 | 12,171 | ~472M |
| dim2816 | 2816 | 16 | 88 | 1.48 | 12,699 | ~513M |
| dim2048 | 2048 | 28 | 64 | 1.55 | 10,947 | ~474M |
| dim3072 | 3072 | 14 | 96 | 1.55 | 12,570 | ~470M |
| dim1536 | 1536 | 50 | 48 | 1.70 | 9,252 | ~476M |

## Key Findings

### 1. Optimal Width: dim=1792-2816
- **Best**: dim=1792, depth=38 achieves 1.44 loss
- Sweet spot range: 1792-2816 (all achieve 1.44-1.48)
- Not "wider is always better"

### 2. Too Narrow Hurts
- dim=1536 (depth=50) achieves only 1.70 loss
- Insufficient per-layer capacity despite more layers

### 3. Too Wide Has Diminishing Returns
- dim=3072 (depth=14) achieves 1.55 loss
- Wider throughput (12.5K tok/s) but worse quality
- Not enough layers for good representation?

### 4. Depth vs Width Trade-off
- At ~500M params, optimal is ~38 layers with dim=1792
- This is deeper than Mamba2 (32 layers) at same scale
- E88 seems to benefit from more layers

## Comparison to Previous Best

| Config | Loss | Improvement |
|--------|------|-------------|
| E88_dim1792 (new) | 1.44 | baseline |
| E88_b56n32 (old best) | 1.55 | +0.11 worse |
| Mamba2 | 1.80 | +0.36 worse |

**The dim=1792, depth=38 config improves E88 by 0.11 nats over our previous best!**

## Recommended Config for 500M

```python
level = 'E88_dim1792'  # or E88_b56n32 renamed
dim = 1792
depth = 38
n_heads = 56
n_state = 32
# ratio = 56 * 32 / 1792 = 1.0 (balanced)
```
