# E88 Matrix State Model Sizing Guide

## Architecture Overview

E88 FLA Hybrid uses expansion=1.0 (non-expansive), meaning:
- `d_inner = n_heads × n_state`
- `head_v_dim = n_state` (square state per head)
- State per layer: `n_heads × n_state²`

## Key Constraint: Avoid Bottlenecks

The model projects: `dim → d_inner → dim`

For balanced architecture: **dim ≈ d_inner** (ratio 0.8 - 1.2)

| dim/d_inner | Status |
|-------------|--------|
| > 2.0 | Severe bottleneck (wasted capacity) |
| 1.2 - 2.0 | Mild bottleneck |
| 0.8 - 1.2 | **Balanced (optimal)** |
| 0.5 - 0.8 | Over-provisioned d_inner |
| < 0.5 | Wasteful (d_inner too large) |

## Parameter Count Formula

```python
def calc_e88_params(dim, n_heads, n_state, depth, vocab_size=256):
    d_inner = n_heads * n_state

    # Components:
    # - Embedding: vocab_size × dim (tied with output)
    # - Per layer: dim × 4 × d_inner (W_kvqb) + d_inner × dim (W_o) + 2×dim (norm)
    # - Final norm: 2 × dim

    embed = vocab_size * dim
    per_layer = dim * 4 * d_inner + d_inner * dim + 2 * dim
    final_norm = 2 * dim

    return embed + depth * per_layer + final_norm
```

## Recommended Configs by Scale

### 100M Parameters (depth=20)

| Config | dim | n_heads | n_state | d_inner | Params | State/Layer |
|--------|-----|---------|---------|---------|--------|-------------|
| h32n32 | 1280 | 32 | 32 | 1024 | 105M | 32K |
| h40n32 | 1152 | 40 | 32 | 1280 | 107M | 40K |
| h24n48 | 1280 | 24 | 48 | 1152 | 106M | 55K |

### 500M Parameters (depth=20)

| Config | dim | n_heads | n_state | d_inner | dim/d | Params | State/Layer |
|--------|-----|---------|---------|---------|-------|--------|-------------|
| h64n32 | 2432 | 64 | 32 | 2048 | 1.19 | 499M | 65K |
| h68n32 | 2304 | 68 | 32 | 2176 | 1.06 | 502M | 70K |
| h72n32 | 2176 | 72 | 32 | 2304 | 0.94 | 502M | 74K |
| h76n32 | 2048 | 76 | 32 | 2432 | 0.84 | 499M | 78K |
| h48n48 | 2176 | 48 | 48 | 2304 | 0.94 | 502M | 110K |
| h40n64 | 1920 | 40 | 64 | 2560 | 0.75 | 492M | 164K |

### 1B Parameters (depth=24)

| Config | dim | n_heads | n_state | d_inner | dim/d | Params | State/Layer |
|--------|-----|---------|---------|---------|-------|--------|-------------|
| h96n32 | 2816 | 96 | 32 | 3072 | 0.92 | 1.01B | 98K |
| h80n48 | 2688 | 80 | 48 | 3840 | 0.70 | 1.02B | 184K |
| h64n64 | 2560 | 64 | 64 | 4096 | 0.63 | 1.03B | 262K |

## Comparison with Baselines

| Model | State/Layer | Notes |
|-------|-------------|-------|
| Mamba2 (500M) | 409K | d_state=128, expand=2 |
| FLA-GDN (500M) | 393K | head_dim=128 |
| E88 h72n32 (500M) | 74K | **5-6x smaller state** |
| E88 h40n64 (500M) | 164K | **2.5x smaller state** |

## n_state Selection by Hardware

Based on profiling (see E88_HEAD_SCALING_ANALYSIS.md):

| GPU | Optimal n_state | Reason |
|-----|-----------------|--------|
| RTX 4090 | 32 | Fits L1 cache, 13-15x faster than n_state=72 |
| A100 | 48-64 | More shared memory, better tensor core utilization |
| H100 | 64-96 | Highest bandwidth, largest shared memory |

## Quick Sizing Procedure

1. **Choose n_state based on hardware** (32 for consumer, 64 for datacenter)
2. **Calculate d_inner** = target_dim (for balanced architecture)
3. **Derive n_heads** = d_inner / n_state
4. **Verify params** using formula above
5. **Adjust dim** to hit exact param target

Example for 500M on RTX 4090:
```
n_state = 32 (optimal for RTX 4090)
target d_inner ≈ 2048-2432 (reasonable for 500M)
n_heads = 2048 / 32 = 64 to 2432 / 32 = 76
→ Use h64n32 dim=2432 or h72n32 dim=2176
```

## Anti-patterns to Avoid

1. **Small n_heads with large dim**: h20n32 dim=8192 → dim/d_inner = 12.8 (severe bottleneck)
2. **Large n_state on consumer GPU**: n_state=96 → 13-15x slower backward pass
3. **Mismatched dim and d_inner**: Creates projection bottleneck, wastes parameters
