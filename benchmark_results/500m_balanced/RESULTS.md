# 500M Balanced E88 Benchmark Results

**Date**: January 22, 2026
**Training Time**: 10 minutes per model
**Dataset**: data/pile.txt (byte-level)
**Batch Size**: 32, Chunk Size: 512

## Key Finding

**E88 with balanced configurations beats Mamba2!**

The n_state=32 variants (E88_b56n32, E88_b60n32) achieve loss ~1.55 vs Mamba2's 1.80, while maintaining reasonable throughput (~10K tok/s vs Mamba2's 15K tok/s).

## Results Table

| Model | Loss | Steps | Tok/s | Params | State/Layer | Ratio |
|-------|------|-------|-------|--------|-------------|-------|
| **fla-gdn** | **1.29** | 793 | 21,137 | ~500M | 1,327,104 | - |
| **E88_b56n32** | **1.55** | 421 | 10,766 | 504M | 57,344 | 0.82 |
| E88_b60n32 | 1.55 | 423 | 10,887 | 508M | 61,440 | 0.94 |
| E88_b64n32 | 1.60 | 390 | 10,140 | 508M | 65,536 | 1.07 |
| E88_b44n48 | 1.70 | 331 | 8,476 | 488M | 101,376 | 1.18 |
| E88_b28n64 | 1.70 | 279 | 7,259 | 502M | 114,688 | 0.82 |
| E88_b40n48 | 1.71 | 356 | 9,120 | 507M | 92,160 | 0.94 |
| E88_b32n64 | 1.72 | 245 | 6,538 | 506M | 131,072 | 1.07 |
| mamba2 | 1.80 | 596 | 15,557 | 508M | 409,600 | - |

## Analysis

### Quality vs State Size Trade-off

1. **n_state=32 is optimal for E88**: Best loss (1.55) with highest throughput (~10K tok/s)
2. **n_state=48**: Intermediate quality (1.70), moderate throughput (~8-9K tok/s)
3. **n_state=64**: Slightly worse quality (1.72), lower throughput (~6-7K tok/s)

### E88 vs Baselines

- **E88 beats Mamba2** by 0.25 nats (1.55 vs 1.80) - significant improvement!
- **FLA-GDN still leads** at 1.29, but has 23× more state per layer
- E88_b56n32 achieves 1.55 loss with only 57K state/layer (0.14× of Mamba2's 410K)

### Balanced Configuration Benefits

The balanced configs (ratio ~1.0) achieve:
- Stable training (no numerical issues)
- Good throughput (10K+ tok/s for n_state=32)
- Competitive quality

This validates the design principle: **n_heads × n_state ≈ dim**

## Configurations Used

| Config | dim | n_heads | n_state | Projection Ratio |
|--------|-----|---------|---------|------------------|
| E88_b56n32 | 2176 | 56 | 32 | 0.82 |
| E88_b60n32 | 2048 | 60 | 32 | 0.94 |
| E88_b64n32 | 1920 | 64 | 32 | 1.07 |
| E88_b40n48 | 2048 | 40 | 48 | 0.94 |
| E88_b44n48 | 1792 | 44 | 48 | 1.18 |
| E88_b28n64 | 2176 | 28 | 64 | 0.82 |
| E88_b32n64 | 1920 | 32 | 64 | 1.07 |

## Recommendations

1. **For best quality**: Use E88_b56n32 or E88_b60n32 (n_state=32 variants)
2. **For more state**: Use E88_b32n64 if you need larger recurrent state
3. **Avoid**: Unbalanced configs with ratio > 2.0 (causes slowdowns)

See `E88_BALANCED_CONFIG_GUIDE.md` for the full configuration methodology.
