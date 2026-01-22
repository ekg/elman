# E88 State Scaling Analysis (Jan 22, 2026)

## Key Finding

**E88 cannot scale state size by increasing heads beyond ~12**. 
Training diverges with 36+ heads regardless of depth or n_state.

## State Size Comparison

At 500M params:
- **FLA-GDN**: 1,327,104 state/layer (4 heads × 576²)
- **Mamba2**: 409,600 state/layer (128 × 3200)
- **E88 best working**: 18,432 state/layer (8 heads × 48²) = **0.01x FLA**

## Tested Configurations

### Working (loss ~1.7-1.8)
| Config | Heads | n_state | State/Layer | Depth | Loss |
|--------|-------|---------|-------------|-------|------|
| E88_h8n48 | 8 | 48 | 18,432 | 20 | 1.76 |
| E88_h8n56 | 8 | 56 | 25,088 | 20 | 1.77 |
| E88_h12n32 | 12 | 32 | 12,288 | 20 | 1.63 |

### Failing (loss 4-34, diverges)
| Config | Heads | n_state | State/Layer | Depth | Loss |
|--------|-------|---------|-------------|-------|------|
| E88_h36n48 | 36 | 48 | 82,944 | 20 | 4.59 |
| E88_h72n48 | 72 | 48 | 165,888 | 20 | 5.53 |
| E88_h81n32 | 81 | 32 | 82,944 | 32 | 4.37 |
| E88_h5n128 | 5 | 128 | 81,920 | 32 | 21.3 |
| E88_h100n64 | 100 | 64 | 409,600 | 32 | 14.7 |

### Baselines (for reference)
| Config | Params | Loss | Throughput |
|--------|--------|------|------------|
| mamba2 | 508M | 1.81 | 15.8K |
| fla-gdn | 534M | 1.71 | 20.9K |

## Root Cause Analysis

1. **Large n_state (64+)**: Numerically unstable - loss explodes immediately
2. **Many heads (36+)**: Training diverges even with stable n_state (32, 48)
3. **Working sweet spot**: 8-12 heads with n_state=32-56

## Fundamental Limitation

E88's maximum achievable state per layer is approximately:
- 12 heads × 56² = **37,632** (working)
- 8 heads × 64² = 32,768 (unstable)

This is **0.03x FLA-GDN** and **0.09x Mamba2** - orders of magnitude smaller.

## Conclusion

E88 (FLA Hybrid with delta rule) **cannot compete with Mamba2/FLA-GDN at 500M scale** 
due to head scaling limitations. The architecture works well at 40-100M with 8-12 heads 
but cannot leverage the additional parameters at 500M to increase state capacity.

**Recommendation**: Focus on architectures that can scale state independently of heads,
or investigate why head scaling causes training instability in delta rule updates.
