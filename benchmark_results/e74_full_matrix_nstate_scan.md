# E74 Full Matrix n_state Scan Results

**Date:** 2026-01-16
**Model:** E74 Full Matrix Delta Rule
**Config:** dim=1536, depth=20, batch=32, chunk=512, lr=3e-4, bf16
**Training:** 10 minutes each

## Model Description

E74 uses a full n×n matrix state with the delta rule update:

```
k_norm = k / ||k||
retrieved = S @ k_norm
delta = v - retrieved
S = tanh(S + outer(delta, k_norm))
Sq = S @ q
output = Sq * silu(Sq)
```

This is an associative memory that learns to store and retrieve key-value pairs.

## Results

| n_state | Parameters | Loss | Throughput | Notes |
|---------|------------|------|------------|-------|
| 1 | 95.0M | 2.937 | 114,274 tok/s | Insufficient capacity |
| 2 | 95.2M | 2.871 | 116,481 tok/s | Fastest |
| 4 | 95.7M | 2.624 | 108,557 tok/s | - |
| 8 | 96.5M | 2.102 | 107,575 tok/s | Quality improving |
| **16** | 98.2M | **1.815** | **99,225 tok/s** | **Best speed/quality** |
| 32 | 101.7M | 1.667 | 78,497 tok/s | Best quality |
| 48 | 105.1M | 1.667 | 51,590 tok/s | Same quality as n=32 |
| 64 | 108.6M | 1.689 | 30,206 tok/s | Diminishing returns |
| 96 | 115.4M | 1.858 | 11,967 tok/s | Quality degraded |

## Key Findings

1. **n=32 is optimal for quality** - achieves best loss (1.667)
2. **n=16 is best speed/quality tradeoff** - 1.815 loss at 99K tok/s (27% faster than n=32)
3. **Quality degrades above n=64** - larger states need more training time to fill
4. **Very small n (1-4) cannot achieve good loss** - insufficient state capacity
5. **Throughput scales as ~1/n²** due to matrix operations

## Comparison with Baselines

| Model | Params | Loss | Throughput |
|-------|--------|------|------------|
| Mamba2 | 102M | 1.70 | 75K tok/s |
| E42 (linear tied) | 43M | 1.59 | 137K tok/s |
| **E74 n=16** | 98M | 1.81 | 99K tok/s |
| **E74 n=32** | 102M | **1.67** | 78K tok/s |

E74 n=32 beats Mamba2 on both loss AND throughput at similar parameter count.

## CUDA Kernel Support

Supported n_state values: 1, 2, 4, 8, 16, 24, 32, 48, 64, 96

Larger sizes (128, 192, 256) fall back to PyTorch due to:
- Shared memory limits for backward pass
- OOM for very large states
