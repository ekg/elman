# E4 (Low-Rank Elman) Experiments Summary

## Overview
E4 uses low-rank U @ V factorization instead of full W_h matrix, enabling larger hidden states with similar parameter count.

## Configuration
- Dataset: pile.txt (byte tokenization, vocab_size=256)
- Training: 2000 steps, batch_size=32, chunk_size=512, lr=3e-4
- Target params: ~50M

## Results

| Model | Hidden Expansion | Backend | Loss@2k | PPL | Tok/s | Params | Notes |
|-------|-----------------|---------|---------|-----|-------|--------|-------|
| E1 | 1x (baseline) | CUDA | 1.58 | 4.8 | 43k | 49.7M | Reference |
| E4 | 1x | CUDA | 2.14 | 8.5 | 16k | 50.6M | rank=64 too small |
| E4 | 2x | CUDA | 2.03 | 7.6 | 26k | 49.2M | Auto rank=128 |
| E4 | 2x (SN) | CUDA | 2.22 | 9.2 | 25k | 49.2M | SN + CUDA backward bug |
| **E4** | **2x (SN)** | **PyTorch** | **1.65** | **5.2** | 3.5k | 49.2M | **Competitive with E1!** |
| E4 | 4x (SN) | PyTorch | 1.67 | 5.3 | 6k | 53.4M | Similar to 2x |

## Key Findings

1. **E4 PyTorch with spectral norm achieves competitive results**
   - E4_2x: Loss 1.65, PPL 5.2 (vs E1: Loss 1.58, PPL 4.8)
   - Gap: +0.07 loss, +0.4 PPL

2. **CUDA backward kernel has bugs**
   - CUDA E4 produces significantly worse results (Loss 2.0-2.2 vs 1.65)
   - Forward pass works correctly
   - Backward gradients are incorrect

3. **2x hidden expansion is optimal**
   - E4_2x and E4_4x perform similarly (1.65 vs 1.67)
   - More hidden state doesn't help if rank is reduced proportionally

4. **Speed tradeoff**
   - PyTorch E4: 3.5k tok/s (7x slower than E1)
   - CUDA E4 (buggy): 26k tok/s (1.7x slower than E1)
   - Need to fix CUDA kernel for practical use

## Model Architecture Comparison

### E1 (MambaGatedElman)
- d_inner = 512
- W_h: 512 x 512 = 262k params
- 21 layers for 50M model

### E4_2x (LowRankElman)
- d_inner = 1024 (2x E1)
- U: 1024 x 128 = 131k params
- V: 128 x 1024 = 131k params
- U+V = 262k (same as E1 W_h)
- 17 layers for 50M model

## CUDA Backward Bug Analysis

Gradient comparison (small test case, T=16, B=2, D=64):
```
dx diff: max=0.015625  (OK)
dz diff: max=0.015625  (OK)
dW_x diff: max=17.0    (BUG)
dU diff: max=6.5       (BUG)
dV diff: max=5.7       (BUG)
db diff: max=0.047     (minor)
```

The gradient norms match but individual values differ, suggesting matrix
multiplication transpose issues in the cuBLAS calls. Key areas to fix:
- `dW_x += dv.T @ x` computation
- `dU += dv.T @ Vh` computation
- `dV += dVh.T @ h_prev` computation

## Next Steps

1. **Fix CUDA backward kernel** - Debug matmul transposes in dW_x, dU, dV
2. **Test with subword tokenization** - Compare on proper language modeling task
3. **Scale experiments** - Test at 100M+ params to see if E4 advantage grows

## Files
- E1 results: `output/comparison_50m/e1/`
- E4 base: `output/comparison_50m/e4/`
- E4 2x CUDA: `output/comparison_50m/e4_2x/`
- E4 2x+SN CUDA: `output/comparison_50m/e4_2x_sn/`
- E4 2x PyTorch: `output/comparison_50m/e4_2x_pytorch/`
- E4 4x PyTorch: `output/comparison_50m/e4_4x/`
