# E25: Dual-Memory with 1.5-Entmax

E25 = E23 with softmax replaced by 1.5-entmax for sparse attention.

## Performance

- E25: 430K tok/s (D=512, N=32)
- E23c: 2310K tok/s (same config)
- Ratio: 0.19x (5x slower due to O(N²) sorting in entmax)

## Optimization opportunities

- Parallel sorting (odd-even, bitonic)
- Warp shuffles for N≤32
