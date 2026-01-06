# Elman RNN Research

**Can simple Elman RNNs compete with modern SSMs like Mamba2?**

Yes. With proper CUDA kernels and hyperparameter tuning, a 6-layer Elman RNN beats Mamba2 in wall-clock training time.

## Key Results (50M params, 10 minutes training)

| Model | Loss | Throughput | Notes |
|-------|------|------------|-------|
| **E1 d1280×6** | **1.43** | **254K tok/s** | Simple gated Elman |
| E1 d1024×10 | 1.45 | 214K tok/s | Deeper but slower |
| Mamba2 | 1.53 | 101K tok/s | SSM baseline |
| E5 d1536 r270 | 1.61 | 81K tok/s | Low-rank Elman |

**Finding**: E1's 3× throughput advantage over Mamba2 dominates. Simpler is faster.

## Model Variants

### E0: Stock Elman
```
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
```

### E1: Gated Elman (best)
```
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
y_t = h_t * silu(W_g @ [h_t, x_t] + b_g)  # h+x selective gating
```

### E5: Low-Rank Elman
```
h_t = tanh(U_h @ V_h @ h_{t-1} + U_x @ V_x @ x_t + b)
```
Uses rank-r factorization. Fewer params per layer but slower due to extra matmuls.

## Architecture Insights

1. **Wider + shallower wins** - d1280×6 beats d1024×10 beats d768×18
2. **But not too shallow** - depth=6 is optimal; depth=2-3 hurts loss significantly
3. **Low-rank hurts throughput** - E5's factorization adds latency that outweighs param savings
4. **CUDA GEMMs are key** - All ops must be batched matmuls, not element-wise

## Quick Start

```bash
# Activate environment
micromamba activate mingru

# Build CUDA kernels
cd elman/cuda && make && pip install -e . && cd ../..

# Train E1 (best model)
python train_ladder.py --level 1 --params 50m --data data/pile.txt
```

## Data Format

Raw text file, byte-level (vocab_size=256). No pretokenization - we mmap and sample randomly:

```python
with open('data/pile.txt', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
pos = np.random.randint(0, len(mm) - 513, size=batch_size)
```

## Repository Structure

```
elman/
├── elman/
│   ├── models/       # E0, E1, E5, Mamba2 wrapper
│   ├── cuda/         # CUDA kernels (haste-based)
│   └── data/         # Mmap data loading
├── benchmark_results/ # All experiment logs
├── train_ladder.py   # Main training script
└── CLAUDE.md         # Development guidelines
```

## Current Research

Exploring sparse/structured matrices for larger hidden states without O(n²) cost:
- Monarch matrices (O(n√n))
- Sparse + low-rank decomposition
- Learned random projections

## Citation

This is research code exploring the question: how far can simple RNNs go with modern engineering?
