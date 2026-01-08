# Step-Matched 400M Comparison

**Date**: 2026-01-08
**Setup**: 1000 steps × 32 batch × 512 tokens = 16.4M tokens per model

All models ~400M parameters, same data, same optimizer (AdamW lr=1e-4), bf16.

## Results

| Model | Params | Loss | Time | Tok/s |
|-------|--------|------|------|-------|
| Mamba2 | 402M | **1.4189** | 11.6 min | 23.6K |
| E1 (exp2.5) | 399M | 1.5084 | 9.5 min | 28.9K |
| minLSTM | 399M | 1.6190 | 13.0 min | 21.0K |
| E16 s4 | 400M | 1.6249 | 12.0 min | 22.8K |
| minGRU | 398M | 1.6291 | 13.2 min | 20.7K |

## Key Findings

1. **Mamba2 learns better per token** - 0.09 nats better than E1 on identical data
2. **E1 is fastest** (28.9K tok/s) but worse sample efficiency
3. **Linear vs nonlinear is not the key factor**:
   - minLSTM (linear) does worse than E1 (nonlinear)
   - Mamba2 (linear) does best
4. **Diagonal state expansion hurts** - E16 trails E1 despite similar structure

## Model Details

- **E1**: h = tanh(W_h @ h + W_x @ x) with selective gating, dense W_h
- **E16**: h = tanh(A ⊙ h + B @ x), diagonal A, state_expansion=4
- **Mamba2**: h = A @ h + B @ x (linear), input-dependent A/B/C via selectivity
- **minGRU**: h = (1-z) ⊙ h + z ⊙ x̃, linear in h
- **minLSTM**: h = f ⊙ h + i ⊙ x̃, linear in h

## Conclusions

The advantage of Mamba2 is NOT simply "linear recurrence enables parallel scan".
minLSTM/minGRU are also linear but perform poorly.

Mamba2's advantage likely comes from:
1. Selective mechanism (input-dependent A, B, C projections)
2. State structure and initialization (S4D-style)
3. Discretization approach (delta rule)
4. Larger effective state dimension
