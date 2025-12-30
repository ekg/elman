# Log-Space Gradient Solution for Deep GRU

## Problem

Gradient vanishing in deep GRU sequences (T > 64 steps).

With δ = 0.12 (typical value):
- **Linear backward**: gradients decay as (1-δ)^T = 0.88^64 ≈ 0.0003
- This means gradients at t=0 are ~3000x smaller than at t=T

## Root Cause Analysis

Log-space storage helps ONLY if the **entire gradient path** stays in log space.

Current implementation breaks this:

```
Forward:
  h = exp(log_h)           <- convert to linear
  output = compete(h) * silu(W_out @ h)

Backward:
  d_h = ∂loss/∂output * ∂output/∂h
  d_log_h = d_h * h         <- MULTIPLIES BY TINY h!
```

The `* h` at the output boundary reintroduces the small magnitude,
canceling all log-space benefits.

## Solution: Log-Space Output

Compute output from `log_h` directly, NOT from `h = exp(log_h)`:

```
Forward:
  output = compete(log_h) * silu(W_out @ log_h)

Backward:
  d_log_h = ∂loss/∂output * ∂output/∂log_h   <- NO h factor!
```

## Experimental Results

| Approach | d_h_0 or d_log_h_0 | Improvement |
|----------|-------------------|-------------|
| Linear backward | 1.4e-06 | baseline |
| Log-space output | 4.3e-02 | **30,000x** |

## Semantic Change

- **Current**: `compete = softmax(h)` where h can be 1e-10
  - Competition based on tiny magnitudes
  - Numerical instability when h underflows

- **New**: `compete = softmax(log_h)` where log_h ≈ -8
  - Competition based on log-magnitudes
  - Numerically stable (log values stay in reasonable range)

## Implementation Changes Required

1. **Output projection**: `W_out @ log_h` instead of `W_out @ h`
   - Compute matmul before calling output kernel

2. **Compete mechanism**: `softmax(log_h)` instead of `softmax(exp(log_h))`
   - Same softmax code, different input

3. **Backward kernel**: Remove `* h` multiplication
   - `d_log_h = softmax_jacobian @ d_compete` (direct log-space gradient)

## Additional Finding: w1 Softmax Weights

When state decays (candidate is small):
- w1 = (1-δ) * |h_prev| / |h_new| → 1
- Gradients DON'T decay in log-space recurrence!

| Candidate | Prod(w1) | vs Linear |
|-----------|----------|-----------|
| 0.00 | 1.0 | 3574x better |
| 0.01 | 0.027 | 97x better |
| 0.10 | 0.003 | 10x better |
| 1.00 | 0.0003 | same |

For long-range memory (small updates), log-space is much better.

## Implementation Status

**IMPLEMENTED** in `elman/cuda/lib/log_storage_diagonal_gpu.cu.cc`:
- `LogStorageSelectiveOutput`: Now computes softmax(log_h) instead of softmax(exp(log_h))
- `LogStorageSelectiveOutputBackward`: Gradient flows directly to log_h (no * h factor)
- `Forward::Run`: Computes W_out @ log_h instead of W_out @ h
- `Backward::Run`: Uses log_h directly throughout

## Limitations Discovered

**The log-space output change helps with the OUTPUT BOUNDARY but does NOT fully solve gradient vanishing.**

### Why Gradients Still Decay

In the recurrence, gradient flows as:
```
d_log_h_{t-1} = d_log_h_t * w1
```

Where w1 = (1-δ) * |h_{t-1}| / |h_t|

When candidate ≠ 0:
- w1 < 1 (state is being updated, not just decaying)
- Prod(w1) over T steps still decays exponentially

Test results with w1 mean = 0.83:
| T | Prod(w1) | Gradient ratio |
|---|----------|----------------|
| 8 | 0.22 | 0.90 |
| 16 | 0.05 | 0.58 |
| 32 | 0.0026 | 0.05 |
| 64 | 6.7e-6 | 8.6e-7 |
| 256 | 1.6e-21 | 3.3e-37 |

### When Log-Space Actually Helps

1. **Pure memory mode** (candidate ≈ 0): w1 → 1, gradients preserved
2. **Numerical stability**: Represent tiny gradients without underflow
3. **Output boundary**: No * h factor at output layer

### Fundamental Insight

For nonlinear recurrences (GRU with tanh(r_h * h_prev)):
- **Cannot** use parallel scan (linear recurrence required)
- **Cannot** avoid exponential gradient decay (multiplicative chain rule)
- Log-space changes REPRESENTATION, not MATHEMATICS

### What Mamba2 Actually Does Differently

Mamba2 uses:
1. **Linear** state space model: h_t = A * h_{t-1} + B * x_t
2. **Parallel scan** for both forward and backward (O(log T) depth)
3. Log-space for **numerical stability** during scan

The key is LINEAR structure, not log-space.

### Alternatives for Nonlinear RNNs

1. **TBPTT**: Truncate gradients at chunk boundaries (prevents vanishing by limiting depth)
2. **Careful initialization**: δ close to 0 → (1-δ) close to 1 → slower decay
3. **Gated architectures**: LSTM/GRU gates can learn to preserve gradients
4. **Skip connections**: Direct gradient paths bypassing recurrence

## Final Status

Implemented in HASTE kernel:
- ✅ Log-space output: softmax(signed_log) where signed_log = log|h| * sign(h)
- ✅ W_out @ log_h instead of W_out @ h
- ✅ Gradient flows directly to log_h (no * h factor)
- ✅ Correctly handles sign for tanh outputs

Results:
- Short sequences (T ≤ 16): gradients preserved reasonably (ratio ~0.5)
- Long sequences (T ≥ 64): gradients still decay exponentially
- w1 weights average ~0.82 when candidates are nonzero

Conclusion:
- Log-space helps with numerical stability and output boundary
- Does NOT prevent gradient vanishing for nonlinear recurrences
- For true long-range learning, use linear recurrence (Mamba2-style) or TBPTT
