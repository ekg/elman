# Elman Ladder Stability Analysis - Handoff Report

**Date**: 2026-01-01
**Prepared for**: Formal Analysis Agent
**Dataset**: 500MB FineWeb-Edu (shuffled), 1000 steps, ~8M tokens/level

## Executive Summary

We benchmarked 13 RNN architectures across the "Elman Ladder" - a progression from simple Elman RNNs to log-space polynomial models. **Three architectures are unstable** (level 6, log_0, log_1/log_2 marginal), while **log_5 achieves the best performance** with perfect gradient stability.

## Benchmark Results

| Level | Architecture | Params | Loss@1k | Grad Norm | Status |
|-------|-------------|--------|---------|-----------|--------|
| 0 | Stock Elman | 53M | 1.96 | 1.55 | ✓ Stable |
| 1 | Gated Elman | 69M | 1.99 | 1.30 | ✓ Stable |
| 2 | Selective Elman | 85M | 1.87 | 1.59 | ✓ Stable |
| 3 | Diagonal Selective | 69M | 1.87 | 1.75 | ✓ Stable |
| 4 | Full Recurrence | 85M | 1.89 | 1.77 | ✓ Stable |
| 5 | Linear Triple-R | 101M | 1.87 | 2.73 | ✓ Stable |
| 6 | Linear Polynomial | 85M | NaN | NaN→∞ | ✗ **UNSTABLE** |
| log_0 | LogSpace Polynomial | 69M | 4.34 | 10⁹→10¹² | ✗ **UNSTABLE** |
| log_1 | LogSpace Selective | 85M | 3.14 | ~250 | ⚠ Marginal |
| log_2 | LogSpace Diag Selective | 85M | 3.12 | ~190 | ⚠ Marginal |
| log_3 | LogSpace + Selective | 69M | 1.91 | 2.06 | ✓ Stable |
| log_4 | LogSpace + Full Recurrence | 85M | 1.89 | 1.98 | ✓ Stable |
| log_5 | LogSpace Triple-R | 101M | **1.77** | **1.13** | ✓ **BEST** |

## Architecture Definitions

### Stable Architectures (Levels 0-5, log_3-log_5)

These use standard activations (tanh, sigmoid, ReLU) or log-space operations with **selective output**:

```
# Selective output mechanism (key stabilizer)
compete = softmax(h.reshape(n_groups, group_size), dim=-1)  # Soft attention
output = compete * silu(W_out @ h)  # Gated output
```

The selective mechanism bounds gradient flow because:
1. `softmax` outputs sum to 1 within each group → bounded activations
2. `silu` is smooth and bounded for negative inputs
3. Product `compete * silu` naturally limits gradient magnitude

### Unstable Architectures (Level 6, log_0)

These use **polynomial activation** without selective output:

```
# Level 6 (Linear Polynomial)
alpha = 1 + softplus(W_alpha @ x + b_alpha)  # alpha ∈ [1, ∞)
v = W_x @ x + r_h * h_prev + b
candidate = sign(v) * |v|^alpha  # POLYNOMIAL ACTIVATION
h_new = (1-delta) * h_prev + delta * candidate

# log_0 (LogSpace Polynomial)
log_candidate = alpha * log|v|  # Equivalent to |v|^alpha in log-space
```

## Root Cause Analysis

### The Polynomial Gradient Problem

For `f(v) = sign(v) * |v|^alpha` where `alpha > 1`:

```
df/dv = alpha * |v|^(alpha-1)
```

**Critical issue**: As `v → 0`, the gradient `|v|^(alpha-1) → ∞` when `alpha > 1`.

Even with `alpha` clamped to `[1, 2]`:
- At `v = 0.01`, `alpha = 2`: gradient = `2 * 0.01^1 = 0.02` (fine)
- At `v = 0.001`, `alpha = 2`: gradient = `2 * 0.001^1 = 0.002` (fine)
- But the issue is **accumulation through time** via the recurrent path

### Gradient Accumulation Through Time

The recurrent gradient path compounds:
```
dL/dh_t = dL/dh_{t+1} * (∂h_{t+1}/∂h_t)
        = dL/dh_{t+1} * [(1-delta) + delta * alpha * |v|^{alpha-1} * r_h]
```

When `|v|` is small and `alpha > 1`, the term `alpha * |v|^{alpha-1}` can be large, and this **compounds through backpropagation through time (BPTT)**.

### Why Selectivity Helps

Models with selective output (levels 2-5, log_1-log_5) have an additional gradient path:

```
output = compete * silu(W_out @ h)
∂output/∂h = compete * silu'(W_out @ h) * W_out + ∂compete/∂h * silu(...)
```

The `softmax` in `compete` provides:
1. **Normalization**: Gradients are scaled by softmax which sums to 1
2. **Competition**: Only the "winning" units get strong gradients
3. **Sparsity**: Effectively reduces the number of active gradient paths

## Hypothesis: Selectivity as Gradient Regularizer

We hypothesize that adding selective output to level 6 and log_0 would stabilize them:

```python
# Proposed fix for Level 6
class StableLinearPolynomial:
    def forward(self, x, h_prev):
        # Original polynomial update
        alpha = 1 + softplus(W_alpha @ x)
        v = W_x @ x + r_h * h_prev + b
        candidate = sign(v) * |v|^alpha
        h_new = (1-delta) * h_prev + delta * bound(candidate)

        # ADD: Selective output (stabilizer)
        compete = softmax(h_new.reshape(groups, -1), dim=-1).flatten()
        output = compete * silu(W_out @ h_new)
        return output, h_new
```

**Rationale**: The selective output would:
1. Bound the gradient flowing back into `h_new`
2. Provide an alternative gradient path that bypasses the polynomial
3. Allow the model to "soft-select" which hidden units matter

## CUDA Fixes Attempted

We applied extensive gradient clipping in CUDA kernels:

| Fix | Level 6 | log_0 | Effectiveness |
|-----|---------|-------|---------------|
| Cap alpha to [1, 2] | ✓ | ✓ | Partial |
| Clamp r_h ≤ 0.9 | ✓ | ✓ | Partial |
| Clip grad_h to [-10, 10] | ✓ | ✓ | Partial |
| Clip dh_prev to [-10, 10] | ✓ | ✓ | Partial |
| Clip hidden state log values | - | ✓ | Moderate |
| NaN/Inf protection | ✓ | ✓ | Prevents crash |
| Clip atomicAdd contributions | ✓ | ✓ | Partial |

**Result**: Fixes reduce spike frequency but don't eliminate instability. The fundamental issue is architectural.

## Formal Analysis Questions

For the formal framework, we suggest investigating:

1. **Spectral Analysis**: What are the eigenvalues of the recurrent Jacobian `∂h_{t+1}/∂h_t` for each architecture? The polynomial term likely creates eigenvalues > 1.

2. **Lyapunov Stability**: Can we prove Lyapunov stability for the selective architectures but not for polynomial ones?

3. **Gradient Flow Bounds**: Can we derive upper bounds on gradient norms that depend on architecture choice?

4. **Selectivity as Regularization**: Is there a formal sense in which `softmax` attention regularizes gradient flow? (Connection to attention mechanisms in transformers?)

5. **Log-Space Stability**: Why does log_0 (polynomial in log-space) fail while log_5 (Triple-R in log-space) succeeds? Both operate in log-space but have different gradient properties.

## Files of Interest

- `/home/erikg/elman/elman/cuda/lib/linear_polynomial_gpu.cu.cc` - Level 6 CUDA kernel
- `/home/erikg/elman/elman/cuda/lib/logspace_polynomial_gpu.cu.cc` - log_0 CUDA kernel
- `/home/erikg/elman/elman/cuda/lib/logspace_selective_gpu.cu.cc` - log_1 CUDA kernel (has selectivity)
- `/home/erikg/elman/elman/models/` - Python model definitions
- `/home/erikg/elman/outputs/benchmark/benchmark_20260101_132419/` - Latest benchmark logs

## Recommended Next Steps

1. **Implement selective output for level 6 and log_0** as a stabilization mechanism
2. **Formal proof** that selectivity bounds gradient norms
3. **Ablation study** comparing polynomial with/without selectivity
4. **Spectral analysis** of Jacobians across all 13 levels

---

*End of handoff report*
