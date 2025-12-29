# Log-Space Research Frontier

This document details the log-space research challenges and proposed solutions.

## Why Log-Space?

Mamba2 achieves numerical stability through log-space computation. The key insight:

### The Logsumexp Gradient Property

For `z = logsumexp(a, b) = log(exp(a) + exp(b))`:

```
dz/da = exp(a) / (exp(a) + exp(b)) = softmax_a
dz/db = exp(b) / (exp(a) + exp(b)) = softmax_b
```

**These gradients are BOUNDED [0, 1]!**

This means:
- No gradient explosion (can't exceed 1)
- No gradient vanishing from multiplicative decay
- Gradients are "softmax weights" that sum to 1

### Mamba2's Linear State Update

Mamba2's state update is:
```
h_t = A * h_{t-1} + B * x_t
```

In log-space:
```
log|h_t| = logsumexp(log|A| + log|h_{t-1}|, log|B| + log|x_t|)
```

Gradient flows through logsumexp with bounded softmax weights. No vanishing!

## The Elman Problem

Elman uses tanh:
```
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
```

### Why tanh Breaks Log-Space

1. **tanh expects LINEAR input**: The argument `W_h @ h + ...` is in linear space
2. **Log-space matmul gives linear output**: Even if we compute `W_h @ h` in log-space, we get a linear-space result
3. **Forced conversion**: We must do log→linear→tanh→log, losing gradient benefits

### Gradient Through tanh

```
d(tanh(v))/dv = 1 - tanh(v)^2
```

When tanh saturates (|v| large), this derivative → 0. Gradient vanishes!

## Attempted Solutions

### Attempt 1: Log-Storage with Linear Backward (Level 4)

**Idea**: Store h in log-space for numerical stability, but compute backward in linear space.

**Implementation**:
```python
# Forward: compute in log-space
log_h_t, sign_h_t = log_space_update(log_h_prev, sign_h_prev, x_t)

# Backward: convert to linear, use standard gradients
h_t = sign_h_t * exp(log_h_t)
d_h_prev = standard_backward(d_h_t, h_t, ...)
```

**Result**: Forward stable, but backward still vanishes. The linear backward inherits all the problems.

### Attempt 2: Log-Space Matrix Multiply (Level 5)

**Idea**: Do the full `W @ h` in log-space, get bounded gradients through logsumexp.

**Implementation**:
```python
# W @ h where h is (log_h, sign_h)
log_contrib = log|W| + log_h  # [out, in] contribution magnitudes
contrib_sign = sign(W) * sign_h

# Separate positive/negative
log_sum_pos = logsumexp(log_contrib where sign > 0)
log_sum_neg = logsumexp(log_contrib where sign < 0)

result = exp(log_sum_pos) - exp(log_sum_neg)  # LINEAR output for tanh!
```

**Result**: Matrix multiply gradients are bounded, but we still need linear output for tanh. The gradient through `result = exp(log_sum_pos) - exp(log_sum_neg)` and then tanh still vanishes.

### Attempt 3: Custom Gradient Through Log-Sum

**Idea**: Manually propagate gradients in log-space, never converting to linear.

**Problem**: We're trying to differentiate `tanh(linear_value)`. tanh doesn't have a log-space equivalent. We can't avoid the conversion.

## The Fundamental Issue

**tanh is a linear-space operation**. It takes a real number and returns a real number in [-1, 1]. There's no log-space equivalent.

To use tanh, we MUST:
1. Have linear-space input (convert from log if needed)
2. Apply tanh
3. Convert output to log-space

This conversion breaks the gradient chain. The gradient `d(log|tanh(v)|)/d(log|v|)` involves:
- `d(log|h|)/dh = 1/h` (unbounded for small h!)
- `d(tanh(v))/dv = 1 - tanh^2(v)` (vanishes when saturated!)

## Proposed Solution: Log-Space Native Activation

Instead of tanh, use an activation that works natively in log-space.

### Polynomial Activation

```
h = sign(v) * |v|^alpha
```

In log-space:
```
log|h| = alpha * log|v|
sign(h) = sign(v)
```

**Gradient**:
```
d(log|h|)/d(log|v|) = alpha  # CONSTANT!
```

No vanishing, no explosion. Just a constant scaling factor.

### Properties

| Property | tanh | Polynomial |
|----------|------|------------|
| Bounded output | Yes [-1, 1] | No (unbounded) |
| Log-space native | No | Yes |
| Gradient at saturation | → 0 | Constant alpha |
| Non-linear | Yes | Yes (if alpha != 1) |
| Sign-preserving | Yes (odd function) | Yes |

### Open Questions

1. **Unbounded output**: Polynomial doesn't bound output like tanh. Problem?
   - Maybe use LayerNorm/RMSNorm to control scale
   - Or use softmax output like Mamba2

2. **Optimal alpha**: What value works best?
   - alpha = 1: Linear (no non-linearity)
   - alpha = 2: Quadratic (strong non-linearity)
   - alpha = 0.5: Square root (weak non-linearity)

3. **Expressivity**: Does polynomial provide enough non-linearity?
   - tanh provides universal approximation
   - Polynomial might need multiple layers to match

4. **Stability**: How to handle very large/small values?
   - Log-space naturally handles this
   - But the linear output for predictions needs care

## Implementation Plan

### Phase 1: Pure Polynomial Elman (Level 6)

```python
# In log-space:
log_v = logsumexp(log_W + log_h, log_input)  # Pre-activation in log
log_h_new = alpha * log_v  # Polynomial activation
sign_h_new = sign_v
```

Test:
- Gradient flow at depth 1000+
- Training stability
- Loss curves vs Levels 0-5

### Phase 2: Hybrid Architectures

- Polynomial for state update (stable gradients)
- tanh/softmax for output (bounded predictions)

### Phase 3: CUDA Kernels

Once PyTorch prototype works:
- Fused log-space polynomial kernel
- Benchmark vs Mamba2

## Scope Questions: Is the Ladder Complete?

The current 7-level ladder tests: pure Elman → gating → selectivity → diagonal → log-storage → log-compute → polynomial activation.

But there are orthogonal dimensions we haven't addressed:

### 1. Discretization in Log-Space

The current ladder adds discretization at Level 1 (linear space):
```
h_t = (1 - delta) * h_{t-1} + delta * candidate
```

This is similar to Mamba2's discretization of continuous dynamics. **But the delta gate itself is computed in linear space** with sigmoid.

**Question**: Should delta also be log-space?

```python
# Current (linear):
delta = sigmoid(W_delta @ x)  # Linear sigmoid

# Log-space delta?
log_delta = -softplus(-W_delta @ x)  # log(sigmoid(x))
# Then discretization becomes:
# log|h_t| = logsumexp(log(1-delta) + log|h_{t-1}|, log_delta + log|candidate|)
```

**Experiment needed**: Does log-space discretization help gradient flow through the gate itself?

### 2. Multi-Head Variants

The current ladder is single-head. Multi-head would:
- Split hidden state into H heads of dimension D/H
- Each head has independent recurrence weights
- Heads can attend to different "aspects" of the sequence

**Multi-head is orthogonal to log-space** - we could test:
- Level 3-MH: Diagonal Selective Multi-Head
- Level 6-MH: Log-Space Polynomial Multi-Head

**Question**: Does multi-head help with non-linear RNNs like it helps with attention?

**Experiment needed**: Compare single-head vs 8-head at same parameter count.

### 3. Additional Gating Mechanisms

The current ladder only has the delta (update) gate. GRU has:
- **Update gate**: How much to update (we have this as delta)
- **Reset gate**: How much to reset hidden state before candidate computation

LSTM has:
- **Input gate**: How much new info to add
- **Forget gate**: How much old state to keep
- **Output gate**: How much hidden state to expose

**Question**: Are multiple gates necessary in log-space, or does polynomial activation make them unnecessary?

**Hypothesis**: Polynomial activation with constant gradient might eliminate the need for reset/output gates that were designed to prevent gradient vanishing.

### 4. Diagonal vs Full R in Log-Space

The ladder goes diagonal at Level 3, and stays diagonal through Levels 4-6.

**But**: Level 5 shows we CAN do log-space matmul with full R via logsumexp decomposition:
```python
# Full R @ h in log-space:
log_sum_pos = logsumexp(log|R| + log|h|, where sign(R*h) > 0)
log_sum_neg = logsumexp(log|R| + log|h|, where sign(R*h) < 0)
```

**Question**: Should we test:
- Level 5a: Log-compute with diagonal r_h
- Level 5b: Log-compute with full R via logsumexp

**Experiment needed**: Does full R provide expressivity benefits that outweigh the computational cost of logsumexp?

### 5. Convolution (Not in Current Scope)

Mamba has a 1D depthwise convolution before the SSM:
```
x_conv = conv1d(x, kernel_size=4)
x_ssm = ssm(x_conv)
```

This provides local context and helps with induction heads.

**Decision**: Convolution is NOT in current scope because:
1. It's a separate hypothesis (local context vs recurrence)
2. Adding it confounds the log-space experiment
3. Can be tested independently later

### 6. Connection to Triple R

The gruboros research explored "Triple R" architecture:
- R_input: Input recurrence
- R_hidden: Hidden recurrence
- R_output: Output recurrence

The current ladder focuses on R_hidden (the `W_h` matrix becoming diagonal or log-space).

**Question**: Should the ladder test input/output recurrence separately?

**Current answer**: Level 2's compete×silu provides output selectivity. Input processing is handled by W_x (not recurrent). Triple R concepts are partially represented, but a full comparison needs:
- Level 2+: Compare W_x variations
- Level 3+: Compare output projection variations

---

## Experimental Matrix

Based on scope analysis, here are the experiments needed:

### Phase 1: Complete Current Ladder (Levels 0-6)

| Experiment | Level | Question | Status |
|------------|-------|----------|--------|
| E1.0 | 0 | Baseline pure Elman | TODO |
| E1.1 | 1 | + Discretization | TODO |
| E1.2 | 2 | + Compete output | TODO |
| E1.3 | 3 | + Diagonal | TODO |
| E1.4 | 4 | + Log-storage | NaN at ~260 |
| E1.5 | 5 | + Log-compute | TODO |
| E1.6 | 6 | + Polynomial | TODO |

### Phase 2: Debug Log-Space Issues

| Experiment | Question |
|------------|----------|
| E2.1 | Why does Level 4 NaN at step ~260? |
| E2.2 | Is it the discretization interaction? |
| E2.3 | Is it the tanh → log conversion? |
| E2.4 | Does skipping to Level 6 (polynomial) avoid the issue? |

### Phase 3: Orthogonal Extensions

| Experiment | Dimension | Question |
|------------|-----------|----------|
| E3.1 | Multi-head | Does 8-head help at Level 3? |
| E3.2 | Multi-head | Does 8-head help at Level 6? |
| E3.3 | Full vs Diagonal | Does Level 5b (full R log-space) beat 5a (diagonal)? |
| E3.4 | Alpha sweep | What polynomial alpha works best? (0.5, 0.8, 1.0, 1.5, 2.0) |

### Phase 4: vs Mamba2 Comparison

| Experiment | Question |
|------------|----------|
| E4.1 | Best Elman level vs Mamba2 at 100M params |
| E4.2 | Best Elman level vs Mamba2 at 500M params |
| E4.3 | Best Elman level vs Mamba2 at 1B params |
| E4.4 | Scaling law comparison |

---

## Decision Log

Decisions made about scope:

1. **Convolution excluded**: Test recurrence separately from local context
2. **Focus on R_hidden first**: Input/output recurrence can come later
3. **Single-head for ladder, multi-head as extension**: Keep main ladder simple
4. **Debug Level 4 before Level 5-6**: Understand what's breaking

---

## References

- **Mamba2**: [Transformers are SSMs](https://arxiv.org/abs/2405.21060) - Log-space SSM computation
- **Log-sum-exp trick**: Numerical stability via max subtraction
- **Softmax gradient**: Why logsumexp gradients are bounded
- **GRU**: Gated Recurrent Unit - reset + update gates
- **LSTM**: Long Short-Term Memory - input/forget/output gates
