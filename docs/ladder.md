# The Elman Ladder: A 7-Level Ablation Study

This document details each level of the Elman Ladder, an ablation study designed to understand which architectural choices are necessary for competitive performance with linear SSMs.

## Level 0: Stock Elman

The pure, original Elman network from 1990:

```
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
output = h_t
```

**Properties:**
- No gating mechanism
- Full dense recurrent weight `W_h`
- Non-linear, non-associative
- Gradient vanishes through depth due to tanh saturation

**Parameters:** `W_x` [D, D], `W_h` [D, D], `b` [D]

**Why test this:** Establishes the baseline for pure non-linear recurrence.

---

## Level 1: Gated Elman

Adds discretization (NOT a GRU - just one gate):

```
delta = sigmoid(W_delta @ x_t + b_delta)
candidate = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
h_t = (1 - delta) * h_{t-1} + delta * candidate
```

**Properties:**
- Single gate controls interpolation between old and new
- delta ~= 0 means "keep old state"
- delta ~= 1 means "use new candidate"
- Similar to Mamba's discretization

**Parameters:** + `W_delta` [D, D], `b_delta` [D]

**Why test this:** Does simple gating help gradient flow?

---

## Level 2: Selective Elman

Adds compete softmax for selective output:

```
# Same recurrence as Level 1
h_grouped = h_t.view(B, n_groups, group_size)
compete = softmax(h_grouped, dim=-1)  # Within-group competition
output = compete * silu(W_out @ h_t)
```

**Properties:**
- Hidden state grouped into `n_groups` groups
- Softmax within each group creates competition
- Output is sparse-ish (only winners contribute strongly)
- Similar to Mamba's selective output mechanism

**Parameters:** + `W_out` [D, D]

**Why test this:** Does selective output help?

---

## Level 3: Diagonal Selective

Replaces full `W_h` with diagonal `r_h`:

```
delta = sigmoid(W_delta @ x_t + b_delta)
candidate = tanh(W_x @ x_t + r_h * h_{t-1} + b)  # Element-wise!
h_t = (1 - delta) * h_{t-1} + delta * candidate
```

**Properties:**
- `r_h` is a vector [D], not a matrix [D, D]
- Much fewer parameters (D vs D^2)
- Each hidden unit only depends on itself (plus input)
- Similar to Mamba's diagonal state matrix

**Parameters:** Replace `W_h` [D, D] with `r_h` [D]

**Why test this:** Is the full recurrent matrix necessary?

---

## Level 4: Log-Storage Diagonal

Same computation as Level 3, but hidden state stored in log-space:

```
# Store h as (log|h|, sign(h)) where h = sign * exp(log|h|)

# For update h_t = (1-delta) * h_{t-1} + delta * candidate:
log_term1 = log(1-delta) + log|h_{t-1}|
log_term2 = log(delta) + log|candidate|

# If same sign: use logaddexp
# If different sign: use log-subtract
log|h_t| = logaddexp(log_term1, log_term2)  # simplified
sign(h_t) = determined by dominant term
```

**Properties:**
- Prevents numerical underflow when h decays over long sequences
- Forward pass identical results to Level 3 (just different representation)
- Uses Mamba2-style log-sigmoid: `log(sigmoid(x)) = -softplus(-x)`

**Why test this:** Does log storage help numerical stability at depth?

---

## Level 5: Log-Compute Full

Log-space matrix-vector multiply:

```
# For y = W @ h where h is in log-space:
# y[i] = sum_j W[i,j] * h[j]

# In log-space with sign tracking:
log_contrib[i,j] = log|W[i,j]| + log|h[j]|
contrib_sign[i,j] = sign(W[i,j]) * sign(h[j])

# Separate positive and negative contributions
log_sum_pos[i] = logsumexp(log_contrib[i,:] where contrib_sign > 0)
log_sum_neg[i] = logsumexp(log_contrib[i,:] where contrib_sign < 0)

y[i] = exp(log_sum_pos[i]) - exp(log_sum_neg[i])
```

**Properties:**
- Full matrix multiply in log-space
- Gradients through logsumexp use softmax weights (bounded [0,1])
- Should prevent gradient vanishing

**The Problem:** tanh expects LINEAR input, so we must:
1. Compute W @ h in log-space → linear result
2. Apply tanh (linear operation)
3. Convert result back to log-space

This conversion breaks the gradient flow benefits!

---

## Level 6: Log-Space Polynomial (Research Frontier)

Replace tanh with a log-space-native activation:

```
# Instead of: h = tanh(v)
# Use polynomial: h = sign(v) * |v|^alpha

# In log-space:
log|h| = alpha * log|v|
sign(h) = sign(v)

# Gradient is CONSTANT: d(log|h|)/d(log|v|) = alpha
```

**Properties:**
- Activation works natively in log-space
- No linear-space conversion needed
- Constant gradient factor (no vanishing!)
- Provides non-linearity without tanh's gradient issues

**Open Questions:**
- What alpha value works best?
- Does polynomial provide enough non-linearity?
- Can this compete with tanh's expressivity?

---

## Summary: What Each Level Tests

| Level | Question |
|-------|----------|
| 0 | How does pure Elman perform? |
| 1 | Does discretization help gradient flow? |
| 2 | Does selective output improve quality? |
| 3 | Can we use diagonal instead of full W_h? |
| 4 | Does log storage help numerical stability? |
| 5 | Can log-space matrix multiply bound gradients? |
| 6 | Can polynomial activation replace tanh? |

## Implementation Status

| Level | Forward | Backward | CUDA Kernel | Training Verified |
|-------|---------|----------|-------------|-------------------|
| 0 | Yes | Yes | Yes (Haste) | Yes |
| 1 | Yes | Yes | Yes (Haste) | Yes |
| 2 | Yes | Yes | Yes (Haste) | Yes |
| 3 | Yes | Yes | Yes (Haste) | Yes |
| 4 | Yes | Partial | Yes (Haste) | NaN at step ~260 |
| 5 | Yes | No | No | No |
| 6 | Theoretical | No | No | No |
