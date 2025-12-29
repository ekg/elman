# Log-Space RNN Implementation Handoff

**Date**: December 29, 2025
**Context**: Mamba2 beats Triple R by 7% at 1000 steps. Investigation suggests log-space computation as a potential key factor.
**Reference Implementation**: `~/elman/` - Clean ablation ladder with Levels 0-3 working

---

## Executive Summary

Mamba2 uses log-space computation for numerical stability. Our Elman variants lack this, which **may** explain:

1. Mamba2's widening lead over training (2.7% → 7%)
2. Gradient death after ~350k steps
3. Why adding Mamba2's selective gates to Triple R had zero effect

**The Fundamental Problem**: tanh is a linear-space operation. To use tanh, we must convert from log-space, breaking the gradient benefits. This led us to explore **polynomial activation** as a log-space native alternative.

**Implementation Status**:
- Levels 0-3: ✅ Working in `~/elman/`
- Level 4 (Log-Storage): ⚠️ NaN at step ~260
- Levels 5-6: Theoretical / In development

---

## The Elman Ladder

See `~/elman/` for reference implementation.

```
Level 0: Stock Elman         h = tanh(W_x @ x + W_h @ h + b)
    │
    ▼ +input gate (delta)
Level 1: Gated Elman         h = (1-δ)*h + δ*tanh(...)
    │
    ▼ +output selectivity
Level 2: Selective Elman     output = compete(h) * silu(W_out @ h)
    │
    ▼ diagonal W_h
Level 3: Diagonal Selective  h = (1-δ)*h + δ*tanh(W_x @ x + r_h * h + b)
    │
    ▼ +log storage
Level 4: Log-Storage         Store (log|h|, sign(h)), compute in linear ⚠️ NaN
    │
    ▼ +logsumexp matmul OR polynomial activation
Level 5: Log-Compute         Full log-space computation
    │
    ▼ +R_delta
Level 6: Log-Space Triple R  Full architecture in log-space
```

**Target**: Mamba2 achieves 3.924 avg50 @ 1000 steps. Find the simplest level that matches.

---

## Part 1: The Fundamental Problem with tanh

### 1.1 Why Log-Space Helps (In Theory)

For `z = logsumexp(a, b) = log(exp(a) + exp(b))`:

```
dz/da = exp(a) / (exp(a) + exp(b)) = softmax_a
dz/db = exp(b) / (exp(a) + exp(b)) = softmax_b
```

**These gradients are BOUNDED [0, 1]!** No vanishing, no explosion.

### 1.2 Why tanh Breaks Log-Space

tanh is a **linear-space operation**. It takes a real number and returns a value in [-1, 1].

```python
# Elman uses:
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
```

To use log-space for the matmul while keeping tanh, we must:

1. Compute `W_h @ h` in log-space → get linear-space result
2. Apply tanh (linear operation)
3. Convert result to log-space

**This conversion breaks the gradient chain!**

The gradient `d(log|tanh(v)|)/d(log|v|)` involves:
- `d(log|h|)/dh = 1/h` — unbounded for small h!
- `d(tanh(v))/dv = 1 - tanh²(v)` — vanishes when saturated!

### 1.3 Log-Storage Doesn't Fix Gradients

Level 4 (Log-Storage) stores h as `(log|h|, sign(h))` but still computes in linear space:

```python
h_prev = sign * exp(log_h_prev)  # Convert to linear (can underflow!)
# ... linear arithmetic with tanh ...
log_h_new = log(|h_new|)         # Convert back to log
```

**Result**: Forward pass stable, but gradients still vanish through tanh. Level 4 produces NaN at step ~260 in current implementation.

---

## Part 2: The Polynomial Activation Solution

### 2.1 A Log-Space Native Activation

Instead of tanh, use a polynomial that works natively in log-space:

```python
h = sign(v) * |v|^alpha
```

In log-space:
```python
log|h| = alpha * log|v|
sign(h) = sign(v)
```

**Gradient**:
```
d(log|h|)/d(log|v|) = alpha  # CONSTANT!
```

No vanishing, no explosion. Just a constant scaling factor.

### 2.2 Comparison: tanh vs Polynomial

| Property | tanh | Polynomial |
|----------|------|------------|
| Bounded output | Yes [-1, 1] | No (unbounded) |
| Log-space native | No | Yes |
| Gradient at saturation | → 0 (vanishes) | Constant alpha |
| Non-linear | Yes | Yes (if alpha ≠ 1) |
| Sign-preserving | Yes (odd function) | Yes |

### 2.3 Addressing Unbounded Output

Polynomial doesn't bound output like tanh. Solutions:

1. **LayerNorm/RMSNorm** on hidden state to control scale
2. **Softmax output** like Mamba2's C projection
3. **Alpha tuning**: alpha < 1 compresses, alpha > 1 expands

### 2.4 Open Questions

1. What alpha value works best? (0.5? 1? 2?)
2. Does polynomial provide enough non-linearity?
3. Can this compete with tanh's expressivity?

---

## Part 3: Implementation Status in ~/elman

### 3.1 Working Levels (0-3)

All use Haste CUDA kernels with PyTorch fallback.

| Level | File | Status |
|-------|------|--------|
| 0 | `elman/models/stock_elman.py` | ✅ Working |
| 1 | `elman/models/gated_elman.py` | ✅ Working |
| 2 | `elman/models/selective_elman.py` | ✅ Working |
| 3 | `elman/models/diagonal_selective.py` | ✅ Working |

### 3.2 Research Frontier (4-6)

| Level | Status | Issue |
|-------|--------|-------|
| 4 | ⚠️ Broken | NaN at step ~260 |
| 5 | Theoretical | Needs logsumexp matmul implementation |
| 6 | Theoretical | Needs polynomial activation + full log-space |

### 3.3 Training Infrastructure

```bash
# Run ablation ladder
cd ~/elman
./scripts/train.elman_ladder_ablation.sh [level]
./scripts/train.elman_ladder_ablation.sh all  # Run all levels
```

---

## Part 4: Log-Space Matrix Multiplication

For Level 5+, we need log-space matmul to preserve full R matrices.

### 4.1 The Math

For `y = R @ h` where h is in log-space:

```
y_j = Σ_i R_{ji} * sign_h_i * exp(log_h_i)
```

Decompose R into positive/negative:
```python
R_pos = R.clamp(min=0)      # R+ = max(R, 0)
R_neg = (-R).clamp(min=0)   # R- = max(-R, 0)
```

Each sum becomes a logsumexp:
```python
log_pos_contrib = logsumexp(log_R_pos + log_h, dim=-1)  # where signs match
log_neg_contrib = logsumexp(log_R_neg + log_h, dim=-1)  # where signs differ
result = exp(log_pos) - exp(log_neg)
```

### 4.2 Implementation

```python
class LogSpaceMatmul(nn.Module):
    """Log-space matrix-vector multiplication."""

    def __init__(self, dim):
        super().__init__()
        self.R = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, log_h, sign_h):
        B, D = log_h.shape

        # Decompose R
        R_pos = self.R.clamp(min=0)
        R_neg = (-self.R).clamp(min=0)
        log_R_pos = torch.log(R_pos + 1e-10)
        log_R_neg = torch.log(R_neg + 1e-10)

        # Masks for h signs
        pos_mask = (sign_h > 0).float()
        neg_mask = (sign_h < 0).float()

        # Positive contribution: R+ @ (h where sign>0) + R- @ (h where sign<0)
        log_terms_pp = log_R_pos.unsqueeze(0) + log_h.unsqueeze(1)
        log_terms_pn = log_R_neg.unsqueeze(0) + log_h.unsqueeze(1)
        log_terms_pp = log_terms_pp + torch.log(pos_mask + 1e-10).unsqueeze(1)
        log_terms_pn = log_terms_pn + torch.log(neg_mask + 1e-10).unsqueeze(1)
        log_pos_contrib = torch.logsumexp(torch.stack([log_terms_pp, log_terms_pn]), dim=(0, -1))

        # Negative contribution: R- @ (h where sign>0) + R+ @ (h where sign<0)
        log_terms_np = log_R_neg.unsqueeze(0) + log_h.unsqueeze(1)
        log_terms_nn = log_R_pos.unsqueeze(0) + log_h.unsqueeze(1)
        log_terms_np = log_terms_np + torch.log(pos_mask + 1e-10).unsqueeze(1)
        log_terms_nn = log_terms_nn + torch.log(neg_mask + 1e-10).unsqueeze(1)
        log_neg_contrib = torch.logsumexp(torch.stack([log_terms_np, log_terms_nn]), dim=(0, -1))

        # Combine
        result = torch.exp(log_pos_contrib.clamp(max=40)) - torch.exp(log_neg_contrib.clamp(max=40))
        sign_result = torch.sign(result)
        log_result = torch.log(torch.abs(result) + 1e-10)

        return log_result, sign_result
```

### 4.3 Complexity

| Operation | Standard | Log-Space |
|-----------|----------|-----------|
| R @ h | O(D²) matmul | O(D²) logsumexp |
| Memory | O(D²) | O(2D²) for R_pos, R_neg |
| Stability | Underflows | Stable |

~2x overhead, but preserves full R matrices with stability.

---

## Part 5: Level 6 - Full Log-Space Architecture

### 5.1 With Polynomial Activation

```python
class LogSpacePolynomialElman(nn.Module):
    """
    Level 6: Full log-space Elman with polynomial activation.

    Key properties:
    - All computation in log-space (no linear conversion)
    - Polynomial activation: h = sign(v) * |v|^alpha
    - Gradients are CONSTANT: d(log|h|)/d(log|v|) = alpha
    """

    def __init__(self, dim, alpha=0.5, delta_init=-2.0):
        super().__init__()
        self.alpha = alpha
        self.log_R_h = LogSpaceMatmul(dim)  # Recurrent
        self.W_x = nn.Linear(dim, dim, bias=False)  # Input (can be linear)
        self.W_delta = nn.Linear(dim, dim, bias=False)
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x, log_h_prev, sign_h_prev):
        """
        x: [B, D] input (linear space)
        log_h_prev, sign_h_prev: [B, D] hidden state (log space)
        """
        # Get linear h for gate (sigmoid handles this naturally)
        h_prev = sign_h_prev * torch.exp(log_h_prev.clamp(max=20))

        # Delta gate
        delta_raw = self.W_delta(x) + self.b_delta
        delta = torch.sigmoid(delta_raw)
        log_delta = -F.softplus(-delta_raw)  # log(sigmoid(x))
        log_one_minus_delta = -F.softplus(delta_raw)  # log(1-sigmoid(x))

        # Pre-activation in log-space: R_h @ h + W_x @ x + b
        log_Rh, sign_Rh = self.log_R_h(log_h_prev, sign_h_prev)
        Rh = sign_Rh * torch.exp(log_Rh.clamp(max=20))
        v = Rh + self.W_x(x) + self.b

        # Polynomial activation (log-space native!)
        sign_v = torch.sign(v)
        log_abs_v = torch.log(torch.abs(v) + 1e-8)
        log_candidate = self.alpha * log_abs_v  # h = |v|^alpha
        sign_candidate = sign_v

        # State update: h_new = (1-delta)*h_prev + delta*candidate
        # In linear space for now (could be optimized)
        decay = (1 - delta) * h_prev
        candidate = sign_candidate * torch.exp(log_candidate.clamp(max=20))
        h_new = decay + delta * candidate

        # Convert to log
        sign_h_new = torch.sign(h_new)
        sign_h_new = torch.where(sign_h_new == 0, torch.ones_like(sign_h_new), sign_h_new)
        log_h_new = torch.log(torch.abs(h_new) + 1e-10).clamp(max=40)

        return log_h_new, sign_h_new
```

### 5.2 Gradient Properties

With polynomial activation, the gradient through the activation is:

```
d(log|h|)/d(log|v|) = alpha
```

Compare to tanh:
```
d(log|tanh(v)|)/d(log|v|) = (1 - tanh²(v)) * v / tanh(v)
                          → 0 as |v| → ∞ (saturation)
```

Polynomial gives **constant** gradient flow regardless of magnitude.

---

## Part 6: Experiment Plan

### 6.1 The Ablation Ladder

Run each level for 1000 steps at ~1.3B params:

| Level | Model | Key Question |
|-------|-------|--------------|
| 0 | Stock Elman | Baseline |
| 1 | Gated Elman | Does gating help? |
| 2 | Selective Elman | Does output selectivity help? |
| 3 | Diagonal Selective | Can we go diagonal without losing quality? |
| 4 | Log-Storage | Does log storage fix numerical issues? |
| 5 | Log-Compute (tanh) | Does logsumexp matmul help? |
| 5b | Log-Compute (polynomial) | Does polynomial activation work? |
| 6 | Full Log-Space Triple R | Best we can do? |

### 6.2 Decision Tree

```
Level 3 matches Mamba2?
├─ YES → Diagonal is key, skip log-space
│
└─ NO → Run Level 4
        ├─ NaN issues → Debug gradient flow, try polynomial
        └─ Works but still behind → Run Level 5
            ├─ tanh version works → logsumexp was key
            └─ Still behind → Try polynomial (5b)
                ├─ Matches Mamba2 → polynomial is the answer
                └─ Still behind → Problem is elsewhere
```

### 6.3 Metrics to Track

1. **avg50 loss** - Primary metric
2. **NaN occurrence** - When does training break?
3. **Hidden state magnitudes** - per layer `|h|` stats
4. **Gradient norms** - per layer, detect vanishing
5. **Throughput** - tok/s penalty for log-space

---

## Part 7: Key Insights from ~/elman

### 7.1 The Logsumexp Gradient Property

For logsumexp, gradients are softmax weights bounded [0, 1]. This is why Mamba2 is stable.

### 7.2 tanh Fundamentally Breaks Log-Space

We cannot keep gradients bounded while using tanh. The conversion to/from linear space introduces unbounded gradients.

### 7.3 Polynomial is the Log-Space Native Alternative

```
log|h| = alpha * log|v|
d(log|h|)/d(log|v|) = alpha  # Constant!
```

This is the only way to stay fully in log-space with a nonlinear activation.

### 7.4 Level 4 NaN at Step 260

Current implementation has issues. Likely causes:
- Gradient through `exp(log_h)` when log_h is very negative
- Sign tracking errors when h ≈ 0
- Numerical issues in backward pass

---

## Part 8: What We Know Works

### 8.1 Output Selectivity (compete×silu)

**This helps.** Keep it in all levels.

```python
compete = F.softmax(h.view(B, n_groups, -1), dim=-1).view(B, D)
output = compete * F.silu(W_out @ h)
```

### 8.2 Input-Dependent Gating (delta)

**This helps.** Level 1+ all have this.

```python
delta = sigmoid(W_delta @ x + b_delta)
h_new = (1 - delta) * h_prev + delta * candidate
```

### 8.3 Full Matrices for Non-Recurrent Paths

Only the recurrent `h → h` transition needs to be diagonal (for simple log-space) or use logsumexp matmul (for full expressivity). Keep full matrices for:
- `W_x`: Input projection
- `W_delta`: Gate input
- `W_out`: Output projection

---

## Part 9: Implementation Checklist

### Phase 1: Debug Level 4 NaN

- [ ] Add gradient clipping per-layer
- [ ] Track hidden state magnitudes during training
- [ ] Identify step where NaN first appears
- [ ] Check backward pass for numerical issues

### Phase 2: Implement Polynomial Activation

- [ ] Create `polynomial_elman.py` in `~/elman/elman/models/`
- [ ] Test gradient flow through 32+ layers
- [ ] Compare loss curves to tanh variants

### Phase 3: Implement Log-Space Matmul

- [ ] Pure PyTorch version (slow but correct)
- [ ] Validate numerics match standard matmul
- [ ] Triton kernel for efficiency

### Phase 4: Full Level 6

- [ ] Combine logsumexp matmul + polynomial
- [ ] Add R_delta modulation
- [ ] Benchmark against Mamba2

---

## Appendix: Quick Reference

### Log-Space Identities

```python
log(sigmoid(x)) = -softplus(-x)
log(1 - sigmoid(x)) = -softplus(x)
log(a * b) = log(a) + log(b)
log(a + b) = logaddexp(log(a), log(b))  # Use torch.logaddexp
```

### Stability Thresholds

```python
log_h = log_h.clamp(min=-40, max=40)  # Prevent over/underflow
log_x = torch.log(x + 1e-8)           # Prevent log(0)
```

### Polynomial Activation

```python
def polynomial_activation(v, alpha=0.5):
    """Log-space native activation: h = sign(v) * |v|^alpha"""
    sign = torch.sign(v)
    log_abs = torch.log(torch.abs(v) + 1e-8)
    log_h = alpha * log_abs
    return log_h, sign
```

---

## References

### Code
- `~/elman/` - Reference implementation of ablation ladder
- `~/elman/docs/ladder.md` - Detailed level documentation
- `~/elman/docs/logspace.md` - Log-space research notes
- `mingru/diagonal_mhtr.py` - Diagonal MHTR variant

### Papers
- Mamba-2 (SSD): Dao & Gu, 2024, arXiv:2405.21060
- Mamba: Gu & Dao, 2023, arXiv:2312.00752

### Blog Posts
- [SSD Algorithm (Tri Dao)](https://tridao.me/blog/2024/mamba2-part3-algorithm/)

---

## Final Notes

### The Core Insight

> "tanh is a linear-space operation. There's no log-space equivalent."

To achieve true log-space gradient flow, we need:
1. Polynomial activation (or similar log-space native function)
2. Logsumexp matrix multiply (for full R matrices)
3. Careful sign tracking throughout

### What to Try First

1. Run Levels 0-3 ablation (already working in ~/elman)
2. Debug Level 4 NaN issue
3. Implement polynomial activation for Level 5b/6
4. If polynomial works, we have a path to matching Mamba2

### If Nothing Works

The problem might not be numerical at all. Investigate:
- Initialization schemes
- Learning rate schedules
- Architecture (multi-head? different depth?)
- The SSD parallel computation itself (not just numerics)
