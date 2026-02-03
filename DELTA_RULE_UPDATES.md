# Delta Rule Updates for E70-E73 Matrix Elman Models

This document specifies updates to convert E70-E73 from global decay to delta rule memory updates. These are patches to existing models, not new variants.

---

## 1. Motivation

Current E70-E73 use global decay which wastes ~85% of matrix capacity:
- With decay=0.9, effective rank utilization is ~15%
- Old information decays even when not being overwritten
- Capacity scales poorly with sequence length

Delta rule provides selective update and better capacity utilization:
- Only overwrites information at the queried key direction
- Preserves orthogonal information indefinitely
- Full rank utilization possible
- Exact retrieval for orthogonal keys

These are patches to existing models, not new variants.

---

## 2. The Delta Rule

Mathematical formulation:

**Standard:**
```
S = S + outer(v - S@k, k)
```

**With learning rate:**
```
S = S + β * outer(v - S@k, k)
```

**Key normalization for stability:**
```
k_norm = k / (||k|| + ε)
```

The delta rule can be understood as:
1. Retrieve current value at key: `retrieved = S @ k_norm`
2. Compute error: `delta = v - retrieved`
3. Write correction: `S = S + outer(delta, k_norm)`

This is equivalent to one step of gradient descent on the associative memory loss.

---

## 3. E70 Updates (E42-style → DeltaNet-style)

### Current Implementation
```python
S = decay * S + outer(v, k)
S = tanh(S)
out = (S @ q) * silu(S @ q)
```

### Proposed Implementation
```python
k_norm = k / (||k|| + eps)
retrieved = S @ k_norm
S = S + beta * outer(v - retrieved, k_norm)
# No tanh - use spectral norm on W_k, W_v instead
out = (S @ q) * silu(S @ q)
```

### Key Changes
- **Remove tanh**: Kills gradients, breaks E42 philosophy of bounded contributions
- **Add spectral norm on W_k, W_v**: Bounds input contribution properly
- **Replace decay with delta rule**: Capacity efficiency
- **Add learnable beta parameter**: Learning rate (init ~0.5-1.0)

---

## 4. E71 Updates (State-dependent gate)

### Current Implementation
```python
retrieved = S @ k
α = sigmoid(W_α @ x + d_α * retrieved)
S = α * S + (1 - α) * outer(v, k)
```

### Proposed Implementation
```python
k_norm = k / (||k|| + eps)
retrieved = S @ k_norm
β = sigmoid(W_β @ x + d_β * retrieved)  # State-dependent learning rate
S = S + β * outer(v - retrieved, k_norm)
out = (S @ q) * silu(S @ q)
```

### Key Insight
β is now a learned, state-dependent learning rate:
- **High β** = "I need to learn this" (large update)
- **Low β** = "I already know this" (small/no update)

This preserves E71's philosophy of adaptive gating while gaining delta rule benefits.

---

## 5. E72 Updates (State-dependent value gate)

### Current Implementation
```python
retrieved = S @ k
g = sigmoid(d_g * retrieved + b_g)
v_gated = v * g
S = α * S + (1 - α) * outer(v_gated, k)
```

### Proposed Implementation
```python
k_norm = k / (||k|| + eps)
retrieved = S @ k_norm
g = sigmoid(d_g * retrieved + b_g)  # What to write
v_gated = v * g
S = S + outer(v_gated - retrieved, k_norm)  # Delta with gated value
out = (S @ q) * silu(S @ q)
```

### Key Insight
Separation of concerns:
- **g** controls what VALUE to write (content selection)
- **delta rule** handles the memory update mechanics (how to write)

The gate decides "what aspects of v are relevant", the delta rule ensures efficient storage.

---

## 6. E73 Updates (Input modulation)

### Current Implementation
```python
z = sigmoid(W_z @ x + b_z)
S_mod = S * z.unsqueeze(1)
S = tanh(S_mod + outer(v, k))
```

### Proposed Implementation

**Option A - Modulated retrieval (recommended):**
```python
z = W_z @ x + b_z  # Remove sigmoid (per earlier discussion)
k_norm = k / (||k|| + eps)
retrieved = (S * z.unsqueeze(1)) @ k_norm  # Query modulated state
S = S + outer(v - retrieved, k_norm)
S = tanh(S)  # Keep tanh here since z is unbounded
out = (S @ q) * silu(S @ q)
```

**Option B - Modulated write:**
```python
z = W_z @ x + b_z
k_norm = k / (||k|| + eps)
retrieved = S @ k_norm
v_mod = v * z  # Modulate what we write
S = S + outer(v_mod - retrieved, k_norm)
out = (S @ q) * silu(S @ q)
```

### Recommendation
Option A preferred - modulating retrieval means "query different aspects of memory based on context", which aligns better with E73's original intent.

---

## 7. Stability Analysis

### Delta Rule Stability Properties

**Key normalization:** `||k_norm|| = 1` bounds update magnitude
```
||S_new - S|| = ||outer(v - retrieved, k_norm)|| ≤ ||v - retrieved||
```

**Spectral norm on W_k, W_v:** Bounds `||v||`, `||k||`
- If spectral_norm(W_v) ≤ 1, then ||v|| ≤ ||x||
- Prevents unbounded growth

**Optional beta < 1:** Dampens updates for additional stability

### Jacobian Analysis

Jacobian of delta update:
```
∂S_new/∂S = I - outer(k_norm, k_norm)
```

This is a projection matrix:
- Eigenvalue 0 in k direction (information at k is replaced)
- Eigenvalue 1 elsewhere (orthogonal information preserved)

**Implication:** Stable but selective. Gradients flow through orthogonal directions.

### Gradient Flow Comparison

| Method | Gradient through S | Issue |
|--------|-------------------|-------|
| tanh(S) | sech²(S) | Vanishes for |S| > 2 |
| decay * S | decay^T | Vanishes exponentially |
| delta rule | I - kk^T | Preserves orthogonal gradients |

---

## 8. Implementation Notes

### Required Changes by Model

**All models:**
- Add k normalization: `k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)`
- Add beta parameter (learnable scalar, init 1.0, or per-head)

**E70, E71, E72:**
- Remove tanh on S
- Add spectral norm on W_k, W_v, W_q
- Replace decay/interpolation with delta update

**E73:**
- Keep tanh (z is unbounded) OR switch to spectral norm + remove tanh
- Test both options

### Parameter Initialization

```python
# Beta initialization
self.beta = nn.Parameter(torch.ones(1))  # or per-head: torch.ones(num_heads)

# Spectral norm application
self.W_k = nn.utils.spectral_norm(nn.Linear(d_model, d_head))
self.W_v = nn.utils.spectral_norm(nn.Linear(d_model, d_head))
```

### CUDA Kernel Updates

Existing CUDA kernels will need modification:
- Remove decay multiplication
- Add k normalization (can be fused)
- Change outer product accumulation to delta form
- Retrieve step now required before write

---

## 9. Expected Benefits

### 1. Capacity
- Full rank utilization vs 15% with decay=0.9
- O(d²) effective storage vs O(d²/10) with decay

### 2. Retrieval Accuracy
- Exact retrieval for orthogonal keys: `S @ k_i = v_i`
- Graceful degradation for non-orthogonal keys

### 3. Gradient Flow
- Better gradient propagation without tanh crushing
- Selective gradient blocking (only in k direction)

### 4. Expressivity
- Nonlinear gates preserved where they matter (E71, E72)
- Gating controls what/when to write, not memory mechanics

---

## 10. Summary Table

| Model | Current | Proposed | Key Change |
|-------|---------|----------|------------|
| E70 | decay + tanh | delta + spectral norm | True E42 analog |
| E71 | α*S + (1-α)*outer | S + β*outer(delta) | State-dependent LR |
| E72 | α*S + (1-α)*outer(v*g) | S + outer(v*g - ret) | Delta + value gate |
| E73 | tanh(S*z + outer) | S + outer(v - modulated_ret) | Modulated retrieval |

---

## 11. Testing Strategy

### Unit Tests
1. Verify exact retrieval with orthogonal keys
2. Check spectral norm bounds ||v||, ||k||
3. Validate gradient flow through S across long sequences

### Integration Tests
1. Compare perplexity: delta vs decay versions
2. Memory utilization analysis (effective rank of S)
3. Gradient norm statistics during training

### Ablations
1. Beta fixed vs learnable
2. Spectral norm vs no spectral norm (with tanh fallback)
3. E73: Option A vs Option B

---

## 12. Migration Checklist

- [ ] Add k normalization to all models
- [ ] E70: Remove tanh, add spectral norm, implement delta
- [ ] E71: Convert α-interpolation to β-scaled delta
- [ ] E72: Implement delta with gated value
- [ ] E73: Implement modulated retrieval delta (Option A)
- [ ] Update CUDA kernels
- [ ] Add beta parameter with proper initialization
- [ ] Run comparison experiments
- [ ] Update documentation
