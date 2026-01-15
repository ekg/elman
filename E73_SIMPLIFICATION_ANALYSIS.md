# E73 Simplification Analysis: Toward E42/E68-like Architecture

This document analyzes E73's current projections and explores simplifications to make it more similar to the high-performing E42 and E68 architectures.

---

## 1. Current Architecture Comparison

### E73 (Matrix Nonlinear with Delta Rule)
```
Projections (4 separate):
  k = W_k @ x              # key for addressing
  v = W_v @ x              # value to write
  q = W_q @ x              # query for output
  z = W_z @ x + b_z        # modulation

State: S ∈ [B, n, n]       # n×n matrix

Update:
  k_norm = k / (||k|| + ε)
  retrieved = (S * z_mod) @ k_norm
  S = tanh(S + outer(v - retrieved, k_norm))

Output:
  out = (S @ q) * silu(S @ q)
```

### E42 (Linear Tied)
```
Projections (1 tied):
  W                        # tied for both input and recurrence

State: h ∈ [B, d]          # vector

Update:
  h = W @ x + W @ h_prev + b   # LINEAR (no tanh!)

Output:
  out = h * silu(h)
```

### E68 (Self-Gating)
```
Projections (2):
  W_alpha                  # retain gate
  W_x                      # value

State: h ∈ [B, d]          # vector

Update:
  α = sigmoid(W_alpha @ x + b_alpha)
  g = sigmoid(d_g * h + b_g)     # state-dependent gate
  v = tanh(W_x @ x) * g
  h = α * h + (1 - α) * v

Output:
  out = h * silu(h)
```

---

## 2. Projection Count Analysis

| Model | Inner Cell Projections | Total Projections | State Size |
|-------|----------------------|-------------------|------------|
| E42   | 1 (W)                | 3 (in_proj, W, out_proj) | O(d) vector |
| E68   | 2 (W_alpha, W_x)     | 4 (in_proj, W_α, W_x, out_proj) | O(d) vector |
| E73   | 4 (W_k, W_v, W_q, W_z) | 6 (in_proj, W_k,v,q,z, out_proj) | O(n²) matrix |

**E73 has 2-4x more projections than E42/E68** plus O(n²) state vs O(d) state.

---

## 3. What Each E73 Projection Does

1. **W_k (key)**: Addresses WHERE to write in the matrix
2. **W_v (value)**: WHAT to write
3. **W_q (query)**: WHERE to read from the matrix
4. **W_z (modulation)**: HOW to weight retrieval (context-dependent)

### Which projections are essential?

For delta rule to work, we minimally need:
- **k**: To address memory (essential for selective update)
- **v**: What to write (essential)
- **q**: What to query (essential for output, but could be tied to k)

The **z modulation** is E73-specific. It's inspired by E1's input modulation but may be unnecessary.

---

## 4. Simplification Options

### Option A: Tie k = v = q (Maximum Simplification)
```python
# Single projection W
w = W @ x
k_norm = w / (||w|| + ε)
retrieved = S @ k_norm
S = tanh(S + outer(w - retrieved, k_norm))
out = (S @ k_norm) * silu(S @ k_norm)
```

**Benefits**:
- Only 1 inner projection (like E42)
- Parameters: n×d instead of 4×n×d
- Semantically: "write at k, read at k" (associative memory)

**Risks**:
- Key/query entanglement limits expressivity
- Can't query different than what was just written

### Option B: Tie k = q, keep v separate
```python
k = W_k @ x
v = W_v @ x
k_norm = k / (||k|| + ε)
retrieved = S @ k_norm
S = tanh(S + outer(v - retrieved, k_norm))
out = (S @ k_norm) * silu(S @ k_norm)
```

**Benefits**:
- 2 inner projections (like E68)
- Key/query alignment: read from what you address
- Write flexibility: v can be different from k

### Option C: Keep k, v, q separate, remove z modulation
```python
k = W_k @ x
v = W_v @ x
q = W_q @ x
k_norm = k / (||k|| + ε)
retrieved = S @ k_norm          # No modulation!
S = tanh(S + outer(v - retrieved, k_norm))
out = (S @ q) * silu(S @ q)
```

**Benefits**:
- 3 inner projections
- Removes complex z variant logic
- Simpler CUDA kernels (no column/row/full variants)

### Option D: Linear recurrence like E42
```python
k = W_k @ x
v = W_v @ x
q = W_q @ x
k_norm = k / (||k|| + ε)
retrieved = S @ k_norm
S = S + outer(v - retrieved, k_norm)  # NO tanh!
out = (S @ q) * silu(S @ q)
```

**Benefits**:
- Linear like E42 - better gradient flow
- Requires spectral norm on W_k, W_v for stability

**Risks**:
- Unbounded S without tanh - need careful normalization

---

## 5. E42-Style Simplification (Recommended)

To make E73 most E42-like while keeping matrix state benefits:

```python
class E74LinearDelta(nn.Module):
    """E42-style matrix state with delta rule."""

    def __init__(self, dim, n_state=64):
        # Single tied projection (E42 philosophy)
        self.W = nn.Parameter(torch.empty(n_state, dim))
        # OR separate k/v if needed:
        # self.W_k = nn.Parameter(...)
        # self.W_v = nn.Parameter(...)

        # Spectral norm for stability (E42's key insight)
        self.spectral_radius = 0.999

    def forward(self, x, S):
        # Single projection
        w = x @ self.get_W().T  # [B, n]

        # Key normalization
        w_norm = w / (w.norm(dim=-1, keepdim=True) + 1e-6)

        # Delta rule update - LINEAR (E42 style, no tanh)
        retrieved = torch.einsum('bij,bj->bi', S, w_norm)
        delta = w - retrieved
        S = S + torch.einsum('bi,bj->bij', delta, w_norm)

        # Self-gating output (E42's only nonlinearity)
        out = torch.einsum('bij,bj->bi', S, w_norm)
        out = out * F.silu(out)

        return out, S
```

**Key E42 principles applied**:
1. **Minimal projections**: Tie k=v=q or at most k/v separate
2. **Linear recurrence**: No tanh inside (only at output self-gate)
3. **Spectral control**: Use spectral norm on W for stability
4. **Self-gating**: out = x * silu(x) as the ONLY nonlinearity

---

## 6. E68-Style Simplification

To make E73 more E68-like (state-dependent gating):

```python
class E74SelfGatedDelta(nn.Module):
    """E68-style self-gating with matrix delta rule."""

    def __init__(self, dim, n_state=64):
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))

        # E68-style state-dependent gate
        self.d_g = nn.Parameter(torch.full((n_state,), 0.5))
        self.b_g = nn.Parameter(torch.zeros(n_state))

    def forward(self, x, S):
        k = x @ self.W_k.T
        v = x @ self.W_v.T
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Retrieve
        retrieved = torch.einsum('bij,bj->bi', S, k_norm)

        # E68-style self-gating: state controls write
        # Flatten S to get "memory fullness" per dimension
        S_energy = (S ** 2).mean(dim=-1)  # [B, n]
        g = torch.sigmoid(self.d_g * S_energy + self.b_g)

        # Gated delta update
        delta = (v - retrieved) * g  # Gate the update
        S = S + torch.einsum('bi,bj->bij', delta, k_norm)
        S = torch.tanh(S)  # Keep tanh for stability

        out = torch.einsum('bij,bj->bi', S, k_norm)
        out = out * F.silu(out)

        return out, S
```

**Key E68 principles applied**:
1. **State-dependent gating**: g depends on S (memory fullness)
2. **Two projections**: W_k and W_v (not 4)
3. **Self-gating at output**: out = x * silu(x)

---

## 7. CUDA Kernel Simplifications

Current E73 CUDA has many kernels:
- `E73NormalizeKKernel_BF16`
- `E73RetrievalKernel_BF16` (3 variants!)
- `E73DeltaUpdateKernel_BF16`
- `E73OutputKernel_BF16`
- Plus backward versions of each

### Simplified kernel set (Option C/D):
1. **NormalizeK**: k_norm = k / ||k|| (keep)
2. **DeltaUpdate**: S = S + outer(v - S@k, k) (simplify - no z modulation)
3. **Output**: out = (S@q) * silu(S@q) (keep)

Removing z modulation eliminates:
- 3 variant codepaths in retrieval
- Complex backward for z gradients
- W_z projection entirely

---

## 8. Memory Analysis

### E73 Checkpointed Memory
```
Forward stores:
- S_checkpoints: [num_cp, B, n, n] = B*n²*(T/K+1)
- k_norm_cache: [T, B, n] = T*B*n
- v_cache: [T, B, n] = T*B*n
- q_cache: [T, B, n] = T*B*n
- z_cache: [T, B, n] = T*B*n
- Sq_cache: [T, B, n] = T*B*n

Total: ~B*n²*(T/K+1) + 5*T*B*n
```

### Simplified E74 Memory (remove z)
```
Forward stores:
- S_checkpoints: [num_cp, B, n, n]
- k_norm_cache: [T, B, n] (or recompute)
- v_cache: [T, B, n] (or tie with k)
- q_cache: [T, B, n] (or tie with k)
- Sq_cache: [T, B, n]

Total: ~B*n²*(T/K+1) + 4*T*B*n  (20% reduction)

With k=v=q tied:
Total: ~B*n²*(T/K+1) + 2*T*B*n  (60% reduction)
```

---

## 9. Recommendations

### For perplexity-focused iteration:
**Start with Option C** (remove z modulation):
- Simplest change from current E73
- Keeps k, v, q separate
- Removes 3-way variant complexity
- Easy to A/B test against E73

### For maximum simplicity:
**Option A or B** (tie projections):
- Mirrors E42's parameter efficiency
- Risks lower expressivity but might be sufficient

### For E68-style gating:
**Add explicit state-dependent gate** before delta update:
- Gives control over "when to write"
- Preserves E68's key insight

---

## 10. Migration Checklist

- [ ] Create E74 variant with z modulation removed (Option C)
- [ ] Benchmark against E73 on standard tasks
- [ ] If comparable: create E74 with tied k=q (Option B)
- [ ] If still comparable: create E74 with k=v=q tied (Option A)
- [ ] For each: update CUDA kernel, remove unused codepaths
- [ ] Consider linear variant (Option D) with spectral norm
- [ ] Document parameter counts and memory usage

---

## 11. Key Insight: Why E42/E68 Perform Well

Both E42 and E68 share a key design principle:

**Minimal architecture + single strong inductive bias**

- **E42**: Linear recurrence + self-gating = gradient flow + nonlinearity
- **E68**: State-dependent gating = adaptive memory retention

E73 combines multiple mechanisms:
- Delta rule (selective update)
- z modulation (context-dependent retrieval)
- Matrix state (key-value storage)
- Self-gating output

This may be **over-parameterized** for what it accomplishes. Simplifying could actually improve performance by:
1. Reducing optimization difficulty
2. Improving gradient flow
3. Increasing effective parameter utilization

The path forward is systematic ablation: remove components one at a time and measure impact.
