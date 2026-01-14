# E43-E55: Radical Simplification Experiments

Building on E42's success (linear + tied + self-gating), we systematically test how much further we can simplify while maintaining or improving performance.

## Critical Lessons from E42 (MUST CARRY FORWARD)

1. **Batched GEMM**: Pre-compute `W @ x` for ALL timesteps in one GEMM, then `W @ h` per step
   ```python
   # SLOW (E37 original):
   for t in range(T):
       h_new = W @ (x[t] + h_prev)  # Cannot batch!

   # FAST (E42 pattern - USE THIS):
   Wx_all = W @ x_all              # ONE batched GEMM
   for t in range(T):
       Wh = W @ h_prev             # Per-step GEMM (unavoidable)
       h_new = Wx_all[t] + Wh + b
   ```

2. **Linear recurrence**: No tanh/nonlinearity in the recurrence itself

3. **Self-gating**: `output = h * silu(h)` provides all necessary nonlinearity

4. **BFloat16**: All implementations must handle bf16 (use `.float()` for eigenvalue computation)

5. **Spectral normalization**: Include but make optional (may not be needed)

---

## Tier 1: Simplify W Matrix Structure

### E43: Scalar Decay
**Hypothesis**: Maybe only the decay rate matters, not the full mixing matrix.

```python
# E42:
h_t = W @ (x_t + h_{t-1}) + b        # W is d×d matrix

# E43:
h_t = λ * (x_t + h_{t-1}) + b        # λ is single scalar!
output = h_t * silu(h_t)
```

**Implementation**:
```python
class E43Cell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(0.0))  # log for stability
        self.b = nn.Parameter(torch.zeros(dim))

    @property
    def lambda_(self):
        return torch.sigmoid(self.log_lambda)  # Constrain to (0, 1)

    def forward(self, x, h_prev):
        # x: [T, B, D], h_prev: [B, D]
        # No GEMM needed! Just scalar multiply
        h_new = self.lambda_ * (x + h_prev) + self.b
        output = h_new * F.silu(h_new)
        return output, h_new
```

**Parameters saved**: d² - 1 per layer (~2.4M for d=1536)
**Expected speed**: MUCH faster (no GEMM in recurrence)

---

### E44: Diagonal W (Mamba2-style)
**Hypothesis**: Per-dimension decay rates, but no cross-dimension mixing.

```python
# E42:
h_t = W @ (x_t + h_{t-1}) + b        # Full matrix

# E44:
h_t = d * (x_t + h_{t-1}) + b        # d is [dim] vector, element-wise
output = h_t * silu(h_t)
```

**Implementation**:
```python
class E44Cell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_d = nn.Parameter(torch.zeros(dim))  # Per-dim decay
        self.b = nn.Parameter(torch.zeros(dim))

    @property
    def d(self):
        return torch.sigmoid(self.log_d)  # Constrain to (0, 1)

    def forward(self, x, h_prev):
        h_new = self.d * (x + h_prev) + self.b
        output = h_new * F.silu(h_new)
        return output, h_new
```

**Parameters saved**: d² - d per layer (~2.4M for d=1536)
**Note**: This is similar to Mamba2's diagonal state decay

---

### E45: Pure Accumulation (W = I)
**Hypothesis**: The most extreme - just accumulate tokens, let self-gate do all the work.

```python
# E42:
h_t = W @ (x_t + h_{t-1}) + b

# E45:
h_t = x_t + h_{t-1}                  # Just add! No parameters!
output = h_t * silu(h_t)
```

**Implementation**:
```python
class E45Cell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # NO PARAMETERS in recurrence!

    def forward(self, x, h_prev):
        h_new = x + h_prev           # Pure accumulation
        output = h_new * F.silu(h_new)
        return output, h_new
```

**Parameters saved**: d² + d per layer (entire recurrence is parameter-free!)
**Risk**: Hidden state may grow unbounded. May need normalization.

**E45b variant**: Add learned scalar decay
```python
h_t = x_t + α * h_{t-1}  # α ∈ (0, 1) learned
```

---

## Tier 2: Simplify Projections

### E46: Remove in_proj
**Hypothesis**: If W mixes everything, why have a separate input projection?

```python
# E42:
x_proj = silu(in_proj(x))            # [B, T, d_inner]
h_t = W @ (x_proj + h_{t-1}) + b
output = h_t * silu(h_t)
y = out_proj(output)

# E46:
# NO in_proj! Recurrence on raw embeddings
h_t = W @ (x + h_{t-1}) + b          # W is dim×dim (not d_inner×d_inner)
output = h_t * silu(h_t)
y = out_proj(output)
```

**Parameters saved**: dim × d_inner (~2M)
**Note**: W size changes from d_inner×d_inner to dim×dim

---

### E47: Remove out_proj
**Hypothesis**: Self-gated output goes directly to residual stream.

```python
# E42:
output = h_t * silu(h_t)
y = out_proj(output)                 # Project back to dim

# E47:
output = h_t * silu(h_t)
y = output                           # Direct to residual (needs dim match)
```

**Note**: Requires d_inner = dim for residual connection to work.

---

### E48: Remove BOTH Projections
**Hypothesis**: Full recurrence directly on embedding space.

```python
# E48:
# x is [B, T, dim] embeddings directly
h_t = W @ (x + h_{t-1}) + b          # W is dim×dim
output = h_t * silu(h_t)
y = output                           # Direct to residual
```

**Parameters**: Only W (dim×dim) and b (dim) per layer
**This is the minimal recurrent layer.**

---

## Tier 3: Simplify Depth/Sharing

### E49: Share W Across All Layers
**Hypothesis**: Same mixing transformation at every depth level.

```python
# E42: Each layer has its own W
layers[0].W, layers[1].W, ...        # 6 different W matrices

# E49: One W shared across all layers
shared_W = nn.Parameter(...)
for layer in layers:
    layer.W = shared_W               # Same W everywhere
```

**Parameters saved**: (depth - 1) × d² (~12M for 6 layers)

---

### E50: Single Wide Layer
**Hypothesis**: With short memory, maybe depth doesn't matter.

```python
# E42: depth=6, d_inner=1536
# E50: depth=1, d_inner=4096 (same params, different shape)
```

**Tests**: Is the stacking of layers important, or just total capacity?

---

## Tier 4: Simplify Self-Gate

### E51: No Self-Gate (Linear Output)
**Hypothesis**: Is the self-gate actually necessary?

```python
# E42:
output = h_t * silu(h_t)             # Self-gating

# E51:
output = h_t                         # Linear! No gating!
```

**Risk**: May lose important nonlinearity. But worth testing.

---

### E52: Pure Quadratic (h²)
**Hypothesis**: Maybe sigmoid doesn't matter, just h².

```python
# E42:
output = h_t * silu(h_t)             # = h² * sigmoid(h)

# E52:
output = h_t * h_t                   # = h² (pure quadratic)
```

**Note**: Always non-negative. May need to handle sign.

**E52b**: Signed quadratic
```python
output = h_t * torch.abs(h_t)        # Preserves sign
```

---

### E53: Sigmoid Gate Only
**Hypothesis**: Maybe the quadratic doesn't matter.

```python
# E42:
output = h_t * silu(h_t)             # h * (h * sigmoid(h)) = h² * sigmoid(h)

# E53:
output = h_t * torch.sigmoid(h_t)    # h * sigmoid(h) = silu(h)
```

**Note**: This is just SiLU activation, not self-gating!

---

## Tier 5: Combination Experiments

### E54: Diagonal + No Projections
Combine E44 (diagonal W) with E48 (no projections):

```python
# Minimal recurrent layer:
h_t = d * (x + h_{t-1})              # Element-wise decay
output = h_t * silu(h_t)
# No in_proj, no out_proj
```

**Parameters per layer**: Only d (dim) for decay + optional bias
**This is approximately Mamba2's core without the complexity.**

---

### E55: Scalar + Shared + No Projections
The ultimate minimal model:

```python
# ONE scalar λ shared across ALL layers
# No in_proj, no out_proj
h_t = λ * (x + h_{t-1})
output = h_t * silu(h_t)
```

**Total recurrence parameters**: 1 (just λ!)
**Everything else**: embeddings, layer norms, final head

---

## Implementation Priority

### Phase 1: W Structure (Critical)
1. **E43** - Scalar decay (tests if W structure matters)
2. **E44** - Diagonal W (middle ground)
3. **E45** - Pure accumulation (extreme test)

### Phase 2: Projections
4. **E46** - Remove in_proj
5. **E48** - Remove both projections

### Phase 3: Sharing/Depth
6. **E49** - Shared W across layers
7. **E50** - Single wide layer

### Phase 4: Self-Gate
8. **E51** - No self-gate
9. **E52** - Pure quadratic
10. **E53** - Sigmoid gate only

### Phase 5: Combinations
11. **E54** - Diagonal + no projections
12. **E55** - Ultimate minimal (scalar + shared + no proj)

---

## Expected Results Table

| Model | W Type | Projections | Self-Gate | Params vs E42 | Speed vs E42 |
|-------|--------|-------------|-----------|---------------|--------------|
| E42 | Full d×d | Both | h*silu(h) | Baseline | Baseline |
| E43 | Scalar λ | Both | h*silu(h) | -80% | +200%? |
| E44 | Diagonal | Both | h*silu(h) | -80% | +150%? |
| E45 | Identity | Both | h*silu(h) | -80% | +300%? |
| E46 | Full | No in_proj | h*silu(h) | -7% | Same |
| E48 | Full | None | h*silu(h) | -15% | Same |
| E49 | Shared | Both | h*silu(h) | -70% | Same |
| E54 | Diagonal | None | h*silu(h) | -85% | +200%? |
| E55 | Scalar shared | None | h*silu(h) | -95% | +400%? |

---

## CUDA Kernel Notes

For E43-E45 (simplified W), the CUDA kernels become MUCH simpler:

```cpp
// E43/E44/E45: No GEMM in recurrence!
// Can fuse everything into one kernel:

__global__ void e43_recurrence_kernel(
    float* h_out,      // [T, B, D]
    const float* x,    // [T, B, D]
    const float* h0,   // [B, D]
    float lambda,      // Single scalar!
    const float* b,    // [D]
    int T, int B, int D
) {
    // Each thread handles one (batch, dim) pair
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    if (bd >= B * D) return;

    int batch = bd / D;
    int dim = bd % D;

    float h_prev = h0[bd];

    for (int t = 0; t < T; t++) {
        float x_t = x[t * B * D + bd];
        float h_new = lambda * (x_t + h_prev) + b[dim];

        // Self-gate: h * silu(h)
        float sigmoid_h = 1.0f / (1.0f + expf(-h_new));
        float output = h_new * h_new * sigmoid_h;

        h_out[t * B * D + bd] = output;
        h_prev = h_new;
    }
}
```

This is **orders of magnitude simpler** than E42's GEMM-based kernel!

---

## Success Criteria

**A simplification is successful if:**
1. Loss within 0.05 nats of E42 at same training steps
2. Throughput improved OR parameter count reduced significantly
3. Training remains stable

**Home run**: Loss matches E42 with 50%+ fewer parameters or 2x speed.

---

## Baseline Reference

E42 (d1536×6):
- Loss: 1.59 (10 min training)
- Throughput: 137K tok/s
- Params: 42.9M

Target: Match or beat on at least one metric while simplifying.
