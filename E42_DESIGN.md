# E42: Linear Tied Self-Gated Elman

Combines the two best-performing simplifications:
- **E36**: Linear recurrence (no tanh) → best loss (1.6299)
- **E37**: Tied weights (W_x = W_h) → second best loss (1.6012)

## Architecture

```python
# E33 (baseline):
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
output = h_t * silu(h_t)

# E36 (linear recurrence):
h_t = W_x @ x_t + W_h @ h_{t-1} + b  # No tanh!
output = h_t * silu(h_t)

# E37 (tied weights):
h_t = tanh(W @ x_t + W @ h_{t-1} + b)  # W_x = W_h = W
output = h_t * silu(h_t)

# E42 (linear + tied):
h_t = W @ x_t + W @ h_{t-1} + b        # Linear AND tied!
    = W @ (x_t + h_{t-1}) + b          # Equivalent: project the sum
output = h_t * silu(h_t)
```

## Data Flow

```
x: [B, T, dim]
       │
       ▼
in_proj(x)              ← Linear(dim → d_inner)
       │
       ▼
    silu(·)             ← Pre-activation
       │
       ▼
┌──────────────────────────────────────────┐
│     W @ (x_t + h_{t-1}) + b              │  ← 1 GEMM, linear, tied
│              │                            │
│     (no tanh - linear recurrence)        │
│              │                            │
│        h_t * silu(h_t)                   │  ← Self-gating (only nonlinearity)
└──────────────────────────────────────────┘
       │
       ▼
out_proj(·)             ← Linear(d_inner → dim)
```

## Why This Should Work

### From E36 (Linear Recurrence)
- **Better gradient flow**: No tanh saturation in recurrence
- **Longer memory**: Linear systems don't suffer vanishing gradients
- **Mamba2 alignment**: SSM is linear, nonlinearity only at input/output
- **Self-gate provides nonlinearity**: `h * silu(h)` is highly nonlinear

### From E37 (Tied Weights)
- **Unified representation**: x and h processed identically
- **Double gradient signal**: W gets updates from both input and recurrence paths
- **Regularization**: Fewer parameters, less overfitting
- **Matches language structure**: Current input ≈ past context in representation

### Combined Benefits
- **Minimal GEMMs**: 1 per step (just W @ sum)
- **Minimal parameters**: Only one d_inner × d_inner matrix
- **Clean formulation**: `h = W @ (x + h_prev) + b`
- **All nonlinearity in gate**: Cleaner separation of concerns

## Implementation

```python
class E42Cell(nn.Module):
    """
    E42: Linear recurrence with tied weights and self-gating.

    h_t = W @ (x_t + h_{t-1}) + b    # Linear, tied
    output = h_t * silu(h_t)         # Self-gating
    """

    def __init__(self, dim, spectral_radius=0.99):
        super().__init__()
        self.dim = dim
        self.spectral_radius = spectral_radius

        # Single weight matrix (tied: W_x = W_h = W)
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init scaled to spectral radius
        # Critical for linear recurrence stability!
        nn.init.orthogonal_(self.W)
        self.W.data.mul_(self.spectral_radius)

    def get_W(self):
        """Apply spectral normalization for stability."""
        # For linear recurrence, MUST constrain spectral radius < 1
        # Otherwise hidden state explodes
        with torch.no_grad():
            u = getattr(self, '_u', None)
            if u is None:
                u = torch.randn(self.dim, device=self.W.device)
                u = u / u.norm()
            for _ in range(3):  # Power iteration
                v = self.W.T @ u
                v = v / (v.norm() + 1e-8)
                u = self.W @ v
                u = u / (u.norm() + 1e-8)
            self._u = u
            sigma = (u @ self.W @ v).abs()
        return self.W * (self.spectral_radius / (sigma + 1e-8))

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input (pre-activated with silu)
            h0: [B, dim] initial hidden state
        """
        T, B, D = x.shape
        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        W = self.get_W()

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # E42: Linear recurrence with tied weights
            # h_t = W @ (x + h_prev) + b
            combined = x_t + h_prev
            h_new = combined @ W.T + self.b
            h_list.append(h_new)

            # Self-gating (only nonlinearity!)
            output = h_new * F.silu(h_new)
            output_list.append(output)

        return torch.stack(output_list), torch.stack(h_list)
```

## Stability Analysis

**Risk**: Without tanh, hidden state can grow unbounded.

**Mitigation**: Spectral normalization ensures `||W|| < 1`:
```
||h_t|| = ||W @ (x + h_{t-1}) + b||
       ≤ ||W|| * ||x + h_{t-1}|| + ||b||
       < ||x + h_{t-1}|| + ||b||  (since ||W|| < 1)
```

With bounded inputs and spectral radius < 1, the system is stable:
- Hidden state converges to bounded attractor
- No explosion even over long sequences

**Additional safety**: The self-gate `h * silu(h)` naturally suppresses extreme values:
- When |h| is large, silu(h) ≈ h, so output ≈ h²
- But for very large h, the output projection + layer norm will clip

## Expected Results

| Metric | E33 | E36 | E37 | E42 (expected) |
|--------|-----|-----|-----|----------------|
| Loss | 1.665 | 1.630 | 1.601 | **~1.58-1.62?** |
| Throughput | 140K | 138K | 92K* | **~160K?** |
| Params | 39M | 39M | 37.4M | **~37.4M** |

*E37 slowness was due to GEMM batching issue, now fixed.

**Optimistic scenario**: E42 combines benefits, achieves both best loss AND best speed.

**Pessimistic scenario**: Linear + tied is too constrained, hurts capacity.

## Comparison to Mamba2

```
Mamba2 SSM:
h_t = decay * h_{t-1} + dt * x    # Linear, scalar decay

E42:
h_t = W @ (x + h_{t-1}) + b       # Linear, matrix decay (tied)

Key differences:
- Mamba2: scalar/diagonal decay, no tied input
- E42: full matrix W, tied input transform
- Both: linear recurrence, nonlinearity only in gating
```

E42 is like Mamba2 but with:
- Full matrix instead of diagonal (more expressive)
- Tied input/hidden transform (structural prior)

## Variants to Consider

### E42a: No bias
```python
h_t = W @ (x + h_prev)  # Even simpler
```

### E42b: Diagonal W (Mamba2-like)
```python
h_t = d * (x + h_prev) + b  # d is [dim] vector, element-wise
```
Would be MUCH faster (no GEMM in recurrence) but less expressive.

### E42c: Low-rank W
```python
# W = U @ V where U, V are [dim, rank]
h_t = U @ (V @ (x + h_prev)) + b
```
Compromise between full and diagonal.

## Implementation Checklist

- [ ] Create `elman/models/e42_linear_tied.py`
- [ ] Add Python fallback implementation
- [ ] Create CUDA kernel `elman/cuda/lib/e42_linear_tied_gpu.cu.cc`
- [ ] Add to `elman/cuda/lib/hasty/elman_ladder.h`
- [ ] Add to `elman/cuda/Makefile`
- [ ] Add Python bindings in `elman/cuda/pytorch/elman_ladder.cc`
- [ ] Register in `ladder_lm.py`
- [ ] Benchmark against E33, E36, E37

## CUDA Kernel Notes

The E42 kernel is simpler than E33:

```cpp
// E33 kernel (per timestep):
// 1. x_proj = W_x @ x
// 2. h_proj = W_h @ h
// 3. raw = x_proj + h_proj + b
// 4. h_new = tanh(raw)
// 5. output = h_new * silu(h_new)

// E42 kernel (per timestep):
// 1. combined = x + h_prev
// 2. h_new = W @ combined + b     // Single GEMM!
// 3. output = h_new * silu(h_new)
```

Fewer operations, should be faster.

## Summary

**E42 = E36 + E37 = Linear recurrence + Tied weights + Self-gating**

```python
h_t = W @ (x_t + h_{t-1}) + b    # Linear, tied, 1 GEMM
output = h_t * silu(h_t)          # All nonlinearity here
```

**Why it should work:**
1. Linear recurrence: better gradients (E36 showed this)
2. Tied weights: unified representation (E37 showed this)
3. Self-gating: sufficient nonlinearity (E33 showed this)
4. Fewer params + fewer GEMMs: faster and more regularized

**Risk:** May be too constrained. But E36 and E37 individually worked, so combination is promising.

**Success criteria:** Loss ≤ 1.63, Throughput ≥ 150K tok/s
