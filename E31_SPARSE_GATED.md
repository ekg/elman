# E31: Sparse-Gated Elman

## TL;DR

E1 with sparse output gating via entmax instead of silu.

```python
# E1 (dense):
output = h * silu(W_gate @ x + b_gate)

# E31 (sparse):
output = h * entmax_1_5(W_gate @ x + b_gate)
```

One line change. Zero new parameters. Differentiable.

---

## Motivation

E1 dominates benchmarks (133k tps, loss 1.626 vs Mamba2's 98.6k tps, loss 1.656).

But E1's output gate is dense - all D dimensions contribute equally. This may limit learning of "program-like" computations where only specific dimensions should be active.

**Hypothesis**: Sparse gating creates "register-like" behavior that makes program learning easier, especially for longer training runs.

---

## Architecture

```
Input x
   │
   ├──────────────────────┐
   │                      │
   ▼                      ▼
┌──────────┐         ┌─────────┐
│ W_x @ x  │         │ W_gate  │
└────┬─────┘         │  @ x    │
     │               └────┬────┘
     ▼                    │
┌─────────────────┐       │
│ + W_h @ h_prev  │       │
│ + b             │       │
└────────┬────────┘       │
         │                │
         ▼                ▼
      tanh(·)        entmax(·)   ◄── SPARSE!
         │                │
         │    ┌───────────┘
         │    │
         ▼    ▼
       h_new * gate
         │
         ▼
      [output]
```

**Key difference from E1**: Gate uses entmax (sparse) instead of silu (dense).

---

## Forward Pass

```python
def e31_step(x, h_prev, W_x, W_h, W_gate, b, b_gate):
    # Hidden state update (SAME AS E1)
    h_new = tanh(x @ W_x.T + h_prev @ W_h.T + b)

    # Output gating (CHANGED: entmax instead of silu)
    gate_logits = x @ W_gate.T + b_gate
    gate = entmax_1_5(gate_logits, dim=-1)  # sparse!

    output = h_new * gate
    return output, h_new
```

---

## Variants

| Variant | Gate | Sparsity | Notes |
|---------|------|----------|-------|
| **E31a** | `sparsemax` (α=2) | High | Most sparse, may be hard to train |
| **E31b** | `entmax_1.5` (α=1.5) | Moderate | **RECOMMENDED** |
| **E31c** | `top_k + softmax` | Exact k | Requires STE, most TM-like |
| **E31d** | `entmax_α` (learned α) | Adaptive | Most flexible |

**Start with E31b** (1.5-entmax) - proven in attention, moderate sparsity.

---

## Parameters

```python
# E31 Parameters (SAME as E1!)
W_x: [D, D]         # Input projection
W_h: [D, D]         # Recurrence
W_gate: [D, D]      # Gate projection
b: [D]              # Hidden bias
b_gate: [D]         # Gate bias

# Total: exactly same as E1
```

**Zero additional parameters over E1.**

---

## Implementation

### Cell

```python
class E31SparseGatedCell(nn.Module):
    def __init__(self, dim, alpha=1.5):
        super().__init__()
        self.dim = dim
        self.alpha = alpha

        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_gate)
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.9)

    def forward(self, x, h0=None):
        T, B, D = x.shape
        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Hidden update (same as E1)
            raw = x_t @ self.W_x.T + h_prev @ self.W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Sparse gating (NEW)
            gate_logits = x_t @ self.W_gate.T + self.b_gate
            gate = entmax_1_5(gate_logits, dim=-1)

            output = h_new * gate
            output_list.append(output)

        return torch.stack(output_list), torch.stack(h_list)
```

### Entmax

Use `entmax` package or implement:

```python
# pip install entmax
from entmax import entmax15

# Or implement 1.5-entmax:
def entmax_1_5(x, dim=-1):
    """1.5-entmax: sparse softmax with α=1.5"""
    # Sort descending
    x_sorted, _ = x.sort(dim=dim, descending=True)

    # Compute cumsum for threshold
    cumsum = x_sorted.cumsum(dim=dim)
    k = torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype)

    # Find threshold τ
    # For α=1.5: τ satisfies sum(max(0, x - τ)^0.5) = 1
    # Simplified bisection or closed-form for 1.5-entmax
    ...
```

---

## CUDA Kernel

Minimal change from E1 kernel:

```cpp
// E1: silu gate
float gate = gate_input / (1.0f + expf(-gate_input));

// E31: entmax gate (need entmax kernel)
// Option 1: Separate entmax kernel call
// Option 2: Fused entmax in output computation
```

**Recommendation**: Start with Python entmax, optimize later if E31 shows promise.

---

## Expected Behavior

1. **Sparsity emerges**: Watch avg gate sparsity over training
   - Early: may be dense (model exploring)
   - Later: should become sparse (model found useful dims)

2. **Register-like activation**: Some dimensions consistently selected

3. **Training dynamics**:
   - May need lower LR initially (sparse gradients)
   - Should stabilize faster once patterns found

---

## Metrics to Track

```python
# Add to training loop:
gate_sparsity = (gate == 0).float().mean()  # fraction of zeros
active_dims = (gate > 0).sum(dim=-1).float().mean()  # avg active
gate_entropy = -(gate * gate.log().clamp(min=-100)).sum(dim=-1).mean()
```

---

## Experiment Plan

### Phase 1: Validation (quick)
```bash
# Compare E1 vs E31b at 500 steps
python benchmark_perstep.py --models E1,E31 --steps 500 --seed 42
```

Check:
- Loss curves similar?
- Gate becoming sparse?
- Throughput acceptable?

### Phase 2: Extended training
```bash
# 5000 steps to test long-run hypothesis
python benchmark_perstep.py --models E1,E31 --steps 5000 --seed 42
```

Key question: Does E31 catch up or surpass E1?

### Phase 3: Sparsity sweep
```bash
# Test α values
for alpha in 1.2 1.5 2.0; do
    python benchmark.py --model E31 --alpha $alpha
done
```

---

## Files to Create

```
elman/models/e31_sparse_gated.py       # Main implementation
elman/cuda/lib/e31_sparse_gated_gpu.cu.cc  # CUDA kernel (later)
```

---

## Comparison: E1 vs E31

| Aspect | E1 | E31 |
|--------|-----|-----|
| Gate | `silu(·)` | `entmax(·)` |
| Sparsity | None | Learned |
| Params | N | N (same) |
| Throughput | 133k tps | ~120k tps (est.) |
| Gradient flow | All dims | Sparse dims |
| TM-like | No | Yes |

---

## Why This Might Work

1. **Cleaner states**: Sparse gates force discrete "modes"
2. **Better long-term**: Less state blurring over time
3. **Focused gradients**: Learning concentrated on active dims
4. **Program induction**: Discrete selection = easier to compose

## Why This Might Not Work

1. **Capacity loss**: Fewer active dims = less information
2. **Training instability**: Sparse gradients can be noisy
3. **E1 is already great**: Dense may just be better

---

## Summary

**E31 = E1 + sparse output gating**

```diff
- gate = silu(W_gate @ x + b_gate)
+ gate = entmax_1_5(W_gate @ x + b_gate)
```

- Zero new parameters
- Differentiable
- Creates register-like behavior
- Tests hypothesis: sparse gating helps program learning

**Start with E31b (α=1.5), measure gate sparsity, compare loss curves.**
