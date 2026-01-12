# E38-E41: Alignment Simplifications

Building on E33's success (self-gating), we identify and remove misalignments with Mamba2's design philosophy.

## The Problem: E33 Has Redundant Structure

```
E33 data flow:

x: [B, T, dim]
       │
       ▼
in_proj(x)         ← Linear(dim → d_inner)      [GEMM 1]
       │
       ▼
    silu(·)        ← Nonlinearity 1
       │
       ▼
W_x @ x_t          ← Linear(d_inner → d_inner)  [GEMM 2] ← REDUNDANT!
       +
W_h @ h_{t-1}      ←                            [GEMM 3]
       +
       b
       │
       ▼
    tanh(·)        ← Nonlinearity 2
       │
       ▼
  h * silu(h)      ← Nonlinearity 3
       │
       ▼
out_proj(·)        ← Linear(d_inner → dim)      [GEMM 4]
```

**Issues identified:**
1. `in_proj` and `W_x` both project into d_inner space - redundant!
2. Three nonlinearities in sequence: silu → tanh → silu
3. Bias only in one place (probably fine)

**Mamba2 comparison:**
- Single input projection, split into components
- SSM recurrence has NO internal W_x equivalent
- Minimal nonlinearities (softplus on dt only)

---

## E38: Remove W_x (Primary Experiment)

**Hypothesis**: W_x is redundant since in_proj already projects to d_inner.

### Architecture
```python
# E33 (current):
x_proj = silu(in_proj(x))
h_t = tanh(W_x @ x_proj + W_h @ h_{t-1} + b)
output = h_t * silu(h_t)

# E38 (proposed):
x_proj = silu(in_proj(x))
h_t = tanh(x_proj + W_h @ h_{t-1} + b)  # Direct add, no W_x!
output = h_t * silu(h_t)
```

### Changes
- Remove `W_x` parameter from cell
- Change recurrence: `tanh(x + W_h @ h + b)` instead of `tanh(W_x @ x + W_h @ h + b)`

### Expected Impact
| Metric | E33 | E38 (expected) |
|--------|-----|----------------|
| Params | 39M | **37.4M** (-1.6M) |
| GEMMs/step | 2 | **1** |
| Throughput | 152K | **~170K?** |

### Implementation
```python
class E38Cell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # NO W_x!
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x, h_prev):
        # x comes pre-projected from in_proj
        h_new = torch.tanh(x + h_prev @ self.W_h.T + self.b)
        output = h_new * F.silu(h_new)
        return output, h_new
```

### Why This Should Work
1. `in_proj` already learns the optimal input transformation
2. `W_x` is learning a redundant d_inner → d_inner mapping
3. Mamba2's SSM works without internal input projection
4. Fewer parameters often = better generalization

---

## E39: Remove Bias

**Hypothesis**: The bias term may not be needed.

### Architecture
```python
# E38:
h_t = tanh(x_proj + W_h @ h_{t-1} + b)

# E39:
h_t = tanh(x_proj + W_h @ h_{t-1})  # No bias!
```

### Rationale
- Mamba2's SSM has no bias in state update
- `in_proj` could absorb any needed bias via its weight matrix
- Simpler = potentially faster + regularizing

### Expected Impact
- Params: -d_inner (negligible, ~1280)
- May help or hurt - quick experiment

---

## E40: Remove Pre-SiLU

**Hypothesis**: Triple nonlinearity (silu → tanh → silu) is excessive.

### Architecture
```python
# E38:
x_proj = silu(in_proj(x))
h_t = tanh(x_proj + W_h @ h_{t-1} + b)

# E40:
x_proj = in_proj(x)  # No silu!
h_t = tanh(x_proj + W_h @ h_{t-1} + b)  # tanh provides nonlinearity
```

### Note
This differs from E32 (which kept W_x but removed pre-silu).
E40 tests: with W_x removed, is pre-silu still needed?

### Rationale
- tanh already provides nonlinearity
- silu(h) in output provides another
- Pre-silu may be redundant now

### Risk
- E32 showed removing pre-silu hurts WITH W_x
- But E40 is different: no W_x, so input goes directly to tanh
- May need the silu to "prepare" input for tanh range

---

## E41: Diagonal W_x (Compromise)

**Hypothesis**: If W_x helps, maybe only element-wise scaling is needed.

### Architecture
```python
# E38: no W_x at all
# E41: diagonal W_x (element-wise)

class E41Cell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_x = nn.Parameter(torch.ones(dim))  # Diagonal "W_x"
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x, h_prev):
        h_new = torch.tanh(self.d_x * x + h_prev @ self.W_h.T + self.b)
        output = h_new * F.silu(h_new)
        return output, h_new
```

### Rationale
- If E38 hurts performance, W_x may be doing something useful
- But maybe it only needs element-wise scaling, not full matrix
- Saves d_inner² - d_inner params (~1.6M - 1.3K ≈ 1.6M)
- Still only 1 GEMM per step (W_h only)

---

## Experiment Matrix

| Variant | W_x | Pre-silu | Bias | Params vs E33 |
|---------|-----|----------|------|---------------|
| E33 (base) | Full matrix | Yes | Yes | 0 |
| **E38** | **None** | Yes | Yes | **-1.6M** |
| E39 | None | Yes | No | -1.6M - 1K |
| E40 | None | No | Yes | -1.6M |
| E41 | Diagonal | Yes | Yes | -1.6M + 1K |

---

## Implementation Plan

### Phase 1: E38 (highest priority)
```bash
# Create model
cp elman/models/e33_self_gate.py elman/models/e38_no_wx.py
# Edit to remove W_x

# Test
python -c "from elman.models.e38_no_wx import E38NoWx; ..."

# Benchmark
python benchmark_perstep.py --models E33,E38 --steps 500
```

### Phase 2: E39, E40, E41 (if E38 works)
Only proceed if E38 shows promise.

### Phase 3: CUDA kernel (if significant)
If E38 improves throughput meaningfully, create optimized kernel.

---

## Success Criteria

**E38 is successful if:**
- Loss within 0.02 nats of E33 at 500 steps
- Throughput improved by >5%
- No training instability

**Bonus success:**
- Loss BETTER than E33 (fewer params = regularization)
- Throughput improved by >10%

---

## Theoretical Justification

### Why W_x Exists in Standard Elman
The classic Elman RNN:
```
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
```
Has W_x because x comes from raw input space. The W_x learns to project input into hidden space.

### Why W_x is Redundant in E33
E33 already has:
```
x_proj = in_proj(x)  # Projects dim → d_inner
```
So x_proj is ALREADY in hidden space. Then:
```
h_t = tanh(W_x @ x_proj + ...)  # W_x: d_inner → d_inner
```
This W_x is projecting FROM hidden space TO hidden space - redundant!

### Mamba2 Analogy
Mamba2's SSM:
```
h_t = decay * h_{t-1} + dt * x
```
Where x comes from projection. No additional matrix inside recurrence.

E38 aligns with this:
```
h_t = tanh(x + W_h @ h_{t-1} + b)
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| E38 hurts loss | Medium | Try E41 (diagonal) as fallback |
| E38 unstable | Low | Keep spectral norm on W_h |
| E40 hurts loss | High | E32 showed pre-silu matters |
| All fail | Low | E33 is already good baseline |

---

## Summary

**Key insight**: W_x is a d_inner × d_inner matrix that projects from hidden space to hidden space. Since in_proj already handles input→hidden, W_x is redundant.

**Primary experiment**: E38 removes W_x entirely.

**Expected outcome**: Faster (1 fewer GEMM), smaller (1.6M fewer params), same or better loss.

**Alignment with Mamba2**: SSM has no internal input matrix, only decay (analogous to W_h).

```
E33:  h = tanh(W_x @ x + W_h @ h + b)   # 2 GEMMs
E38:  h = tanh(x + W_h @ h + b)          # 1 GEMM  ← Target
```
