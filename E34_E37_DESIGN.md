# E34-E37 Simplification Variants Design Doc

Building on E33's success (self-gating: `output = h * silu(h)`), we test four further simplifications.

## Base Architecture (E33)
```
x = in_proj(x)                           # [B, T, d_inner]
x = silu(x)                              # pre-activation
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  # Elman recurrence
output = h_t * silu(h_t)                 # self-gating
output = out_proj(output)                # [B, T, dim]
```

## Variants

### E34: Diagonal W_h (Mamba2-style)
Replace dense W_h with element-wise scaling vector `d`:
```
h_t = tanh(W_x @ x_t + d * h_{t-1} + b)  # d is [dim] vector, element-wise
output = h_t * silu(h_t)
```
**Benefits**: Only 1 GEMM per timestep (vs 2), much faster sequential loop.
**Parameters removed**: dim² - dim (e.g., 1280² - 1280 = 1.6M params saved)

### E35: Cubic Gating (h³)
Replace `h * silu(h)` with `h³`:
```
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
output = h_t³  # or h_t * h_t² for numerical stability
```
**Benefits**: No sigmoid/exp in gating, simpler backward pass.
**Risk**: May have gradient issues at large h values (tanh bounds to [-1,1] so h³ ∈ [-1,1]).

### E36: Linear Recurrence + Self-Gate
Remove tanh from recurrence, keep nonlinearity only in output gate:
```
h_t = W_x @ x_t + W_h @ h_{t-1} + b      # LINEAR recurrence
output = h_t * silu(h_t)                  # nonlinearity here
```
**Benefits**: Simpler recurrence, self-gate provides nonlinearity.
**Risk**: May need careful initialization to prevent exploding/vanishing.

### E37: Tied Weights (W_x = W_h)
Use same weight matrix for input and hidden:
```
h_t = tanh(W @ x_t + W @ h_{t-1} + b)    # same W for both
output = h_t * silu(h_t)
```
**Benefits**: Halves recurrence parameters.
**Alternative formulation**: `h_t = tanh(W @ (x_t + h_{t-1}) + b)`

## Implementation Checklist

For each variant (E34, E35, E36, E37):
- [ ] Create CUDA kernel: `elman/cuda/lib/e{N}_*.cu.cc`
- [ ] Add header declarations: `elman/cuda/lib/hasty/elman_ladder.h`
- [ ] Add to Makefile: `elman/cuda/Makefile`
- [ ] Add Python bindings: `elman/cuda/pytorch/elman_ladder.cc`
- [ ] Create Python model: `elman/models/e{N}_*.py`
- [ ] Register in ladder_lm.py
- [ ] Build and test numerical correctness (CUDA vs Python fallback)

## Expected Results

| Variant | Params vs E33 | Speed vs E33 | Hypothesis |
|---------|---------------|--------------|------------|
| E34 | -1.6M (fewer) | Faster | Diagonal W_h like Mamba2 |
| E35 | Same | Similar | Simpler gate, same capacity |
| E36 | Same | Similar | Linear RNN + nonlinear gate |
| E37 | -1.6M (fewer) | Same | Weight tying regularization |
