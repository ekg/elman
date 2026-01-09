# E19: Simplified Gate Variants

## Background

E18-A showed that adding h to the gate improves loss (1.4932 vs E1's 1.5025) for FREE.

Current E18-A:
```python
Wx = W_x @ x                              # batched GEMM
gate_proj = W_gate @ x                    # batched GEMM (separate)
Rh = W_h @ h_prev                         # per-step GEMM
h_t = tanh(Wx + Rh + b)
output = h_t * silu(gate_proj + h_t + b_gate)
```

Question: Do we need W_gate at all? Can we simplify further?

## E19 Variants

### E19-A: Reuse Wx in gate (DROP W_gate)

```python
Wx = W_x @ x                              # already computed
Rh = W_h @ h_prev                         # already computed
h_t = tanh(Wx + Rh + b)
output = h_t * silu(Wx + h_t + b_gate)    # reuse Wx!
```

**Savings**:
- Removes W_gate: **-d² parameters**
- Removes gate_proj GEMM: **-1 batched GEMM**
- Faster forward AND backward

**Rationale**: W_x @ x already projects input to hidden space. Why project again for the gate?

### E19-B: h-only gate (DROP all x from gate)

```python
Wx = W_x @ x
Rh = W_h @ h_prev
h_t = tanh(Wx + Rh + b)
output = h_t * silu(h_t + b_gate)         # gate sees ONLY h
```

**Savings**: Same as E19-A

**Risk**: Gate can't see current input at all. May hurt.

### E19-C: Scaled h in gate

```python
# E18-A but with learnable scale on h
alpha = nn.Parameter(torch.tensor(1.0))   # init to 1.0
output = h_t * silu(W_gate @ x + alpha * h_t + b_gate)
```

**Cost**: +1 parameter

**Rationale**: h and W_gate @ x may have different magnitudes. Let the model learn the balance.

### E19-D: Residual h in recurrence

```python
Wx = W_x @ x
Rh = W_h @ h_prev
h_t = tanh(Wx + Rh + h_prev + b)          # residual connection!
output = h_t * silu(W_gate @ x + h_t + b_gate)  # E18-A gate
```

**Cost**: Free (just add h_prev)

**Rationale**:
- Stronger gradient flow through residual path
- Like ResNet/DenseNet but for RNN
- h_prev passes through without W_h transformation

### E19-E: Combined (A + D)

```python
Wx = W_x @ x
Rh = W_h @ h_prev
h_t = tanh(Wx + Rh + h_prev + b)          # residual (from D)
output = h_t * silu(Wx + h_t + b_gate)    # reuse Wx (from A)
```

**Savings**: -d² params, -1 GEMM, +residual gradient flow

Best of both worlds if both ideas help.

## CUDA Implementation

### E19-A: Remove W_gate GEMM

**Forward changes in StockElmanForward::Run**:

```cpp
// BEFORE (E18-A):
// Pre-compute: tmp_Wx, gate_proj
// Workspace: [tmp_Wx: T*BD] [gate_proj: T*BD] [tmp_Rh: BD]

// AFTER (E19-A):
// Pre-compute: tmp_Wx only
// Workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]

// Remove this GEMM entirely:
// blas<T>::gemm(..., W_gate, x, gate_proj);  // DELETE

// In the loop, pass Wx_t instead of gate_proj_t:
SelectiveOutputForwardWithH<T><<<...>>>(
    batch_size_, dim_, h_t, Wx_t, b_gate, out_t, gate_t);  // Wx_t not gate_proj_t
```

**Backward changes**:

```cpp
// d_gate_proj gradient now goes to Wx, which means it adds to dW_x

// BEFORE: dW_gate = x^T @ d_gate_proj_all (separate gradient)
// AFTER:  dW_x += x^T @ d_gate_proj_all  (combined gradient)

// In backward, accumulate gate gradient into dW_x:
// dW_x = x^T @ (dv_all + d_gate_proj_all)
```

### E19-B: h-only gate kernel

```cpp
template<typename T>
__global__ void SelectiveOutputHOnly(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ b_gate,       // [dim]
    T* __restrict__ output,
    T* __restrict__ gate_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        // gate = silu(h + b_gate)
        float gate_raw = h_val + b_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(gate_raw);
    }
}
```

### E19-D: Residual h in FusedTanhKernel

```cpp
template<typename T>
__global__ void FusedTanhWithResidual(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,        // [B, dim]
    const T* __restrict__ Rh,        // [B, dim]
    const T* __restrict__ h_prev,    // [B, dim] for residual  <-- NEW
    const T* __restrict__ b,         // [dim]
    T* __restrict__ h_out,
    T* __restrict__ v_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // E19-D: val = Wx + Rh + h_prev + b
        float val = static_cast<float>(Wx[idx])
                  + static_cast<float>(Rh[idx])
                  + static_cast<float>(h_prev[idx])  // <-- residual
                  + static_cast<float>(b[d]);

        if (v_cache) v_cache[idx] = static_cast<T>(val);
        h_out[idx] = static_cast<T>(tanhf(val));
    }
}
```

**E19-D Backward**:

The residual creates an additional gradient path:
```
Forward: val = Wx + Rh + h_prev + b
         h_t = tanh(val)

Backward:
dval = dh * (1 - h_t²)
d_h_prev_from_residual = dval  // direct gradient through residual
d_h_prev_from_Rh = W_h^T @ dval  // gradient through W_h
d_h_prev_total = d_h_prev_from_residual + d_h_prev_from_Rh
```

In the backward kernel, dh_recurrent needs both paths:

```cpp
// After computing dv_t (gradient to pre-activation):

// Path 1: Through W_h (existing)
// dh_recurrent = W_h @ dv_t

// Path 2: Through residual (NEW)
// dh_recurrent += dv_t  (direct add)

// Combined in one operation:
// dh_recurrent = W_h @ dv_t + dv_t
```

```cpp
// Option 1: Separate add kernel after GEMM
blas<T>::gemm(..., W_h, dv_t, dh_recurrent);  // dh_recurrent = W_h @ dv_t
VectorAddInplace<T><<<...>>>(BD, dh_recurrent, dv_t);  // dh_recurrent += dv_t

// Option 2: Use beta=1 and initialize dh_recurrent = dv_t first
cudaMemcpyAsync(dh_recurrent, dv_t, BD * sizeof(T), cudaMemcpyDeviceToDevice);
blas<T>::gemm(..., W_h, dv_t, dh_recurrent, alpha=1, beta=1);  // dh_recurrent = W_h @ dv_t + dh_recurrent
```

## Python Config

```python
class E19Config:
    gate_input: str = "wx_plus_h"  # "gate_proj_plus_h" (E18-A), "wx_plus_h" (E19-A),
                                   # "h_only" (E19-B), "gate_proj_plus_scaled_h" (E19-C)
    h_scale: float = 1.0           # for E19-C
    use_residual: bool = False     # for E19-D
```

## Parameter Counts (d=1024)

| Variant | W_gate | W_x | W_h | Total hidden | vs E18-A |
|---------|--------|-----|-----|--------------|----------|
| E18-A | d² | d² | d² | 3d² | baseline |
| E19-A | 0 | d² | d² | 2d² | **-33%** |
| E19-B | 0 | d² | d² | 2d² | **-33%** |
| E19-C | d² | d² | d² | 3d² + 1 | same |
| E19-D | d² | d² | d² | 3d² | same |
| E19-E | 0 | d² | d² | 2d² | **-33%** |

## Workspace Layouts

### E19-A/B (no W_gate)
```
Forward:  [tmp_Wx: T*BD] [tmp_Rh: BD]
Backward: [dv_all: T*BD] [dh: BD] [dh_recurrent: BD] [db_float: dim]
```
Smaller than E18-A (no gate_proj, no d_gate_proj).

### E19-D (with residual)
Same as E18-A, but backward needs dv_t available for residual gradient.

## Experiment Plan

Run on same setup as E18 comparison (10 min, 400M):

| Variant | Description | Hypothesis |
|---------|-------------|------------|
| E19-A | reuse Wx in gate | Faster, similar loss |
| E19-B | h-only gate | Risky - may need x |
| E19-D | residual h | Better gradient flow |
| E19-E | A + D combined | Best case |

**Baseline**: E18-A (1.4932 loss, 30.6K tok/s)

## Success Criteria

- **E19-A**: If loss ≤ E18-A AND faster → new default
- **E19-B**: Probably worse, but worth testing
- **E19-D**: If loss < E18-A → residual helps
- **E19-E**: If beats both A and D → compound improvement

## Initialization

### E19-A/B/E
No W_gate needed. Remove from model:
```python
# Remove these:
# self.W_gate = nn.Parameter(...)
# self.b_gate = nn.Parameter(...)  # keep this one!
```

### E19-D
No changes, same init as E18-A.

### E19-C
```python
self.h_scale = nn.Parameter(torch.tensor(1.0))  # init to 1.0
```

## Backward Gradient Summary

| Variant | dW_x source | dW_h source | dW_gate source |
|---------|-------------|-------------|----------------|
| E18-A | dv | dv | d_gate_proj |
| E19-A | dv + d_gate | dv | N/A |
| E19-B | dv | dv | N/A |
| E19-D | dv | dv | d_gate_proj |
| E19-E | dv + d_gate | dv | N/A |

For E19-A/E, the gate gradient adds to dW_x since we reuse Wx.
