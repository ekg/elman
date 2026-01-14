# dt Discretization for E1: Input-Dependent Recurrence Gating

## Summary

Add Mamba2-style dt discretization to E1, making the recurrence **input-dependent**. This allows the model to selectively remember or forget based on the current input content.

**Current E1:**
```
h_t = tanh(W_x @ x + W_h @ h_prev + b)
```

**Proposed E1 + dt:**
```
decay = sigmoid(W_dt @ x + b_dt)              # input-dependent gate
h_t = tanh(W_x @ x + decay * (W_h @ h_prev) + b)
```

The decay modulates how much the transformed hidden state contributes. When decay → 0, the model ignores history. When decay → 1, full recurrence.

## Why This Should Help

1. **Selectivity**: Mamba2's key advantage is input-dependent state transitions. This adds that to E1.
2. **Content-based forgetting**: Model can learn to "reset" on delimiters, punctuation, etc.
3. **Preserves nonlinearity**: Unlike Mamba2, we keep tanh on the full expression.

## Implementation

### New Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| W_dt | [dim, dim] | dt projection matrix |
| b_dt | [dim] | dt bias, init to ~2.2 so sigmoid ≈ 0.9 |

**Parameter overhead**: d² + d (same as adding another projection)

### Forward Pass Changes

#### 1. Pre-compute Phase (HASTE pattern)

Add `dt_proj = x @ W_dt.T` batched with existing projections:

```cpp
// In StockElmanForward::Run

// Workspace layout: [tmp_Wx: T*BD] [gate_proj: T*BD] [dt_proj: T*BD] [tmp_Rh: BD]
T* tmp_Wx = workspace;
T* gate_proj = workspace + steps * BD;
T* dt_proj = workspace + 2 * steps * BD;      // NEW
T* tmp_Rh = workspace + 3 * steps * BD;       // shifted

// Pre-compute dt_proj = x @ W_dt.T for all timesteps
blas<T>::gemm(
    blas_handle_,
    CUBLAS_OP_T, CUBLAS_OP_N,
    dim_, steps * batch_size_, dim_,
    &alpha,
    W_dt, dim_,
    x, dim_,
    &beta_zero,
    dt_proj, dim_);
```

#### 2. Modified Fused Kernel

Replace `FusedTanhKernel` with `FusedTanhWithDecay`:

```cpp
template<typename T>
__global__ void FusedTanhWithDecay(
    const int batch_size,
    const int dim,
    const T* __restrict__ Wx,        // [B, dim] pre-computed W_x @ x
    const T* __restrict__ Rh,        // [B, dim] W_h @ h_prev
    const T* __restrict__ dt_proj,   // [B, dim] pre-computed W_dt @ x
    const T* __restrict__ b,         // [dim] main bias
    const T* __restrict__ b_dt,      // [dim] decay bias
    T* __restrict__ h_out,           // [B, dim] output
    T* __restrict__ v_cache,         // [B, dim] pre-activation cache
    T* __restrict__ decay_cache) {   // [B, dim] decay values for backward

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Input-dependent decay: sigmoid(W_dt @ x + b_dt)
        float dt_val = static_cast<float>(dt_proj[idx]) + static_cast<float>(b_dt[d]);
        float decay = 1.0f / (1.0f + expf(-dt_val));

        // Modulated recurrence: Wx + decay * Rh + b
        float rh_val = static_cast<float>(Rh[idx]);
        float val = static_cast<float>(Wx[idx]) + decay * rh_val + static_cast<float>(b[d]);

        // Cache for backward
        if (v_cache) v_cache[idx] = static_cast<T>(val);
        if (decay_cache) decay_cache[idx] = static_cast<T>(decay);

        h_out[idx] = static_cast<T>(tanhf(val));
    }
}
```

#### 3. Forward Loop Update

```cpp
for (int t = 0; t < steps; ++t) {
    const T* Wx_t = tmp_Wx + t * BD;
    const T* dt_proj_t = dt_proj + t * BD;        // NEW
    const T* h_prev = h + t * BD;
    const T* gate_proj_t = gate_proj + t * BD;
    T* h_t = h + (t + 1) * BD;
    T* out_t = output + t * BD;
    T* v_t = training_ ? (v + t * BD) : nullptr;
    T* decay_t = training_ ? (decay_cache + t * BD) : nullptr;  // NEW
    T* gate_t = training_ ? (gate_cache + t * BD) : nullptr;

    // tmp_Rh = h_prev @ W_h.T
    blas<T>::gemm(..., W_h, h_prev, tmp_Rh);

    // h_t = tanh(Wx_t + decay * tmp_Rh + b)  -- MODIFIED
    FusedTanhWithDecay<T><<<num_blocks, block_size, 0, stream_>>>(
        batch_size_, dim_, Wx_t, tmp_Rh, dt_proj_t, b, b_dt, h_t, v_t, decay_t);

    // output = h * silu(gate_proj + b_gate)  -- unchanged
    SelectiveOutputForward<T><<<num_blocks, block_size, 0, stream_>>>(
        batch_size_, dim_, h_t, gate_proj_t, b_gate, out_t, gate_t);
}
```

### Backward Pass Changes

#### 1. Gradient Through Decay

The forward was:
```
decay = sigmoid(dt_proj + b_dt)
val = Wx + decay * Rh + b
h = tanh(val)
```

Backward:
```
dval = dh * (1 - h²)                          # already computed as dv
d_Rh = dval * decay                           # gradient to Rh (for dW_h)
d_decay = dval * Rh                           # gradient to decay
d_dt_proj = d_decay * decay * (1 - decay)     # sigmoid derivative
d_b_dt += d_dt_proj                           # accumulated bias gradient
```

#### 2. New Backward Kernel

```cpp
template<typename T>
__global__ void DecayBackwardKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ v,           // [B, dim] pre-activation
    const T* __restrict__ Rh,          // [B, dim] cached W_h @ h_prev
    const T* __restrict__ decay,       // [B, dim] cached decay values
    const T* __restrict__ dh,          // [B, dim] gradient from above
    const T* __restrict__ dh_recurrent,// [B, dim] gradient from next timestep
    T* __restrict__ dv,                // [B, dim] gradient w.r.t. pre-activation
    T* __restrict__ d_Rh,              // [B, dim] gradient to Rh (for dW_h)
    T* __restrict__ d_dt_proj,         // [B, dim] gradient to dt_proj
    float* __restrict__ db,            // [dim] bias gradient
    float* __restrict__ db_dt) {       // [dim] dt bias gradient

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Combine gradients
        float grad = static_cast<float>(dh[idx]);
        if (dh_recurrent) grad += static_cast<float>(dh_recurrent[idx]);

        // dtanh
        float h = tanhf(static_cast<float>(v[idx]));
        float dtanh = 1.0f - h * h;
        float dv_val = grad * dtanh;

        // Gradient through decay * Rh
        float decay_val = static_cast<float>(decay[idx]);
        float rh_val = static_cast<float>(Rh[idx]);

        // d_Rh = dv * decay
        float d_rh_val = dv_val * decay_val;

        // d_decay = dv * Rh
        float d_decay = dv_val * rh_val;

        // d_dt_proj = d_decay * sigmoid'(dt) = d_decay * decay * (1 - decay)
        float d_dt_val = d_decay * decay_val * (1.0f - decay_val);

        // Write outputs
        dv[idx] = static_cast<T>(dv_val);
        d_Rh[idx] = static_cast<T>(d_rh_val);
        d_dt_proj[idx] = static_cast<T>(d_dt_val);

        // Accumulate bias gradients
        atomicAdd(&db[d], dv_val);
        atomicAdd(&db_dt[d], d_dt_val);
    }
}
```

#### 3. Additional Weight Gradients

After the BPTT loop:

```cpp
// dW_dt = x^T @ d_dt_proj_all
blas<T>::gemm(
    blas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_T,
    dim_, dim_, steps * batch_size_,
    &alpha,
    x, dim_,
    d_dt_proj_all, dim_,
    &beta_one,
    dW_dt, dim_);

// dx += W_dt @ d_dt_proj_all
blas<T>::gemm(
    blas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_N,
    dim_, steps * batch_size_, dim_,
    &alpha,
    W_dt, dim_,
    d_dt_proj_all, dim_,
    &beta_one,
    dx, dim_);
```

#### 4. dW_h Modification

**Important**: The gradient to W_h now uses `d_Rh` instead of `dv`:

```cpp
// OLD: dW_h = h^T @ dv_all
// NEW: dW_h = h^T @ d_Rh_all

blas<T>::gemm(
    blas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_T,
    dim_, dim_, steps * batch_size_,
    &alpha,
    h, dim_,
    d_Rh_all, dim_,   // CHANGED from dv_all
    &beta_one,
    dW_h, dim_);
```

And the recurrent gradient propagation:

```cpp
// OLD: dh_recurrent = W_h @ dv_t
// NEW: dh_recurrent = W_h @ d_Rh_t

blas<T>::gemm(
    blas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_N,
    dim_, batch_size_, dim_,
    &alpha,
    W_h, dim_,
    d_Rh_t, dim_,    // CHANGED from dv_t
    &beta_zero,
    dh_recurrent, dim_);
```

### Workspace Layout

**Forward:**
```
[tmp_Wx: T*BD] [gate_proj: T*BD] [dt_proj: T*BD] [tmp_Rh: BD]
```
Size: (3T + 1) * BD

**Backward:**
```
[dv_all: T*BD] [d_gate_proj: T*BD] [d_dt_proj: T*BD] [d_Rh_all: T*BD]
[dh: BD] [dh_recurrent: BD] [db_float: dim] [db_gate_float: dim] [db_dt_float: dim]
```
Size: (4T + 2) * BD + 3 * dim

### Cache Layout (Training)

```
[v: T*BD] [decay: T*BD] [gate_cache: T*BD] [Rh_cache: T*BD]
```
Need to cache `Rh` for backward through decay. Size: 4T * BD

### Python Binding Changes

```cpp
// Forward signature
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
stock_elman_forward(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor W_x,
    torch::Tensor W_h,
    torch::Tensor W_dt,      // NEW
    torch::Tensor W_gate,
    torch::Tensor b,
    torch::Tensor b_dt,      // NEW
    torch::Tensor b_gate,
    bool training);

// Backward signature - add dW_dt, db_dt to outputs
```

### Initialization

```python
# W_dt: standard Xavier/Kaiming init
nn.init.xavier_uniform_(self.W_dt)

# b_dt: initialize so sigmoid(b_dt) ≈ 0.9 (slow decay by default)
# sigmoid(2.2) ≈ 0.9
nn.init.constant_(self.b_dt, 2.2)
```

### Cost Analysis

| Metric | Current E1 | E1 + dt | Overhead |
|--------|------------|---------|----------|
| Parameters | 3d² + 2d | 4d² + 3d | +33% |
| Forward GEMMs | 3 batched + T per-step | 4 batched + T per-step | +1 batched |
| Backward GEMMs | 6 | 8 | +2 |
| Fused kernel ops | sigmoid, tanh, silu | +1 sigmoid, +1 mul | minimal |

The main cost is one additional d² GEMM per forward/backward, which is ~25% more compute for the projections but negligible compared to the T per-step GEMMs.

### Naming

Call this variant **E1-dt** or **E1-selective** to distinguish from base E1.

### Testing

1. Gradient check: Verify backward pass is correct
2. Compare loss at matched params with base E1
3. Profile throughput overhead (should be <25%)
4. Test different b_dt initializations (0.5, 0.9, 0.99 default decay)

### Alternative: Scalar dt

For lower overhead, use a single scalar decay per position:

```python
dt_proj = W_dt @ x        # [B, 1] instead of [B, dim]
decay = sigmoid(dt_proj)  # scalar broadcast
h_t = tanh(W_x @ x + decay * (W_h @ h) + b)
```

This adds only d parameters (W_dt is [1, d]) but is less expressive. Try per-dimension first.
