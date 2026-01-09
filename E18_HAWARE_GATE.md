# E18: h-Aware Output Gating

## Problem

E1's output gate only sees input x:
```python
output = h_t * silu(W_gate @ x + b_gate)  # gate is x-only
```

This prevents the model from "querying" attractor states in h. If h encodes long-range memory via nonlinear dynamics, the x-only gate can't selectively read it out.

LSTM's output gate sees both x AND h:
```python
o_t = sigmoid(W_o @ [x_t, h_prev])  # LSTM gate sees both
```

## E18 Variants

Test three variants, all cheap or free:

### E18-A: Add h_t directly to gate (FREE)

```python
Rh = W_h @ h_prev                              # already computed
h_t = tanh(W_x @ x + Rh + b)
output = h_t * silu(W_gate @ x + h_t + b_gate)  # add h_t directly
                                ^^^^
```

**Cost**: Zero extra compute. h_t is already in registers/cache.

**Rationale**: Gate can see raw hidden state to decide what to output.

### E18-B: Add Rh to gate (FREE)

```python
Rh = W_h @ h_prev                              # already computed
h_t = tanh(W_x @ x + Rh + b)
output = h_t * silu(W_gate @ x + Rh + b_gate)  # add Rh (transformed h)
                                ^^
```

**Cost**: Zero extra compute. Rh is already computed for the update.

**Rationale**: Gate sees W_h-transformed h, which may be more useful than raw h.

### E18-E: No output gate (ABLATION)

```python
Rh = W_h @ h_prev
h_t = tanh(W_x @ x + Rh + b)
output = h_t                                   # no gating at all
```

**Cost**: Negative! Removes W_gate GEMM entirely. Faster + fewer params.

**Rationale**: Test if x-only gate is actively hurting. If E18-E matches or beats E1, the gate is useless or harmful.

## CUDA Implementation

### E18-A: Modify SelectiveOutputForward

```cpp
template<typename T>
__global__ void SelectiveOutputForwardWithH(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,            // [B, dim]
    const T* __restrict__ gate_proj,    // [B, dim] pre-computed W_gate @ x
    const T* __restrict__ b_gate,       // [dim]
    T* __restrict__ output,
    T* __restrict__ gate_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float gp_val = static_cast<float>(gate_proj[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        // E18-A: gate_raw = W_gate @ x + h + b_gate
        float gate_raw = gp_val + h_val + b_val;  // <-- ADD h_val

        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(gate_raw);
    }
}
```

### E18-B: Pass Rh to SelectiveOutputForward

```cpp
template<typename T>
__global__ void SelectiveOutputForwardWithRh(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,            // [B, dim]
    const T* __restrict__ Rh,           // [B, dim] pre-computed W_h @ h_prev  <-- NEW
    const T* __restrict__ gate_proj,    // [B, dim] pre-computed W_gate @ x
    const T* __restrict__ b_gate,       // [dim]
    T* __restrict__ output,
    T* __restrict__ gate_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float rh_val = static_cast<float>(Rh[idx]);  // <-- NEW
        float gp_val = static_cast<float>(gate_proj[idx]);
        float b_val = static_cast<float>(b_gate[d]);

        // E18-B: gate_raw = W_gate @ x + Rh + b_gate
        float gate_raw = gp_val + rh_val + b_val;  // <-- ADD rh_val

        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        output[idx] = static_cast<T>(h_val * silu_val);
        if (gate_cache) gate_cache[idx] = static_cast<T>(gate_raw);
    }
}
```

**Note**: For E18-B, need to keep tmp_Rh alive until after SelectiveOutputForward, or cache it.

### E18-E: Remove gate entirely

```cpp
// Just copy h to output, no gating
template<typename T>
__global__ void DirectOutput(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    T* __restrict__ output) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        output[idx] = h[idx];
    }
}
```

Or simply: `cudaMemcpyAsync(output, h, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);`

**Forward changes for E18-E**:
- Remove W_gate GEMM in pre-compute phase
- Remove SelectiveOutputForward kernel
- Just copy h_t to output

## Backward Pass Changes

### E18-A Backward

Forward: `output = h * silu(W_gate @ x + h + b_gate)`

```
gate_raw = gp + h + b_gate
silu = gate_raw * sigmoid(gate_raw)
output = h * silu
```

Backward:
```
d_silu = d_output * h
d_h_from_output = d_output * silu

dsilu_draw = sigmoid * (1 + gate_raw * (1 - sigmoid))
d_gate_raw = d_silu * dsilu_draw

d_gp = d_gate_raw           # for dW_gate
d_h_from_gate = d_gate_raw  # h appears in gate_raw too!
d_b_gate += d_gate_raw

# Total gradient to h:
d_h = d_h_from_output + d_h_from_gate
```

**Key difference**: h now contributes to gradient through TWO paths (output multiply AND gate).

### E18-A Backward Kernel

```cpp
template<typename T>
__global__ void SelectiveOutputBackwardWithH(
    const int batch_size,
    const int dim,
    const T* __restrict__ h,
    const T* __restrict__ gate_cache,   // gate_raw from forward
    const T* __restrict__ d_output,
    T* __restrict__ dh,                 // gradient to h (BOTH paths)
    T* __restrict__ d_gate_proj,
    float* __restrict__ d_b_gate) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        float h_val = static_cast<float>(h[idx]);
        float gate_raw = static_cast<float>(gate_cache[idx]);
        float dout = static_cast<float>(d_output[idx]);

        float sigmoid_val = 1.0f / (1.0f + expf(-gate_raw));
        float silu_val = gate_raw * sigmoid_val;

        // d_output/d_silu = h
        float d_silu = dout * h_val;

        // d_silu/d_gate_raw
        float dsilu = sigmoid_val * (1.0f + gate_raw * (1.0f - sigmoid_val));
        float d_gate_raw = d_silu * dsilu;

        // d_output/d_h has TWO components:
        // 1. h appears in output = h * silu  -->  d_h = dout * silu
        // 2. h appears in gate_raw = gp + h + b  -->  d_h += d_gate_raw
        float dh_val = dout * silu_val + d_gate_raw;  // <-- BOTH paths

        // d_gate_proj = d_gate_raw (W_gate @ x contributes)
        float dg_val = d_gate_raw;

        dh[idx] = static_cast<T>(dh_val);
        d_gate_proj[idx] = static_cast<T>(dg_val);
        atomicAdd(&d_b_gate[d], d_gate_raw);
    }
}
```

### E18-B Backward

Forward: `output = h * silu(W_gate @ x + Rh + b_gate)`

Similar to E18-A, but gradient flows to Rh instead of h directly:
```
d_Rh_from_gate = d_gate_raw  # Rh appears in gate_raw
```

This gradient then flows back through `Rh = W_h @ h_prev` to affect dW_h and dh_prev.

### E18-E Backward

Trivial: `d_h = d_output` (no gate to backprop through)

Remove all gate-related gradient computation. Simpler and faster.

## Python Module Changes

```python
class E18Config:
    gate_mode: str = "x_only"  # "x_only" (E1), "x_plus_h" (A), "x_plus_Rh" (B), "none" (E)

# Forward dispatch based on gate_mode
if config.gate_mode == "none":
    output = h_t  # E18-E
elif config.gate_mode == "x_plus_h":
    output = h_t * silu(gate_proj + h_t + b_gate)  # E18-A
elif config.gate_mode == "x_plus_Rh":
    output = h_t * silu(gate_proj + Rh + b_gate)  # E18-B
else:
    output = h_t * silu(gate_proj + b_gate)  # E1 baseline
```

## Workspace Changes

### E18-B workspace

Need to cache Rh for use in SelectiveOutputForward:
```
Forward workspace: [tmp_Wx: T*BD] [gate_proj: T*BD] [Rh_cache: T*BD] [tmp_Rh: BD]
                                                     ^^^^^^^^^^^^^ NEW for E18-B
```

Or: pass tmp_Rh directly to SelectiveOutputForward before it gets overwritten (reorder operations).

### E18-E workspace

Smaller! No gate_proj needed:
```
Forward workspace: [tmp_Wx: T*BD] [tmp_Rh: BD]
```

## Experiment Plan

Run all three on same setup as 6-hour comparison:

| Variant | Description | Expected |
|---------|-------------|----------|
| E18-A | gate = x + h | May help attractor readout |
| E18-B | gate = x + Rh | May help with transformed h |
| E18-E | no gate | Ablation - is gate hurting? |

Compare against E1 baseline (1.162 loss at 400M, 6hr).

## Parameter Counts

All variants have same or fewer params than E1:
- E18-A: Same as E1 (no new weights)
- E18-B: Same as E1 (no new weights)
- E18-E: **Fewer** - removes W_gate (d²) and b_gate (d)

## Initialization

No changes needed for E18-A/B (same weights as E1).

For E18-E, just remove W_gate and b_gate initialization.

## Success Criteria

- If E18-A or E18-B beats E1 by >0.01 nats: h-aware gating helps
- If E18-E matches E1: gate was useless, remove it
- If E18-E beats E1: gate was actively hurting!
- If all worse than E1: output gating structure is fine, problem is elsewhere
