# CUDA Implementation Spec: E61-E63m Gated Delta Elman Family

## Overview

Implement CUDA kernels for four new Elman variants with gated delta dynamics. All kernels must support bfloat16 and match PyTorch reference implementations exactly.

**Models to implement:**
- E61: Decay-gated (linear, **PARALLELIZABLE via associative scan**)
- E62: Selective write (linear, **PARALLELIZABLE via associative scan**)
- E63: Nonlinear delta (UTM-class, **SEQUENTIAL ONLY**)
- E63m: Matrix nonlinear (UTM-class, **SEQUENTIAL ONLY**)

---

## Associative Scan Analysis

### What Makes a Recurrence Parallelizable?

A linear recurrence `h_t = A_t * h_{t-1} + B_t` can use **associative scan** when:
1. `A_t` depends only on `x_t` (not on `h_{t-1}`)
2. `B_t` depends only on `x_t` (not on `h_{t-1}`)

The scan combines tuples `(A, B)` via: `(A1, B1) ⊗ (A2, B2) = (A1*A2, A1*B2 + B1)`

This is associative, enabling O(log T) parallel reduction instead of O(T) sequential.

### E61: ✅ CAN USE ASSOCIATIVE SCAN
```
h_t = α_t * h_{t-1} + (1 - α_t) * v_t

Where:
  α_t = sigmoid(x_t @ W_alpha.T + b_alpha)  ← depends only on x ✓
  v_t = x_t @ W_v.T + b_v                   ← depends only on x ✓

Scan form: A_t = α_t, B_t = (1 - α_t) * v_t
Both computable from x alone → PARALLELIZABLE
```

### E62: ✅ CAN USE ASSOCIATIVE SCAN
```
h_t = (1 - k_t) * h_{t-1} + k_t * v_t

Where:
  k_t = sigmoid(x_t @ W_k.T + b_k)   ← depends only on x ✓
  v_t = tanh(x_t @ W_v.T + b_v)      ← depends only on x ✓

Scan form: A_t = (1 - k_t), B_t = k_t * v_t
Both computable from x alone → PARALLELIZABLE
```

### E63: ❌ CANNOT USE ASSOCIATIVE SCAN
```
h_t = α_t * h_{t-1} + (1 - α_t) * v_t

Where:
  α_t = sigmoid(x_t @ W_alpha.T + b_alpha)  ← depends only on x ✓
  v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b) ← DEPENDS ON h_{t-1} ✗

B_t depends on h_{t-1} → NOT PARALLELIZABLE
Must compute sequentially: need h_{t-1} to compute v_t
```

### E63m: ❌ CANNOT USE ASSOCIATIVE SCAN
```
retrieved = tanh(S_{t-1} @ k_t)  ← DEPENDS ON S_{t-1} ✗
v_t = tanh(W_r @ retrieved + ...)
S_t = α_t * S_{t-1} + (1 - α_t) * v_t @ k_t.T

Both retrieval and value depend on previous state → NOT PARALLELIZABLE
```

### Summary Table

| Model | A_t depends on h? | B_t depends on h? | Parallelizable? |
|-------|-------------------|-------------------|-----------------|
| E61   | No (α from x)     | No (v from x)     | ✅ Yes (scan)   |
| E62   | No (1-k from x)   | No (k*v from x)   | ✅ Yes (scan)   |
| E63   | No (α from x)     | **Yes (v uses h)**| ❌ No           |
| E63m  | No (α from x)     | **Yes (v uses S)**| ❌ No           |

### Why This Matters

- **E61/E62**: Can achieve Mamba2-like throughput (~150K+ tok/s) via parallel scan
- **E63/E63m**: Limited to ~100K tok/s due to sequential dependency, BUT have UTM expressivity

The tradeoff is fundamental: **nonlinear h-dependence gives expressivity but breaks parallelism**.

---

## File Locations

```
elman/cuda/lib/e61_decay_gated_gpu.cu.cc
elman/cuda/lib/e62_selective_write_gpu.cu.cc
elman/cuda/lib/e63_nonlinear_delta_gpu.cu.cc
elman/cuda/lib/e63m_matrix_nonlinear_gpu.cu.cc

elman/cuda/lib/hasty/elman_ladder.h  (add declarations)
elman/cuda/pytorch/elman_ladder.cc  (add bindings)
```

---

## E61: Decay-Gated Elman

### Forward
```
Input: x [T, B, D], h0 [B, D], W_alpha [D, D], b_alpha [D], W_v [D, D], b_v [D]

For t = 0..T-1:
    alpha_t = sigmoid(x[t] @ W_alpha.T + b_alpha)    # [B, D]
    v_t = x[t] @ W_v.T + b_v                         # [B, D]
    h[t+1] = alpha_t * h[t] + (1 - alpha_t) * v_t    # [B, D]
    output[t] = h[t+1] * silu(h[t+1])                # [B, D]

Output: h [T+1, B, D], output [T, B, D]
```

### Backward
```
Gradient through: d_alpha, dW_alpha, db_alpha, dW_v, db_v, dx
Key: d_h[t] += d_h[t+1] * alpha_t  (gradient flows through retain gate)
```

### Notes
- Batch compute `x @ W_alpha.T` and `x @ W_v.T` for all T upfront (single GEMM each)
- Sequential loop for recurrence
- **Potential parallel scan**: Since linear in h, could use associative scan

---

## E62: Selective Write Elman

### Forward
```
Input: x [T, B, D], h0 [B, D], W_k [D, D], b_k [D], W_v [D, D], b_v [D]

For t = 0..T-1:
    k_t = sigmoid(x[t] @ W_k.T + b_k)    # [B, D] selection mask
    v_t = tanh(x[t] @ W_v.T + b_v)       # [B, D] new values
    h[t+1] = (1 - k_t) * h[t] + k_t * v_t
    output[t] = h[t+1] * silu(h[t+1])

Output: h [T+1, B, D], output [T, B, D]
```

### Backward
```
Key: d_h[t] += d_h[t+1] * (1 - k_t)  (gradient through retain path)
```

### Notes
- Very similar structure to E61
- Batch GEMM for x projections upfront

---

## E63: Nonlinear Delta Elman (Priority!)

### Forward
```
Input: x [T, B, D], h0 [B, D], W_alpha [D, D], b_alpha [D],
       W_h [D, D], W_x [D, D], b [D]

For t = 0..T-1:
    alpha_t = sigmoid(x[t] @ W_alpha.T + b_alpha)      # [B, D]
    Wh = h[t] @ W_h.T                                  # [B, D] - NONLINEAR PATH
    Wx = x[t] @ W_x.T                                  # [B, D]
    v_t = tanh(Wh + Wx + b)                            # [B, D]
    h[t+1] = alpha_t * h[t] + (1 - alpha_t) * v_t
    output[t] = h[t+1] * silu(h[t+1])

Output: h [T+1, B, D], output [T, B, D]
Save for backward: h, tanh_cache (pre-activation values)
```

### Backward
```
d_h[t] += d_h[t+1] * alpha_t                          # Retain path
d_h[t] += d_h[t+1] * (1-alpha_t) * tanh'(...) * W_h   # Nonlinear path
dW_h += sum over t: d_v[t].T @ h[t]
dW_x += sum over t: d_v[t].T @ x[t]
```

### Notes
- Cannot parallelize: h[t] needed to compute v_t
- Two GEMMs per timestep: `h @ W_h.T` and `x @ W_x.T`
- Batch `x @ W_x.T` upfront, but `h @ W_h.T` must be per-step

---

## E63m: Matrix Nonlinear Elman

### Forward
```
Input: x [T, B, D], S0 [B, N, D],
       W_k [D, D], W_q [D, D], W_x [N, D], W_r [N, N], b [N],
       W_alpha [N, D], b_alpha [N]

For t = 0..T-1:
    k_t = x[t] @ W_k.T                                # [B, D]
    q_t = x[t] @ W_q.T                                # [B, D]

    # Nonlinear retrieval (KEY!)
    Sk = bmm(S[t], k_t.unsqueeze(-1)).squeeze(-1)     # [B, N]
    retrieved = tanh(Sk)                              # [B, N]

    # Value computation
    Wr_ret = retrieved @ W_r.T                        # [B, N]
    Wx = x[t] @ W_x.T                                 # [B, N]
    v_t = tanh(Wr_ret + Wx + b)                       # [B, N]

    # Gated update
    alpha_t = sigmoid(x[t] @ W_alpha.T + b_alpha)     # [B, N]
    v_outer_k = bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))  # [B, N, D]
    S[t+1] = alpha_t.unsqueeze(-1) * S[t] + (1 - alpha_t.unsqueeze(-1)) * v_outer_k

    # Nonlinear output
    Sq = bmm(S[t+1], q_t.unsqueeze(-1)).squeeze(-1)   # [B, N]
    output[t] = tanh(Sq)                              # [B, N]

Output: S [T+1, B, N, D], output [T, B, N]
```

### Notes
- Most complex kernel
- Multiple batched matrix multiplies per step
- Consider E63m-lite variant first (smaller N)

---

## Testing Protocol

For each kernel, verify against PyTorch reference:

### 1. Forward Pass Test
```python
# Generate random inputs
x = torch.randn(T, B, D, device='cuda', dtype=torch.bfloat16)
h0 = torch.randn(B, D, device='cuda', dtype=torch.bfloat16)

# Run both implementations
out_cuda, h_cuda = cuda_forward(x, h0, ...)
out_py, h_py = pytorch_forward(x, h0, ...)

# Compare
assert torch.allclose(out_cuda, out_py, rtol=1e-2, atol=1e-2)
assert torch.allclose(h_cuda, h_py, rtol=1e-2, atol=1e-2)
```

### 2. Backward Pass Test
```python
# Forward with gradients
x.requires_grad_(True)
out_cuda, _ = cuda_forward(x, h0, ...)
out_py, _ = pytorch_forward(x.clone(), h0, ...)

# Backward
loss_cuda = out_cuda.sum()
loss_py = out_py.sum()
loss_cuda.backward()
loss_py.backward()

# Compare gradients
assert torch.allclose(x.grad_cuda, x.grad_py, rtol=1e-2, atol=1e-2)
# Also check weight gradients
```

### 3. Numerical Gradient Check
```python
# For critical operations, verify with finite differences
torch.autograd.gradcheck(cuda_function, inputs, eps=1e-3)
```

---

## Implementation Priority

1. **E63** - Most important (UTM-class vector state)
2. **E61** - Simple baseline (potential parallel scan)
3. **E62** - Similar to E61
4. **E63m-lite** - Matrix state with reduced N
5. **E63m-full** - Full matrix state (memory intensive)

---

## Performance Targets

| Model | Target Throughput | Notes |
|-------|------------------|-------|
| E61 | 150K+ tok/s | Could parallelize |
| E62 | 150K+ tok/s | Could parallelize |
| E63 | 100K+ tok/s | Sequential, 2 GEMMs/step |
| E63m | 50K+ tok/s | Sequential, heavy matrix ops |

Reference: E42 achieves ~137K tok/s at d1536×6

---

## Key Optimizations

1. **Batch GEMM**: Pre-compute all `x @ W.T` projections in one GEMM
2. **Fused kernels**: Combine sigmoid/tanh with element-wise ops
3. **Memory layout**: Keep tensors contiguous, avoid transposes
4. **Shared memory**: Cache W matrices if they fit
5. **Stream parallelism**: Overlap x projections with recurrence

---

## Existing Patterns

Reference implementations:
- `elman/cuda/lib/e42_linear_tied_gpu.cu.cc` - Linear recurrence pattern
- `elman/cuda/lib/e45_pure_accumulation_gpu.cu.cc` - Simple accumulation
- `elman/cuda/lib/e33_self_gate_gpu.cu.cc` - Self-gating pattern

---

## Deliverables

For each model (E61, E62, E63, E63m):

1. [ ] Forward CUDA kernel
2. [ ] Backward CUDA kernel
3. [ ] Header declarations in `hasty/elman_ladder.h`
4. [ ] Python bindings in `elman_ladder.cc`
5. [ ] Autograd Function wrapper in Python model file
6. [ ] Forward pass test (matches PyTorch)
7. [ ] Backward pass test (matches PyTorch)
8. [ ] Benchmark throughput

---

## Commands

Build:
```bash
cd elman/cuda && ./build.sh
```

Test single model:
```bash
python -c "from elman.models.e63_nonlinear_delta import E63NonlinearDelta; ..."
```

Benchmark:
```bash
python benchmark.py --model 63 --dim 1536 --depth 6 --batch 32 --seq 512
```
