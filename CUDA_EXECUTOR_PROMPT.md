# CUDA Kernel Implementation Task

## Task
Implement CUDA kernels for E61, E62, E63, E63m Elman variants. Priority order: E63 > E61 > E62 > E63m.

## Reference
- Full spec: `CUDA_IMPLEMENTATION_SPEC.md`
- Example kernels: `elman/cuda/lib/e42_linear_tied_gpu.cu.cc`, `e45_pure_accumulation_gpu.cu.cc`
- Python models: `elman/models/e61_decay_gated.py`, `e62_selective_write.py`, `e63_nonlinear_delta.py`, `e63m_matrix_nonlinear.py`

## Parallelization Analysis

**Key Question:** Can we use associative scan (like Mamba2)?

A recurrence `h_t = A_t * h_{t-1} + B_t` is parallelizable IFF both A_t and B_t depend only on x (not h).

| Model | A_t | B_t | Scan? | Why |
|-------|-----|-----|-------|-----|
| **E61** | α (from x) | (1-α)*v (from x) | ✅ YES | v = W @ x, no h |
| **E62** | 1-k (from x) | k*v (from x) | ✅ YES | v = tanh(W @ x), no h |
| **E63** | α (from x) | (1-α)*v where **v = tanh(W_h @ h + ...)** | ❌ NO | v depends on h! |
| **E63m** | α (from x) | v where **v = f(S @ k)** | ❌ NO | v depends on S! |

**Implementation strategy:**
- E61/E62: Implement BOTH sequential (simple) AND parallel scan versions
- E63/E63m: Sequential only (the h-dependence is the whole point - UTM expressivity)

## E63 (Priority 1) - Nonlinear Delta

**Forward:**
```
alpha = sigmoid(x @ W_alpha.T + b_alpha)
v = tanh(h @ W_h.T + x @ W_x.T + b)        # h-dependent!
h_new = alpha * h + (1 - alpha) * v
output = h_new * silu(h_new)
```

**Key:** `h @ W_h.T` must be computed per-step (cannot batch). Batch `x @ W_x.T` upfront.

**Files to create:**
- `elman/cuda/lib/e63_nonlinear_delta_gpu.cu.cc`

## E61 (Priority 2) - Decay Gated

**Forward:**
```
alpha = sigmoid(x @ W_alpha.T + b_alpha)
v = x @ W_v.T + b_v
h_new = alpha * h + (1 - alpha) * v
output = h_new * silu(h_new)
```

**Key:** Linear in h - potential for parallel scan optimization.

## E62 (Priority 3) - Selective Write

**Forward:**
```
k = sigmoid(x @ W_k.T + b_k)
v = tanh(x @ W_v.T + b_v)
h_new = (1 - k) * h + k * v
output = h_new * silu(h_new)
```

## E63m (Priority 4) - Matrix Nonlinear

**Forward:**
```
retrieved = tanh(S @ k)                    # Nonlinear retrieval!
v = tanh(W_r @ retrieved + W_x @ x + b)
S_new = alpha * S + (1 - alpha) * outer(v, k)
output = tanh(S_new @ q)
```

**Key:** S is [B, N, D] matrix state. Start with E63m-lite (N=32 or 64).

## Testing Requirements

For each kernel:
1. Forward matches PyTorch within rtol=1e-2
2. Backward matches PyTorch within rtol=1e-2
3. No NaN/Inf on random inputs
4. Works with bfloat16

## Build & Test
```bash
cd elman/cuda && ./build.sh
python -m elman.models.e63_nonlinear_delta  # Has built-in test
```

## Output Files
```
elman/cuda/lib/e61_decay_gated_gpu.cu.cc
elman/cuda/lib/e62_selective_write_gpu.cu.cc
elman/cuda/lib/e63_nonlinear_delta_gpu.cu.cc
elman/cuda/lib/e63m_matrix_nonlinear_gpu.cu.cc
+ header declarations + python bindings
```
