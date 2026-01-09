# E21: Structured Elman (MIMO with Nonlinear Mixing)

## Summary

E21 combines Mamba2's efficient state structure with the **nonlinear state transition** that gives Elman its theoretical expressivity advantage. The key insight: a nonlinear state with N=32 may achieve "effective state capacity" matching linear N=128.

## The Core Hypothesis

**Linear SSMs (Mamba2)**: State evolution is linear, meaning:
- Two different histories that produce the same state are forever indistinguishable
- Requires massive state (N=128) to preserve history distinctions
- Computational class: TC⁰ (cannot compute parity)

**Nonlinear SSMs (Elman)**: State evolution through nonlinearity (tanh/silu):
- Different histories can be "folded" into distinct attractor basins
- Smaller state can preserve more history information
- Computational class: TC¹+ (can compute parity, modular arithmetic)

**E21 Goal**: Get Mamba2's efficient structure with Elman's nonlinear advantage.

## Architecture

### State Structure (like Mamba2)

```
State: H ∈ ℝ^{batch × nheads × d_state × headdim}
     = [B, H, N, P]

For typical config: H=16, N=32, P=64
State elements: 16 × 32 × 64 = 32,768 (32× E1, 1/8 Mamba2)
```

### The Key Equation

```python
# Mamba2 (LINEAR):
H_t = α_t * H_{t-1} + B_t ⊗ X_t

# E21 (NONLINEAR):
H_t = SiLU(α_t * H_{t-1} + B_t @ X_t.T)
      ^^^^^
      THE DIFFERENCE
```

The SiLU wrapping the entire update creates nonlinear state dynamics.

### MIMO Formulation

Instead of outer product (rank-1), use rank-R MIMO update:

```python
# Rank-1 (Mamba2 style):
update = outer(B_t, X_t)  # [N] ⊗ [P] → [N, P]

# Rank-R (E21 MIMO):
update = einsum('bnr,bpr->bnp', B_t, X_t)  # [N, R] @ [P, R].T → [N, P]
```

With R=8 or R=16, this allows richer per-step updates while staying efficient.

## Implementation

### Configuration

```python
@dataclass
class E21Config:
    d_model: int = 1024
    nheads: int = 16          # H: number of heads
    d_state: int = 32         # N: state dimension (smaller than Mamba2's 128!)
    headdim: int = 64         # P: head dimension
    mimo_rank: int = 8        # R: rank of MIMO update
    expand: int = 2

    @property
    def d_inner(self):
        return self.d_model * self.expand

    @property
    def state_size(self):
        return self.nheads * self.d_state * self.headdim
```

### Forward Pass

```python
class E21Layer(nn.Module):
    def __init__(self, cfg: E21Config):
        super().__init__()
        self.cfg = cfg

        # Combined input projection (ONE GEMM)
        # Output: [x, z, B, X, α]
        d_proj = (cfg.d_inner +                    # x for output path
                  cfg.d_inner +                    # z for gate
                  cfg.nheads * cfg.d_state * cfg.mimo_rank +  # B: [H, N, R]
                  cfg.nheads * cfg.headdim * cfg.mimo_rank +  # X: [H, P, R]
                  cfg.nheads)                      # α: [H] scalar per head
        self.in_proj = nn.Linear(cfg.d_model, d_proj, bias=False)

        # Output projection
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)

        # Decay bias (initialize for ~0.9 retention)
        self.alpha_bias = nn.Parameter(torch.full((cfg.nheads,), 2.2))

    def forward(self, x, H=None):
        B, T, D = x.shape
        cfg = self.cfg

        # Single projection
        proj = self.in_proj(x)  # [B, T, d_proj]

        # Split
        sizes = [
            cfg.d_inner,
            cfg.d_inner,
            cfg.nheads * cfg.d_state * cfg.mimo_rank,
            cfg.nheads * cfg.headdim * cfg.mimo_rank,
            cfg.nheads
        ]
        x_path, z, B_flat, X_flat, alpha_raw = proj.split(sizes, dim=-1)

        # Reshape for MIMO
        B_proj = B_flat.view(B, T, cfg.nheads, cfg.d_state, cfg.mimo_rank)
        X_proj = X_flat.view(B, T, cfg.nheads, cfg.headdim, cfg.mimo_rank)

        # Initialize state
        if H is None:
            H = torch.zeros(B, cfg.nheads, cfg.d_state, cfg.headdim,
                          device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            # Input-dependent decay (scalar per head)
            alpha = torch.sigmoid(-F.softplus(alpha_raw[:, t] + self.alpha_bias))
            # Shape: [B, H]

            # MIMO update: B @ X.T with rank R
            B_t = B_proj[:, t]  # [B, H, N, R]
            X_t = X_proj[:, t]  # [B, H, P, R]

            # update[b,h,n,p] = sum_r B[b,h,n,r] * X[b,h,p,r]
            update = torch.einsum('bhnr,bhpr->bhnp', B_t, X_t)

            # THE KEY: Nonlinear state transition
            H = F.silu(
                alpha[:, :, None, None] * H +  # decay: [B, H, 1, 1] * [B, H, N, P]
                update                          # update: [B, H, N, P]
            )

            # Output via matmul with C (reuse X as query)
            # y[b,h,p] = sum_n H[b,h,n,p] (simple sum, or use learned C)
            y_t = H.sum(dim=2)  # [B, H, P]
            y_t = y_t.view(B, cfg.d_inner)  # [B, d_inner]

            # E18-A style h-aware gating
            y_t = y_t * F.silu(z[:, t] + y_t)

            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)  # [B, T, d_inner]
        output = self.out_proj(output)  # [B, T, d_model]

        return output, H
```

### Variant: With Learned Output Query (E21-Q)

```python
# Add C projection for more expressive output
C_flat = ...  # add to in_proj output
C_proj = C_flat.view(B, T, cfg.nheads, cfg.d_state)

# Output: y = C @ H (instead of sum)
y_t = torch.einsum('bhn,bhnp->bhp', C_proj[:, t], H)
```

## CUDA Implementation

### Kernel Structure

```cpp
// Pre-compute all projections (one big GEMM)
proj_all = x @ in_proj.T  // [T*B, d_proj]

// Split and reshape
x_path, z, B_all, X_all, alpha_all = split_and_reshape(proj_all)

// Per-timestep loop
for (int t = 0; t < T; ++t) {
    // Compute decay (scalar per head)
    // alpha[h] = sigmoid(-softplus(alpha_raw[t,h] + alpha_bias[h]))
    DecayKernel<<<...>>>(alpha_raw_t, alpha_bias, alpha);  // [B, H]

    // MIMO update: einsum('bhnr,bhpr->bhnp', B_t, X_t)
    // This is a batched matrix multiply per head
    MIMOUpdateKernel<<<...>>>(B_t, X_t, update);  // [B, H, N, P]

    // Nonlinear state update (fused)
    // H = silu(alpha * H + update)
    NonlinearStateUpdateKernel<<<...>>>(
        H, alpha, update, H  // in-place update
    );

    // Output reduction + gating
    OutputKernel<<<...>>>(H, z_t, y_t);
}
```

### Fused Nonlinear State Update Kernel

```cpp
template<typename T>
__global__ void NonlinearStateUpdateKernel(
    T* __restrict__ H,           // [B, nheads, d_state, headdim]
    const T* __restrict__ alpha, // [B, nheads]
    const T* __restrict__ update,// [B, nheads, d_state, headdim]
    const int batch_size,
    const int nheads,
    const int d_state,
    const int headdim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * nheads * d_state * headdim;

    if (idx < total) {
        // Decompose index
        const int p = idx % headdim;
        const int n = (idx / headdim) % d_state;
        const int h = (idx / (headdim * d_state)) % nheads;
        const int b = idx / (headdim * d_state * nheads);

        // Get scalar decay for this head
        float a = static_cast<float>(alpha[b * nheads + h]);
        float h_val = static_cast<float>(H[idx]);
        float u_val = static_cast<float>(update[idx]);

        // Nonlinear update: H = silu(alpha * H + update)
        float pre_act = a * h_val + u_val;
        float sigmoid_val = 1.0f / (1.0f + expf(-pre_act));
        float silu_val = pre_act * sigmoid_val;

        H[idx] = static_cast<T>(silu_val);
    }
}
```

### MIMO Update Kernel

```cpp
// For each (b, h), compute: update[n,p] = sum_r B[n,r] * X[p,r]
// This is essentially a batched gemm: B @ X.T
// Can use cublas batched gemm for efficiency

template<typename T>
__global__ void MIMOUpdateKernel(
    const T* __restrict__ B,      // [B, H, N, R]
    const T* __restrict__ X,      // [B, H, P, R]
    T* __restrict__ update,       // [B, H, N, P]
    const int batch_size,
    const int nheads,
    const int d_state,
    const int headdim,
    const int mimo_rank) {

    // Each thread computes one (b, h, n, p) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * nheads * d_state * headdim;

    if (idx < total) {
        const int p = idx % headdim;
        const int n = (idx / headdim) % d_state;
        const int h = (idx / (headdim * d_state)) % nheads;
        const int b = idx / (headdim * d_state * nheads);

        float sum = 0.0f;
        const int B_base = ((b * nheads + h) * d_state + n) * mimo_rank;
        const int X_base = ((b * nheads + h) * headdim + p) * mimo_rank;

        #pragma unroll
        for (int r = 0; r < mimo_rank; ++r) {
            sum += static_cast<float>(B[B_base + r]) *
                   static_cast<float>(X[X_base + r]);
        }

        update[idx] = static_cast<T>(sum);
    }
}
```

## Backward Pass

### Gradient Through Nonlinear State

The key complexity: gradient must flow through the SiLU at each timestep.

```python
# Forward:
pre_act = alpha * H_prev + update
H = silu(pre_act)

# Backward:
# dsilu/dpre_act = sigmoid(pre_act) * (1 + pre_act * (1 - sigmoid(pre_act)))
d_pre_act = dH * dsilu_derivative(pre_act)

# Gradients to components:
d_alpha = (d_pre_act * H_prev).sum(dim=[-2, -1])  # reduce over N, P
d_H_prev = d_pre_act * alpha[..., None, None]
d_update = d_pre_act
```

### Chunked BPTT

For long sequences, use chunked backpropagation:

```python
def chunked_forward(x, chunk_size=64):
    B, T, D = x.shape
    H = None
    outputs = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = x[:, start:end]

        if H is not None:
            H = H.detach()  # Stop gradient at chunk boundary

        out, H = e21_layer(chunk, H)
        outputs.append(out)

    return torch.cat(outputs, dim=1), H
```

Chunk size 64-128 balances memory vs gradient flow.

## Parameter Counts

For d_model=1024, H=16, N=32, P=64, R=8:

```
in_proj:  1024 × (2048 + 2048 + 16*32*8 + 16*64*8 + 16)
        = 1024 × (2048 + 2048 + 4096 + 8192 + 16)
        = 1024 × 16400
        = 16.8M

out_proj: 2048 × 1024 = 2.1M

alpha_bias: 16

Per-layer: ~18.9M params
```

Compare to:
- E1 at d=1024: ~4M params per layer
- Mamba2 at d=1024: ~6.5M params per layer
- E20 proposed: ~6.4M params per layer

E21 is larger due to MIMO rank parameters. Can reduce with smaller R.

### Smaller Config (E21-S)

```python
E21SConfig:
    nheads = 16
    d_state = 32
    headdim = 64
    mimo_rank = 4  # Reduced from 8

in_proj: 1024 × (2048 + 2048 + 2048 + 4096 + 16) = 10.5M
```

## Ablations

### 1. Nonlinearity Type

```python
# E21-A: SiLU (proposed)
H = F.silu(alpha * H_prev + update)

# E21-B: Tanh
H = torch.tanh(alpha * H_prev + update)

# E21-C: GeLU
H = F.gelu(alpha * H_prev + update)

# E21-L: Linear (ablation - should match Mamba2 scaling)
H = alpha * H_prev + update
```

### 2. MIMO Rank Sweep

| Config | R | State Update Cost | Expected |
|--------|---|-------------------|----------|
| E21-R1 | 1 | O(N + P) | Like Mamba2 |
| E21-R4 | 4 | O(4(N + P)) | Moderate |
| E21-R8 | 8 | O(8(N + P)) | Rich |
| E21-R16 | 16 | O(16(N + P)) | Very rich |

### 3. State Size vs Nonlinearity

Test the core hypothesis: nonlinear small state vs linear large state

| Config | N | Nonlinear? | Hypothesis |
|--------|---|------------|------------|
| E21-N32 | 32 | Yes | Should match E21-L-N128 |
| E21-N64 | 64 | Yes | Should beat E21-L-N128 |
| E21-L-N128 | 128 | No | Mamba2-like baseline |

### 4. Nonlinearity Location

```python
# E21: On full state (proposed)
H = silu(alpha * H_prev + update)

# E21-U: Only on update
H = alpha * H_prev + silu(update)

# E21-D: Only on decay path
H = silu(alpha * H_prev) + update
```

## Expressivity Tests

Beyond language modeling loss, test computational expressivity:

### 1. Parity Task

```python
# Input: sequence of 0s and 1s
# Output: XOR of all inputs (parity)
# Linear SSMs CANNOT solve this (TC⁰)
# Nonlinear SHOULD solve this (TC¹)

def parity_dataset(seq_len=100, n_samples=10000):
    x = torch.randint(0, 2, (n_samples, seq_len))
    y = x.sum(dim=1) % 2
    return x.float(), y
```

### 2. Modular Arithmetic

```python
# Input: sequence of digits
# Output: sum mod k
# Tests ability to maintain finite-state information

def modular_sum_dataset(seq_len=50, mod=7):
    x = torch.randint(0, 10, (n_samples, seq_len))
    y = x.sum(dim=1) % mod
    return x.float(), y
```

### 3. Bracket Matching

```python
# Input: sequence with ( and )
# Output: whether brackets are balanced
# Requires counter - nonlinear should help

def bracket_matching_dataset(max_depth=10):
    # Generate balanced and unbalanced sequences
    ...
```

## Expected Outcomes

### Optimistic

If nonlinear state expansion is the key insight:
- E21 with N=32 matches Mamba2 with N=128 on language modeling
- E21 solves parity/modular arithmetic where Mamba2 fails
- 2-4× smaller state for same quality

### Realistic

- E21 beats E18-A by 0.02-0.05 nats (MIMO + nonlinear helps)
- E21 is 1.5-2× slower than Mamba2 (no parallel scan)
- E21 shows clear advantage on expressivity tests

### Pessimistic

- Nonlinearity in state hurts gradient flow too much
- Chunked BPTT limits effective sequence length
- No improvement over E20 despite added complexity

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Loss vs E18-A | ≤ -0.02 nats | Improvement from MIMO+nonlinear |
| Loss vs Mamba2 | ≤ +0.05 nats | Competitive despite smaller state |
| Parity accuracy | >90% | Linear SSMs get ~50% |
| Modular arith | >80% | Linear SSMs struggle |
| Throughput | ≥ 0.5× Mamba2 | Acceptable slowdown |

## Implementation Plan

### Phase 1: Basic E21

1. Implement E21Layer in Python
2. Test on small scale (d=256, 1 layer)
3. Verify gradient flow works
4. Confirm parity task advantage

### Phase 2: CUDA Kernel

1. Fused nonlinear state update kernel
2. MIMO update kernel (or batched cublas)
3. Benchmark vs Python implementation

### Phase 3: Full Training

1. 50M token comparison (10 min)
2. Ablation sweep (nonlinearity type, rank, state size)
3. Expressivity benchmarks

### Phase 4: Scale Up

1. 400M+ token runs
2. Compare wall-clock times
3. Tune chunk size for optimal gradient flow

## Comparison Summary

| Aspect | E1 | Mamba2 | E21 |
|--------|-------|--------|-----|
| State size | d | H×N×P | H×N×P |
| State transition | Nonlinear (tanh) | Linear | Nonlinear (silu) |
| Update type | Rank-1 (W_h @ h) | Rank-1 (outer) | Rank-R (MIMO) |
| Decay | Per-element | Scalar/head | Scalar/head |
| Parallel scan | No | Yes | No |
| TC class | TC¹+ | TC⁰ | TC¹+ |
| Effective capacity | High (nonlinear) | Low (linear) | High (nonlinear) |

## Key Insight

The fundamental bet of E21:

**A nonlinear RNN with N=32 state can match a linear SSM with N=128 state** because:

1. Nonlinear dynamics create "attractor basins" that encode discrete history distinctions
2. Linear dynamics must use raw state dimensions to preserve history
3. Effective state capacity scales with computational expressivity, not just dimensions

If true, E21 gets Mamba2's efficiency (scalar decay, MIMO updates) with Elman's expressivity (nonlinear dynamics) in a smaller state space.
