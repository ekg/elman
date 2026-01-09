# E20: Mamba2-Informed Elman

## Summary

E20 applies lessons learned from analyzing Mamba2's actual implementation.
The key insight: **Mamba2 has 256× more state than E1, with cheap transitions.**

## Key Findings from Mamba2 Analysis

### 1. State Size Difference (CRITICAL)

| Model | State per layer | Elements |
|-------|-----------------|----------|
| E1 (d=1024) | `h ∈ ℝ^d` | 1,024 |
| Mamba2 | `H ∈ ℝ^{nheads × headdim × d_state}` | 32 × 64 × 128 = **262,144** |

**Mamba2 has 256× more state!** This is the dominant factor.

### 2. Decay Structure

| Model | Decay | Parameters |
|-------|-------|------------|
| E14 (our attempt) | Per-row: `decay ∈ ℝ^d` | 1024 |
| Mamba2 | Per-head SCALAR: `decay ∈ ℝ^{nheads}` | 32 |

Mamba2 uses **32× fewer decay parameters**. All 64×128 = 8192 elements in a head share ONE decay.

### 3. Projection Structure

| Model | Projections | GEMMs |
|-------|-------------|-------|
| E14 | W_key, W_val, W_query, W_decay (separate) | 4 |
| Mamba2 | Combined in_proj, then split | 1 |

### 4. Nonlinearity Location

| Model | Nonlinearity in state update? |
|-------|------------------------------|
| E14 | YES: `key = tanh(W_key @ x)` |
| Mamba2 | NO: state update is purely linear |

Mamba2's nonlinearity comes from how `dt`, `B`, `C` are computed from `x` (via silu after in_proj), NOT from the state update itself.

## E20 Design

### Core Idea

Combine Mamba2's efficient structure with a minimal nonlinearity:

```python
# Single combined projection (one GEMM)
xzBCdt = in_proj(input)  # [B, T, 2*d_inner + 2*d_state + nheads]
x, z, B, C, dt = split(xzBCdt)

# Pre-activation
x = silu(x)  # [B, T, d_inner]
x = rearrange(x, "b t (h p) -> b t h p", h=nheads)  # [B, T, nheads, headdim]

# Input-dependent decay (scalar per head)
decay = sigmoid(dt)  # [B, T, nheads]

# State update (per timestep)
for t in range(T):
    # Outer product update with SCALAR decay broadcast
    # H[h,p,n] = decay[h] * H[h,p,n] + x[h,p] * B[n]
    H = decay[t].unsqueeze(-1).unsqueeze(-1) * H + outer(x[t], B[t])

    # Output via matmul
    y[t] = einsum("bhpn,bn->bhp", H, C[t])

# Gate and output
y = y * silu(z)  # E18-A style h-aware gating
output = out_proj(y)
```

### Key Differences from E14

| Aspect | E14 | E20 |
|--------|-----|-----|
| Decay | Per-row (d params) | Per-head scalar (nheads params) |
| Projections | 4 separate GEMMs | 1 combined in_proj |
| tanh | On key in state update | NONE in state update |
| State shape | [B, d, k] | [B, nheads, headdim, d_state] |
| Gate | x-only | h-aware (E18-A) |

### Key Differences from Mamba2

| Aspect | Mamba2 | E20 |
|--------|--------|-----|
| Parallel scan | Yes | No (sequential) |
| Inner loop | Tensor cores | CUDA kernel with k loop |
| Gating | RMSNorm + z gate | E18-A style h-aware silu gate |

## Implementation

### Configuration

```python
@dataclass
class E20Config:
    d_model: int = 1024
    nheads: int = 16        # Fewer than Mamba2's 32
    headdim: int = 64       # Same as Mamba2
    d_state: int = 64       # Smaller than Mamba2's 128
    expand: int = 2

    @property
    def d_inner(self):
        return self.d_model * self.expand

    @property
    def state_size(self):
        return self.nheads * self.headdim * self.d_state
```

With this config:
- State size: 16 × 64 × 64 = 65,536 (64× more than E1)
- Still less than Mamba2 (262K) but much more than E14

### Forward Pass

```python
class E20Layer(nn.Module):
    def __init__(self, cfg: E20Config):
        super().__init__()
        self.cfg = cfg

        # Combined input projection (ONE GEMM)
        # Output: [x, z, B, C, dt]
        d_proj = 2 * cfg.d_inner + 2 * cfg.d_state + cfg.nheads
        self.in_proj = nn.Linear(cfg.d_model, d_proj, bias=False)

        # Output projection
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)

        # dt bias (for decay initialization)
        self.dt_bias = nn.Parameter(torch.full((cfg.nheads,), 2.2))  # sigmoid ≈ 0.9

    def forward(self, x, H=None):
        B, T, D = x.shape
        cfg = self.cfg

        # Single projection
        proj = self.in_proj(x)  # [B, T, d_proj]

        # Split
        x_proj, z, B_proj, C_proj, dt = proj.split([
            cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state, cfg.nheads
        ], dim=-1)

        # Pre-activation
        x_proj = F.silu(x_proj)  # [B, T, d_inner]
        x_proj = x_proj.view(B, T, cfg.nheads, cfg.headdim)  # [B, T, nheads, headdim]

        # Initialize state
        if H is None:
            H = torch.zeros(B, cfg.nheads, cfg.headdim, cfg.d_state,
                          device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            # Input-dependent SCALAR decay per head
            decay = torch.sigmoid(dt[:, t] + self.dt_bias)  # [B, nheads]

            # State update: H = decay * H + x ⊗ B (outer product)
            # decay: [B, nheads] -> [B, nheads, 1, 1]
            # x_t: [B, nheads, headdim] -> [B, nheads, headdim, 1]
            # B_t: [B, d_state] -> [B, 1, 1, d_state]
            x_t = x_proj[:, t]  # [B, nheads, headdim]
            B_t = B_proj[:, t]  # [B, d_state]
            C_t = C_proj[:, t]  # [B, d_state]

            H = (decay.unsqueeze(-1).unsqueeze(-1) * H +
                 x_t.unsqueeze(-1) * B_t.unsqueeze(1).unsqueeze(1))
            # H: [B, nheads, headdim, d_state]

            # Output: y = H @ C
            y_t = torch.einsum("bhpn,bn->bhp", H, C_t)  # [B, nheads, headdim]
            y_t = y_t.view(B, cfg.d_inner)  # [B, d_inner]

            # E18-A style h-aware gating
            y_t = y_t * F.silu(z[:, t] + y_t)  # gate sees both z and h

            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)  # [B, T, d_inner]
        output = self.out_proj(output)  # [B, T, d_model]

        return output, H
```

### CUDA Kernel Structure

```cpp
// Pre-compute all projections (one big GEMM)
proj_all = x @ in_proj.T  // [T*B, d_proj]

// Split
x_all, z_all, B_all, C_all, dt_all = split(proj_all)

// Apply silu to x
x_all = silu(x_all)

// Per-timestep loop
for (int t = 0; t < T; ++t) {
    // Compute decay (scalar per head)
    // decay[h] = sigmoid(dt[t,h] + dt_bias[h])
    DecayKernel<<<...>>>(dt_t, dt_bias, decay);  // [B, nheads]

    // State update + output (fused)
    // For each (b, h, p):
    //   For each n:
    //     H[b,h,p,n] = decay[b,h] * H[b,h,p,n] + x[b,h,p] * B[b,n]
    //   y[b,h,p] = sum_n(H[b,h,p,n] * C[b,n])
    MatrixStateUpdateKernel<<<...>>>(
        H, x_t, B_t, C_t, decay, y_t
    );

    // Apply h-aware gate
    // output[b,i] = y[b,i] * silu(z[b,i] + y[b,i])
    HAwareGateKernel<<<...>>>(y_t, z_t, out_t);
}
```

## Parameter Count

For d_model=1024, nheads=16, headdim=64, d_state=64, expand=2:

```
in_proj:  1024 × (2*2048 + 2*64 + 16) = 1024 × 4240 = 4.3M
out_proj: 2048 × 1024 = 2.1M
dt_bias:  16

Per-layer: ~6.4M params
```

Compare to E1 at same d_model:
- E1: ~4M params per layer (in_proj + W_h + W_gate + out_proj)
- E20: ~6.4M params per layer (larger state)

## Expected Outcomes

### Optimistic

If state expansion is the key:
- E20 should significantly beat E18-A
- May approach Mamba2 performance
- 64× state expansion (vs E1) should help

### Pessimistic

If the issues are elsewhere:
- Sequential scan still limits throughput
- bf16 precision may hurt state accumulation
- Outer product kernel may be memory-bound

### Realistic

- E20 should beat E14 (simpler, more efficient)
- May beat E18-A by 0.02-0.05 nats
- Still won't match Mamba2 (no parallel scan)

## Ablations to Run

1. **State size sweep**: nheads × headdim × d_state
   - 8×32×32 = 8K (8× E1)
   - 16×64×64 = 64K (64× E1)
   - 32×64×128 = 256K (Mamba2 scale)

2. **Decay structure**:
   - Scalar per head (proposed)
   - Per-headdim (nheads × headdim params)
   - Per-element (full diagonal like E14)

3. **Nonlinearity location**:
   - None in state update (proposed)
   - tanh on x before outer product
   - tanh on H after update

4. **Gate structure**:
   - E18-A style (h-aware)
   - Mamba2 style (z only)
   - None

## Success Criteria

- **E20 > E18-A by ≥0.02 nats**: State expansion helps
- **E20 > E14**: Simpler structure wins
- **E20 within 0.05 nats of Mamba2**: Competitive despite sequential
