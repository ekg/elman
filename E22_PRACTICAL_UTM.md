# E22: Practical UTM (Nonlinear Elman + State Attention)

## Summary

E22 extends E21 with **periodic state attention** to achieve Universal Turing Machine (UTM) computational class. The key insight: TC¹ (nonlinear RNN) is necessary but not sufficient for UTM—you also need **arbitrary routing** where any state position can influence any other based on state content.

## The Theoretical Gap

### What Each Class Can Compute

| Class | Example Problems | Architecture |
|-------|------------------|--------------|
| TC⁰ | Threshold functions | Linear SSM (Mamba) |
| TC¹ | Parity, mod arithmetic | Nonlinear RNN (E21) |
| UTM | Sorting, graph algorithms, general computation | E22 |

### Why E21 Isn't UTM

E21's update:
```python
H_t = SiLU(α_t * H_{t-1} + B_t @ X_t.T)
```

- **Diagonal decay**: Position (n, p) only sees its own history
- **MIMO update**: Mixing is **input-controlled**, not **state-controlled**
- **No routing**: State position i cannot read from position j based on what's at i

For UTM, you need: **state-dependent routing** — the ability to "follow pointers" within the state.

### The Solution: State Attention

Add periodic self-attention over state dimensions:

```python
# Every K steps:
H_routed = StateAttention(H)  # Positions can read from each other
```

This provides the "head movement" capability of a Turing machine.

## Architecture

### Core Equation

```python
# Per timestep:
H_updated = SiLU(α_t * H_{t-1} + B_t @ X_t.T)  # E21: nonlinear MIMO

# Every K timesteps:
H_routed = H_updated + StateAttn(H_updated)     # E22: routing
```

### State Attention Design

The challenge: full attention over state is O((N×P)²) = O(4M) for typical sizes.

Solution: **Low-rank attention in head space**

```python
def state_attention(H):
    """
    H: [B, nheads, N, P] - state tensor

    Instead of attention over all N×P positions (expensive),
    do attention over heads, where each head's full N×P state
    is compressed to a low-rank representation.
    """
    B, heads, N, P = H.shape

    # Flatten each head's state
    H_flat = H.view(B, heads, N * P)  # [B, H, NP]

    # Project to low-rank queries/keys/values
    Q = H_flat @ W_q  # [B, H, R]
    K = H_flat @ W_k  # [B, H, R]
    V = H_flat @ W_v  # [B, H, R]

    # Attention: heads attend to each other
    attn = softmax(Q @ K.T / sqrt(R))  # [B, H, H]
    V_out = attn @ V  # [B, H, R]

    # Project back to state shape
    H_delta = V_out @ W_o  # [B, H, NP]
    H_delta = H_delta.view(B, heads, N, P)

    return H + H_delta  # Residual connection
```

This is O(H² × R + H × NP × R) ≈ O(H × NP × R) per attention step.

### Alternative: Attention Over State Positions

For maximum routing power, attend over the N dimension directly:

```python
def state_attention_over_N(H):
    """
    H: [B, nheads, N, P]

    Each of the N state positions can attend to all other N positions.
    This is O(N²) but N is small (32-64).
    """
    B, heads, N, P = H.shape

    # Reshape: treat P as features, N as sequence
    H_seq = H.permute(0, 1, 3, 2)  # [B, H, P, N] - P features, N positions
    H_seq = H_seq.reshape(B * heads, P, N)  # [BH, P, N]

    # Standard attention over N dimension
    # Q, K, V all derived from the P-dimensional features at each position
    Q = einsum('bpn,pd->bdn', H_seq, W_q)  # [BH, d_k, N]
    K = einsum('bpn,pd->bdn', H_seq, W_k)  # [BH, d_k, N]
    V = einsum('bpn,pd->bdn', H_seq, W_v)  # [BH, d_v, N]

    # Attention: position i attends to position j
    attn = softmax(einsum('bdi,bdj->bij', Q, K) / sqrt(d_k))  # [BH, N, N]
    V_out = einsum('bij,bdj->bdi', attn, V)  # [BH, d_v, N]

    # Project back
    H_delta = einsum('bdn,dp->bpn', V_out, W_o)  # [BH, P, N]
    H_delta = H_delta.reshape(B, heads, P, N).permute(0, 1, 3, 2)  # [B, H, N, P]

    return H + H_delta
```

Cost: O(N² × d_k + N × P × d_k) per head. For N=32, d_k=32, P=64: ~100K FLOPs per head.

## Implementation

### Configuration

```python
@dataclass
class E22Config:
    # E21 base config
    d_model: int = 1024
    nheads: int = 16
    d_state: int = 32         # N
    headdim: int = 64         # P
    mimo_rank: int = 8        # R for MIMO
    expand: int = 2

    # E22 state attention config
    state_attn_period: int = 8        # K: attend every K steps
    state_attn_type: str = "over_N"   # "over_heads" or "over_N"
    state_attn_dim: int = 32          # d_k for attention

    @property
    def d_inner(self):
        return self.d_model * self.expand

    @property
    def state_size(self):
        return self.nheads * self.d_state * self.headdim
```

### Full Layer Implementation

```python
class E22Layer(nn.Module):
    def __init__(self, cfg: E22Config):
        super().__init__()
        self.cfg = cfg

        # === E21 components ===

        # Combined input projection
        d_proj = (cfg.d_inner +                                    # x path
                  cfg.d_inner +                                    # z gate
                  cfg.nheads * cfg.d_state * cfg.mimo_rank +       # B
                  cfg.nheads * cfg.headdim * cfg.mimo_rank +       # X
                  cfg.nheads)                                      # α
        self.in_proj = nn.Linear(cfg.d_model, d_proj, bias=False)

        # Output projection
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)

        # Decay bias
        self.alpha_bias = nn.Parameter(torch.full((cfg.nheads,), 2.2))

        # === E22 state attention ===

        if cfg.state_attn_type == "over_N":
            # Attention over N dimension (more powerful)
            self.attn_q = nn.Linear(cfg.headdim, cfg.state_attn_dim, bias=False)
            self.attn_k = nn.Linear(cfg.headdim, cfg.state_attn_dim, bias=False)
            self.attn_v = nn.Linear(cfg.headdim, cfg.state_attn_dim, bias=False)
            self.attn_o = nn.Linear(cfg.state_attn_dim, cfg.headdim, bias=False)
        else:
            # Attention over heads (cheaper)
            state_dim = cfg.d_state * cfg.headdim
            self.attn_q = nn.Linear(state_dim, cfg.state_attn_dim, bias=False)
            self.attn_k = nn.Linear(state_dim, cfg.state_attn_dim, bias=False)
            self.attn_v = nn.Linear(state_dim, cfg.state_attn_dim, bias=False)
            self.attn_o = nn.Linear(cfg.state_attn_dim, state_dim, bias=False)

        self._attn_scale = cfg.state_attn_dim ** -0.5

    def state_attention_over_N(self, H):
        """
        H: [B, nheads, N, P]
        Attention over N positions, P is feature dim
        """
        B, heads, N, P = H.shape

        # [B, heads, N, P] -> [B*heads, N, P]
        H_flat = H.view(B * heads, N, P)

        # Project to Q, K, V
        Q = self.attn_q(H_flat)  # [BH, N, d_k]
        K = self.attn_k(H_flat)  # [BH, N, d_k]
        V = self.attn_v(H_flat)  # [BH, N, d_v]

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) * self._attn_scale  # [BH, N, N]
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        V_out = torch.bmm(attn, V)  # [BH, N, d_v]

        # Project back
        H_delta = self.attn_o(V_out)  # [BH, N, P]
        H_delta = H_delta.view(B, heads, N, P)

        return H + H_delta

    def state_attention_over_heads(self, H):
        """
        H: [B, nheads, N, P]
        Attention over heads, flattened N*P is feature dim
        """
        B, heads, N, P = H.shape

        # Flatten state per head
        H_flat = H.view(B, heads, N * P)  # [B, H, NP]

        # Project to Q, K, V
        Q = self.attn_q(H_flat)  # [B, H, d_k]
        K = self.attn_k(H_flat)  # [B, H, d_k]
        V = self.attn_v(H_flat)  # [B, H, d_v]

        # Attention: heads attend to each other
        scores = torch.bmm(Q, K.transpose(1, 2)) * self._attn_scale  # [B, H, H]
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        V_out = torch.bmm(attn, V)  # [B, H, d_v]

        # Project back
        H_delta = self.attn_o(V_out)  # [B, H, NP]
        H_delta = H_delta.view(B, heads, N, P)

        return H + H_delta

    def state_attention(self, H):
        if self.cfg.state_attn_type == "over_N":
            return self.state_attention_over_N(H)
        else:
            return self.state_attention_over_heads(H)

    def forward(self, x, H=None, step_offset=0):
        """
        x: [B, T, d_model]
        H: [B, nheads, N, P] or None
        step_offset: for tracking when to apply state attention
        """
        B, T, D = x.shape
        cfg = self.cfg

        # Single projection
        proj = self.in_proj(x)

        # Split
        sizes = [
            cfg.d_inner,
            cfg.d_inner,
            cfg.nheads * cfg.d_state * cfg.mimo_rank,
            cfg.nheads * cfg.headdim * cfg.mimo_rank,
            cfg.nheads
        ]
        x_path, z, B_flat, X_flat, alpha_raw = proj.split(sizes, dim=-1)

        # Reshape MIMO components
        B_proj = B_flat.view(B, T, cfg.nheads, cfg.d_state, cfg.mimo_rank)
        X_proj = X_flat.view(B, T, cfg.nheads, cfg.headdim, cfg.mimo_rank)

        # Initialize state
        if H is None:
            H = torch.zeros(B, cfg.nheads, cfg.d_state, cfg.headdim,
                          device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            global_t = step_offset + t

            # === E21: Nonlinear MIMO update ===

            # Decay (scalar per head)
            alpha = torch.sigmoid(-F.softplus(alpha_raw[:, t] + self.alpha_bias))

            # MIMO update
            B_t = B_proj[:, t]  # [B, H, N, R]
            X_t = X_proj[:, t]  # [B, H, P, R]
            update = torch.einsum('bhnr,bhpr->bhnp', B_t, X_t)

            # Nonlinear state transition
            H = F.silu(alpha[:, :, None, None] * H + update)

            # === E22: Periodic state attention ===

            if (global_t + 1) % cfg.state_attn_period == 0:
                H = self.state_attention(H)

            # === Output ===

            # Sum over N dimension (or use learned query)
            y_t = H.sum(dim=2)  # [B, H, P]
            y_t = y_t.view(B, cfg.d_inner)

            # E18-A style h-aware gating
            y_t = y_t * F.silu(z[:, t] + y_t)

            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)
        output = self.out_proj(output)

        return output, H
```

## CUDA Kernel Structure

### Fused State Attention Kernel

```cpp
// For "over_N" attention with small N (32-64), can fuse entirely

template<typename T, int N, int P, int D_K>
__global__ void StateAttentionOverN(
    T* __restrict__ H,           // [B, nheads, N, P] - in/out
    const T* __restrict__ W_q,   // [P, D_K]
    const T* __restrict__ W_k,   // [P, D_K]
    const T* __restrict__ W_v,   // [P, D_K]
    const T* __restrict__ W_o,   // [D_K, P]
    const int batch_size,
    const int nheads) {

    // Each block handles one (batch, head) pair
    const int bh = blockIdx.x;
    const int b = bh / nheads;
    const int h = bh % nheads;

    if (b >= batch_size) return;

    // Shared memory for Q, K, V, attention scores
    __shared__ float Q[N][D_K];
    __shared__ float K[N][D_K];
    __shared__ float V[N][D_K];
    __shared__ float scores[N][N];
    __shared__ float V_out[N][D_K];

    // Load H into shared memory and compute Q, K, V
    // Each thread handles one (n, p) or (n, d_k) element

    const int H_offset = ((b * nheads + h) * N) * P;

    // Step 1: Compute Q, K, V projections
    // Q[n, d] = sum_p H[n, p] * W_q[p, d]
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        for (int d = 0; d < D_K; d++) {
            float q_sum = 0, k_sum = 0, v_sum = 0;
            for (int p = 0; p < P; p++) {
                float h_val = static_cast<float>(H[H_offset + n * P + p]);
                q_sum += h_val * static_cast<float>(W_q[p * D_K + d]);
                k_sum += h_val * static_cast<float>(W_k[p * D_K + d]);
                v_sum += h_val * static_cast<float>(W_v[p * D_K + d]);
            }
            Q[n][d] = q_sum;
            K[n][d] = k_sum;
            V[n][d] = v_sum;
        }
    }
    __syncthreads();

    // Step 2: Compute attention scores
    // scores[i, j] = sum_d Q[i, d] * K[j, d] / sqrt(D_K)
    float scale = 1.0f / sqrtf(static_cast<float>(D_K));
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float max_score = -INFINITY;
        for (int j = 0; j < N; j++) {
            float score = 0;
            for (int d = 0; d < D_K; d++) {
                score += Q[i][d] * K[j][d];
            }
            scores[i][j] = score * scale;
            max_score = fmaxf(max_score, scores[i][j]);
        }

        // Softmax
        float sum_exp = 0;
        for (int j = 0; j < N; j++) {
            scores[i][j] = expf(scores[i][j] - max_score);
            sum_exp += scores[i][j];
        }
        for (int j = 0; j < N; j++) {
            scores[i][j] /= sum_exp;
        }
    }
    __syncthreads();

    // Step 3: Apply attention to V
    // V_out[i, d] = sum_j scores[i, j] * V[j, d]
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        for (int d = 0; d < D_K; d++) {
            float sum = 0;
            for (int j = 0; j < N; j++) {
                sum += scores[i][j] * V[j][d];
            }
            V_out[i][d] = sum;
        }
    }
    __syncthreads();

    // Step 4: Project back and add residual
    // H[n, p] += sum_d V_out[n, d] * W_o[d, p]
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        for (int p = 0; p < P; p++) {
            float delta = 0;
            for (int d = 0; d < D_K; d++) {
                delta += V_out[n][d] * static_cast<float>(W_o[d * P + p]);
            }
            H[H_offset + n * P + p] = static_cast<T>(
                static_cast<float>(H[H_offset + n * P + p]) + delta
            );
        }
    }
}
```

### Kernel Launch

```cpp
void LaunchStateAttention(
    void* H,
    const void* W_q, const void* W_k, const void* W_v, const void* W_o,
    int batch_size, int nheads, int N, int P, int D_K,
    cudaStream_t stream) {

    int num_blocks = batch_size * nheads;
    int threads_per_block = min(N, 256);

    // Template dispatch based on N, P, D_K
    if (N == 32 && P == 64 && D_K == 32) {
        StateAttentionOverN<__nv_bfloat16, 32, 64, 32><<<num_blocks, threads_per_block, 0, stream>>>(
            (bf16*)H, (bf16*)W_q, (bf16*)W_k, (bf16*)W_v, (bf16*)W_o,
            batch_size, nheads
        );
    }
    // ... other template instantiations
}
```

## Cost Analysis

### Per-Step FLOPs

For H=16, N=32, P=64, R=8, d_k=32, K=8:

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| E21 MIMO update | ~65K | Same as E21 |
| State attention (per call) | ~200K | O(N² × d_k + N × P × d_k) |
| State attention (amortized over K=8) | ~25K | 200K / 8 |
| **E22 total per step** | **~90K** | |
| Mamba-2 (reference) | ~50K | |

E22 is ~1.8× Mamba-2 cost. Acceptable for UTM capability.

### Memory

State attention adds:
- W_q, W_k, W_v: 3 × P × d_k = 3 × 64 × 32 = 6K params per layer
- W_o: d_k × P = 32 × 64 = 2K params per layer
- Total: 8K extra params per layer (~0.05% overhead)

## Ablations

### 1. Attention Period K

| K | Amortized Cost | Routing Frequency | Expected |
|---|----------------|-------------------|----------|
| 1 | +200K/step | Every step | Maximum UTM power, slowest |
| 4 | +50K/step | Every 4 steps | Good balance |
| 8 | +25K/step | Every 8 steps | Proposed default |
| 16 | +12K/step | Every 16 steps | Minimal routing |
| ∞ | 0 | Never | Falls back to TC¹ (E21) |

### 2. Attention Type

| Type | Cost | Routing Power |
|------|------|---------------|
| over_N | O(N² × d_k) | Position i reads from j within head |
| over_heads | O(H² × d_k) | Head i reads from head j |
| both | Higher | Maximum routing |

### 3. Attention Dimension d_k

| d_k | Cost | Capacity |
|-----|------|----------|
| 16 | Lower | May bottleneck |
| 32 | Medium | Proposed |
| 64 | Higher | More expressive |

## Computational Class Experiments

### Tier 1: TC⁰ vs TC¹ (Mamba vs E21)

```python
def tier1_experiments():
    """Tasks that separate linear (TC⁰) from nonlinear (TC¹)"""

    tasks = {
        'parity': {
            'fn': lambda n: (torch.randint(0, 2, (B, n)),
                            lambda x: x.sum(1) % 2),
            'lengths': [32, 64, 128, 256],
            'expected_tc0': 0.50,  # Random guessing
            'expected_tc1': 0.95,  # Should solve
        },
        'majority': {
            'fn': lambda n: (torch.randint(0, 2, (B, n)),
                            lambda x: (x.sum(1) > n/2).long()),
            'lengths': [32, 64, 128, 256],
            'expected_tc0': 0.60,  # Slight bias learning
            'expected_tc1': 0.95,
        },
        'mod_sum': {
            'fn': lambda n, k=7: (torch.randint(0, 10, (B, n)),
                                 lambda x: x.sum(1) % k),
            'lengths': [20, 50, 100],
            'expected_tc0': 0.20,  # ~1/k + some
            'expected_tc1': 0.90,
        },
        'first_one': {
            # Find position of first 1 in binary sequence
            'fn': lambda n: make_first_one_task(n),
            'lengths': [32, 64, 128],
            'expected_tc0': 0.30,
            'expected_tc1': 0.90,
        },
    }
    return tasks
```

### Tier 2: TC¹ vs UTM (E21 vs E22)

```python
def tier2_experiments():
    """Tasks that require routing - separate TC¹ from UTM"""

    tasks = {
        'permutation_composition': {
            # Given σ, τ as sequences, compute σ∘τ
            # Requires: read τ[i], use result to index σ
            'fn': make_permutation_task,
            'sizes': [4, 6, 8, 10],
            'expected_tc1': 0.40,  # Can memorize small cases
            'expected_utm': 0.85,
        },
        'indirect_addressing': {
            # SET x[i] = v, GET x[i], COPY x[i] = x[j]
            # COPY requires pointer chasing
            'fn': make_indirect_addressing_task,
            'n_vars': [4, 8, 16],
            'expected_tc1': 0.50,
            'expected_utm': 0.80,
        },
        'in_context_sort': {
            # Sort small arrays
            # Comparison sort needs routing
            'fn': make_sort_task,
            'sizes': [4, 6, 8],
            'expected_tc1': 0.30,
            'expected_utm': 0.70,
        },
        'bracket_depth': {
            # Track nesting depth of brackets
            # Easy for counter, tests state capacity
            'fn': make_bracket_task,
            'max_depth': [4, 8, 16],
            'expected_tc1': 0.70,  # Counter is TC¹
            'expected_utm': 0.90,
        },
        'graph_reachability': {
            # Is node t reachable from s?
            # BFS/DFS needs routing
            'fn': make_reachability_task,
            'nodes': [4, 6, 8],
            'expected_tc1': 0.40,
            'expected_utm': 0.75,
        },
    }
    return tasks
```

### Tier 3: Scaling Behavior

```python
def tier3_scaling():
    """
    Test the key UTM property: performance scales with generation budget

    Hypothesis:
    - TC⁰/TC¹: Performance plateaus regardless of sequence length
    - UTM: Performance improves with more "thinking" tokens
    """

    def test_with_scratchpad(model, task, scratchpad_lengths):
        """
        Give model extra tokens to "think" before answering.
        Input: [task tokens] [<scratchpad>] [<answer>]
        """
        results = {}
        for sp_len in scratchpad_lengths:
            acc = evaluate_with_scratchpad(model, task, sp_len)
            results[sp_len] = acc
        return results

    tasks = ['permutation_composition', 'sort', 'reachability']
    scratchpad_lengths = [0, 16, 32, 64, 128, 256]

    for task in tasks:
        for model in ['mamba2', 'e21', 'e22']:
            results = test_with_scratchpad(model, task, scratchpad_lengths)
            # Plot results[sp_len] vs sp_len
            # UTM signature: positive slope
            # TC⁰/TC¹ signature: flat line
```

### Task Implementations

```python
def make_permutation_task(n):
    """
    Input: [σ_0, σ_1, ..., σ_{n-1}, SEP, τ_0, τ_1, ..., τ_{n-1}]
    Output: [σ[τ[0]], σ[τ[1]], ..., σ[τ[n-1]]]
    """
    batch_size = 256
    sigma = torch.stack([torch.randperm(n) for _ in range(batch_size)])
    tau = torch.stack([torch.randperm(n) for _ in range(batch_size)])

    # Compose: (σ∘τ)[i] = σ[τ[i]]
    sigma_tau = torch.gather(sigma, 1, tau)

    # Format as sequence
    SEP = n  # Use n as separator token
    x = torch.cat([sigma, torch.full((batch_size, 1), SEP), tau], dim=1)
    y = sigma_tau

    return x.float(), y


def make_indirect_addressing_task(n_vars, seq_len=20):
    """
    Operations:
    - SET i v: x[i] = v
    - GET i: output x[i]
    - COPY i j: x[i] = x[j]  (the routing test!)

    Input: sequence of operations
    Output: sequence of GET results
    """
    # ... implementation ...


def make_sort_task(n):
    """
    Input: [a_0, a_1, ..., a_{n-1}]
    Output: sorted array
    """
    batch_size = 256
    x = torch.randint(0, 100, (batch_size, n))
    y = torch.sort(x, dim=1).values
    return x.float(), y


def make_reachability_task(n_nodes, density=0.3):
    """
    Input: [edge list] [SEP] [source] [SEP] [target]
    Output: 1 if reachable, 0 otherwise
    """
    # ... implementation ...
```

## Expected Results

### Language Modeling

| Model | Loss (50M tok) | Throughput | Notes |
|-------|----------------|------------|-------|
| Mamba-2 | 1.29 | 117K tok/s | Baseline |
| E21 | 1.32 | 100K tok/s | Nonlinear helps slightly |
| E22 (K=8) | 1.30 | 85K tok/s | Routing helps more |

### Computational Tasks

| Task | Mamba-2 (TC⁰) | E21 (TC¹) | E22 (UTM) |
|------|---------------|-----------|-----------|
| Parity | 50% | 95% | 95% |
| Mod-7 sum | 20% | 90% | 90% |
| Permutation comp | 15% | 40% | 85% |
| Indirect addressing | 25% | 45% | 80% |
| Sort (n=6) | 10% | 30% | 70% |
| Graph reach | 50% | 50% | 75% |

### Scaling Signature

```
Task: Permutation Composition (n=8)

Scratchpad Length:    0    16    32    64   128
Mamba-2:            15%   15%   15%   15%   15%  (flat)
E21:                40%   42%   43%   44%   44%  (slight improvement)
E22:                60%   70%   78%   83%   86%  (clear scaling)
```

The UTM signature: performance improves with generation budget.

## Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| E22 parity | >90% | Confirms TC¹ base works |
| E22 permutation | >80% | Confirms routing works |
| E22 > E21 on routing tasks | >20% gap | State attention helps |
| E22 scaling with scratchpad | Positive slope | UTM property verified |
| E22 throughput | >0.6× Mamba-2 | Acceptable slowdown |

## Implementation Plan

### Phase 1: E22 Layer (Python)
1. Implement state attention module
2. Integrate with E21 base
3. Verify gradients flow correctly
4. Test on small scale (d=256)

### Phase 2: Computational Tasks
1. Implement tier 1 tasks (TC⁰ vs TC¹)
2. Verify E21 beats Mamba-2 on parity
3. Implement tier 2 tasks (TC¹ vs UTM)
4. Compare E21 vs E22 on routing tasks

### Phase 3: CUDA Kernel
1. Fused state attention kernel
2. Benchmark vs Python
3. Optimize for small N (32-64)

### Phase 4: Full Evaluation
1. Language modeling comparison
2. Full tier 1/2/3 benchmark suite
3. Ablations (K, d_k, attention type)
4. Scaling experiments with scratchpad

## Summary

E22 = E21 + Periodic State Attention

| Aspect | E21 | E22 |
|--------|-----|-----|
| Base | Nonlinear MIMO | Nonlinear MIMO |
| Routing | None (diagonal + low-rank) | Attention every K steps |
| Computational class | TC¹ | UTM |
| Cost | ~65K FLOPs/step | ~90K FLOPs/step |
| Can solve | Parity, mod arithmetic | + Sorting, graph algorithms |

The key insight: **TC¹ is not UTM**. For true universal computation, you need state-dependent routing. Periodic state attention provides this at ~1.4× cost overhead.

If E22 works, you have an architecture that is:
1. More expressive than Transformers (can run indefinitely)
2. More efficient than Transformers (O(1) memory in state)
3. Provably UTM (with routing + nonlinearity + unbounded generation)
