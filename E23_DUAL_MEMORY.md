# E23: Dual-Memory Elman

## Summary

E23 introduces a two-tier memory architecture that provides massive state expansion with modest compute overhead. The key insight: **separate large linear storage (tape) from small nonlinear computation (working memory)**, connected via lightweight attention.

- **Tape**: Large, cheap, linear accumulator (like RAM)
- **Working Memory**: Small, expensive, full routing (like CPU registers)

This gives 64-130× more state than E1 at similar or lower cost.

## Motivation

The gap between E1 and Mamba2 comes partly from state size:
- E1: 1K state elements
- Mamba2: 262K state elements (256× more)

But naively scaling E1's hidden state is expensive: doubling D quadruples W_h cost (D²).

E23 decouples storage from computation:
- Tape operations are O(N × D) - linear in tape size
- Working operations are O(D²) - quadratic but D is fixed
- Total state: N × D + D, but cost doesn't explode

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  TAPE (Linear Memory)                               │
│  h_tape ∈ ℝ^{N × D}                                │
│                                                     │
│  - N slots, each of dimension D                     │
│  - Learnable per-slot decay α ∈ ℝ^N                │
│  - Updated via rank-1 outer products               │
│  - Read/written via dot-product attention          │
└─────────────────────────────────────────────────────┘
         ↑ write                    ↓ read
         │ O(N × D)                 │ O(N × D)
         │                          │
┌─────────────────────────────────────────────────────┐
│  WORKING MEMORY                                     │
│  h_work ∈ ℝ^D                                      │
│                                                     │
│  - Standard Elman update with tape input           │
│  - h = tanh(W_h @ h + W_x @ x + read + b)         │
│  - Full expressivity via W_h (D × D)              │
└─────────────────────────────────────────────────────┘
                      ↓
                   Output
```

## Core Equations

```python
def e23_step(x_t, h_tape, h_work, params):
    """
    x_t: [B, D_in] - input
    h_tape: [B, N, D] - tape memory (N slots of dimension D)
    h_work: [B, D] - working memory

    Returns: y_t, h_tape_new, h_work_new
    """

    # 1. TAPE DECAY (learnable per-slot)
    #    α ∈ (0, 1) learned, initialized near 1 for long memory
    h_tape = params.α[:, None] * h_tape  # [N, 1] * [B, N, D] → [B, N, D]

    # 2. TAPE UPDATE (rank-1 from input)
    #    Input writes to tape via outer product
    key = params.W_k @ x_t      # [B, D_in] → [B, N]
    value = params.W_v @ x_t    # [B, D_in] → [B, D]
    h_tape = h_tape + key[:, :, None] * value[:, None, :]  # outer product

    # 3. READ (working memory queries tape)
    #    Simple dot-product attention, no learned projections
    scores = einsum('bnd,bd->bn', h_tape, h_work) / sqrt(D)  # [B, N]
    read_attn = softmax(scores, dim=-1)  # [B, N]
    read = einsum('bn,bnd->bd', read_attn, h_tape)  # [B, D]

    # 4. WORKING MEMORY UPDATE (standard Elman + tape read)
    pre_act = (params.W_h @ h_work.T).T + (params.W_x @ x_t.T).T + read + params.b
    h_work = tanh(pre_act)  # [B, D]

    # 5. WRITE (working memory updates tape)
    #    Tape attends to working memory, gets weighted update
    write_scores = einsum('bnd,bd->bn', h_tape, h_work) / sqrt(D)  # [B, N]
    write_attn = softmax(write_scores, dim=-1)  # [B, N]
    h_tape = h_tape + write_attn[:, :, None] * h_work[:, None, :]  # outer product

    # 6. OUTPUT
    y = params.W_out @ h_work.T  # [D_out, B]

    return y.T, h_tape, h_work
```

## Parameters

### Learnable Parameters per Layer

| Parameter | Shape | Count (D=1024, N=64) |
|-----------|-------|----------------------|
| α (decay) | [N] | 64 |
| W_k (tape key) | [N, D_in] | 64K |
| W_v (tape value) | [D, D_in] | 1M |
| W_h (recurrence) | [D, D] | 1M |
| W_x (input) | [D, D_in] | 1M |
| b (bias) | [D] | 1K |
| W_out (output) | [D_out, D] | 1M |
| **Total** | | **~4M** |

Compare to E1: ~3M per layer (no W_k, W_v)

### Hyperparameters

| Name | Symbol | Default | Notes |
|------|--------|---------|-------|
| Working dimension | D | 1024 | Same as E1 |
| Number of tape slots | N | 64 | Main state expansion knob |
| Input dimension | D_in | D | Usually same as D |
| Output dimension | D_out | D | Usually same as D |

## Cost Analysis

### Per-Step FLOPs (D=1024, N=64)

| Operation | Formula | FLOPs | Notes |
|-----------|---------|-------|-------|
| Tape decay | N × D | 64K | Element-wise |
| Tape update (key) | D_in × N | 64K | |
| Tape update (value) | D_in × D | 1M | |
| Tape update (outer) | N × D | 64K | |
| Read (scores) | N × D | 64K | |
| Read (weighted sum) | N × D | 64K | |
| Working (W_h) | D × D | 1M | |
| Working (W_x) | D × D | 1M | |
| Write (scores) | N × D | 64K | |
| Write (outer) | N × D | 64K | |
| Output | D × D | 1M | |
| **Total** | | **~4.5M** | |

### Comparison

| Model | State | FLOPs/step | State/FLOP |
|-------|-------|------------|------------|
| E1 (D=1024) | 1K | 2M | 0.0005 |
| **E23 (D=1024, N=64)** | **65K** | **4.5M** | **0.014** |
| Mamba2 | 262K | 500K | 0.52 |

E23 is **28× more state-efficient** than E1, though still behind Mamba2.

### Cost vs State Tradeoff

| Config | D | N | State | FLOPs | vs E1 State | vs E1 Cost |
|--------|---|---|-------|-------|-------------|------------|
| E1 | 1024 | - | 1K | 2M | 1× | 1× |
| E23-S | 1024 | 32 | 33K | 3.5M | 33× | 1.75× |
| **E23-M** | **1024** | **64** | **65K** | **4.5M** | **65×** | **2.25×** |
| E23-L | 1024 | 128 | 131K | 6.5M | 131× | 3.25× |
| E23-XL | 1024 | 256 | 262K | 10.5M | 262× | 5.25× |

## Initialization

### Decay (α)

Initialize for long-term memory retention:

```python
# Initialize α such that sigmoid(α_raw) ≈ 0.99
# After 100 steps, retention is 0.99^100 ≈ 0.37
α_raw = nn.Parameter(torch.full((N,), 4.6))  # sigmoid(4.6) ≈ 0.99

# In forward:
α = torch.sigmoid(α_raw)  # Constrain to (0, 1)
```

Alternative: different decay rates per slot for multi-scale memory:

```python
# Slot 0: fast decay (0.9), Slot N-1: slow decay (0.999)
α_init = torch.linspace(0.9, 0.999, N)
α_raw = torch.logit(α_init)  # inverse sigmoid
```

### Tape

Initialize to zero:

```python
h_tape = torch.zeros(batch_size, N, D)
```

### Working Memory

Initialize to zero (like E1):

```python
h_work = torch.zeros(batch_size, D)
```

### Weight Matrices

Standard initialization:

```python
# W_h: careful initialization for gradient flow
nn.init.orthogonal_(W_h)
W_h *= 0.9  # Scale down slightly

# W_x, W_k, W_v, W_out: Xavier
nn.init.xavier_uniform_(W_x)
nn.init.xavier_uniform_(W_k)
nn.init.xavier_uniform_(W_v)
nn.init.xavier_uniform_(W_out)
```

## CUDA Implementation

### Kernel Structure

```cpp
// Pre-compute input projections (batched GEMMs)
key_all = x_all @ W_k.T      // [T*B, N]
value_all = x_all @ W_v.T    // [T*B, D]
Wx_all = x_all @ W_x.T       // [T*B, D]

// Per-timestep loop
for (int t = 0; t < T; ++t) {
    // 1. Tape decay (element-wise, parallelizes trivially)
    TapeDecayKernel<<<...>>>(h_tape, alpha, B, N, D);

    // 2. Tape update (outer product)
    TapeUpdateKernel<<<...>>>(h_tape, key_t, value_t, B, N, D);

    // 3. Read attention
    ReadScoresKernel<<<...>>>(h_tape, h_work, read_scores, B, N, D);
    SoftmaxKernel<<<...>>>(read_scores, read_attn, B, N);
    ReadKernel<<<...>>>(h_tape, read_attn, read, B, N, D);

    // 4. Working update (GEMM + fused add + tanh)
    Rh = h_work @ W_h.T       // GEMM
    FusedTanhKernel<<<...>>>(Wx_t, Rh, read, b, h_work, B, D);

    // 5. Write attention
    WriteScoresKernel<<<...>>>(h_tape, h_work, write_scores, B, N, D);
    SoftmaxKernel<<<...>>>(write_scores, write_attn, B, N);
    WriteKernel<<<...>>>(h_tape, write_attn, h_work, B, N, D);

    // 6. Output
    y_t = h_work @ W_out.T    // GEMM
}
```

### Fused Kernels

Several operations can be fused:

1. **Tape decay + update**: Single kernel that decays then adds
2. **Read scores + softmax**: Fuse into one kernel
3. **Write scores + softmax + update**: Fuse all write operations

### Memory Layout

```
Tape:    [B, N, D] - batch, slots, features
Working: [B, D] - batch, features
```

For coalesced memory access, ensure D is the innermost dimension.

## Backward Pass

### Gradient Flow

The dual-memory structure has two gradient paths:

1. **Through working memory**: Standard BPTT through h_work
2. **Through tape**: Gradients flow through read/write attention

```
Forward:  x_t → tape_update → read → h_work → write → tape
Backward: Same path in reverse, with attention gradient distribution
```

### Key Gradients

**Read attention backward:**
```python
# Forward: read = attn @ h_tape
# Backward:
d_attn = d_read @ h_tape.T      # [B, N]
d_h_tape += attn.T @ d_read     # [B, N, D]
```

**Write attention backward:**
```python
# Forward: h_tape += outer(attn, h_work)
# Backward:
d_attn = (d_h_tape * h_work[:, None, :]).sum(dim=-1)  # [B, N]
d_h_work += (d_h_tape * attn[:, :, None]).sum(dim=1)  # [B, D]
```

**Softmax backward:**
```python
# Standard softmax gradient
d_scores = attn * (d_attn - (attn * d_attn).sum(dim=-1, keepdim=True))
```

## Variants

### E23-A: Attention with Learned Projections

Add learned Q, K, V projections for more expressive attention:

```python
# Read
Q_read = W_q_read @ h_work      # [B, D] → [B, d_k]
K_read = W_k_read @ h_tape      # [B, N, D] → [B, N, d_k]
V_read = W_v_read @ h_tape      # [B, N, D] → [B, N, d_v]
scores = Q_read @ K_read.T / sqrt(d_k)
read = softmax(scores) @ V_read
```

Cost: Adds d_k × D + d_k × D + d_v × D per read/write

### E23-B: Multi-Head Tape

Split tape into H heads, each with N/H slots:

```python
h_tape: [B, H, N//H, D//H]  # H heads, fewer slots per head, smaller dimension
```

This allows different heads to specialize (some fast, some slow).

### E23-C: Gated Write

Add a gate to control how much working memory affects tape:

```python
write_gate = sigmoid(W_wg @ h_work)  # [B, 1] scalar gate
h_tape = h_tape + write_gate * outer(write_attn, h_work)
```

### E23-D: Linear Tape (No Decay Nonlinearity)

Keep tape completely linear (like Mamba2's state):

```python
# No sigmoid on alpha, just clamp to (0, 1)
α = torch.clamp(α_raw, 0.01, 0.99)
```

## Theoretical Properties

### Computational Class

E23 maintains **TC¹** or higher because:
- Working memory has nonlinear activation (tanh)
- This provides composition depth via recurrence

With periodic self-attention on tape (E22-style), could achieve **UTM**.

### Memory Capacity

- **Tape**: N × D elements of linear storage
- **Working**: D elements of nonlinear "register"
- **Effective capacity**: Higher than raw count due to attention-based addressing

### Gradient Flow

Two paths help gradient flow:
1. **Direct path**: Through W_h @ h_work (standard RNN)
2. **Tape path**: Through read/write attention (longer range, but softer)

The tape path allows gradients to skip many timesteps via attention weights.

## Experimental Plan

### Phase 1: Baseline Comparison

Compare E23-M (D=1024, N=64) against:
- E1 (D=1024): Same working memory, no tape
- E18-A: Current best E-series
- Mamba2: State-of-the-art SSM

Metrics: Loss, throughput, memory usage

### Phase 2: Ablations

| Ablation | What we learn |
|----------|---------------|
| N = 32, 64, 128, 256 | How much does tape size help? |
| No write-back | Is bidirectional tape access needed? |
| No decay | Does forgetting help? |
| Linear tape | Does tape nonlinearity matter? |
| Learned vs fixed decay | Is per-slot decay important? |

### Phase 3: Scale Up

If Phase 1/2 are promising:
- Train at 400M+ tokens
- Compare to Mamba2 at same compute
- Test on long-context tasks

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Loss vs E18-A | ≤ -0.02 nats | Tape helps |
| Loss vs Mamba2 | ≤ +0.05 nats | Competitive |
| Throughput vs E1 | ≥ 0.4× | Acceptable slowdown |
| Long-context improvement | Measurable | Main hypothesis |

## Summary

E23 introduces dual memory to Elman networks:

| Aspect | E1 | E23 |
|--------|-------|-----|
| State | D | N×D + D |
| State size (D=1024, N=64) | 1K | 65K |
| Cost | O(D²) | O(D² + N×D) |
| Memory access | Dense (W_h) | Sparse (attention) |
| Long-term storage | Limited | Large tape |

The key insight: **decouple storage from computation**. The tape provides cheap, large memory; working memory provides expensive, powerful computation. Attention bridges them efficiently.
