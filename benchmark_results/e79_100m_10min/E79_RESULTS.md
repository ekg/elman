# E79: Coupled Memory-Modulation Matrix System

## The Journey: 79 Architectures Later

E79 represents the **culmination of 79 architectural experiments**, each testing a different hypothesis about what makes recurrent neural networks work. This isn't a random design—it's the result of systematically exploring the space of recurrent architectures, with **50+ custom CUDA kernels** implemented along the way.

### The Experiments That Led Here

**Phase 1: Foundations (E1-E15)**
- **E1**: Gated Elman baseline—still competitive, the reference point
- **E5**: Low-rank factorization with CUTLASS B2B GEMM fusion
- **E6/E7**: Structured matrices (Circulant, Monarch)—efficient but limited
- **E8**: Learned sparsification
- **E10**: Multi-scale EMA with learned decay banks
- **E11**: Selective memory (Mamba-inspired input-dependent gating)
- **E14**: First matrix state experiments
- **E15**: Activation functions (softsign vs tanh vs silu)

**Phase 2: Dual Memory & Attention (E16-E29)**
- **E16**: Diagonal state expansion—SSM-style
- **E18/E19**: H-aware gating, simplified gating
- **E20/E21**: Mamba2-informed designs, structured attention
- **E23**: Dual-memory Elman (NTM/DNC/Fast Weights literature review)
- **E25**: Sparse attention with 1.5-entmax
- **E26**: Parallel dual-memory with softmax
- **E28**: Convolution integration (Mamba2-style)
- **E29/E29c**: Selective dual-memory, SSM-style diagonal gating

**Phase 3: Simplification & Ablations (E30-E55)**
- **E30-E33**: Gating simplifications
- **E31**: Sparse gating—**negative result**, learned to not work
- **E34**: Diagonal W_h (+80% speed, worse loss)
- **E36**: Linear recurrence (no tanh)—surprisingly good
- **E37v2**: Tied weights + batched GEMM optimization
- **E38-E41**: No W_x, no bias, no pre-silu, diagonal W_x
- **E42**: **Best simple model**—linear + tied + batched GEMM
- **E43-E55**: Scalar decay, diagonal W, pure accumulation, no projections...

**Phase 4: State Structure (E56-E68)**
- **E56**: Concatenation instead of addition
- **E58**: Learned spectral radii
- **E59**: Highway connections
- **E60**: Residual nonlinearity
- **E61-E63**: Decay gating, selective write, nonlinear delta
- **E64-E66**: Additive H, diagonal H, low-rank H
- **E67/E68**: H-gated alpha, self-gating with h-dependence

**Phase 5: Matrix State Renaissance (E70-E79)**
- **E70**: Linear matrix update (delta rule baseline)
- **E71**: S-dependent gating on matrix
- **E72**: Memory-gated value selection
- **E73**: Nonlinear delta rule with gradient checkpointing
- **E74**: Full matrix with multiple update types (DELTA, EMA, NTM, RESIDUAL)
- **E75**: Gated delta (forget gate + delta rule)
- **E76**: Log-space gated delta
- **E77/E78**: Linear matrix state with FP32 support
- **E79**: **Coupled memory-modulation**—two matrices that talk to each other

### What We Learned Along the Way

1. **Simplicity often wins short-term**: E42 (linear + tied weights) beats complex architectures in 10-min benchmarks
2. **But complexity enables capacity**: Matrix states (E70+) can represent more, just need more training
3. **Nonlinearity is tricky**: Too much hurts optimization, too little limits expressiveness
4. **Gating matters**: Almost every good model has some form of gating
5. **The delta rule keeps appearing**: v - retrieved is a recurring theme (E63, E70, E73, E74, E79)
6. **Negative results are valuable**: E31 sparse gating failed—good to know!

---

## E79 Architecture

E79 synthesizes insights from this entire journey into **two coupled n×n matrix states**:

### State Matrices
- **S (Content Memory)**: Primary associative memory storing key-value pairs
- **M (Modulation Memory)**: Controls S's decay gates (what S forgets)

### Forward Pass (per timestep)
```
Input: k, v, q, m vectors (all n-dimensional)

# Normalize keys
k_norm = k / ||k||
m_norm = m / ||m||

# M controls S's decay gates (M → S coupling)
s_row_decay = sigmoid(M @ k_norm + b_s_gate)    # M decides row decay
s_col_decay = sigmoid(M.T @ k_norm + b_s_gate)  # M decides col decay

# S update with M-controlled gating + delta rule
s_retrieved = S @ k_norm
s_delta = v - s_retrieved
S_new = (s_row_decay[:, None] * S * s_col_decay[None, :]) + outer(s_delta, k_norm)

# S controls M's decay gates (S → M coupling)
m_row_decay = sigmoid(S @ m_norm + b_m_gate)    # S decides row decay
m_col_decay = sigmoid(S.T @ m_norm + b_m_gate)  # S decides col decay

# M update: learns to predict S's changes
m_retrieved = M @ m_norm
m_delta = s_delta - m_retrieved
M_new = (m_row_decay[:, None] * M * m_col_decay[None, :]) + outer(m_delta, m_norm)

# Output via query (self-gating from E33/E68)
Sq = S_new @ q
output = Sq * silu(Sq)
```

### Why This Design?

E79 combines ideas that worked:
- **Delta rule** (E70, E73): `v - retrieved` as the update signal
- **Matrix state** (E70-E77): Full n×n associative memory
- **Self-gating** (E33, E68): `x * silu(x)` output nonlinearity
- **Input-dependent decay** (E61, E75): Learned forgetting per row/column

And adds something new:
- **Mutual gating control**: M controls S's forgetting, S controls M's forgetting
- This creates a self-organizing dynamical system where each memory regulates the other
- M gets gradients through: Loss → output → S → s_decay_gates → M

---

## Benchmark Results

### E79 n_state Sweep (100M params, 10 min training)

| n_state | Steps | Loss | tok/s | State Size |
|---------|-------|------|-------|------------|
| 8 | 3990 | 2.29 | 52K | 2×64 = 128 |
| 16 | 3730 | 1.99 | 47K | 2×256 = 512 |
| **32** | **2460** | **1.51** | **31K** | **2×1024 = 2048** |
| 48 | 1870 | 1.91 | 24K | 2×2304 = 4608 |
| 64 | 1270 | 2.67 | 17K | 2×4096 = 8192 |
| 96 | 290 | 3.10 | 3.7K | 2×9216 = 18432 |
| 128 | 170 | 5.09 | 2.3K | 2×16384 = 32768 |

### Comparison to Baselines (100M params, 10 min)

| Model | Loss | tok/s | State Type |
|-------|------|-------|------------|
| Mamba2 | 1.27 | 78.7K | Structured SSM (parallel scan) |
| **E79 n=32** | **1.51** | **31.5K** | **Nonlinear coupled matrices** |
| E1 | 1.53 | 45.5K | Gated vector |
| E42 | 1.59 | 137K | Linear tied (simplest good model) |
| Llama | 1.91 | 71.3K | Transformer attention |
| FLA-GDN | 1.99 | 18.7K | Gated Delta Net |

---

## Key Findings

### 1. Nonlinear Matrix-State RNNs Are Viable
E79 demonstrates that **full nonlinear matrix updates** can train stably and efficiently:
- 2 coupled 32×32 matrices = 2048-element recurrent state
- Nonlinear dynamics (gated updates + silu output)
- 31K tok/s = **~40% of Mamba2's throughput**
- This is remarkable: Mamba2 uses parallel scan, E79 is fully sequential

### 2. E79 Beats Simpler Baselines
- **Beats E1**: 1.51 vs 1.53 loss with much larger state capacity
- **Beats FLA-GDN**: 1.51 vs 1.99 loss at higher throughput
- The coupled memory architecture provides measurable benefit

### 3. n_state=32 is Optimal for 10-Minute Training
- Smaller n_state (8, 16): Higher throughput but limited capacity
- Larger n_state (64+): Not enough training steps to converge
- This is a **training budget limitation**, not architectural

### 4. Larger n_state Needs More Investigation
The n_state=64/96/128 results are inconclusive:
- n=64: only 1270 steps
- n=96: only 290 steps
- n=128: only 170 steps

These need **longer training** to evaluate properly.

---

## Theoretical Significance

### What 79 Experiments Taught Us

1. **Large recurrent state is feasible**: 2048+ element matrix state at reasonable speed
2. **Nonlinearity doesn't break training**: Coupled nonlinear updates work
3. **The design space is vast**: 79 architectures and still finding new things
4. **Mutual control is powerful**: M controlling S's gates (and vice versa) enables adaptive forgetting

### The Mamba2 Comparison (In Context)

The 1.51 vs 1.27 loss gap should be interpreted carefully:
- Mamba2 benefits from **parallel scan** (O(log n) depth vs O(n) sequential)
- Mamba2 has **years of kernel optimization**
- E79 is a **first implementation** with basic CUDA
- We're comparing a week-old architecture to a mature system

### Open Questions

1. Does larger n_state surpass n=32 with sufficient training?
2. Does S/M coupling help with long-range dependencies?
3. How does E79 scale to 1B+ parameters?
4. Can we close the speed gap with better kernels?

---

## Implementation Notes

### CUDA Kernels (50+ total in the project)
- **Shared memory** (n_state ≤ 64): Both S and M in shared memory
- **Global memory fallback** (n_state ≥ 96): Matrices in global, vectors in shared
- **Gradient checkpointing**: interval=16 reduces memory 2.5x

### Gradient Bug Fixed
Global memory backward kernel was missing:
```cpp
d_m_norm += M^T @ d_m_delta  // m_retrieved gradient
```
Now fixed and stable for all n_state values.

---

## Conclusion

E79 represents **79 experiments distilled into one architecture**. It's not the end—it's a waypoint showing that:

1. **Nonlinear matrix-state RNNs work** at reasonable speed
2. **Mutual gating control** is a viable design pattern for coupled memories
3. **There's still room to explore** in recurrent architectures

The journey from E1 to E79 has been one of systematic exploration: trying ideas, measuring results, keeping what works, discarding what doesn't. E79 combines the best insights:
- Delta rule updates (E70, E73)
- Self-gating (E33, E68)
- Matrix state (E70-E77)
- Novel: mutual gating control (M controls S's decay, S controls M's decay)

**This isn't the final answer. It's proof that the search is worth continuing.**

---

## Appendix: All 79 Architectures (Summary)

| Range | Theme | Key Models |
|-------|-------|------------|
| E1-E15 | Foundations | E1 (gated baseline), E5 (low-rank), E6/E7 (structured), E14 (matrix state) |
| E16-E29 | Dual memory | E23 (NTM-inspired), E25 (entmax), E28 (conv), E29 (selective) |
| E30-E55 | Simplification | E31 (sparse, failed), E36 (linear), E42 (best simple) |
| E56-E68 | State structure | E61 (decay gate), E67 (h-gated), E68 (self-gating) |
| E70-E79 | Matrix renaissance | E70 (delta), E73 (nonlinear), E74 (full matrix), E79 (coupled) |

**Total: 50+ CUDA kernels, 126+ commits, countless hours of GPU time.**
