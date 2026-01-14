# Gated Delta Elman: Unifying DeltaNet and Elman

## Executive Summary

This document formalizes the connection between Gated DeltaNet's memory management and Elman RNNs, proposing new hybrid architectures that combine Elman's simplicity with DeltaNet's selective write/erase capabilities.

**Core Insight**: Gated DeltaNet operates on **matrix state** S ∈ ℝ^(d×d), while Elman operates on **vector state** h ∈ ℝ^d. We can derive vector analogs of the gated delta rule that preserve the key properties (selective erasure, targeted writes) while maintaining O(d) state complexity.

---

## Part 1: Gated DeltaNet Formalization

### 1.1 The Gated Delta Rule (Matrix Form)

The full Gated DeltaNet update rule:

```
S_t = α_t · S_{t-1} · (I - β_t · k_t · k_t^T) + β_t · v_t · k_t^T
```

Where:
- `S_t ∈ ℝ^(d×d)` - state matrix (key-value memory)
- `α_t ∈ (0, 1)` - **decay gate** (uniform memory erasure)
- `β_t ∈ (0, 1)` - **write gate** (selective update strength)
- `k_t ∈ ℝ^d` - **key** (what to update/query)
- `v_t ∈ ℝ^d` - **value** (new content to write)

### 1.2 Decomposition of Effects

Breaking down the update:

```
S_t = α_t · S_{t-1}                    # (1) Global decay
    - α_t · β_t · S_{t-1} @ k_t @ k_t^T # (2) Selective erasure of key direction
    + β_t · v_t @ k_t^T                 # (3) Write new value at key
```

**Effect 1 - Global Decay (α_t · I):**
- When α_t → 0: Rapid memory clearing
- When α_t → 1: Full memory retention
- This is Mamba2's mechanism

**Effect 2 - Selective Erasure (-β_t · k_t · k_t^T):**
- Removes old content associated with key k_t
- Prevents memory collisions
- This is DeltaNet's key innovation

**Effect 3 - Selective Write (+v_t · k_t^T):**
- Writes new value v_t at key k_t
- Standard associative memory update

### 1.3 Jacobian Analysis

For query output `y_t = S_t @ q_t`:

```
∂S_t/∂S_{t-1} = α_t · (I - β_t · k_t · k_t^T)
```

This is a **Householder-like** transformation:
- Eigenvalues: α_t (multiplicity d-1), α_t(1 - β_t·||k_t||²) (multiplicity 1)
- When β_t → 0: Pure decay, eigenvalues all α_t
- When β_t → 1: One direction decays faster (the key direction)

---

## Part 2: Existing Elman Models Through Delta Lens

### 2.1 E42: Linear Tied Self-Gated

```python
h_t = W @ (h_{t-1} + x_t) + b        # Linear recurrence
output = h_t * silu(h_t)              # Self-gating
```

**Delta Interpretation:**
- No explicit decay gate (implicit in W's spectral radius)
- No selective erasure (W mixes all dimensions equally)
- No key-value separation (x_t is both key and value)

**Limitations:**
- Can't selectively forget specific information
- W controls everything - coupling decay and transformation

### 2.2 E1: Mamba-Gated Elman

```python
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
gate = sigmoid(z_t)
output = h_t * gate
```

**Delta Interpretation:**
- tanh provides bounded state (implicit decay through saturation)
- Gate is on output, not on state update
- No selective write/erase in the recurrence itself

### 2.3 E56: Concat Elman

```python
h_t = W @ [x_t; h_{t-1}]  # Concatenation instead of sum
```

**Delta Interpretation:**
- Separate pathways for x and h (like having separate W_k, W_v)
- But still no gating in the recurrence
- W controls both key (from h) and value (from x) simultaneously

---

## Part 3: New Architectures - Gated Delta Elman

### 3.1 E61: Decay-Gated Elman (Mamba2-style)

The simplest delta-inspired model - just add input-dependent decay:

```python
α_t = sigmoid(W_α @ x_t + b_α)        # Input-dependent decay
h_t = α_t * h_{t-1} + (1 - α_t) * W @ x_t
output = h_t * silu(h_t)
```

**Properties:**
- Jacobian: `∂h_t/∂h_{t-1} = diag(α_t)`
- Gradient through T steps: `prod_{t=1}^T diag(α_t)`
- **Linear in h** → parallelizable with associative scan!
- No selective erasure (uniform decay)

**Comparison to Mamba2:**
- Mamba2: α is per-dimension but scalar-identity structure
- E61: α is per-dimension with full input dependence

### 3.2 E62: Selective Write Elman (DeltaNet-style)

Add selective writing - choose WHICH dimensions to update:

```python
k_t = sigmoid(W_k @ x_t)              # Selection mask (0-1 per dimension)
v_t = tanh(W_v @ x_t)                 # New values

h_t = (1 - k_t) * h_{t-1} + k_t * v_t # Selective replacement
output = h_t * silu(h_t)
```

**Properties:**
- Jacobian: `∂h_t/∂h_{t-1} = diag(1 - k_t)`
- k_t selects dimensions to overwrite
- **Direct vector analog of delta rule!**
- Linear in h → parallelizable

**This is the key insight:**
```
DeltaNet matrix:  S_t = S_{t-1} - β·S@k@k^T + β·v@k^T
E62 vector:       h_t = h_{t-1} - k⊙h_{t-1} + k⊙v_t
                      = (1-k)⊙h + k⊙v
```

The sigmoid k_t plays the role of `β·k·k^T` projected onto the diagonal.

### 3.3 E63: Full Gated Delta Elman

Combine decay gate AND selective write:

```python
α_t = sigmoid(W_α @ x_t + b_α)        # Decay gate (Mamba2)
β_t = sigmoid(W_β @ x_t + b_β)        # Write gate (DeltaNet)
k_t = W_k @ x_t                       # Key (what to update)
v_t = W_v @ x_t                       # Value (new content)

# Gated delta rule for vectors:
# h_t = α·h·(1 - β·k²) + β·v·|k|

erase = β_t * k_t.pow(2)              # Selective erase strength
write = β_t * v_t * k_t.abs()         # Selective write

h_t = α_t * h_{t-1} * (1 - erase) + write
output = h_t * silu(h_t)
```

**Properties:**
- Two gates: α for global decay, β for selective strength
- k_t.pow(2) ensures non-negative erase
- Still linear in h! The nonlinearity is in the gates/keys

**Alternative formulation (simpler):**

```python
α_t = sigmoid(W_α @ x_t)              # Retain gate
k_t = sigmoid(W_k @ x_t)              # Select gate (0-1)
v_t = tanh(W_v @ x_t)                 # Value

h_t = α_t * (1 - k_t) * h_{t-1} + k_t * v_t
```

This combines:
- α_t: How much to retain globally
- k_t: Which dimensions to overwrite
- The interaction α_t * (1 - k_t) gives selective retention

### 3.4 E64: Tied Gated Delta Elman

Minimize parameters by tying weights:

```python
# Single projection, split into gates and values
proj = W @ x_t
α_t = sigmoid(proj[:d])               # First half: retain gate
k_t = sigmoid(proj[d:2d])             # Second half: select gate
v_t = tanh(proj[2d:3d])               # Third part: value

h_t = α_t * (1 - k_t) * h_{t-1} + k_t * v_t
output = h_t * silu(h_t)
```

Or even simpler - derive gates from the same signal:

```python
v_t = W @ x_t
α_t = sigmoid(v_t)                    # Retain = sigmoid(new value)
k_t = 1 - α_t                         # Select = 1 - retain (complementary)

h_t = α_t * h_{t-1} + k_t * tanh(v_t) # = α·h + (1-α)·tanh(Wx)
```

This is actually very similar to GRU's update gate but with a single projection!

---

## Part 4: Gradient Analysis

### 4.1 Jacobian Comparison

| Model | Recurrence | Jacobian | Gradient T steps |
|-------|------------|----------|------------------|
| E42 | W @ (h + x) | W | W^T (vanishes) |
| E59 | h + W @ x | I | I (preserved) |
| E60 | h + tanh(Wh + Ux) | I + tanh'·W | ~I (mostly preserved) |
| **E61** | α·h + (1-α)·Wx | diag(α) | prod(α) (controlled) |
| **E62** | (1-k)·h + k·v | diag(1-k) | prod(1-k) (controlled) |
| **E63** | α·(1-k)·h + k·v | diag(α·(1-k)) | prod(α·(1-k)) (controlled) |

### 4.2 Why Gated Models Win

The key insight from DeltaNet/Mamba2: **input-dependent gates provide selectivity without destroying gradients**.

- Fixed W in E42: Gradient = W^T → vanishes exponentially
- Gated E61-63: Gradient = prod(gates) → controlled by input

When the model needs to remember (α → 1, k → 0):
- Jacobian → I
- Perfect gradient preservation

When the model needs to forget/overwrite (α → 0 or k → 1):
- Jacobian → 0 in those dimensions
- Gradient vanishes BUT that's intentional - the model chose to forget

**This is exactly what LSTM's cell state does!** The forget gate controls gradient flow through time.

---

## Part 5: Implementation Priority

### Recommended Order:

1. **E62 (Selective Write)** - Simplest direct translation of delta rule
   - Just 2 projections (W_k, W_v)
   - Clear semantics: k selects, v provides new values

2. **E61 (Decay-Gated)** - Mamba2-style decay
   - Single projection for α
   - Easiest to parallelize

3. **E63 (Full Gated Delta)** - Most expressive
   - 4 projections (W_α, W_β, W_k, W_v)
   - May be over-parameterized

4. **E64 (Tied)** - Efficiency variant
   - Shared projections
   - Test if expressivity is maintained

### Key Design Decisions:

1. **Complement gates vs independent?**
   - GRU uses complementary (z and 1-z)
   - Gated DeltaNet uses independent (α and β)
   - Independent is more expressive but harder to learn

2. **How to derive k (select mask)?**
   - sigmoid(Wx) - learnable per-dimension
   - softmax(Wx) - sparse selection (1-hot like)
   - k² from linear projection (DeltaNet style)

3. **Nonlinearity placement?**
   - E61-64: Only in gates (preserves linearity in h)
   - E60: tanh in update (nonlinear in h)
   - Gates-only is more parallelizable

---

## Part 6: Theoretical Connection to Fast Weight Programmers

### 6.1 Elman as Degenerate FWP

Standard Elman can be viewed as a Fast Weight Programmer where:
- Slow net: Fixed W_x, W_h
- Fast weights: h itself (vector, not matrix)
- Update: h_t = f(W_h @ h_{t-1} + W_x @ x_t)

The key limitation: **h is a vector, so "fast weights" can only encode O(d) information**, not O(d²) like matrix-state models.

### 6.2 E62/E63 as Vector FWP

The gated delta Elman variants make the FWP structure explicit:
- Slow net: Produces k, v, α (key, value, gates)
- Fast weights: h (vector state)
- Delta rule: h_t = (1-k) ⊙ h_{t-1} + k ⊙ v_t

This is exactly the **diagonal restriction** of DeltaNet's matrix state!

### 6.3 Why O(d) Might Be Enough

Hypothesis: For many tasks, we don't need O(d²) state - O(d) suffices if we have:
1. Good input-dependent gating
2. Enough layers (depth compensates for width)
3. Proper residual connections between layers

The Elman advantage: **O(d) state = O(d) memory, O(d²) compute per step**, while DeltaNet has **O(d²) state = O(d²) memory, O(d³) compute per step**.

---

## Part 7: Proposed Experiments

### 7.1 Direct Comparison (40M params, T=512)

| Model | Description | Expected |
|-------|-------------|----------|
| E42 | Baseline (linear tied) | ~1.59 loss |
| E61 | +Decay gate | Better long-range |
| E62 | +Selective write | Better retrieval |
| E63 | +Both gates | Best? Or overfit? |
| E64 | Tied version | Efficiency check |

### 7.2 Scaling Test (100M → 1B)

Key question: Do gated variants scale better than E42?

Hypothesis: Yes, because:
- E42's W^T vanishes at long sequences
- E61-64's gates preserve gradients selectively
- This should matter MORE at scale (longer contexts, deeper networks)

### 7.3 Ablations

1. **Gate initialization**: Start with α ≈ 1 (preserve) vs α ≈ 0.5 (neutral)
2. **k activation**: sigmoid vs softmax vs linear+square
3. **Complementary vs independent gates**: (α, 1-α) vs (α, β)

---

## References

1. Irie et al. (2021). Going Beyond Linear Transformers with Recurrent Fast Weight Programmers. NeurIPS.
2. Irie et al. (2022). A Modern Self-Referential Weight Matrix. ICML.
3. Yang et al. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025.
4. Schlag et al. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML.

---

*Created: 2026-01-14*
