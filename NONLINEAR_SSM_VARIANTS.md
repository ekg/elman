# Nonlinear SSM Variants: Softsign Activation and Mamba2-Inspired Architectures

## Part 1: Activation Function Alternatives

### The Activation Spectrum

```
Cheapest                                                    Most Expensive
   │                                                              │
   ▼                                                              ▼
hardtanh ──── softsign ──── silu/swish ──── tanh ──── gelu
(clamp)      (x/(1+|x|))    (x*sigmoid)   (exp-based)  (erf-based)
```

### Softsign: The Sweet Spot

```python
softsign(x) = x / (1 + |x|)
```

**Properties:**
- Range: (-1, 1) - bounded like tanh
- Derivative: `1 / (1 + |x|)²` - always positive, never zero
- Smooth everywhere (C^∞)
- No exp() or erf() - just division and abs

**Comparison:**

| Property | tanh | softsign | hardtanh |
|----------|------|----------|----------|
| Range | (-1, 1) | (-1, 1) | [-1, 1] |
| Derivative at 0 | 1 | 1 | 1 |
| Derivative at ±∞ | → 0 | → 0 | = 0 |
| Smooth | yes | yes | no |
| Dead units | no | no | **yes** |
| Computational cost | high | **low** | lowest |

**Implementation:**

```python
def softsign(x):
    return x / (1 + torch.abs(x))

# Derivative (for reference, autograd handles this)
def softsign_grad(x):
    denom = 1 + torch.abs(x)
    return 1 / (denom * denom)
```

### Softsign Elman Layer

```python
class SoftsignElmanLayer(nn.Module):
    """E1 with softsign instead of tanh."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.W_h = nn.Linear(d_model, d_model, bias=False)
        self.W_x = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

        # Spectral norm on W_h for stability
        self.W_h = nn.utils.parametrizations.spectral_norm(self.W_h)

    def forward(self, x, h=None):
        batch, seq_len, d = x.shape

        if h is None:
            h = torch.zeros(batch, d, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            pre = self.W_h(h) + self.W_x(x_t)

            # Softsign instead of tanh
            h_new = pre / (1 + torch.abs(pre))

            # Output gating (unchanged)
            gate = torch.sigmoid(self.gate(x_t))
            out = h_new * gate

            h = h_new
            outputs.append(out)

        return torch.stack(outputs, dim=1), h
```

---

## Part 2: Nonlinear Mamba2 Variants

### Why Mamba2 is Linear

Mamba2's state update:
```
h' = A(x) ⊙ h + B(x) @ x
```

This is **linear in h**, which enables:
1. Parallel associative scan (O(log n) depth)
2. Hardware-efficient implementation
3. Stable gradient flow (no activation saturation)

But it limits composition depth to 1 per layer (from our proofs).

### The Nonlinear SSM Family

What if we add nonlinearity to SSM-style architectures?

#### Variant 1: Nonlinear Diagonal SSM

```python
class NonlinearDiagonalSSM(nn.Module):
    """
    Like Mamba2 but with nonlinearity in the state update.
    Loses parallel scan but gains composition depth.

    h' = softsign(A ⊙ h + B @ x)

    where A is diagonal (d parameters, not d²)
    """

    def __init__(self, d_model: int, d_state: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state or d_model

        # Diagonal decay (like Mamba2's A)
        self.log_A = nn.Parameter(torch.zeros(self.d_state))

        # Input projection (like Mamba2's B)
        self.B = nn.Linear(d_model, self.d_state)

        # Output projection (like Mamba2's C)
        self.C = nn.Linear(self.d_state, d_model)

        # Optional: input-dependent A (selectivity)
        self.A_proj = nn.Linear(d_model, self.d_state)

    def forward(self, x, h=None):
        batch, seq_len, d = x.shape

        if h is None:
            h = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]

            # Selective A (input-dependent decay)
            A = torch.sigmoid(self.log_A + self.A_proj(x_t))  # (batch, d_state)

            # State update with nonlinearity
            pre = A * h + self.B(x_t)
            h = pre / (1 + torch.abs(pre))  # softsign

            # Output
            out = self.C(h)
            outputs.append(out)

        return torch.stack(outputs, dim=1), h
```

**Key differences from Mamba2:**
- Nonlinearity (softsign) in state update
- Sequential (no parallel scan)
- But: composition depth = seq_len per layer

**Key differences from E1:**
- Diagonal A instead of full W_h (d params vs d²)
- State expansion (d_state can be > d_model)
- Selective (A depends on input)

#### Variant 2: Selective Elman

```python
class SelectiveElman(nn.Module):
    """
    E1 with Mamba2-style selectivity.

    h' = softsign(A(x) @ h + B(x) @ x)

    where A(x) is input-dependent (selective).
    A can be full matrix, low-rank, or diagonal.
    """

    def __init__(self, d_model: int, rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.rank = rank  # None = full, int = low-rank

        if rank is None:
            # Full selective matrix: A(x) = base_A + x @ W_A
            self.base_A = nn.Parameter(torch.zeros(d_model, d_model))
            self.W_A = nn.Linear(d_model, d_model * d_model)
        else:
            # Low-rank selective: A(x) = base_A + U(x) @ V
            self.base_A = nn.Parameter(torch.zeros(d_model, d_model))
            self.U_proj = nn.Linear(d_model, d_model * rank)
            self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

        self.W_x = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

        # Initialize base_A with small spectral radius
        nn.init.xavier_uniform_(self.base_A, gain=0.5)

    def forward(self, x, h=None):
        batch, seq_len, d = x.shape

        if h is None:
            h = torch.zeros(batch, d, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]

            # Compute selective A(x)
            if self.rank is None:
                A_delta = self.W_A(x_t).view(batch, d, d)
                A = self.base_A + A_delta * 0.1  # scale down delta
            else:
                U = self.U_proj(x_t).view(batch, d, self.rank)
                A_delta = torch.bmm(U, self.V.unsqueeze(0).expand(batch, -1, -1))
                A = self.base_A + A_delta

            # State update: h' = softsign(A(x) @ h + W_x @ x)
            pre = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1) + self.W_x(x_t)
            h_new = pre / (1 + torch.abs(pre))

            # Output gating
            gate = torch.sigmoid(self.gate(x_t))
            out = h_new * gate

            h = h_new
            outputs.append(out)

        return torch.stack(outputs, dim=1), h
```

#### Variant 3: State-Expanded Elman

```python
class StateExpandedElman(nn.Module):
    """
    E1 with Mamba2-style state expansion.

    Internal state n > output dimension d.
    Like Mamba2's d_state > d_model.

    h ∈ ℝ^n (expanded state)
    y ∈ ℝ^d (output)

    h' = softsign(W_h @ h + W_x @ x)
    y = C @ h
    """

    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_model * expansion

        # State transition (n x n) - this is expensive!
        # Could use diagonal or low-rank for efficiency
        self.W_h = nn.Linear(self.d_state, self.d_state, bias=False)

        # Input projection (n x d)
        self.W_x = nn.Linear(d_model, self.d_state)

        # Output projection (d x n)
        self.C = nn.Linear(self.d_state, d_model)

        # Gating
        self.gate = nn.Linear(d_model, d_model)

        # Spectral norm for stability
        self.W_h = nn.utils.parametrizations.spectral_norm(self.W_h)

    def forward(self, x, h=None):
        batch, seq_len, d = x.shape

        if h is None:
            h = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]

            # Expanded state update
            pre = self.W_h(h) + self.W_x(x_t)
            h = pre / (1 + torch.abs(pre))  # softsign

            # Project to output dimension
            y = self.C(h)

            # Output gating
            gate = torch.sigmoid(self.gate(x_t))
            out = y * gate

            outputs.append(out)

        return torch.stack(outputs, dim=1), h
```

**Note:** The W_h here is (d_state × d_state), which is expensive if expansion > 1. For efficiency, use diagonal or structured W_h.

#### Variant 4: Diagonal State-Expanded Elman

```python
class DiagonalStateExpandedElman(nn.Module):
    """
    State expansion with DIAGONAL recurrence.

    Combines:
    - Mamba2's state expansion (n > d)
    - Mamba2's diagonal transitions (O(n) not O(n²))
    - E1's nonlinearity (softsign)

    h' = softsign(A ⊙ h + B @ x)  where A is diagonal
    y = C @ h
    """

    def __init__(self, d_model: int, d_state: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state or (d_model * 2)

        # Diagonal decay (n parameters, not n²)
        self.log_A = nn.Parameter(torch.zeros(self.d_state))

        # Input projection
        self.B = nn.Linear(d_model, self.d_state)

        # Output projection
        self.C = nn.Linear(self.d_state, d_model)

        # Gating
        self.gate = nn.Linear(d_model, d_model)

        # Optional selectivity
        self.selective = True
        if self.selective:
            self.A_proj = nn.Linear(d_model, self.d_state)

    def forward(self, x, h=None):
        batch, seq_len, d = x.shape

        if h is None:
            h = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]

            # Diagonal A (optionally selective)
            if self.selective:
                A = torch.sigmoid(self.log_A + self.A_proj(x_t))
            else:
                A = torch.sigmoid(self.log_A)

            # State update with nonlinearity
            pre = A * h + self.B(x_t)
            h = pre / (1 + torch.abs(pre))  # softsign

            # Output
            y = self.C(h)
            gate = torch.sigmoid(self.gate(x_t))
            out = y * gate

            outputs.append(out)

        return torch.stack(outputs, dim=1), h
```

---

## Part 3: Architecture Comparison

### Computational Cost per Step

| Architecture | State Size | Recurrence Cost | Total Cost |
|--------------|------------|-----------------|------------|
| E1 (tanh) | d | O(d²) W_h@h | O(d²) |
| E1 (softsign) | d | O(d²) W_h@h | O(d²) |
| Mamba2 | n | O(n) diagonal | O(nd) |
| Nonlinear Diagonal SSM | n | O(n) diagonal | O(nd) |
| Selective Elman (full) | d | O(d²) A(x)@h | O(d²) |
| Selective Elman (rank-r) | d | O(dr) | O(dr) |
| State-Expanded (full) | n | O(n²) | O(n²) |
| State-Expanded (diag) | n | O(n) | O(nd) |

### Expressivity Properties

| Architecture | Nonlinear in h | Composition Depth | Selective | State Expansion |
|--------------|---------------|-------------------|-----------|-----------------|
| E1 | ✓ | seq_len | ✗ | ✗ |
| Mamba2 | ✗ | 1 | ✓ | ✓ |
| Nonlinear Diagonal SSM | ✓ | seq_len | ✓ | ✓ |
| Selective Elman | ✓ | seq_len | ✓ | ✗ |
| State-Expanded (diag) | ✓ | seq_len | ✓ | ✓ |

---

## Part 4: Recommended Experiments

### Experiment 1: Activation Function

Compare at fixed architecture (E1 d26 ~400M):
- tanh (baseline)
- softsign
- hardtanh
- silu

Metrics: loss, throughput, gradient statistics

### Experiment 2: Nonlinear Diagonal SSM

Compare at ~400M params:
- Mamba2 d22 (baseline, linear)
- Nonlinear Diagonal SSM with softsign
- Vary d_state: 64, 128, 256

Question: Does nonlinearity help enough to offset loss of parallel scan?

### Experiment 3: Selective Elman

Compare to E1 d26:
- E1 baseline (fixed W_h)
- Selective Elman (full A(x))
- Selective Elman (rank-16 A(x))
- Selective Elman (diagonal A(x))

Question: Does selectivity help E1?

### Experiment 4: The Best of Both Worlds

Try: **Diagonal State-Expanded Elman with Softsign**
- d_state = 2 * d_model (like Mamba2)
- Diagonal recurrence (O(n) like Mamba2)
- Softsign nonlinearity (composition depth like E1)
- Selective A(x) (like Mamba2)

This combines:
- Mamba2's efficiency (diagonal, state expansion)
- E1's expressivity (nonlinearity)
- Softsign's speed (no exp)

```python
# The "best of both" architecture
class HybridSSM(nn.Module):
    """
    Mamba2's structure + E1's nonlinearity + softsign efficiency.
    """
    def __init__(self, d_model, d_state=None):
        self.d_state = d_state or d_model * 2
        self.log_A = nn.Parameter(...)      # diagonal
        self.A_proj = nn.Linear(...)        # selectivity
        self.B = nn.Linear(...)             # input proj
        self.C = nn.Linear(...)             # output proj
        self.gate = nn.Linear(...)          # gating

    def forward(self, x, h):
        A = sigmoid(self.log_A + self.A_proj(x))  # selective diagonal
        pre = A * h + self.B(x)                    # O(n) recurrence
        h = softsign(pre)                          # nonlinearity!
        y = self.C(h) * sigmoid(self.gate(x))     # gated output
        return y, h
```

---

## Part 5: Implementation Priority

### High Priority (try first)
1. **Softsign E1**: Drop-in replacement, easy to test
2. **Diagonal State-Expanded Elman**: Best theoretical properties

### Medium Priority
3. **Selective Elman (diagonal)**: Adds selectivity cheaply
4. **Nonlinear Diagonal SSM**: Direct Mamba2 comparison

### Lower Priority (more speculative)
5. **Selective Elman (low-rank)**: Complex, unclear benefit
6. **Full State-Expanded**: Too expensive

---

## Summary

The key insight: **Mamba2's advantages (selectivity, state expansion) are orthogonal to its linearity.**

We can build architectures that have:
- Mamba2's selectivity (input-dependent transitions)
- Mamba2's state expansion (n > d)
- E1's nonlinearity (composition depth)
- Softsign's efficiency (no exp)

The most promising variant is **Diagonal State-Expanded Elman with Softsign**:
```
h' = softsign(A(x) ⊙ h + B @ x)    # O(n) recurrence with nonlinearity
y = C @ h * gate(x)                 # gated output
```

This should give us the best of both worlds: E1's expressivity with Mamba2's efficiency.
