# Log-Space Research Plan: Getting It Right

**Date**: December 29, 2025
**Status**: Active Research
**Goal**: Achieve Mamba2-competitive performance with nonlinear recurrence using log-space gradient benefits

---

## Executive Summary

The core bet: **Mamba2's linearity leaves expressivity on the table**. If we can achieve nonlinear recurrence with log-space gradient benefits, we might win.

### Current State

| Level | Status | Issue |
|-------|--------|-------|
| 0-3 | Working | Linear-space, gradient vanishing at depth |
| 4 | Broken | NaN at step ~260, tanh breaks log-space |
| 5-6 | Theoretical | CUDA exists but tanh still used |

### Core Insight

**tanh is a linear-space operation**. The current log-space implementations all convert log→linear→tanh→log, which:
1. Breaks gradient bounds (1/h unbounded for small h)
2. Reintroduces vanishing gradients (tanh'(v) → 0 when saturated)

### Solution Paths (Priority Order)

1. **Positive/Negative Channel Decomposition** (cleanest, GPU-friendly)
2. **Polynomial Activation** (documented, simpler math)
3. **Complex Log-Polar** (elegant but complex)
4. **Multiplicative Gating Only** (simplest, may lack expressivity)

---

## Part 1: The Mathematics

### 1.1 Why Log-Space Works for Mamba2

For `z = logsumexp(a, b) = log(exp(a) + exp(b))`:
```
dz/da = exp(a) / (exp(a) + exp(b)) = softmax_a ∈ [0, 1]
dz/db = exp(b) / (exp(a) + exp(b)) = softmax_b ∈ [0, 1]
```

**Bounded gradients** = no vanishing, no exploding.

Mamba2: `h_t = A * h_{t-1} + B * x_t` (linear)
In log-space: All operations are logsumexp → bounded gradients throughout.

### 1.2 Why tanh Breaks Log-Space

Elman: `h_t = tanh(W @ h_{t-1} + ...)` (nonlinear)

To use tanh:
1. Convert log_h → h = sign * exp(log_h)
2. Compute W @ h (can be log-space, but result is linear)
3. Apply tanh(v)
4. Convert h_new → log_h_new

**The gradient chain breaks at step 3**:
```
d(log|h_new|)/d(log|h_prev|) involves:
  - d(log|y|)/dy = 1/y  (unbounded for small y!)
  - d(tanh(v))/dv = 1 - tanh²(v)  (→ 0 when saturated!)
```

### 1.3 The Signed Log Addition Problem

For signed values stored as (log|x|, sign(x)):

**Addition** a + b where a = s_a * exp(log_a), b = s_b * exp(log_b):
```python
if s_a == s_b:  # Same sign
    log_result = logsumexp(log_a, log_b)  # Easy!
    sign_result = s_a
else:  # Opposite signs - HARD
    # Need: log|exp(log_a) - exp(log_b)|
    # When log_a ≈ log_b: catastrophic cancellation
    if log_a > log_b:
        log_result = log_a + log1p(-exp(log_b - log_a))
        sign_result = s_a
    else:
        log_result = log_b + log1p(-exp(log_a - log_b))
        sign_result = s_b
```

**Problems**:
1. Branching (GPU-unfriendly)
2. Near-cancellation numerically unstable: `log(1 - 0.999999)` loses precision

---

## Part 2: Proposed Approaches

### 2.1 Positive/Negative Channel Decomposition (RECOMMENDED)

**Representation**: Instead of (log|x|, sign(x)), use (log_pos, log_neg):
```
value = exp(log_pos) - exp(log_neg)
```

**Key Insight**: Addition becomes fully additive, no branching:
```python
def add(log_pos_a, log_neg_a, log_pos_b, log_neg_b):
    """Add two numbers in pos/neg representation."""
    log_pos_out = logaddexp(log_pos_a, log_pos_b)  # Positive parts add
    log_neg_out = logaddexp(log_neg_a, log_neg_b)  # Negative parts add
    return log_pos_out, log_neg_out
```

**Multiplication** a * b:
```python
def multiply(log_pos_a, log_neg_a, log_pos_b, log_neg_b):
    """Multiply two numbers in pos/neg representation.
    (p_a - n_a) * (p_b - n_b) = p_a*p_b + n_a*n_b - p_a*n_b - n_a*p_b
                              = (p_a*p_b + n_a*n_b) - (p_a*n_b + n_a*p_b)
    """
    log_pos_out = logaddexp(log_pos_a + log_pos_b, log_neg_a + log_neg_b)
    log_neg_out = logaddexp(log_pos_a + log_neg_b, log_neg_a + log_pos_b)
    return log_pos_out, log_neg_out
```

**Matrix-vector product** y = R @ h where h is (log_pos_h, log_neg_h):
```python
def log_matvec(R, log_pos_h, log_neg_h):
    """Log-space matrix-vector multiplication."""
    # Decompose R into positive and negative parts
    log_R_pos = log(max(R, 0) + eps)
    log_R_neg = log(max(-R, 0) + eps)

    # For each output dimension i:
    # y_i = sum_j R_ij * h_j
    # pos contribution: R+_ij * h+_j + R-_ij * h-_j
    # neg contribution: R+_ij * h-_j + R-_ij * h+_j

    log_pos_contrib_pp = logsumexp(log_R_pos + log_pos_h, dim=-1)
    log_pos_contrib_nn = logsumexp(log_R_neg + log_neg_h, dim=-1)
    log_pos_out = logaddexp(log_pos_contrib_pp, log_pos_contrib_nn)

    log_neg_contrib_pn = logsumexp(log_R_pos + log_neg_h, dim=-1)
    log_neg_contrib_np = logsumexp(log_R_neg + log_pos_h, dim=-1)
    log_neg_out = logaddexp(log_neg_contrib_pn, log_neg_contrib_np)

    return log_pos_out, log_neg_out
```

**The tanh Problem**: Still need to apply tanh to (exp(log_pos) - exp(log_neg)).

**Solution A - Approximate tanh in log-pos/neg space**:
```python
def tanh_log_posneg(log_pos, log_neg):
    """Apply tanh to value represented as (log_pos, log_neg).

    tanh(x) = (e^x - e^-x) / (e^x + e^-x)

    For x = exp(log_pos) - exp(log_neg), this is complex.
    Instead, use soft saturation.
    """
    # Convert to linear, apply tanh, convert back
    # This is the SAME PROBLEM as before - we lose log-space benefits
    pass
```

**Solution B - Replace tanh with gating (PREFERRED)**:
```python
def gated_update_log_posneg(log_pos_h, log_neg_h, log_pos_x, log_neg_x, log_gate):
    """Gated update without tanh.

    h_new = gate * x + (1-gate) * h_prev

    In log-pos/neg space:
    - Multiplication by positive gate is just addition in log
    - Addition of terms uses logaddexp
    """
    log_one_minus_gate = log(-expm1(-exp(log_gate)))  # log(1 - exp(log_gate))

    # gate * x
    log_pos_gx = log_gate + log_pos_x
    log_neg_gx = log_gate + log_neg_x

    # (1-gate) * h_prev
    log_pos_gh = log_one_minus_gate + log_pos_h
    log_neg_gh = log_one_minus_gate + log_neg_h

    # Sum
    log_pos_out = logaddexp(log_pos_gx, log_pos_gh)
    log_neg_out = logaddexp(log_neg_gx, log_neg_gh)

    return log_pos_out, log_neg_out
```

**Nonlinearity from gating**: The learned, input-dependent gate provides nonlinearity. This is the LSTM/GRU philosophy - gates ARE the nonlinearity.

### 2.2 Polynomial Activation

**Representation**: (log|h|, sign(h))

**Activation**: Replace tanh with polynomial
```
h = sign(v) * |v|^alpha
```

In log-space:
```
log|h| = alpha * log|v|
sign(h) = sign(v)
```

**Gradient**:
```
d(log|h|)/d(log|v|) = alpha  (CONSTANT!)
```

**Pros**:
- Simpler representation than pos/neg
- Constant gradient (no vanishing)
- Already documented in codebase

**Cons**:
- Unbounded output (need normalization)
- Still needs signed log addition for the linear combination

**Implementation**:
```python
def polynomial_activation_log(log_v, sign_v, alpha=0.5):
    """Apply polynomial in log space."""
    log_h = alpha * log_v
    sign_h = sign_v  # Sign preserved
    return log_h, sign_h
```

### 2.3 Complex Log-Polar Representation

**Representation**: (log_r, theta) where value = r * e^(i*theta)

Real line lives at theta ∈ {0, π}:
- theta = 0: positive real
- theta = π: negative real

**Multiplication** (fully additive):
```python
log_r_out = log_r_a + log_r_b
theta_out = theta_a + theta_b
```

**Addition** (still hard, but smooth sign via phase):
```python
# Complex addition in polar form requires conversion
# BUT: phase wrapping gives smooth sign changes
```

**Activation** (polynomial-like):
```python
log_r_out = alpha * log_r  # Power law
theta_out = theta  # Phase preserved
```

**Pros**:
- Elegant mathematics
- Smooth sign representation (no branching)
- S4 exploited this

**Cons**:
- Complex numbers add implementation complexity
- Most operations still require conversion for addition

### 2.4 Multiplicative Gating Only (Simplest)

**Core idea**: What if we don't need tanh at all?

```python
def multiplicative_recurrence(log_h, log_x, log_gate, log_decay):
    """All operations stay in log space.

    h_new = decay * h_prev + gate * transform(x)

    In log space (all positive):
    log_h_new = logsumexp(log_decay + log_h, log_gate + log_x)
    """
    return torch.logsumexp(
        torch.stack([log_decay + log_h, log_gate + log_x]),
        dim=0
    )
```

**Soft saturation** (instead of tanh):
```python
def soft_cap(log_h, log_cap):
    """Soft upper bound: h_capped = cap * tanh(h/cap) ≈ h for small h."""
    # Approximation: cap - softplus(cap - h)
    log_h_capped = log_cap - F.softplus(torch.exp(log_cap - log_h))
    return log_h_capped
```

**Pros**:
- Fully additive operations
- No branching
- Hardware-friendly

**Cons**:
- Positive-only (need sign handling separately)
- May lack expressivity compared to tanh

---

## Part 3: Implementation Roadmap

### Phase 1: Pure PyTorch Prototypes (Week 1)

**Goal**: Validate math before CUDA investment.

#### 1.1 Pos/Neg Channel Architecture
```python
# File: elman/models/logspace_posneg.py

class LogPosNegElman(nn.Module):
    """Level X: Positive/Negative channel log-space Elman."""

    def __init__(self, dim):
        super().__init__()
        # Weights stored in linear space, decomposed at forward time
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.W_h = nn.Linear(dim, dim, bias=False)  # Full matrix
        self.b = nn.Parameter(torch.zeros(dim))

        # Gate weights
        self.W_gate = nn.Linear(dim, dim, bias=False)
        self.b_gate = nn.Parameter(torch.full((dim,), -2.0))  # Small gate init

    def forward(self, x, log_pos_h, log_neg_h):
        """
        x: [B, D] input (linear space, can be negative)
        log_pos_h, log_neg_h: [B, D] hidden state in pos/neg representation

        Returns: log_pos_h_new, log_neg_h_new
        """
        B, D = x.shape

        # Convert input to pos/neg representation
        log_pos_x = torch.log(F.relu(self.W_x(x) + self.b) + 1e-10)
        log_neg_x = torch.log(F.relu(-self.W_x(x) - self.b) + 1e-10)

        # W_h @ h in log-pos/neg space
        log_pos_Rh, log_neg_Rh = self._log_matvec(
            self.W_h.weight, log_pos_h, log_neg_h
        )

        # Pre-activation: v = W_x @ x + W_h @ h + b
        log_pos_v = torch.logaddexp(log_pos_x, log_pos_Rh)
        log_neg_v = torch.logaddexp(log_neg_x, log_neg_Rh)

        # Gate (sigmoid in log space)
        gate_linear = self.W_gate(x) + self.b_gate
        log_gate = F.logsigmoid(gate_linear)
        log_one_minus_gate = F.logsigmoid(-gate_linear)

        # Gated update: h_new = (1-g)*h + g*v
        log_pos_new = torch.logaddexp(
            log_one_minus_gate + log_pos_h,
            log_gate + log_pos_v
        )
        log_neg_new = torch.logaddexp(
            log_one_minus_gate + log_neg_h,
            log_gate + log_neg_v
        )

        return log_pos_new, log_neg_new

    def _log_matvec(self, W, log_pos, log_neg):
        """Log-space matrix-vector multiplication."""
        # Decompose W
        W_pos = F.relu(W)
        W_neg = F.relu(-W)
        log_W_pos = torch.log(W_pos + 1e-10)
        log_W_neg = torch.log(W_neg + 1e-10)

        # Contributions: pos*pos + neg*neg → pos output
        log_pp = torch.logsumexp(log_W_pos.unsqueeze(0) + log_pos.unsqueeze(1), dim=-1)
        log_nn = torch.logsumexp(log_W_neg.unsqueeze(0) + log_neg.unsqueeze(1), dim=-1)
        log_pos_out = torch.logaddexp(log_pp, log_nn)

        # Contributions: pos*neg + neg*pos → neg output
        log_pn = torch.logsumexp(log_W_pos.unsqueeze(0) + log_neg.unsqueeze(1), dim=-1)
        log_np = torch.logsumexp(log_W_neg.unsqueeze(0) + log_pos.unsqueeze(1), dim=-1)
        log_neg_out = torch.logaddexp(log_pn, log_np)

        return log_pos_out, log_neg_out
```

#### 1.2 Polynomial Activation Architecture
```python
# File: elman/models/logspace_polynomial.py

class LogPolynomialElman(nn.Module):
    """Level 6: Log-space Elman with polynomial activation."""

    def __init__(self, dim, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        # ... weights ...

    def forward(self, x, log_h, sign_h):
        """
        Returns: log_h_new, sign_h_new
        """
        # Log-space matmul for R @ h
        log_Rh, sign_Rh = self._log_matvec_signed(self.W_h.weight, log_h, sign_h)

        # Convert to linear for combination with x
        Rh = sign_Rh * torch.exp(log_Rh.clamp(max=20))
        v = self.W_x(x) + Rh + self.b

        # Polynomial activation (log-space native!)
        sign_v = torch.sign(v)
        log_v = torch.log(torch.abs(v) + 1e-8)
        log_candidate = self.alpha * log_v  # h = |v|^alpha

        # Gate
        gate = torch.sigmoid(self.W_gate(x) + self.b_gate)

        # Convert candidate to linear for gated update
        candidate = sign_v * torch.exp(log_candidate.clamp(max=20))
        h_prev = sign_h * torch.exp(log_h.clamp(max=20))
        h_new = (1 - gate) * h_prev + gate * candidate

        # Convert back to log
        sign_h_new = torch.sign(h_new)
        log_h_new = torch.log(torch.abs(h_new) + 1e-10)

        return log_h_new, sign_h_new
```

### Phase 2: Numerical Validation (Week 1-2)

**Tests to run**:

1. **Gradient flow test**:
   ```python
   def test_gradient_flow(model, seq_len=1000):
       """Verify gradients don't vanish over long sequences."""
       h = init_hidden()
       for t in range(seq_len):
           h = model(x[t], h)
       loss = h.sum()
       loss.backward()
       return model.W_h.grad.norm()  # Should not be tiny
   ```

2. **Numerical stability test**:
   ```python
   def test_numerical_stability(model, steps=10000):
       """Verify no NaN/Inf over training."""
       for step in range(steps):
           loss = train_step(model)
           assert torch.isfinite(loss), f"NaN at step {step}"
   ```

3. **Expressivity test**:
   ```python
   def test_expressivity(model):
       """Verify model can learn simple patterns."""
       # Train on copy task, reverse task, etc.
       final_accuracy = train_on_synthetic(model)
       assert final_accuracy > 0.9
   ```

### Phase 3: CUDA Kernel Development (Week 2-3)

**Priority order for kernels**:

#### 3.1 Log-Space Matrix-Vector Multiplication
```cuda
// log_matvec_kernel.cu
// Input: log_W_pos[D, D], log_W_neg[D, D], log_pos_h[B, D], log_neg_h[B, D]
// Output: log_pos_out[B, D], log_neg_out[B, D]

template<typename T>
__global__ void LogPosNegMatVecKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_W_pos,
    const T* __restrict__ log_W_neg,
    const T* __restrict__ log_pos_h,
    const T* __restrict__ log_neg_h,
    T* __restrict__ log_pos_out,
    T* __restrict__ log_neg_out) {

    // Each block handles one (batch, output_dim) pair
    // Use shared memory for warp-level logsumexp reduction
    // ...
}
```

#### 3.2 Fused Gated Update
```cuda
// log_gated_update_kernel.cu
// Fuses: gate computation + gated combination

template<typename T>
__global__ void LogPosNegGatedUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_pos_h,
    const T* __restrict__ log_neg_h,
    const T* __restrict__ log_pos_v,
    const T* __restrict__ log_neg_v,
    const T* __restrict__ gate_raw,      // Pre-sigmoid
    T* __restrict__ log_pos_out,
    T* __restrict__ log_neg_out) {

    // log_gate = logsigmoid(gate_raw)
    // log_one_minus_gate = logsigmoid(-gate_raw)
    // out = logaddexp((1-g)*h, g*v) for both channels
}
```

#### 3.3 Output Projection
```cuda
// log_output_kernel.cu
// Convert log-pos/neg to linear for output layer

template<typename T>
__global__ void LogPosNegToLinearKernel(
    const int n,
    const T* __restrict__ log_pos,
    const T* __restrict__ log_neg,
    T* __restrict__ linear_out) {

    // out = exp(log_pos) - exp(log_neg)
    // Use numerically stable version
}
```

### Phase 4: Integration and Benchmarking (Week 3-4)

1. **Integrate into LadderLM**:
   - Add LogPosNegElman as Level 7
   - Add LogPolynomialElman as Level 6b

2. **Run ablation ladder**:
   ```bash
   ./scripts/train_ladder.sh 0 1 2 3 6b 7 --params 100m --steps 1000
   ```

3. **Compare to Mamba2**:
   - Target: Match Mamba2's 3.924 avg50 @ 1000 steps
   - Track: loss curves, gradient norms, throughput

---

## Part 4: Critical Decision Points

### Decision 1: Representation Choice

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| (log\|h\|, sign) | Simpler, documented | Branching for addition | Use for polynomial |
| (log_pos, log_neg) | No branching, GPU-friendly | 2x memory | **Primary choice** |
| (log_r, theta) | Smooth signs | Complex numbers | Defer |

**Decision**: Start with (log_pos, log_neg), it's the cleanest for GPU.

### Decision 2: Nonlinearity

| Option | Gradient | Bounded | GPU-Friendly |
|--------|----------|---------|--------------|
| tanh | Vanishes | Yes | Yes |
| Polynomial | Constant | No | Yes |
| Gating only | Bounded [0,1] | By design | Yes |

**Decision**: Try gating-only first (LSTM philosophy). Add polynomial if expressivity insufficient.

### Decision 3: Full R vs Diagonal

| Option | Expressivity | Compute | Memory |
|--------|--------------|---------|--------|
| Diagonal r_h | Lower | O(D) | O(D) |
| Full R | Higher | O(D²) logsumexp | O(D²) |

**Decision**: Start diagonal (Level 3 already works), add full R if needed.

---

## Part 5: Open Research Questions

1. **Does gating-only provide enough nonlinearity?**
   - Hypothesis: Yes, gates ARE the nonlinearity (LSTM lesson)
   - Test: Compare gating-only vs polynomial on synthetic tasks

2. **What happens to cancellation dynamics?**
   - In pos/neg representation, exact cancellation → both channels large
   - Is this a feature or bug? Does it provide useful gradient signal?

3. **Optimal initialization for log-space?**
   - Current: Standard initialization in linear space
   - Better: Initialize such that log_pos ≈ log_neg (near-zero mean)?

4. **Memory overhead acceptable?**
   - Pos/neg doubles hidden state memory
   - For 1B model: 2x hidden = maybe 20% total memory increase
   - Likely acceptable for correctness

5. **Can we recover from catastrophic cancellation?**
   - When exp(log_pos) ≈ exp(log_neg), value ≈ 0
   - Gradient still flows to both channels
   - Does this help or hurt learning?

---

## Part 6: Immediate Action Items

### Week 1 Deliverables

- [ ] Implement `LogPosNegElman` in PyTorch (elman/models/logspace_posneg.py)
- [ ] Implement `LogPolynomialElman` in PyTorch (elman/models/logspace_polynomial.py)
- [ ] Add gradient flow test (tests/test_gradient_flow.py)
- [ ] Add numerical stability test (tests/test_numerical_stability.py)
- [ ] Run comparison on tiny model (100K params, synthetic data)

### Week 2 Deliverables

- [ ] Debug any numerical issues found in Week 1
- [ ] Begin CUDA kernel for log-space matvec
- [ ] Benchmark PyTorch vs CUDA implementation
- [ ] Run 100M model on real data (FineWeb)

### Week 3-4 Deliverables

- [ ] Complete CUDA kernels (matvec, gated update, output)
- [ ] Integrate into training pipeline
- [ ] Full Mamba2 comparison at 100M, 500M params
- [ ] Write up results

---

## Appendix A: Mathematical Identities

### Log-Space Operations

```python
# Stable logsigmoid
log_sigmoid(x) = -softplus(-x) = -log(1 + exp(-x))

# Stable log(1 - sigmoid)
log_one_minus_sigmoid(x) = -softplus(x) = -log(1 + exp(x))

# Logaddexp (numerically stable)
logaddexp(a, b) = max(a, b) + log1p(exp(-|a - b|))

# Log of sum of exponentials (logsumexp)
logsumexp([a1, ..., an]) = max_i + log(sum_i exp(a_i - max_i))
```

### Gradient Identities

```python
# Gradient of logsumexp
d(logsumexp(a, b))/da = exp(a) / (exp(a) + exp(b)) = softmax_a  # ∈ [0, 1]

# Gradient of polynomial in log space
d(alpha * log|v|)/d(log|v|) = alpha  # CONSTANT!

# Gradient of tanh (for comparison)
d(tanh(v))/dv = 1 - tanh²(v)  # → 0 when |v| large
```

---

## Appendix B: CUDA Kernel Patterns (Haste Style)

### Pattern 1: Element-wise with shared memory
```cuda
template<typename T>
__global__ void ElementWiseKernel(const int n, const T* a, const T* b, T* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple element-wise operation
        out[idx] = op(a[idx], b[idx]);
    }
}
```

### Pattern 2: Reduction (for logsumexp)
```cuda
template<typename T>
__global__ void LogSumExpKernel(
    const int batch, const int dim,
    const T* input, T* output) {

    extern __shared__ float smem[];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    // Load into shared memory
    float val = (tid < dim) ? input[b * dim + tid] : -1e10f;
    smem[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = logaddexp(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[b] = smem[0];
}
```

### Pattern 3: Matrix-vector with log-space
```cuda
// See log_compute_full_gpu.cu.cc for full implementation
// Key: Use warp shuffle for final reduction step
```

---

## References

- `elman/cuda/lib/log_storage_diagonal_gpu.cu.cc` - Current Level 4 implementation
- `elman/cuda/lib/log_compute_full_gpu.cu.cc` - Current Level 5 implementation
- `docs/logspace.md` - Detailed log-space research notes
- `docs/LOG_SPACE_GRADIENT_SOLUTION.md` - Gradient analysis
- Mamba-2 paper: arXiv:2405.21060
