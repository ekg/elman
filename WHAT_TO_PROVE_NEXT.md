# What to Prove Next

## Immediate Targets for Lean Formalization

### 1. Power Law Variance Formula

**Statement**:
```lean
theorem powerLaw_variance_sum (α : ℝ) (hα : α > 1/2) (r : ℕ) (hr : r > 0) :
    ∃ C : ℝ, C > 0 ∧
    (∑ i in Finset.range r, powerLawSigma i α ^ 2) ≤ C * (r : ℝ) ^ (1 - 2*α)
```

**Why**: This is the key lemma for deriving the optimal rank formula. It says that the sum of squared singular values up to rank r scales as r^{1-2α}.

**Approach**:
1. Use integral comparison: Σ ≤ ∫
2. Compute integral of i^{-2α}
3. Handle edge cases for α = 1/2

**Difficulty**: Medium

---

### 2. Variance Ratio Bound

**Statement**:
```lean
theorem variance_ratio_powerLaw (α : ℝ) (hα : α > 1/2) (r d : ℕ) (hrd : r ≤ d) :
    (∑ i in Finset.range r, powerLawSigma i α ^ 2) /
    (∑ i in Finset.range d, powerLawSigma i α ^ 2) ≥
    1 - ((r : ℝ) / d) ^ (2*α - 1)
```

**Why**: This connects rank ratio to variance captured, which is the core of the theory.

**Approach**:
1. Use variance sum formula for numerator and denominator
2. Take ratio
3. Simplify

**Difficulty**: Medium (depends on #1)

---

### 3. Optimal Rank Characterization

**Statement**:
```lean
theorem optimal_rank_formula (α ε : ℝ) (hα : α > 1/2) (hε : 0 < ε ∧ ε < 1) :
    let r_star := ε ^ (1 / (2*α - 1))
    -- At r_star, variance captured is exactly 1 - ε
    True  -- Need to formalize what "optimal" means
```

**Why**: This is THE main theorem - it predicts optimal rank from spectral exponent.

**Challenge**: Need to define what "optimal" means:
- Minimum rank to capture (1-ε) variance?
- Maximum rank before condition number explodes?
- Best tradeoff between capacity and trainability?

**Approach**:
1. First prove: rank r captures variance ≥ 1 - (r/d)^{2α-1}
2. Then: setting (r/d)^{2α-1} = ε gives r/d = ε^{1/(2α-1)}

**Difficulty**: Medium (mostly bookkeeping once #2 is done)

---

### 4. Condition Number - Convergence Connection

**Statement**:
```lean
theorem condition_convergence_rate (κ : ℝ) (hκ : κ ≥ 1) (t : ℕ) :
    -- After t steps of gradient descent on κ-conditioned problem:
    -- error ≤ (1 - 1/κ)^t * initial_error
    True  -- Connect to existing GD proofs
```

**Why**: This connects the spectral theory (condition number grows as r^α) to learning speed.

**Approach**:
1. Use existing `convex_convergence_rate` from Flow.lean
2. Instantiate with condition number from low-rank
3. Show: lower rank → faster convergence

**Difficulty**: Hard (needs to connect different parts of the codebase)

---

### 5. Gradient SNR for Low-Rank

**Statement**:
```lean
theorem lowRank_gradient_snr (d r : ℕ) (hr : r < d) :
    -- Gradient through U·V has higher SNR than through dense W
    -- Because: rank-r projection filters out noise in bottom (d-r) directions
    True  -- Need to formalize SNR
```

**Why**: This explains WHY low-rank helps learning, not just capacity.

**Challenge**: Need to define:
- What is "signal" vs "noise" in gradients?
- How does rank-r projection filter noise?

**Approach**:
1. Define gradient decomposition into signal + noise
2. Show rank-r projection preserves signal (assuming signal is low-rank)
3. Show rank-r projection reduces noise (assuming noise is isotropic)

**Difficulty**: Hard (needs careful definitions)

---

## Medium-Term Targets

### 6. Jacobian Spectrum Bound

**Statement**:
```lean
theorem jacobian_spectrum_bound (W : Matrix n n ℝ) (tanh_deriv : n → ℝ)
    (h_tanh : ∀ i, 0 ≤ tanh_deriv i ∧ tanh_deriv i ≤ 1) :
    -- Jacobian = diag(tanh_deriv) * W
    -- σᵢ(Jacobian) ≤ σᵢ(W)
    True
```

**Why**: Shows that nonlinearity can only REDUCE effective singular values, not increase them.

**Difficulty**: Medium

---

### 7. Effective Rank During Training

**Statement**:
```lean
-- This requires formalizing training dynamics, which is very hard
-- Placeholder for the concept
theorem effective_rank_decreases_during_training :
    -- As training progresses, effective rank tends to decrease
    -- (Implicit regularization effect)
    True
```

**Why**: Would explain why low-rank constraint might not hurt much - models naturally become low-rank.

**Difficulty**: Very hard (open research problem)

---

### 8. Linear vs Nonlinear Optimal Rank

**Statement**:
```lean
theorem linear_vs_nonlinear_optimal_rank :
    -- Optimal rank ratio for linear RNN ≠ optimal rank ratio for Elman
    -- (in general)
    True
```

**Why**: Tests a key prediction of the theory - nonlinearity should change the optimal ratio.

**Challenge**: Need to define "optimal" for both architectures.

**Difficulty**: Hard

---

## Long-Term / Research-Level

### 9. Information-Theoretic Capacity

**Statement**:
```lean
theorem information_capacity_rank_r (d r : ℕ) (precision : ℕ) :
    -- Information storable in rank-r matrix ≤ 2*d*r*precision bits
    True
```

**Why**: Would give a rigorous bound on what can be learned with rank-r.

**Difficulty**: Hard (needs information theory formalization)

---

### 10. Power Law Optimality

**Statement**:
```lean
theorem power_law_is_optimal :
    -- Among all spectral distributions with fixed total variance,
    -- power law maximizes [something] subject to [constraints]
    True
```

**Why**: Would explain WHY power law emerges, not just that it does.

**Difficulty**: Research-level (probably open problem)

---

## Proof Strategy Overview

```
Current state (proven):
    LinearCapacity, Associativity, GradientDynamics,
    ExpansionTradeoff, LowRankCapacity, SpectralLowRank (basic)

Next step (targets 1-3):
    Variance formulas → Optimal rank formula

Then (targets 4-5):
    Connect to convergence → Learning speed bounds

Medium-term (targets 6-8):
    Jacobian analysis → Effective spectra
    Linear vs nonlinear comparison

Long-term (targets 9-10):
    Information theory → Capacity bounds
    Optimality → Why power law?
```

---

## Recommended Order

1. **powerLaw_variance_sum** (Target 1) - foundational lemma
2. **variance_ratio_powerLaw** (Target 2) - builds on #1
3. **optimal_rank_formula** (Target 3) - main theorem, builds on #2
4. **condition_convergence_rate** (Target 4) - connects to learning
5. **jacobian_spectrum_bound** (Target 6) - handles nonlinearity

After these, the core theory would be rigorous. Remaining targets are extensions.

---

## Dependencies

```
                    ┌─────────────────────┐
                    │ powerLaw_decreasing │ (done)
                    │    powerLaw_pos     │ (done)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ powerLaw_variance   │ (Target 1)
                    │       _sum          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ variance_ratio_     │ (Target 2)
                    │    powerLaw         │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
┌─────────▼─────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│ optimal_rank_     │ │ condition_      │ │ lowRank_        │
│    formula        │ │ convergence_    │ │ gradient_snr    │
│   (Target 3)      │ │ rate (Target 4) │ │   (Target 5)    │
└───────────────────┘ └─────────────────┘ └─────────────────┘
```
