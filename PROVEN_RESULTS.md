# Proven Results: Low-Rank Spectral Theory

## Status Summary

This document tracks what has been **formally proven** (in Lean), what is **mathematically derived** (on paper), and what is **empirically observed** (experiments).

---

## Formally Proven (Lean 4)

Location: `~/elman-proofs/ElmanProofs/Expressivity/`

### Linear State Capacity (LinearCapacity.lean)

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| `linear_state_is_sum` | State = Σ A^{T-1-t} B x_t | Explicit formula for linear RNN state |
| `state_additive` | State is additive in inputs | Linearity property |
| `same_state_same_future` | Same state → same future | State sufficiency |
| `reachable_is_subspace` | Reachable states form subspace | Linear structure |
| `reachable_dim_bound` | dim(reachable) ≤ n | **Capacity bound** |

**Link to Low-Rank**: These theorems establish that linear RNNs have bounded capacity = hidden dimension. Low-rank factorization doesn't change the hidden dimension, so **state capacity = d regardless of rank r**.

### Associativity Separation (Associativity.lean)

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| `LinearScanElement.instMonoid` | Linear transitions form monoid | Enables parallel scan |
| `tanh_composition_not_linear` | Tanh RNN ≠ single affine step | **Nonlinearity is real** |
| `polynomial_rnn_not_associative` | |x|^α RNN is non-associative | Separation result |

**Link to Low-Rank**: Establishes that linear and nonlinear RNNs have fundamentally different structure. Spectral theory applies directly to linear; needs modification for nonlinear.

### Gradient Dynamics (GradientDynamics.lean)

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| `tanh_deriv_strict` | \|tanh'(x)\| < 1 for x ≠ 0 | Gradient factor < 1 |
| `mamba2_gradient_h_independent` | Mamba2 gradient independent of h | Simpler gradient |
| `elman_gradient_varies_with_h` | Elman gradient depends on h | More complex |
| `mamba2_tradeoff` | Mamba2 has better gradient quality | **Why Mamba2 learns well** |

**Link to Low-Rank**: Gradient quality affects learning efficiency. Low-rank affects gradient flow through the factorization U·V.

### Expansion Tradeoff (ExpansionTradeoff.lean)

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| `expansion_rank_bottleneck` | Expansion pathway has rank ≤ d | Bottleneck theorem |
| `wide_beats_narrow_on_capacity` | Wider hidden > expansion | **Why no expansion** |
| `diagonal_more_efficient` | Diagonal better than dense | Parameter efficiency |

**Link to Low-Rank**: Establishes that state capacity (= hidden dim) is the key resource, not per-step computation. Motivates: maximize d at fixed params → use low-rank W_h.

### Low-Rank Capacity (LowRankCapacity.lean)

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| `e5_triple_capacity` | E5 has 3x state capacity of E1 | **Why E5 wins** |
| `lowRank_between` | diagonal < low-rank < dense (efficiency) | Ordering |
| `diagonal_most_efficient` | Diagonal is most param-efficient | Upper bound on efficiency |
| `lowRank_increases_snr` | Low-rank increases gradient SNR | Learning benefit |

**Link**: Directly formalizes E5 experimental findings.

### Spectral Low-Rank (SpectralLowRank.lean)

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| `powerLaw_decreasing` | Power law σᵢ ∝ i^{-α} is decreasing | Spectrum structure |
| `powerLaw_pos` | Power law values are positive | Well-defined |
| `condition_grows` | Condition number κ_r = r^α grows with r | **Rank-condition tradeoff** |
| `manifold_dim_mono` | Rank manifold dim increases with r | Geometric structure |

**Link**: Core spectral theory relating power law exponent α to optimal rank.

---

## Mathematically Derived (Paper Proofs)

These are derived but not yet formalized in Lean:

### Optimal Rank Formula

**Claim**: For power law decay σᵢ ∝ i^{-α} and variance threshold ε:
```
r*/d = ε^{1/(2α-1)}
```

**Derivation**:
1. Variance in top-r: Var(r) = Σᵢ≤r σᵢ² ≈ ∫₁ʳ i^{-2α} di = r^{1-2α}/(1-2α)
2. Total variance: Var(d) ≈ d^{1-2α}/(1-2α)
3. Fraction captured: Var(r)/Var(d) = (r/d)^{1-2α}
4. For 1-ε captured: (r/d)^{1-2α} = 1-ε ≈ 1
5. Residual: (r/d)^{2α-1} = ε
6. Solve: r/d = ε^{1/(2α-1)}

**Status**: Derivation complete, awaiting Lean formalization.

### Implied α from Observations

**Claim**: Given observed r*/d and assumed ε:
```
α = (log(ε)/log(r*/d) + 1) / 2
```

**For E5**: r*/d = 0.17, ε = 0.05 → α ≈ 1.35

**Status**: Definition in Lean, numerical verification needed.

### Condition Number Scaling

**Claim**: For power law with exponent α:
```
κ_r = σ_1/σ_r = r^α
```

**Implication**: Higher rank → worse conditioning → slower optimization

**Status**: Proven for power law case in Lean.

### Learning Score

**Claim**: Learning efficiency combines:
```
score = SNR × utilization / √κ
```

Where:
- SNR = signal-to-noise ratio in gradients
- utilization = fraction of gradient used effectively
- κ = condition number

**Status**: Definition in Lean, relationship to rank needs formalization.

---

## Empirically Observed (Experiments)

### E5 Results (from E5_EXPERIMENT_REPORT.md)

| Config | dim | rank | ratio | Loss |
|--------|-----|------|-------|------|
| E5 | 1536 | 270 | 17% | **1.39** |
| E1 | 512 | 512 | 100% | 1.55 |

**Observation**: 3x larger hidden dim with 17% rank ratio beats full-rank baseline.

**Status**: Single experiment, needs replication and controls.

### X-Gated Elman (from ExpressivityGradientTradeoff.lean docs)

| Model | Tok/s | Loss |
|-------|-------|------|
| X-Gated Elman | 166k | 1.89 |
| Mamba2 | 96k | 1.84 |

**Observation**: X-Gated is 1.73x faster with similar loss.

**Status**: Relates to gradient structure, not directly to low-rank.

---

## Dependency Graph

```
LinearCapacity (proven)
    └── "State capacity = d"
            │
            ├── ExpansionTradeoff (proven)
            │       └── "Wider > Expansion"
            │               │
            │               └── Motivates: maximize d
            │
            └── LowRankCapacity (proven)
                    └── "E5 has 3x capacity"
                            │
                            └── SpectralLowRank (proven)
                                    │
                                    ├── powerLaw_decreasing
                                    ├── condition_grows
                                    └── manifold_dim_mono
                                            │
                                            └── Optimal Rank Formula (derived)
                                                    │
                                                    └── r*/d = ε^{1/(2α-1)}
                                                            │
                                                            └── α = 1.35 (observed)
```

---

## What Needs Proving

### High Priority (Required for Theory Completeness)

1. **Variance integral formula**
   - Statement: For power law σᵢ ∝ i^{-α}, Σᵢ≤r σᵢ² ≈ r^{1-2α}/(1-2α)
   - Status: Paper derivation exists
   - Difficulty: Medium (needs careful analysis bounds)

2. **Optimal rank derivation**
   - Statement: r*/d = ε^{1/(2α-1)} minimizes loss
   - Status: Informal derivation
   - Difficulty: Hard (needs loss model)

3. **Learning rate bound**
   - Statement: Lower rank → faster convergence (up to capacity limit)
   - Status: Intuition only
   - Difficulty: Hard (needs optimization theory)

### Medium Priority (Theory Extensions)

4. **Nonlinear correction**
   - Statement: Tanh modifies effective α to α_eff
   - Status: Conjectured
   - Difficulty: Very hard (nonlinear analysis)

5. **Jacobian spectrum**
   - Statement: Effective spectrum from Jacobian differs from static W_h
   - Status: Conjectured
   - Difficulty: Hard (needs stochastic analysis)

6. **Multi-matrix interaction**
   - Statement: W_h and W_x spectra combine to determine capacity
   - Status: Not started
   - Difficulty: Medium

### Lower Priority (Interesting Extensions)

7. **Scaling law connection**
   - Statement: α relates to neural scaling exponent β
   - Status: Speculation
   - Difficulty: Very hard (empirical relationship)

8. **Information-theoretic optimality**
   - Statement: Power law is optimal for some objective
   - Status: Speculation
   - Difficulty: Research-level

9. **Program complexity**
   - Statement: Low-rank ≈ simple programs
   - Status: Philosophical
   - Difficulty: Open problem

---

## Summary Table

| Result | Status | Location | Confidence |
|--------|--------|----------|------------|
| State capacity = d | **Proven** | LinearCapacity.lean | 100% |
| Wider > Expansion | **Proven** | ExpansionTradeoff.lean | 100% |
| E5 3x capacity | **Proven** | LowRankCapacity.lean | 100% |
| Power law decreasing | **Proven** | SpectralLowRank.lean | 100% |
| Condition grows with rank | **Proven** | SpectralLowRank.lean | 100% |
| r*/d = ε^{1/(2α-1)} | Derived | Paper | 90% |
| α ≈ 1.35 for LMs | Observed | E5 experiments | 60% (needs validation) |
| Power law is correct model | Assumed | - | 50% (needs measurement) |
| ε = 0.05 is right threshold | Assumed | - | 40% (needs measurement) |
