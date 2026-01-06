# Spectral Low-Rank Research Plan

## Objective

Validate and extend the spectral theory of optimal low-rank factorization for recurrent neural networks.

**Core Claim**: The optimal rank ratio r*/d ≈ 17% emerges from power law singular value decay with exponent α ≈ 1.35.

---

## Phase 1: Empirical Validation (Immediate)

### 1.1 Measure Singular Value Spectrum
**Goal**: Directly measure α from trained models

**Tasks**:
- [ ] Train baseline dense Elman (d=512) to convergence
- [ ] Train E5 low-rank Elman (d=1536, r=270) to convergence
- [ ] Extract weight matrices, compute SVD
- [ ] Plot log(σᵢ) vs log(i), fit slope to get α
- [ ] Compare measured α to predicted α = 1.35

**Success Criterion**: Measured α within [1.2, 1.5]

### 1.2 Test Power Law Hypothesis
**Goal**: Verify that power law is the correct model

**Tasks**:
- [ ] Fit multiple models: power law, exponential, stretched exponential
- [ ] Compare goodness of fit (R², AIC, BIC)
- [ ] Check for regime changes (knees in spectrum)
- [ ] Document deviations from power law

**Success Criterion**: Power law provides best fit, or identify correct alternative

### 1.3 Find True ε Threshold
**Goal**: Determine the variance threshold that predicts optimal rank

**Tasks**:
- [ ] Train models at rank ratios: 5%, 10%, 15%, 17%, 20%, 25%, 30%
- [ ] For each, compute variance captured by that rank
- [ ] Plot loss vs variance captured
- [ ] Find the "knee" where additional variance stops helping
- [ ] Derive empirical ε from this knee

**Success Criterion**: Identify ε such that r*/d = ε^{1/(2α-1)} holds

---

## Phase 2: Theory Extension (Near-term)

### 2.1 Jacobian Spectrum Analysis
**Goal**: Understand effective (not static) spectral properties

**Tasks**:
- [ ] Compute Jacobian ∂h'/∂h = diag(tanh'(z)) · W_h during forward pass
- [ ] Track Jacobian singular values over many (x, h) pairs
- [ ] Compare static W_h spectrum to average Jacobian spectrum
- [ ] Derive "effective α" from Jacobian analysis

**Deliverable**: Understanding of how nonlinearity modifies spectral properties

### 2.2 Linear vs Nonlinear Comparison
**Goal**: Test prediction that linear/nonlinear have different optimal ratios

**Tasks**:
- [ ] Implement linear RNN with low-rank: h' = A·h + B·x (no tanh)
- [ ] Sweep rank ratios for linear RNN
- [ ] Compare optimal ratio: linear vs Elman
- [ ] Analyze why they differ (if they do)

**Prediction**: Linear RNN optimal ratio ≠ Elman optimal ratio

### 2.3 Multi-Matrix Analysis
**Goal**: Extend theory to full model (W_h, W_x, output)

**Tasks**:
- [ ] Analyze spectrum of W_x
- [ ] Test low-rank W_x independently and jointly with low-rank W_h
- [ ] Understand interaction between input and recurrence spectra

**Deliverable**: Complete spectral characterization of Elman architecture

---

## Phase 3: Scaling Laws (Medium-term)

### 3.1 Scale Invariance Test
**Goal**: Check if optimal ratio is scale-independent

**Tasks**:
- [ ] Test at d ∈ {256, 512, 1024, 1536, 2048}
- [ ] For each d, find optimal rank ratio
- [ ] Check if r*/d ≈ 17% across all scales
- [ ] If not, model how optimal ratio depends on d

**Prediction**: Optimal ratio depends only on α (task property), not d (model property)

### 3.2 Dataset Scaling
**Goal**: Check if optimal ratio depends on data quantity

**Tasks**:
- [ ] Train with 10%, 25%, 50%, 100% of data
- [ ] Find optimal ratio at each data scale
- [ ] Check for interactions between data scale and optimal ratio

**Question**: Does more data require higher or lower rank?

### 3.3 Connect to Neural Scaling Laws
**Goal**: Relate spectral exponent α to loss scaling exponent β

**Tasks**:
- [ ] Fit scaling law: Loss ∝ N^{-β} for various N
- [ ] Check if β relates to α (e.g., β ≈ 1/(2α)?)
- [ ] Understand connection theoretically

**Speculation**: The "bitter lesson" scaling might emerge from spectral structure

---

## Phase 4: Learning Dynamics (Medium-term)

### 4.1 Training Dynamics
**Goal**: Understand how spectrum evolves during training

**Tasks**:
- [ ] Track singular value spectrum every 1000 steps
- [ ] Track effective rank (participation ratio) over training
- [ ] Identify phases: does α change during training?
- [ ] Check if optimal rank at init ≠ optimal rank at convergence

**Deliverable**: Dynamic theory of spectral evolution

### 4.2 Gradient Flow Analysis
**Goal**: Connect spectrum to gradient quality

**Tasks**:
- [ ] Track gradient SNR during training
- [ ] Correlate with condition number κ_r
- [ ] Test if gradient quality predicts convergence speed
- [ ] Verify: lower rank → better gradient flow (up to capacity limit)

**Deliverable**: Understanding of optimization landscape for low-rank

### 4.3 Learning Rate Scaling
**Goal**: Optimal LR as function of rank

**Tasks**:
- [ ] For each rank ratio, sweep learning rates
- [ ] Find optimal LR per rank
- [ ] Model: optimal_LR ∝ f(rank)?
- [ ] Connect to condition number theory

**Practical Output**: LR heuristic for low-rank models

---

## Phase 5: Theoretical Foundations (Long-term)

### 5.1 Formalize in Lean
**Goal**: Rigorous proofs of key relationships

**Tasks**:
- [ ] Prove: power law decay → optimal rank formula
- [ ] Prove: condition number bounds for power law
- [ ] Prove: manifold dimension formula
- [ ] Connect to existing LinearCapacity theorems

**Location**: ~/elman-proofs/ElmanProofs/Expressivity/

### 5.2 Information-Theoretic Bounds
**Goal**: Connect spectral theory to information theory

**Tasks**:
- [ ] Formalize state information capacity
- [ ] Derive bits/step bounds from rank
- [ ] Connect to rate-distortion theory
- [ ] Prove optimality of power law allocation?

**Question**: Is power law decay OPTIMAL for some objective?

### 5.3 Program Learning Connection
**Goal**: Extend to learning programs, not just functions

**Tasks**:
- [ ] Define "program complexity" in spectral terms
- [ ] Connect description length to rank
- [ ] Derive sample complexity from spectral properties
- [ ] Relate to Kolmogorov complexity

**Speculation**: Low-rank ≈ simple programs ≈ fast learning

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 | 1-2 weeks | None |
| Phase 2 | 2-3 weeks | Phase 1 |
| Phase 3 | 2-3 weeks | Phase 1 |
| Phase 4 | 3-4 weeks | Phases 1, 2 |
| Phase 5 | Ongoing | Phases 1-4 |

---

## Resources Needed

1. **Compute**: GPU hours for training sweeps
2. **Code**: Modify elman codebase for spectral analysis
3. **Theory**: Time for Lean proofs

---

## Key Risks

1. **Power law doesn't hold**: Would invalidate core theory
2. **α varies unpredictably**: Would make theory less useful
3. **Optimal ratio is scale-dependent**: Would complicate practical recommendations
4. **Training dynamics dominate**: Static analysis might be insufficient

---

## Success Metrics

1. **Validation**: Measured α within 20% of predicted (1.35 ± 0.27)
2. **Predictive**: Theory correctly predicts optimal ratio for new configurations
3. **Generalizable**: Results hold across scales and architectures
4. **Rigorous**: Key results formalized in Lean
