# Spectral Theory Validation: What Could Be Wrong?

## Background

We developed a spectral theory in `elman-proofs` that unifies three hypotheses about optimal low-rank ratio:
- Singular value concentration
- Gradient condition number
- Manifold curvature

The theory predicts: `r*/d = ε^{1/(2α-1)}` where α is the power law exponent of singular value decay.

For E5's observed r*/d ≈ 17% and assuming ε = 0.05, we derived α ≈ 1.35.

**But we never actually verified any of this empirically.** This document outlines what needs to be checked.

---

## Issue 1: We Never Measured α

**The Problem**: We reverse-engineered α from the observed optimal ratio, assuming ε = 0.05. This is circular.

**What to Check**:
1. Train a model and extract the weight matrices (W_h or U, V)
2. Compute SVD of the matrices
3. Plot log(σᵢ) vs log(i) — should be linear if power law holds
4. Fit the slope to get α directly
5. Compare measured α to predicted α = 1.35

**Experiment**:
```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# After training, extract weights
W_h = model.rnn.weight_hh  # or U @ V for low-rank

# SVD
U, S, Vh = torch.linalg.svd(W_h)
singular_values = S.cpu().numpy()

# Plot log-log
indices = np.arange(1, len(singular_values) + 1)
plt.loglog(indices, singular_values, 'o-')
plt.xlabel('Index i')
plt.ylabel('Singular value σᵢ')
plt.title('Singular Value Spectrum')

# Fit power law: log(σ) = -α*log(i) + c
log_i = np.log(indices)
log_s = np.log(singular_values)
alpha, c = np.polyfit(log_i, log_s, 1)
print(f"Measured α = {-alpha:.3f}")  # Note: negative because decay
```

**Expected**: α ≈ 1.35 if theory is correct.

---

## Issue 2: Power Law Might Not Hold

**The Problem**: Real spectra might not follow power laws. Could be:
- Exponential decay: σᵢ ∝ e^{-βi}
- Multi-modal: different regimes for different i ranges
- Noise floor: small σᵢ all similar magnitude

**What to Check**:
1. Compare fits: power law vs exponential vs other models
2. Look for "knees" in the spectrum where behavior changes
3. Check residuals of power law fit

**Experiment**:
```python
from scipy.optimize import curve_fit

# Power law
def power_law(i, a, alpha):
    return a * i ** (-alpha)

# Exponential
def exponential(i, a, beta):
    return a * np.exp(-beta * i)

# Fit both
popt_power, _ = curve_fit(power_law, indices, singular_values, p0=[1, 1])
popt_exp, _ = curve_fit(exponential, indices, singular_values, p0=[1, 0.01])

# Compare R² or AIC
```

**If power law doesn't hold**: The formula r*/d = ε^{1/(2α-1)} is wrong, and we need a different theory.

---

## Issue 3: The ε = 0.05 Assumption is Arbitrary

**The Problem**: We assumed 95% variance threshold (ε = 0.05 residual). Why 95%? Could be:
- 90% (ε = 0.10) → different α
- 99% (ε = 0.01) → different α
- Information-theoretic threshold (entropy-based) instead of variance

**What to Check**:
1. Vary the rank ratio and measure actual variance captured
2. Plot loss vs variance captured — where does it plateau?
3. The "knee" in loss-vs-variance might reveal the true ε

**Experiment**:
```python
# For each rank ratio r/d
for ratio in [0.05, 0.10, 0.15, 0.17, 0.20, 0.25, 0.30]:
    r = int(ratio * d)
    # Train model with this rank
    model = ElmanLowRank(d=1536, r=r)
    train(model)
    final_loss = evaluate(model)

    # Also compute variance captured by rank-r approximation of trained W
    U, S, Vh = svd(model.get_W())
    variance_captured = sum(S[:r]**2) / sum(S**2)

    print(f"r/d={ratio:.2f}, loss={final_loss:.3f}, var={variance_captured:.3f}")
```

**Look for**: The relationship between variance captured and loss. If loss plateaus at 90% variance, then ε = 0.10, not 0.05.

---

## Issue 4: We Analyzed Static Matrices, But Training is Dynamic

**The Problem**: Optimal rank might change during training:
- Early: high rank for exploration
- Late: low rank for regularization/generalization

**What to Check**:
1. Track singular value spectrum throughout training
2. Does α change during training?
3. Does the "effective rank" (participation ratio) change?

**Experiment**:
```python
# Effective rank = (Σσᵢ)² / Σσᵢ²
def effective_rank(singular_values):
    return (singular_values.sum() ** 2) / (singular_values ** 2).sum()

# Track during training
for step in range(total_steps):
    train_step(model)
    if step % 1000 == 0:
        S = svd(model.get_W())[1]
        eff_r = effective_rank(S)
        print(f"Step {step}: effective_rank = {eff_r:.1f}")
```

**Look for**: Does effective rank decrease during training? If so, the "optimal" rank at initialization might differ from optimal rank at convergence.

---

## Issue 5: U·V Factorization Spectrum ≠ Dense W Spectrum

**The Problem**: For low-rank models, we have W = U·V. The spectrum of U·V might differ from:
- The spectrum we'd get from a dense W trained on the same task
- The spectra of U and V individually

**What to Check**:
1. Train both dense and low-rank models to convergence
2. Compare SVD spectra of dense W vs (U·V)
3. Do they have the same α?

**Experiment**:
```python
# Train dense model
model_dense = ElmanDense(d=512)
train(model_dense)
S_dense = svd(model_dense.W_h)[1]

# Train low-rank model (with same effective capacity?)
model_lowrank = ElmanLowRank(d=1536, r=270)  # Same params roughly
train(model_lowrank)
S_lowrank = svd(model_lowrank.U @ model_lowrank.V)[1]

# Compare spectra
plt.loglog(S_dense / S_dense[0], label='Dense d=512')
plt.loglog(S_lowrank / S_lowrank[0], label='Low-rank d=1536 r=270')
```

---

## Issue 6: Multiple Matrices Interact

**The Problem**: We focused on W_h, but there are multiple matrices:
- W_x (input projection)
- W_h (recurrence) — possibly factored as U·V
- Output projection

The effective spectrum might depend on the PRODUCT of these through the recurrence.

**What to Check**:
1. Analyze the Jacobian of the full recurrence: ∂h'/∂h
2. For Elman: this is diag(1 - tanh²(z)) · W_h
3. The effective spectrum depends on BOTH W_h and the activation pattern

**Experiment**:
```python
# Compute Jacobian spectrum during forward pass
def jacobian_spectrum(model, x, h):
    z = model.W_x @ x + model.W_h @ h
    tanh_deriv = 1 - torch.tanh(z) ** 2
    jacobian = torch.diag(tanh_deriv) @ model.W_h
    return svd(jacobian)[1]

# Average over many (x, h) pairs from real data
spectra = []
for batch in dataloader:
    for x, h in forward_with_states(model, batch):
        spectra.append(jacobian_spectrum(model, x, h))
mean_spectrum = torch.stack(spectra).mean(dim=0)
```

This might reveal that the EFFECTIVE α differs from the STATIC α of W_h.

---

## Issue 7: Linear Theory Applied to Nonlinear Model

**The Problem**: Our spectral theory is about linear algebra. Elman has tanh nonlinearity.

**What to Check**:
1. Compare optimal rank ratio for LINEAR RNN vs Elman
2. Theory predicts linear might need HIGHER ratio (no nonlinear compensation)
3. Or might need LOWER ratio (no gradient vanishing through tanh)

**Experiment**:
```python
# Compare linear vs nonlinear at same hidden dim
results = {}
for model_type in ['linear', 'elman']:
    for ratio in [0.10, 0.15, 0.17, 0.20, 0.25]:
        r = int(ratio * 1536)
        if model_type == 'linear':
            model = LinearRNNLowRank(d=1536, r=r)
        else:
            model = ElmanLowRank(d=1536, r=r)
        train(model)
        results[(model_type, ratio)] = evaluate(model)
```

**Key question**: Is optimal ratio the same for linear and nonlinear?

---

## Issue 8: Scaling Effects

**The Problem**: We only tested at one scale (d=1536, ~50M params). Does optimal ratio change with:
- Model size (d)?
- Dataset size?
- Sequence length?

**What to Check**:
1. Sweep rank ratios at multiple scales
2. Does optimal r*/d stay constant at ~17%?
3. Or does it depend on d?

**Experiment**:
```python
# Test at multiple scales
for d in [256, 512, 1024, 1536, 2048]:
    best_ratio = None
    best_loss = float('inf')
    for ratio in [0.10, 0.15, 0.17, 0.20, 0.25, 0.30]:
        r = int(ratio * d)
        model = ElmanLowRank(d=d, r=r)
        train(model)
        loss = evaluate(model)
        if loss < best_loss:
            best_loss = loss
            best_ratio = ratio
    print(f"d={d}: optimal ratio = {best_ratio}")
```

**Prediction from theory**: Optimal ratio depends only on α, which is a property of the TASK (language structure), not the model size. So optimal ratio should be constant across scales.

If it's NOT constant, then α varies with scale, which would be very interesting.

---

## Issue 9: E5 Experimental Confounds

**The Problem**: Was the E5 experiment well-controlled?

**What to Check**:
1. Was training fully converged for all configurations?
2. Were hyperparameters (LR, batch size) optimized per-configuration or shared?
3. Could there be initialization effects?

**Specific concerns**:
- Lower rank = fewer parameters = might need different learning rate
- Higher rank = more parameters = might need more steps to converge
- If we stopped too early, higher rank might appear worse than it is

**Experiment**: Re-run with:
- Longer training (2-3x steps)
- Per-configuration LR tuning
- Multiple seeds

---

## Issue 10: What About W_x?

**The Problem**: We focused entirely on W_h (the recurrence). But W_x (input projection) also matters.

**What to Check**:
1. Should W_x also be low-rank?
2. What's the optimal rank ratio for W_x vs W_h?
3. Do they share the same α?

**Experiment**:
```python
# Factorial design
for r_h in [0.1, 0.17, 0.25]:  # W_h rank ratio
    for r_x in [0.1, 0.17, 0.25, 1.0]:  # W_x rank ratio (1.0 = full rank)
        model = ElmanLowRank(d=1536, r_h=int(r_h*1536), r_x=int(r_x*1536))
        train(model)
        print(f"r_h={r_h}, r_x={r_x}: loss={evaluate(model)}")
```

---

## Priority Order for Validation

1. **Measure α directly** (Issue 1) — fundamental check
2. **Check power law fit** (Issue 2) — is the model even right?
3. **Vary ε empirically** (Issue 3) — find the real threshold
4. **Compare linear vs nonlinear** (Issue 7) — test key prediction
5. **Check scaling** (Issue 8) — does theory generalize?
6. **Track dynamics** (Issue 4) — understand training
7. **Analyze Jacobian spectrum** (Issue 6) — effective vs static
8. **Re-run E5 carefully** (Issue 9) — experimental validity
9. **Test W_x** (Issue 10) — complete picture
10. **Compare factored vs dense spectrum** (Issue 5) — understand factorization

---

## Summary

The spectral theory is elegant and makes predictions, but rests on unverified assumptions:

| Assumption | Status | Priority to Check |
|------------|--------|-------------------|
| Power law decay | Assumed, not measured | HIGH |
| α ≈ 1.35 | Derived circularly | HIGH |
| ε = 0.05 (95% threshold) | Arbitrary | HIGH |
| Static analysis sufficient | Unknown | MEDIUM |
| Linear theory applies to nonlinear | Plausible but untested | MEDIUM |
| Scale-independent | Unknown | MEDIUM |

The theory could be:
- **Correct**: All checks pass, α ≈ 1.35 measured directly
- **Approximately correct**: Power law is rough fit, α in range [1.2, 1.5]
- **Wrong model, right intuition**: Not power law, but spectral concentration still explains results
- **Coincidence**: 17% works for other reasons entirely

Run the experiments to find out.
