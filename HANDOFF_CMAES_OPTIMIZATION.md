# Handoff: CMA-ES Optimization for E88/FLA-GDN/Mamba2

**Date**: 2026-01-24
**From**: Claude (elman-proofs session)
**To**: Claude (elman session)

## Summary

We explored using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) as an alternative optimization technique for finding better E88 configurations. This was inspired by the GENREG-sine project (https://github.com/A1CST/GENREG-sine) which shows that evolutionary/gradient-free optimization can discover useful properties like saturation patterns.

## Current State

### E88 Performance (500M scale)
- Best config: dim=1792, depth=38, n_heads=56, n_state=32
- Loss: 1.44 (beats Mamba2's 1.50, close to FLA-GDN's 1.36)
- Key finding: **More state HURTS** - this is why CMA-ES is interesting

### CMA-ES Script Created
File: `cmaes_search.py`

Usage:
```bash
pip install --break-system-packages cma  # Already installed
python cmaes_search.py --model e88 --generations 20 --train_minutes 2 --gpu 1
```

### Initial Test Results
- 24 configs evaluated in ~10 min
- Found h36_n48 with 3.72 loss (1-min training)
- Some configs caused NaN (h57_n40) - instability regions exist
- Param estimation needs improvement

## What Needs to Be Done

### 1. Fix the CMA-ES Script (Priority: High)

The current script has issues:

**a) Param estimation is rough** - many valid configs rejected
- Fix `estimate_params()` to use `calc_dim.py` functions
- Import and use `calc_e88_params`, `find_dim_for_params`

**b) Add stability checks**
- Reject n_state values not in CUDA kernel support list: 4, 8, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 80, 96, 128
- Add early stopping if loss is NaN after 20 steps

**c) Run population in parallel**
- Current: sequential (1 config at a time)
- Better: run 8 configs on 8 GPUs simultaneously
- Each CMA-ES generation takes 8 fitness evals

### 2. Run Full CMA-ES Search (Priority: Medium)

After fixing the script:
```bash
# E88 search - most likely to find improvements
python cmaes_search.py --model e88 --generations 50 --train_minutes 2 --gpu 1

# FLA-GDN search - lower priority
python cmaes_search.py --model fla-gdn --generations 30 --train_minutes 2 --gpu 2

# Mamba2 - lowest priority (already well-optimized)
python cmaes_search.py --model mamba2 --generations 20 --train_minutes 2 --gpu 3
```

### 3. Learnable Saturation Experiment (Priority: High)

The GENREG project shows saturation can be beneficial. E88's tanh creates saturation. We should test learnable saturation thresholds:

In `e88_fla_hybrid.py`:
```python
# In __init__:
self.sat_threshold = nn.Parameter(torch.ones(n_heads))

# In forward, replace:
S = tanh(decay * S + outer(delta, k))
# With:
def learnable_tanh(x, threshold):
    return threshold * torch.tanh(x / threshold)
S = learnable_tanh(decay * S + outer(delta, k), self.sat_threshold[h])
```

This lets each head learn its own saturation level.

### 4. Residual State Experiment (Priority: High)

From CLAUDE.md "Next Steps":
```python
# Instead of:
S = tanh(decay * S + update)

# Try:
S = S + tanh(update)  # ResNet-style residual
```

This untangles gradient flow through the state.

## Key Files

- `cmaes_search.py` - CMA-ES optimization script (needs fixes)
- `calc_dim.py` - Parameter calculation functions
- `elman/models/e88_fla_hybrid.py` - E88 model implementation
- `docs/E88_500M_STUDY.md` - Benchmark results
- `CLAUDE.md` - Current best configs and next steps

## Key Findings from Research

From the Explore agent research on alternative optimization:

1. **Zero-order optimization is scaling up** - DeepZero, CMA-ES improvements make gradient-free viable
2. **Hybrid approaches work best** - Evolution for architecture, gradient descent for weights
3. **Saturation can be adaptive** - Recent papers show learnable activation functions improve performance

Papers to reference:
- DeepZero (ICLR 2024): Coordinate-wise gradient estimation
- CMA-ES improvements (2024-2025): Multi-modal, noise-robust variants
- Polynomial activation learning (2025): Learning optimal nonlinearities

## Expected Outcomes

If CMA-ES works well on E88:
- Could find better n_heads/n_state combinations than h56_n32
- Might discover that h40_n36 or similar works better
- Would provide evidence that E88's fitness landscape is indeed tricky

If learnable saturation works:
- Per-head saturation thresholds could improve loss by 0.02-0.05 nats
- Would validate GENREG's finding that saturation is useful

## Commands to Continue

```bash
# 1. First, fix the CMA-ES script to use calc_dim.py
cd /home/erikg/elman
# Edit cmaes_search.py to import and use calc_e88_params

# 2. Run a longer E88 search
python cmaes_search.py --model e88 --generations 50 --train_minutes 2 --gpu 1

# 3. Test learnable saturation (after implementing)
python train.py --level E88 --dim 1792 --depth 38 --n_heads 56 --n_state 32 \
  --learnable_saturation --bf16 --batch_size 32 --train_minutes 10

# 4. Test residual state (after implementing)
python train.py --level E88 --dim 1792 --depth 38 --n_heads 56 --n_state 32 \
  --residual_state --bf16 --batch_size 32 --train_minutes 10
```
