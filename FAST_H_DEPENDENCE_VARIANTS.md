# Fast H-Dependence Variants (E64-E68)

## Overview

E61/E62 are parallelizable via associative scan but NOT UTM-class (no nonlinear h-dependence).
E63 IS UTM-class but requires expensive O(d²) `W_h @ h` per timestep.

E64-E68 explore **fast h-dependence**: ways to get nonlinear h-dependence for UTM expressivity
while keeping cost to O(d) or O(d*rank) per timestep.

## The Tradeoff

| Model | Cost/step | UTM Class? | Cross-dim mixing via h? |
|-------|-----------|------------|-------------------------|
| E61/E62 | O(d*d) (batched) | NO | Yes (W_v @ x) |
| E63 | O(d²) per step | YES | Yes (W_h @ h) |
| **E64** | O(d) | YES | NO |
| **E65** | O(d) | YES | NO |
| **E66** | O(d*rank) | YES | YES |
| **E67** | O(d) or O(d*rank) | YES | Depends on variant |
| **E68** | O(d) | YES | NO |

## E64: Additive H-Dependence

**Formula:**
```
α_t = sigmoid(W_α @ x_t + b_α)
v_t = tanh(h_{t-1} + W_x @ x_t + b)    # h added directly!
h_t = α_t * h_{t-1} + (1 - α_t) * v_t
```

**Key insight:** Adding h directly is O(d) - no matrix multiply needed.

**Jacobian:**
```
∂h_t/∂h_{t-1} = diag(α + (1-α)*(1-v²))
```

Diagonal Jacobian = very efficient gradients. But no cross-dimension mixing through h.

**Use case:** Cheapest possible UTM recurrence. Good baseline.

## E65: Diagonal H-Dependence

**Formula:**
```
α_t = sigmoid(W_α @ x_t + b_α)
v_t = tanh(d_h * h_{t-1} + W_x @ x_t + b)    # d_h is learnable [dim]
h_t = α_t * h_{t-1} + (1 - α_t) * v_t
```

**Key insight:** Learnable per-dimension scaling of h contribution.

**Parameters:** Just +d for the scale vector d_h.

**Use case:** When some dimensions should be more "stateful" than others.

## E66: Low-Rank H-Dependence

**Formula:**
```
α_t = sigmoid(W_α @ x_t + b_α)
v_t = tanh(U @ (V @ h_{t-1}) + W_x @ x_t + b)    # U:[d,r], V:[r,d]
h_t = α_t * h_{t-1} + (1 - α_t) * v_t
```

**Key insight:** Low-rank factorization provides cross-dimension mixing at reduced cost.

**Cost:** O(d*rank) per step vs O(d²) for E63.

**Variants:**
- `66r16`: rank=16 (cheap, limited mixing)
- `66r64`: rank=64 (good balance)
- `66r128`: rank=128 (close to full rank)

**Use case:** When cross-dimension mixing through h is important but O(d²) is too expensive.

## E67: H-Dependent Gate Only

**Formula:**
```
α_t = sigmoid(W_α @ x_t + d_α * h_{t-1} + b_α)   # h in gate!
v_t = tanh(W_x @ x_t + b_v)                       # v is simple
h_t = α_t * h_{t-1} + (1 - α_t) * v_t
```

**Key insight:** Put h-dependence in the GATE, not the value.

The gate controls what to remember vs write. Making it h-dependent allows
state-dependent memory management decisions.

**Jacobian:**
```
∂h_t/∂h_{t-1} = diag(α) + diag(h - v) * diag(σ'(gate) * d_α)
```

The (h - v) term means gradient through gate depends on the "surprise"
between stored value and new value.

**Variants:**
- `67d`: Diagonal h in gate (O(d))
- `67lr`: Low-rank h in gate (O(d*rank))

**Use case:** When you want state-dependent "keep vs overwrite" decisions.

## E68: Self-Gating

**Formula:**
```
α_t = sigmoid(W_α @ x_t + b_α)
g_t = sigmoid(d_g * h_{t-1} + b_g)            # h gates the value
v_t = tanh(W_x @ x_t + b_v) * g_t             # multiplicative gating
h_t = α_t * h_{t-1} + (1 - α_t) * v_t
```

**Key insight:** Hidden state controls its own update resistance through multiplication.

When h is large → sigmoid(d*h) can be high or low depending on d sign.

**Variants:**
- `68s` (standard): g = sigmoid(d*h + b) - h activates the gate
- `68i` (inverse): g = sigmoid(-d*|h| + b) - large |h| closes the gate

The inverse variant creates natural "slot" behavior:
- Empty dimensions (h ≈ 0) are easy to write
- Full dimensions (large |h|) resist overwriting

**Use case:** Tasks requiring persistent storage (slots that "lock" once written).

## Comparison to Gated DeltaNet

**Gated DeltaNet** (ICLR 2025) uses matrix state S:
```
S_t = α_t * S_{t-1} * (I - β_t * k_t * k_t^T) + β_t * v_t * k_t^T
```

This is O(d²) state, O(d²) per step, but parallelizable via associative scan.

**E64-E68** use vector state h:
```
h_t = α_t * h_{t-1} + (1 - α_t) * f(h_{t-1}, x_t)
```

This is O(d) state, O(d) to O(d*rank) per step, but NOT parallelizable
(h-dependence in value breaks associativity).

**When to use what:**
- **Gated DeltaNet**: Need parallelism, can afford O(d²) state
- **E64-E68**: Need UTM expressivity, want minimal state/compute
- **E63**: Full UTM with cross-dim mixing, don't care about parallelism

## Level Keys

| Level | Model | Description |
|-------|-------|-------------|
| 64 | E64AdditiveH | v = tanh(h + Wx) |
| 65 | E65DiagonalH | v = tanh(d*h + Wx) |
| 66 | E66LowRankH | v = tanh(UVh + Wx), default rank=d/4 |
| 66r16 | E66LowRankH(rank=16) | Low-rank r=16 |
| 66r64 | E66LowRankH(rank=64) | Low-rank r=64 |
| 66r128 | E66LowRankH(rank=128) | Low-rank r=128 |
| 67 | E67HGated | α = σ(Wx + d*h), v = tanh(Wx) |
| 67d | E67HGatedDiagonal | Diagonal h in gate |
| 67lr | E67HGatedLowRank | Low-rank h in gate |
| 68 | E68SelfGating | v = tanh(Wx) * σ(h) |
| 68s | E68SelfGatingStandard | Standard self-gating |
| 68i | E68SelfGatingInverse | Inverse (resist overwrite) |
| gdn | GatedDeltaNet | ICLR 2025 matrix state |
| gdn-vec | GatedDeltaNetVector | Simplified vector state |

## Testing

```bash
# Run individual tests
python -m elman.models.e64_additive_h
python -m elman.models.e65_diagonal_h
python -m elman.models.e66_lowrank_h
python -m elman.models.e67_h_gated
python -m elman.models.e68_self_gating
python -m elman.models.gated_delta_net

# Train with new levels
python train.py --model.level=64 --experiment_name=e64_test
python train.py --model.level=66r64 --experiment_name=e66_r64_test
python train.py --model.level=gdn --experiment_name=gated_deltanet_test
```

## Installation for FLA Backend

To use optimized FLA kernels for GatedDeltaNet:

```bash
pip install -U git+https://github.com/fla-org/flash-linear-attention
```

Without FLA, the model falls back to pure PyTorch implementation.
