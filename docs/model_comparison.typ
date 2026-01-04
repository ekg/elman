#set document(title: "Elman ResNet vs Mamba2: Architecture Comparison")
#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")

= Elman ResNet (Level 0) vs Mamba2: Architecture Comparison

This document compares the two sequence models as implemented in the `elman` codebase.

== Overview

#table(
  columns: (1fr, 1fr, 1fr),
  align: (left, center, center),
  [*Property*], [*Elman ResNet (Level 0)*], [*Mamba2*],
  [Core Operation], [RNN (sequential)], [SSM (parallel scan)],
  [Recurrence], [$tanh(W_x x + W_h h + b)$], [Selective state space],
  [Memory], [$O(d)$ hidden state], [$O(d times d_"state")$],
  [Training], [BPTT through time], [Parallel across time],
  [CUDA Kernel], [Custom fused kernel], [`mamba-ssm` package],
)

== Level 0: Elman ResNet Architecture

=== Cell Equations (per timestep $t$)

The core recurrence in `stock_elman.py`:

$ h_t = tanh(W_x dot x_t + W_h dot h_(t-1) + b) $

$ "output"_t = h_t dot.o "silu"(W_"gate" dot x_t + b_"gate") $

Where:
- $W_x in RR^(d times d)$: input projection
- $W_h in RR^(d times d)$: hidden-to-hidden (with spectral normalization, $rho < 0.99$)
- $W_"gate" in RR^(d times d)$: learned gate projection
- $dot.o$: element-wise multiplication

=== Layer Wrapper (`StockElman`)

```
x_proj = in_proj(x)           # [B, T, d_inner]
x_rnn = cell(x_proj, h0)      # Sequential RNN
output = out_proj(x_rnn)      # [B, T, dim]
```

=== Full Model (`LadderLM`)

Pre-norm residual architecture (like Mamba/GPT):

```
x = embedding(tokens)
for layer in layers:
    residual = x
    x = RMSNorm(x)
    x = layer(x)              # StockElman
    x = residual + x          # Residual connection
x = RMSNorm(x)
logits = lm_head(x)
```

=== Key Features

1. *Spectral Normalization*: $W_h$ constrained to spectral radius $< 0.99$ for stability
2. *Learned Gating*: Output gated by $"silu"(W_"gate" dot x)$ (input-dependent)
3. *Pre-norm Residual*: RMSNorm before each layer, residual around
4. *Custom CUDA*: Fused forward/backward kernel in `hasty_pytorch_lib`

=== 500M Model Configuration

```
dim = 1024
depth = 18
d_inner = 1024 (expansion = 1.0)
vocab_size = 50,281 (p50k_base)
params = 499,885,056
```

== Mamba2 Architecture

=== Core Mechanism

Mamba2 uses a *Selective State Space Model* with input-dependent dynamics:

$ h_t = A_t dot.o h_(t-1) + B_t dot.o x_t $
$ y_t = C_t dot.o h_t $

Where $A_t, B_t, C_t$ are computed from $x_t$ via learned projections (selectivity).

=== Mamba2 Block

From `mamba_ssm.Mamba2`:

```
# Input projections
x, z = split(linear(x))       # x for SSM, z for gate
x = conv1d(x)                 # Local context (d_conv=4)
x = silu(x)

# Selective SSM
x = ssm(x)                    # Parallel selective scan

# Output gate
x = x * silu(z)
x = out_proj(x)
```

=== Full Model (`Mamba2LM`)

Same pre-norm residual pattern:

```
x = embedding(tokens)
for layer in layers:
    residual = x
    x = LayerNorm(x)
    x = Mamba2Block(x)
    x = residual + x
x = LayerNorm(x)
logits = lm_head(x)
```

=== Key Features

1. *Selective Dynamics*: $A, B, C$ are input-dependent (not fixed)
2. *Parallel Scan*: $O(log T)$ depth for training (vs $O(T)$ for RNN)
3. *Conv1d*: Depthwise conv for local context before SSM
4. *Dual Gating*: Both SSM output and skip connection gated

=== 500M Model Configuration

```
dim = 1536
depth = 24
d_state = 128
d_conv = 4
expand = 2
headdim = 64
vocab_size = 50,281 (p50k_base)
params = 428,730,240
```

== Implementation Differences

#table(
  columns: (1fr, 1fr, 1fr),
  align: (left, left, left),
  [*Aspect*], [*Elman ResNet*], [*Mamba2*],
  [Forward Pass], [Sequential loop over $T$], [Parallel selective scan],
  [Backward Pass], [BPTT (sequential)], [Parallel scan gradient],
  [Normalization], [RMSNorm (pre-norm)], [LayerNorm (pre-norm)],
  [Weight Tying], [embed = lm_head], [embed = lm_head],
  [Gating], [$h dot.o "silu"(W_"gate" x)$], [SSM output $dot.o$ silu(z)],
  [Nonlinearity], [tanh in recurrence], [Linear SSM + silu gates],
)

== Computational Complexity

#table(
  columns: (1fr, 1fr, 1fr),
  align: (left, center, center),
  [*Operation*], [*Elman ResNet*], [*Mamba2*],
  [Forward (time)], [$O(T dot d^2)$ sequential], [$O(T dot d^2)$ parallel],
  [Backward (time)], [$O(T dot d^2)$ sequential], [$O(T dot d^2)$ parallel],
  [Memory], [$O(T dot d)$ states], [$O(T dot d dot d_"state")$],
  [Parallelism], [Batch only], [Batch + Time],
)

In practice:
- *Elman*: Sequential but highly optimized fused CUDA kernel
- *Mamba2*: Parallel but more memory overhead from state expansion

== Empirical Results (1000s, p50k_base, ~500M params)

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, right, right, right, right),
  [*Model*], [*Params*], [*tok/s*], [*Tokens*], [*Final Loss*],
  [Elman ResNet], [500M], [~27k], [~27M], [5.63],
  [Mamba2], [429M], [~25k], [~25M], [5.34],
)

Despite lower throughput, Mamba2 achieves lower loss in same wall time at 500M scale due to better sample efficiency.

== Source Files

- `elman/models/stock_elman.py` - Level 0 Elman cell and layer
- `elman/models/ladder_lm.py` - LadderLM wrapper with residual connections
- `elman/models/mamba2_baseline.py` - Mamba2LM wrapper
- `elman/cuda/lib/stock_elman_gpu.cu.cc` - Custom CUDA kernel

== Summary

Both models use the same high-level architecture pattern (pre-norm + residual), but differ fundamentally in their sequence processing:

- *Elman ResNet*: Classical RNN with tanh nonlinearity, learned gating, spectral-normalized $W_h$
- *Mamba2*: Selective SSM with input-dependent dynamics, parallel scan for training

The Elman model is simpler and faster per-token, while Mamba2 is more sample-efficient at scale.
