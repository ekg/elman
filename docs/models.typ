#set document(title: "Elman RNN Model Architectures")
#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")

= Elman RNN Model Architectures

This document describes the core RNN models implemented in the `elman` codebase, focusing on their state update equations, output mechanisms, and key parameters.

== Overview

All models follow a common layer structure:
```
x = in_proj(input)        # Project to d_inner
x = silu(x)               # Pre-activation
x, state = cell(x, state) # Recurrent cell
output = out_proj(x)      # Project back to dim
```

The models differ primarily in their recurrent cell implementation. We categorize them by state type:

#table(
  columns: (1fr, 1fr, 1fr),
  align: (left, center, center),
  [*Category*], [*State Size*], [*Models*],
  [Vector State], [$O(d)$], [E1, E42, E61, E68],
  [Matrix State], [$O(n^2)$], [E74, E75, E76, E77, E78],
  [External], [Varies], [FLA-GDN, Mamba2],
)

== FLA-GDN (Gated Delta Network)

*Reference*: ICLR 2025, Flash Linear Attention library

FLA-GDN combines the delta rule for selective memory updates with Mamba2-style gating. It uses chunked linear attention with $O(n)$ complexity.

=== Core Architecture

The model uses multi-head linear attention with short convolutions:

$ Q, K, V = "Conv1d"("proj"(x)) $
$ alpha_t = sigma(W_alpha x_t) quad quad beta_t = sigma(W_beta x_t) $

Per-head state update (linear attention with delta rule):
$ S_t = alpha_t dot.o S_(t-1) + K_t^top V_t $
$ y_t = Q_t S_t $

=== Output

$ "output" = "GateNorm"(y) dot.o "silu"(g) $

Where $g$ is a learned gate projection.

=== Key Parameters

- `hidden_size`: Model dimension
- `expand_v`: Value expansion factor (typically 2.0)
- `num_heads`: Number of attention heads
- `head_dim`: Dimension per head (typically 128)
- `conv_size`: Short convolution kernel size (typically 4)

=== Notable Features

- Uses Triton kernels for high efficiency
- Chunked linear attention enables parallelism
- Short convolutions on Q, K, V provide local context
- Fused RMSNorm on output for stability

== Mamba2 (State Space Model)

*Reference*: `mamba-ssm` package

Mamba2 uses a selective state space model with input-dependent dynamics and parallel scan for training.

=== Core State Update

The selective SSM dynamics:
$ h_t = A_t dot.o h_(t-1) + B_t dot.o x_t $
$ y_t = C_t dot.o h_t $

Where $A_t, B_t, C_t$ are computed from $x_t$ via learned projections (selectivity).

=== Output

$ x, z = "split"("linear"(x)) $
$ x = "silu"("conv1d"(x)) $
$ x = "ssm"(x) $
$ "output" = x dot.o "silu"(z) $

=== Key Parameters

- `d_model`: Model dimension
- `d_state`: SSM state dimension (typically 64-128)
- `d_conv`: Convolution kernel size (typically 4)
- `expand`: Expansion factor (typically 2)
- `headdim`: Head dimension (typically 64)

=== Notable Features

- Parallel scan enables $O(log T)$ depth training
- Input-dependent dynamics (selective mechanism)
- Depthwise conv for local context
- Dual gating on SSM output and skip connection

#pagebreak()

== E1: Mamba-Gated Elman RNN

The base gated Elman RNN, using Mamba2-style split projection for gating.

=== State Update

$ h_t = tanh(W_x x_t + W_h h_(t-1) + b) $

=== Output

Input is split into two branches:
$ x, z = "split"("in_proj"(x)) $
$ x = "silu"(x) quad "(pre-activation)" $
$ "output" = h_t dot.o "silu"(z) $

The hidden state $h_t$ is gated by the other branch $z$, matching Mamba2's gating pattern.

=== Key Parameters

- `dim`: Model dimension
- `d_inner`: Hidden dimension ($= "dim" times "expansion"$)
- `expansion`: Typically 1.0-2.0

=== Notable Features

- Spectral normalization on $W_h$ ($rho < 0.99$) for stability
- Optional conv1d for local context
- Custom CUDA kernel for fused forward/backward

== E42: Linear Tied Self-Gated

*Current best vector-state model*. Combines linear recurrence with tied weights and self-gating.

=== State Update (Linear, No tanh!)

$ h_t = W x_t + W h_(t-1) + b $

Critical: This is a *linear* recurrence with tied weights ($W_x = W_h = W$).

=== Output

$ "output" = h_t dot.o "silu"(h_t) $

The only nonlinearity is in the self-gating output.

=== Key Parameters

- `dim`: Model dimension
- `d_inner`: Hidden dimension
- `spectral_radius`: Target spectral radius for $W$ (default 0.999)

=== Notable Features

- *Linear recurrence*: Better gradient flow, no tanh saturation
- *Tied weights*: Fewer parameters ($W_x = W_h$)
- *Self-gating*: $h dot.o "silu"(h)$ provides sufficient nonlinearity
- *Batched GEMM*: Pre-compute $W x$ for all timesteps
- *Spectral normalization*: Essential for linear recurrence stability ($||W|| < 1$)

#pagebreak()

== E61: Decay-Gated Elman

Mamba2-style input-dependent decay for selective state retention.

=== State Update

$ alpha_t = sigma(W_alpha x_t + b_alpha) $
$ v_t = W_v x_t + b_v $
$ h_t = alpha_t dot.o h_(t-1) + (1 - alpha_t) dot.o v_t $

The complementary gates $(alpha, 1-alpha)$ create a convex combination of old state and new value.

=== Output

$ "output" = h_t dot.o "silu"(h_t) $

=== Key Parameters

- `init_alpha_bias`: Initial bias for decay gate (default 2.0, so $sigma(2) approx 0.88$)
- `use_tanh`: Optional tanh on value

=== Notable Features

- *Input-dependent decay*: $alpha_t$ controls retention vs replacement
- *Linear in $h$*: Jacobian $partial h_t / partial h_(t-1) = "diag"(alpha_t)$
- *Parallelizable*: Could use associative scan (not implemented)

=== Variants

- *E61* (pure): $alpha dot.o h + (1-alpha) dot.o v$
- *E61b* (additive): $alpha dot.o h + v$ (non-complementary, Mamba2-style)
- *E61c* (tied): $alpha$ derived from $v$ itself (single-gate GRU)

== E68: Self-Gating with H-Dependence

Multiplicative interaction where the hidden state gates the new value.

=== State Update

$ alpha_t = sigma(W_alpha x_t + b_alpha) $
$ g_t = sigma(d_g dot.o h_(t-1) + b_g) $
$ v_t = tanh(W_x x_t + b_v) dot.o g_t $
$ h_t = alpha_t dot.o h_(t-1) + (1 - alpha_t) dot.o v_t $

=== Output

$ "output" = h_t dot.o "silu"(h_t) $

=== Key Parameters

- `init_d_g`: Diagonal scaling for h-gating (default 0.5)
- `init_b_g`: Bias for gating (default 0.0)

=== Notable Features

- *State controls updates*: Large $|h|$ affects how much new info can be written
- *Capacity-based gating*: Dimensions with strong existing values resist overwriting
- *O(d) cost*: Just sigmoid on $h$, no matrix multiply
- *UTM-class*: Hidden state is inside sigmoid (nonlinear function of state)

=== Variants

- *Standard*: $g = sigma(d dot.o h + b)$ -- $h$ activates the gate
- *Inverse*: $g = sigma(-d dot.o |h| + b)$ -- large $|h|$ closes the gate (resists overwriting)

#pagebreak()

== E74v2: Full Matrix State with Delta Rule

Matrix state model with associative memory via the delta rule.

=== State Structure

State is a matrix $S in RR^(n times n)$ where $n = n_"state"$.

=== Projections

$ k = W_k x_t, quad v = W_v x_t, quad q = W_q x_t $
$ hat(k) = k / ||k|| quad "(normalized key)" $

=== State Update (Delta Rule)

$ "retrieved" = S hat(k) $
$ delta = v - "retrieved" $
$ S = tanh(S + delta hat(k)^top) $

The delta rule writes the *difference* between the desired value $v$ and what was retrieved.

=== Output

$ S_q = S q $
$ "output" = S_q dot.o "silu"(S_q) quad "(self-gating)" $

=== Key Parameters

- `n_state`: Matrix dimension (32, 48, 64, or 96 for CUDA kernel)
- `proj_type`: 'no_z' (separate K, V, Q), 'tied_kq', or 'tied_kvq'
- `use_tanh`: Whether to apply tanh to state update
- `update_type`: 'delta', 'residual', 'ntm', 'retrieved_gate', or 'ema'
- `gate_type`: 'output' (self-gating) or 'input' (E1-style)

=== Update Type Variants

- *DELTA*: $S = tanh(S + "outer"(v - S hat(k), hat(k)))$
- *RESIDUAL*: $S = S + "scale" dot.o tanh("outer"(delta, hat(k)))$
- *NTM*: $S = S dot.o (1 - "outer"("erase", hat(k))) + "outer"("write" dot.o v, hat(k))$
- *RETRIEVED_GATE*: $S = S + "gate" dot.o "outer"(delta, hat(k))$
- *EMA*: $S = alpha dot.o S + (1-alpha) dot.o "outer"(v, hat(k))$

=== Notable Features

- *Associative memory*: Key-value storage with content-addressable retrieval
- *O($n^2$) state*: Richer state than vector models
- *CUDA kernel support*: Optimized for $n_"state" in {32, 48, 64, 96}$
- *Gradient checkpointing*: For memory efficiency with large states

#pagebreak()

== E75: Gated Delta with Forget Gate

E74's delta rule combined with E61's input-dependent forget gate.

=== Projections

$ k = W_k x_t, quad v = W_v x_t, quad q = W_q x_t $
$ beta_t = sigma(W_beta x_t + b_beta) quad "(per-row forget gate)" $

=== State Update

$ hat(k) = k / ||k|| $
$ "retrieved" = S hat(k) $
$ delta = v - "retrieved" $
$ S = tanh(beta_t dot.o S + "outer"(delta, hat(k))) $

The forget gate $beta_t$ allows fine-grained control over state preservation.

=== Output

$ S_q = S q $
$ "output" = S_q dot.o "silu"(S_q) $

=== Key Parameters

- `n_state`: Matrix dimension (default 64)
- `init_beta_bias`: Initial bias for forget gate (default 2.0)

=== Notable Features

- *Active forgetting*: Critical insight from E61/E68 analysis
- *Per-row control*: $beta_t in RR^n$ allows row-wise decay
- *Learn when to preserve*: $beta -> 1$ preserves, $beta -> 0$ forgets
- *CUDA kernel*: Optimized forward/backward with checkpointing

== E76: Log-Space Gated Delta

E75's nonlinear recurrence with Mamba2/FLA-GDN stability techniques.

=== Log-Space Parameterization

$ A = exp(A_"log") quad "(positive decay factor)" $
$ "dt" = "softplus"("gate" + "dt_bias") $
$ "decay" = exp(-A dot.o "dt") quad "(in [0, 1])" $

=== State Update

$ S = cases(
  tanh("decay" dot.o S + "outer"(delta, hat(k))) & "if use_tanh",
  "decay" dot.o S + "outer"(delta, hat(k)) & "otherwise (linear)"
) $

=== Key Parameters

- `A_init_range`: Range for initializing $A$ (default (1, 16))
- `dt_min`, `dt_max`: Range for timestep initialization
- `use_tanh`: Nonlinear vs linear recurrence (default True)
- `log_space_gate`: Use log-space parameterization (default True)

=== Configuration Matrix

#table(
  columns: (1fr, 1fr, 1fr),
  align: (left, center, center),
  [*Config*], [*use_tanh*], [*log_space_gate*],
  [Default (nonlinear + stable)], [True], [True],
  [E75-style], [True], [False],
  [Linear + stable], [False], [True],
  [Fully linear], [False], [False],
)

=== Notable Features

- *Weight decay exemption*: $A_"log"$ and $"dt_bias"$ marked for no weight decay
- *Training stability*: Log-space parameterization prevents decay explosion
- *Configurable nonlinearity*: Test nonlinear vs linear recurrence

#pagebreak()

== E77: Linear Matrix State

E42's linear recurrence philosophy applied to matrix state.

=== Key Insight

Put the nonlinearity at *output*, not in the state update. This allows gradients to flow through the matrix state unimpeded.

=== State Update (Linear -- No tanh!)

$ "decay" = sigma("gate" + b_"gate") $
$ S = "decay" dot.o S + "outer"(delta, hat(k)) $

=== Output

$ S_q = S q $
$ "output" = S_q dot.o "silu"(S_q) $

=== Key Parameters

- `n_state`: Matrix dimension (default 64)
- Fused projection: $W_"kvqg" in RR^(4n times d)$ for K, V, Q, gate

=== Notable Features

- *Linear state dynamics*: Better gradient flow (like E42)
- *Self-gating output*: Only nonlinearity is at output (like E42)
- *Fused projection*: Single GEMM for all 4 vectors
- *CUDA kernel*: Optimized for fused projections

=== Comparison with E76

#table(
  columns: (1fr, 1fr),
  align: (left, left),
  [*E76*], [*E77*],
  [$S = tanh("decay" dot.o S + "outer")$], [$S = "decay" dot.o S + "outer"$],
  [Nonlinear state], [Linear state],
  [tanh bounds state], [Self-gating bounds output],
)

== E78: Projected Matrix (Sparse)

E77 with random projection for efficient large effective state.

=== Architecture

Store small $S in RR^(n_"small" times n_"small")$ but simulate larger $RR^(n_"eff" times n_"eff")$ via random projection.

$ P in RR^(n_"eff" times n_"small") quad "(fixed random orthogonal projection)" $

=== Projections

$ k, v, q in RR^(n_"eff"} quad "(effective space)" $
$ k_"small" = P^top k, quad v_"small" = P^top v $

=== State Update

$ S = "decay" dot.o S + "outer"(delta_"small", hat(k)_"small") $

=== Output

$ S_(q,"small") = S (P^top q) $
$ S_q = P S_(q,"small") quad "(project back to effective space)" $
$ "output" = S_q dot.o "silu"(S_q) $

=== Key Parameters

- `n_effective`: Virtual large state size (e.g., 128)
- `n_small`: Actual stored state size (e.g., 32)
- Compression ratio: $(n_"eff" / n_"small")^2$

=== Notable Features

- *Johnson-Lindenstrauss*: Random projection preserves distances w.h.p.
- *O($n_"small"^2$) compute*: Despite O($n_"eff"^2$) effective capacity
- *Fixed projection*: $P$ is not learned, just random orthogonal

#pagebreak()

== Model Summary Table

#table(
  columns: (1fr, 2fr, 2fr, 1fr),
  align: (left, left, left, center),
  [*Model*], [*State Update*], [*Output*], [*State*],
  [E1], [$h = tanh(W_x x + W_h h + b)$], [$h dot.o "silu"(z)$], [$O(d)$],
  [E42], [$h = W x + W h + b$ (linear)], [$h dot.o "silu"(h)$], [$O(d)$],
  [E61], [$h = alpha h + (1-alpha) v$], [$h dot.o "silu"(h)$], [$O(d)$],
  [E68], [$h = alpha h + (1-alpha)(v dot.o sigma(h))$], [$h dot.o "silu"(h)$], [$O(d)$],
  [E74], [$S = tanh(S + "outer"(delta, hat(k)))$], [$S q dot.o "silu"(S q)$], [$O(n^2)$],
  [E75], [$S = tanh(beta S + "outer"(delta, hat(k)))$], [$S q dot.o "silu"(S q)$], [$O(n^2)$],
  [E76], [$S = tanh("decay" S + "outer")$ or linear], [$S q dot.o "silu"(S q)$], [$O(n^2)$],
  [E77], [$S = "decay" S + "outer"$ (linear)], [$S q dot.o "silu"(S q)$], [$O(n^2)$],
  [E78], [$S = "decay" S + "outer"$ (projected)], [$P S q dot.o "silu"(...)$], [$O(n_s^2)$],
)

== Key Insights

=== Linear vs Nonlinear Recurrence

- *E1, E74, E75*: Nonlinear state (tanh) -- bounded but may saturate gradients
- *E42, E77*: Linear state -- unbounded gradients, rely on spectral norm / self-gating
- *E76*: Configurable -- test both hypotheses

=== Gating Mechanisms

- *E1*: Input-gated output ($h dot.o "silu"(z)$)
- *E42, E68, E74-E78*: Self-gated output ($h dot.o "silu"(h)$)
- *E61*: Complementary decay gate ($alpha, 1-alpha$)
- *E75, E76*: Learned forget gate for matrix state

=== State Capacity

- Vector models ($O(d)$): Fast but limited memory
- Matrix models ($O(n^2)$): Rich associative memory but more compute
- Projected ($O(n_s^2)$): Efficient approximation of large matrix

== Source Files

- `elman/models/mamba_gated_elman.py` -- E1
- `elman/models/e42_linear_tied.py` -- E42
- `elman/models/e61_decay_gated.py` -- E61
- `elman/models/e68_self_gating.py` -- E68
- `elman/models/e74_v2.py` -- E74v2
- `elman/models/e75_gated_delta.py` -- E75
- `elman/models/e76_logspace_delta.py` -- E76
- `elman/models/e77_linear_matrix.py` -- E77
- `elman/models/e78_projected_matrix.py` -- E78
- `elman/models/fla_gated_delta.py` -- FLA-GDN wrapper
- `elman/models/mamba2_baseline.py` -- Mamba2 wrapper
