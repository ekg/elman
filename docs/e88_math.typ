#set document(title: "E88: FLA-GDN Hybrid with Nonlinear Matrix State", author: "Erik Garrison")
#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 18pt, weight: "bold")[E88: FLA-GDN Hybrid with Nonlinear Matrix State]

  #v(0.5em)
  _A recurrent neural network with associative memory_
  #v(1em)
]

= Introduction

Modern sequence models that support efficient parallel training share a key property: *linear temporal dynamics*. In Mamba2, the state at time $T$ is:

$ bold(h)_T = sum_(t=0)^T alpha^(T-t) dot f(bold(x)_t) $

where $f(bold(x)_t)$ may be computed nonlinearly from $bold(x)_t$, but the accumulation across time is a *linear combination*. FLA-GatedDeltaNet is similar: $bold(S)_T = sum alpha^(T-t) dot bold(k)_t bold(v)_t^top$. The nonlinearities in these models---SiLU activations, gating, input projections---all operate *within* each timestep. Information flows *forward through time* via purely linear operations.

This is the key insight: *nonlinearity flows down (through layers), not forward (through time)*.

Each layer applies nonlinear transformations to its input, but the recurrent accumulation within each layer is linear. The expressivity of deep Mamba2 or FLA-GDN comes from stacking layers, not from temporal dynamics. Within any single layer, the limitations of linear RNNs apply: the output at time $T$ is a linear function of the per-timestep contributions.

E88 takes a different approach: *nonlinear temporal dynamics*.

#align(center)[
  #box(stroke: 1pt, inset: 12pt)[
    $ bold(S) := tanh(alpha bold(S) + bold(delta) bold(k)^top) $
  ]
]

The $tanh$ wrapping the state update means the state at time $T$ is a *nonlinear* function of the entire history. The nonlinearity compounds across timesteps: $tanh$ of $tanh$ of $tanh$... This creates qualitatively different dynamics than the linear accumulation in Mamba2/FLA-GDN.

*Concrete consequence:* In a linear system, if state component $i$ has value 100 and component $j$ has value 1, the ratio 100:1 is preserved through the recurrence. In E88, the $tanh$ compresses this to roughly 1.3:1 (since $tanh(100)/tanh(1) approx 1.3$). This compression creates implicit decision boundaries---values that were far apart become close, values near zero stay distinguishable. These boundaries accumulate and interact across timesteps.

*The Lean4 proofs* in `elman-proofs/ElmanProofs/Expressivity/LinearLimitations.lean` establish that linear RNNs cannot compute threshold functions or XOR, because their output is a linear (hence continuous) function of inputs. These proofs apply directly to the *temporal dimension* of Mamba2 and FLA-GDN within each layer. E88's $tanh$ breaks out of these limitations---at the cost of parallelism.

*What does it cost?* The $tanh$ is not associative, so E88 cannot use parallel scans. At 480M scale: E88 runs at ~14K tok/s versus ~22K tok/s for FLA-GDN. Sequential processing is the price of nonlinear temporal dynamics.

= Synopsis

E88 is a *multi-head associative memory* with nonlinear dynamics. Each layer maintains $H$ independent matrix states $bold(S)_h in RR^(n times n)$, where each head learns to store and retrieve key-value associations using the delta rule.

*Key features:*

- *Nonlinear state:* The $tanh$ in the update provides bounded dynamics, gradient regularization, and computational expressivity beyond linear RNNs.

- *Delta rule:* Rather than directly writing $bold(v)$ to memory, E88 computes $bold(delta) = bold(v) - bold(S) bold(k)$ and writes only the _difference_. This is provably more efficient for associative storage.

- *Multi-head parallelism:* 98 independent heads with small (32×32) states provide high capacity through parallelism rather than raw state size.

- *FLA-GDN components:* Mamba2-style exponential decay, SiLU gating, L2-normalized keys/queries, and depthwise convolutions.

*Comparison to FLA-GDN:* E88 is architecturally similar to FLA-GatedDeltaNet but differs in three critical ways:

#table(
  columns: 3,
  align: (left, left, left),
  [*Aspect*], [*FLA-GDN*], [*E88*],
  [State update], [$bold(S) := alpha bold(S) + bold(k) bold(v)^top$], [$bold(S) := tanh(alpha bold(S) + bold(delta) bold(k)^top)$],
  [Nonlinearity], [None (linear state)], [$tanh$ on full state],
  [Write rule], [Direct write ($bold(k) bold(v)^top$)], [Delta rule ($bold(delta) = bold(v) - bold(S) bold(k)$)],
  [Parallelism], [Chunk-wise parallel scan], [Fully sequential],
)

The cost of nonlinearity is speed: E88 runs at ~14K tok/s vs FLA-GDN's ~22K tok/s at 480M scale. The benefit is expressivity and stability at scale.

= Architecture Details

== Input Projections

For input $bold(x) in RR^d$ at each timestep, we compute per-head queries, keys, and values:

$ bold(k)_h &= "silu"("conv"(bold(W)_k^h bold(x))) in RR^n $
$ bold(v)_h &= "silu"("conv"(bold(W)_v^h bold(x))) in RR^n $
$ bold(q)_h &= "silu"("conv"(bold(W)_q^h bold(x))) in RR^n $

where $bold(W)_k^h, bold(W)_v^h, bold(W)_q^h in RR^(n times d)$ are learned projections, "conv" is a depthwise 1D convolution (kernel size 4), and "silu" is the SiLU activation $"silu"(x) = x dot sigma(x)$.

== Mamba2-Style Exponential Decay

The decay gate uses Mamba2's parameterization:

$ g_h = -exp(bold(A)_"log"^h) dot "softplus"(bold(a)^h dot bold(x) + b_"dt"^h) $ <decay>

where $bold(A)_"log"^h in RR$ is a learned log-eigenvalue, $bold(a)^h in RR^d$ projects the input, and $b_"dt"^h in RR$ is a learned time-step bias.

The decay factor for the state update is then:

$ alpha_h = exp(g_h) in (0, 1) $

== L2 Normalization

Keys and queries are L2-normalized before retrieval:

$ hat(bold(k))_h = bold(k)_h / norm(bold(k)_h)_2 $
$ hat(bold(q))_h = bold(q)_h / norm(bold(q)_h)_2 $

This improves training stability by preventing the state matrix from growing unboundedly.

== Matrix State Update (Core Innovation)

The state update follows a delta-rule with nonlinear saturation:

$ bold(r)_h &= bold(S)_h hat(bold(k))_h quad &"(retrieve)" $ <retrieve>

$ bold(delta)_h &= bold(v)_h - bold(r)_h quad &"(compute delta)" $ <delta>

$ bold(S)_h &:= tanh(alpha_h bold(S)_h + bold(delta)_h hat(bold(k))_h^top) quad &"(update with nonlinearity)" $ <update>

*Key insight:* The $tanh$ in @update bounds the state entries to $[-1, 1]$, preventing gradient explosion and providing natural regularization. This differs from FLA-GDN which uses a linear state.

== Output Computation

The output for each head is computed by querying the state:

$ bold(o)_h = bold(S)_h hat(bold(q))_h in RR^n $ <query>

== Output Gating (FLA-GDN Style)

E88 uses SiLU-gated output following FLA-GDN:

$ bold(g)_h = bold(W)_g^h bold(x) in RR^n $
$ tilde(bold(o))_h = bold(o)_h dot.o "silu"(bold(g)_h) $

where $dot.o$ denotes element-wise multiplication.

== Final Output

The gated outputs from all heads are concatenated and projected:

$ bold(y) = bold(W)_"out" ["tilde"(bold(o))_0; tilde(bold(o))_1; dots.c; tilde(bold(o))_(H-1)] + bold(b)_"out" $

where $bold(W)_"out" in RR^(d times H n)$ and $bold(b)_"out" in RR^d$.

= Summary of Operations Per Timestep

For each head $h in {0, dots, H-1}$:

#align(center)[
#table(
  columns: 2,
  stroke: none,
  align: left,
  [*Step*], [*Operation*],
  [1. Project], [$bold(k)_h, bold(v)_h, bold(q)_h = "silu"("conv"(bold(W) bold(x)))$],
  [2. Normalize], [$hat(bold(k))_h = bold(k)_h \/ norm(bold(k)_h), quad hat(bold(q))_h = bold(q)_h \/ norm(bold(q)_h)$],
  [3. Decay], [$alpha_h = exp(-exp(A_"log") dot "softplus"(bold(a) dot bold(x) + b))$],
  [4. Retrieve], [$bold(r)_h = bold(S)_h hat(bold(k))_h$],
  [5. Delta], [$bold(delta)_h = bold(v)_h - bold(r)_h$],
  [6. Update], [$bold(S)_h := tanh(alpha_h bold(S)_h + bold(delta)_h hat(bold(k))_h^top)$],
  [7. Query], [$bold(o)_h = bold(S)_h hat(bold(q))_h$],
  [8. Gate], [$tilde(bold(o))_h = bold(o)_h dot.o "silu"(bold(W)_g bold(x))$],
)
]

= Comparison with Baselines

#table(
  columns: 4,
  align: (left, center, center, center),
  [*Model*], [*State Update*], [*State Size*], [*Parallel Scan*],
  [E88], [$tanh(alpha S + delta k^top)$], [$H times n^2$], [No],
  [FLA-GDN], [$alpha S + k v^top$], [$H times n^2$], [Yes],
  [Mamba2], [$A h + B x$], [$d times n_"state"$], [Yes],
  [GRU], [$z dot.o h + (1-z) dot.o tilde(h)$], [$d$], [No],
)

= Hyperparameters (480M Scale)

Optimal configuration found via CMA-ES search:

- $d = 2176$ (model dimension)
- $H = 98$ (number of heads)
- $n = 32$ (state dimension per head)
- Depth = 14 layers
- Gate activation: SiLU
- Learning rate: $3 times 10^(-4)$

Total state memory per layer: $H times n^2 = 98 times 32^2 approx 100"K"$ floats.

= Implementation Notes

1. *CUDA Kernel:* E88 uses a custom CUDA kernel with gradient checkpointing (checkpoint every 16 steps) for O(T) memory complexity.

2. *No Parallel Scan:* Unlike Mamba2 and FLA-GDN, E88 cannot use parallel scan due to the nonlinear $tanh$ in the state update. This makes it ~2.5x slower but enables richer dynamics.

3. *Supported State Sizes:* The CUDA kernel efficiently supports $n in {16, 32, 48, 64}$.

= References

- FLA-GatedDeltaNet: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025)
- Mamba2: Dao & Gu, "Transformers are SSMs" (2024)
- Delta Rule: Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers" (ICML 2021)

#pagebreak()

= Appendix A: Mamba2 SSM

Mamba2 uses a selective state space model (SSM) with diagonal state transitions:

== State Equation

$ bold(h)_t = bold(A) dot.o bold(h)_(t-1) + bold(B)_t bold(x)_t $
$ bold(y)_t = bold(C)_t bold(h)_t $

where:
- $bold(h)_t in RR^(d times n)$ is the hidden state (expanded dimension $times$ state size)
- $bold(A) in RR^(d times n)$ is a diagonal decay matrix (discretized from continuous $bold(A)$)
- $bold(B)_t in RR^(d times n)$ is the input-dependent input projection
- $bold(C)_t in RR^(d times n)$ is the input-dependent output projection

== Discretization

The continuous SSM parameters are discretized using the zero-order hold:

$ overline(bold(A)) = exp(Delta bold(A)) $
$ overline(bold(B)) = (exp(Delta bold(A)) - bold(I)) bold(A)^(-1) bold(B) approx Delta bold(B) $

where $Delta$ is the step size, computed as:
$ Delta = "softplus"(bold(W)_Delta bold(x) + bold(b)_Delta) $

== Parallel Scan

Because the state update is _linear_ in $bold(h)$, Mamba2 can compute all timesteps in parallel using an associative scan:

$ bold(h)_t = sum_(i=1)^t (product_(j=i+1)^t overline(bold(A))_j) overline(bold(B))_i bold(x)_i $

This gives O(log T) parallel depth, enabling efficient GPU utilization.

== Key Differences from E88

#table(
  columns: 3,
  align: (left, left, left),
  [*Aspect*], [*Mamba2*], [*E88*],
  [State structure], [Diagonal SSM], [Full matrix per head],
  [Update], [Linear], [Nonlinear ($tanh$)],
  [Parallelization], [Associative scan], [Sequential],
  [State size], [$d times n_"state"$], [$H times n^2$],
  [Memory mechanism], [Implicit (SSM)], [Explicit (key-value)],
)

#pagebreak()

= Appendix B: FLA-GatedDeltaNet

FLA-GatedDeltaNet (ICLR 2025) combines linear attention with the delta rule for associative memory.

== Architecture

For each head $h$:

$ bold(k)_h &= bold(W)_k bold(x) in RR^n $
$ bold(v)_h &= bold(W)_v bold(x) in RR^m $
$ bold(q)_h &= bold(W)_q bold(x) in RR^n $
$ bold(beta)_h &= sigma(bold(W)_beta bold(x)) in (0, 1) quad "(forget gate)" $

== Linear State Update

$ bold(S)_h := bold(beta)_h bold(S)_h + bold(k)_h bold(v)_h^top $

Note: This is a _linear_ update (no nonlinearity applied to $bold(S)_h$).

== Output

$ bold(o)_h = bold(S)_h bold(q)_h $

== Chunk-wise Parallel Algorithm

Because the update is linear, FLA-GDN uses a chunk-wise parallel algorithm:

1. Divide sequence into chunks of size $C$
2. Within each chunk, compute cumulative states using parallel scan
3. Across chunks, propagate boundary states

This achieves O(T/C + C) complexity with high parallelism.

== Gated Output (Key Feature)

FLA-GDN applies SiLU gating to the output:

$ tilde(bold(o))_h = "RMSNorm"(bold(o)_h) dot.o "silu"(bold(g)_h) $

where $bold(g)_h = bold(W)_g bold(x)$.

E88 adopts this gating mechanism.

== Key Differences from E88

#table(
  columns: 3,
  align: (left, left, left),
  [*Aspect*], [*FLA-GDN*], [*E88*],
  [State update], [$beta S + k v^top$], [$tanh(alpha S + delta k^top)$],
  [Nonlinearity], [None (linear)], [$tanh$ on state],
  [Delta rule], [No (direct write)], [Yes ($delta = v - S k$)],
  [Parallelization], [Chunk-wise scan], [Sequential],
  [Decay], [Sigmoid $beta in (0,1)$], [Exp decay $alpha = e^g$],
)

== Why E88 Uses Nonlinear State

The linear state in FLA-GDN allows parallel computation but has limitations:

1. *Unbounded growth:* Without $tanh$, state entries can grow without bound
2. *Gradient issues:* Large state values cause gradient explosion at scale
3. *Limited dynamics:* Linear updates may have limited expressivity

E88's $tanh$ provides:
- Bounded state entries $in [-1, 1]$
- Natural gradient regularization
- Richer nonlinear dynamics

The cost is sequential computation (no parallel scan possible).

#pagebreak()

= Appendix C: Recurrent State Size Analysis

A critical factor in recurrent model design is the _state size_: the amount of memory carried forward between timesteps. This determines the model's capacity for long-range dependencies.

== State Size Formulas

#table(
  columns: 3,
  align: (left, left, left),
  [*Model*], [*State Structure*], [*Total State (per layer)*],
  [E88], [$H$ matrices of $n times n$], [$H dot n^2$],
  [FLA-GDN], [$H$ matrices of $n_k times n_v$], [$H dot n_k dot n_v$],
  [Mamba2], [Vector of $d_"inner" times n_"state"$], [$d_"inner" dot n_"state"$],
  [GRU/LSTM], [Vector of $d_"inner"$], [$d_"inner"$ (or $2 d_"inner"$ for LSTM)],
  [Transformer], [KV cache of $T times d$], [$T dot d$ (grows with sequence)],
)

== Concrete Numbers (480M Scale)

Using optimal configurations from CMA-ES search:

#table(
  columns: 4,
  align: (left, right, right, right),
  [*Model*], [*Config*], [*State/Layer*], [*vs E88*],
  [E88], [$H=98, n=32$], [$98 times 32^2 = 100,352$], [$1 times$],
  [FLA-GDN], [$H=24, n_k=80, n_v=160$], [$24 times 80 times 160 = 307,200$], [$3.1 times$],
  [Mamba2], [$d=1792, "exp"=2, n=96$], [$3584 times 96 = 344,064$], [$3.4 times$],
  [Transformer], [KV at $T=512$], [$512 times 1792 = 917,504$], [$9.1 times$],
)

== Interpretation

*E88 achieves competitive performance with 3-4x less state memory than SSM baselines.*

This is significant because:

1. *Memory efficiency:* E88 requires less memory during inference
2. *Parallel heads:* E88's state is distributed across many small independent heads ($98 times 32^2$) rather than one large state
3. *Nonlinear compression:* The $tanh$ bounds may enable more efficient information packing

== State Capacity vs Performance

Despite having less state, E88 trails Mamba2/FLA-GDN by ~0.12 nats at 480M scale:

#align(center)[
#table(
  columns: 3,
  align: (left, right, right),
  [*Model*], [*Loss*], [*State/Layer*],
  [Mamba2], [1.27], [344K],
  [FLA-GDN], [1.27], [307K],
  [E88], [1.39], [100K],
)
]

The ~0.12 nat gap appears to be the _cost of memory efficiency_. E88's nonlinear update provides expressivity but cannot match the raw capacity of larger linear SSM states.

== Bits per State Entry

Another way to analyze: how many "bits" of information can each state entry store?

- *Linear state (Mamba2, FLA-GDN):* Unbounded floats, limited by numerical precision (~23 bits for FP32 mantissa)
- *E88 nonlinear state:* Bounded to $[-1, 1]$ by $tanh$, effectively compressing to a soft "bit"

E88 compensates with more state entries (100K) but each entry stores less information than unbounded linear states.

== State Utilization

An open question: do linear SSMs actually _use_ their full state capacity? The effective information may be much lower than the theoretical maximum.

Future work could analyze:
- Singular value distribution of learned states
- Information-theoretic capacity utilization
- Whether E88's compression is actually optimal

#pagebreak()

= Appendix D: Formal Expressivity Analysis

This appendix presents formal results (proven in Lean4) explaining why the $tanh$ nonlinearity in E88's state update is essential for computational expressivity.

== Linear RNNs Cannot Compute Threshold Functions

*Theorem (LinearLimitations):* A linear recurrent neural network with update rule $bold(h)_t = bold(A) bold(h)_(t-1) + bold(B) bold(x)_t$ cannot compute any discontinuous function of its inputs.

*Proof sketch:* The output of a linear RNN is a continuous function of its initial state and input sequence (composition of linear maps is linear, hence continuous). Threshold functions like $f(x) = 1 "if" sum x_i > theta "else" 0$ are discontinuous at the boundary. Therefore, linear RNNs cannot compute threshold functions.

*Corollary:* Linear RNNs cannot compute XOR, parity, or any function requiring discrete decision boundaries.

This limitation applies to:
- Mamba2's diagonal SSM (linear in state)
- FLA-GDN's linear attention state

E88's $tanh$ nonlinearity breaks this limitation by introducing a nonlinear compression at each step.

== The tanh Compression Property

*Theorem (tanh_compresses_ratio):* For any $a, b > 0$ with $a > b$:
$ (tanh(a)) / (tanh(b)) < a / b $

*Proof:* Since $tanh$ is concave on $(0, infinity)$ and $tanh(0) = 0$, by concavity:
$ tanh(a) < a / b dot tanh(b) $
Rearranging gives the result.

*Implication:* The $tanh$ in E88's update @update prevents any single input from dominating the state. If the linear combination $alpha bold(S) + bold(delta) bold(k)^top$ produces a large value (say 100), while another direction has value 1, linear dynamics would preserve the 100:1 ratio. With $tanh$:
$ (tanh(100)) / (tanh(1)) approx 1 / 0.76 approx 1.3 << 100 $

This "squashing" creates decision boundaries that linear systems cannot express.

== Jacobian Spectrum and Gradient Flow

For the state update $bold(S)' = tanh(alpha bold(S) + bold(delta) bold(k)^top)$, the Jacobian with respect to $bold(S)$ is:

$ (partial bold(S)') / (partial bold(S)) = alpha dot "diag"(1 - tanh^2(bold(Z))) $

where $bold(Z) = alpha bold(S) + bold(delta) bold(k)^top$ (pre-activation).

*Key properties:*

1. *Bounded eigenvalues:* Since $tanh'(z) = 1 - tanh^2(z) in (0, 1]$:
   $ lambda_i in (0, alpha] $

2. *Vanishing gradients:* Over $T$ timesteps, gradient magnitude scales as:
   $ norm((partial cal(L)) / (partial bold(S)_0)) <= alpha^T dot product_(t=1)^T (1 - tanh^2(bold(Z)_t)) $

   This vanishes exponentially, but $alpha approx 1$ (from Mamba2-style decay) keeps it manageable.

3. *Stability guarantee:* Unlike linear systems where eigenvalues can be arbitrarily large, E88's state is bounded:
   $ norm(bold(S))_infinity <= 1 $

== Residual Connections and Condition Number

E88 (via LadderLM) uses residual connections: $bold(y) = bold(x) + f(bold(x))$.

*Theorem (Gradient Condition Number):*

For a residual block $bold(y) = bold(x) + f(bold(x))$ where $norm((partial f) / (partial bold(x))) <= 1$:

$ kappa = (lambda_max) / (lambda_min) = 2 / 1 = 2 $

For a stock (non-residual) Elman network:

$ kappa = (lambda_max) / (lambda_min) = 1 / 0 = infinity $

*Interpretation:* The condition number $kappa$ measures how much gradients can vary in magnitude across different directions. Lower $kappa$ means more uniform gradient flow.

- *Residual E88:* $kappa = 2$ means gradients in any direction are within 2x of each other
- *Stock Elman:* $kappa = infinity$ means some directions get zero gradient (vanishing)

This explains why E88 trains stably at 480M+ scale despite using $tanh$.

== Expressivity-Gradient Tradeoff

#align(center)[
#table(
  columns: 4,
  align: (left, center, center, center),
  [*Model*], [*Expressivity*], [*Gradient Bound*], [*Condition $kappa$*],
  [Linear SSM], [Limited], [$[0, infinity)$], [Unbounded],
  [Stock Elman], [Full (tanh)], [$[0, 1]$], [$infinity$],
  [E88 (residual)], [Full (tanh)], [$[1, 2]$], [$2$],
)
]

E88 achieves the best of both worlds:
- *Full expressivity* from the $tanh$ nonlinearity (can compute any continuous function)
- *Bounded gradients* from residual connections ($kappa = 2$)

The cost is sequential computation (no parallel scan), but the gradient stability enables training at arbitrary scale.

== Why Not Just Use Linear + Post-hoc Nonlinearity?

One might ask: why not use a linear state (like Mamba2) and apply nonlinearity only at output?

*Answer:* The _composition_ of nonlinearities across timesteps creates exponentially more complex decision boundaries than a single final nonlinearity.

Consider two timesteps:
- Linear: $bold(y) = bold(A)_2 (bold(A)_1 bold(x) + bold(b)_1) + bold(b)_2 = bold(A)_2 bold(A)_1 bold(x) + "const"$ (still linear!)
- Nonlinear: $bold(y) = tanh(bold(A)_2 tanh(bold(A)_1 bold(x) + bold(b)_1) + bold(b)_2)$ (nested nonlinearity)

The nested $tanh$ creates decision boundaries that cannot be replicated by any linear system with a single output nonlinearity.

== Information Capacity Bounds

From capacity analysis, the effective rank of E88's state after $T$ timesteps saturates at:

$ "eff_rank"(bold(S)_T) approx min(n, 1 / (1 - alpha)) $

For $alpha = 0.99$ (typical decay), this gives $"eff_rank" approx 100$, meaning even an $n = 32$ state can effectively store information across 100+ timesteps.

The delta rule (@delta) further improves capacity by only updating the component of $bold(v)$ not already stored:
$ bold(delta) = bold(v) - bold(S) bold(k) $

This is provably more efficient than direct writes ($bold(S) := alpha bold(S) + bold(v) bold(k)^top$) used in simpler attention mechanisms.

#pagebreak()

= Appendix E: CMA-ES Hyperparameter Search Results

We performed systematic hyperparameter optimization using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) across 8 model architectures at the 480-500M parameter scale. All runs used 10 minutes of training time on The Pile dataset with learning rate $3 times 10^(-4)$ and bfloat16 precision.

== Summary Table

#align(center)[
#table(
  columns: 7,
  align: (left, right, right, right, right, right, left),
  [*Model*], [*Loss*], [*Steps*], [*Tok/s*], [*Dim*], [*Params*], [*Best Config*],
  [Mamba2], [1.27], [3140], [22.7K], [1792], [494M], [n_state=96, exp=2, d=25],
  [FLA-GDN], [1.27], [3110], [22.3K], [1920], [502M], [exp=2, d=17, H=24],
  [E88], [1.39], [2050], [14.0K], [2176], [481M], [H=98, n=32, d=14],
  [Transformer], [1.51], [4820], [34.1K], [1536], [491M], [H=8, exp=4, d=13],
  [MinGRU], [1.53], [4050], [27.8K], [2944], [486M], [exp=1, d=14],
  [MinLSTM], [1.56], [3190], [21.3K], [1792], [498M], [exp=1, d=31],
  [MoM-E88], [1.76], [740], [20.4K], [3840], [480M], [H=40, k=8, n=64, d=12],
  [E90], [1.79], [420], [13.8K], [3072], [497M], [H=114, fast=8, slow=16, d=13],
  [GRU/LSTM], [--], [--], [--], [--], [480M], [Training failed at scale],
)
]

Loss values are last-100 step averages. Steps and tok/s are from the best configuration's 10-minute run.

== Key Findings

=== Tier 1: SSM Baselines (Loss ~1.27)

*Mamba2* and *FLA-GDN* achieve identical performance at this scale, both reaching 1.27 loss. Key CMA-ES discoveries:

- *Mamba2:* n_state=96 beats default 64. Depth 25 is optimal.
- *FLA-GDN:* Shallower is better (depth=17 vs typical 24). 24 heads optimal.

=== Tier 2: E88 (Loss ~1.39)

E88 trails SSMs by ~0.12 nats but uses 3-4× less state memory per layer. CMA-ES found:

- *Shallow + Wide:* depth=14, dim=2176 beats deeper/narrower configs
- *n_state=32 optimal:* Consistently outperforms 16 or 48
- *~98 heads:* More heads than expected (vs ~64 in initial configs)

The gap to SSMs appears architectural: E88's nonlinear state cannot use parallel scan.

=== Tier 3: Attention (Loss ~1.51)

*Transformer* achieves 1.51 loss but with highest throughput (34K tok/s). CMA-ES found:

- Very shallow (depth=13) with wide FFN (expansion=4)
- Only 8 attention heads needed at this scale

=== Tier 4: Minimal RNNs (Loss 1.53-1.56)

*MinGRU* and *MinLSTM* perform similarly at 1.53-1.56 loss:

- Both prefer expansion=1 (no expansion, just wider dims)
- MinGRU prefers shallow (d=14), MinLSTM deep (d=31)

=== Failed: CUDA GRU/LSTM

Standard CUDA GRU and LSTM failed to train at 480M scale (gradient instability). This highlights the training difficulty of traditional gated RNNs at scale.

=== Experimental: MoM-E88 and E90

*MoM-E88* (Mixture of Memory) routes to top-k heads instead of updating all heads. At 1.76 loss, it's not yet competitive but shows potential for scaling to more heads.

*E90* uses dual-speed memory (fast k=8 + slow k=16). At 1.79 loss, the architecture needs further optimization.

== Search Space and Methodology

Each CMA-ES run used population size 8 with 15-30 generations (120-240 total evaluations).

#table(
  columns: 3,
  align: (left, left, left),
  [*Model*], [*Search Parameters*], [*Evals*],
  [Mamba2], [n_state $in$ [64,160], expand $in$ [1,3], depth $in$ [16,40]], [120],
  [FLA-GDN], [expansion $in$ [1,3], depth $in$ [16,40], H $in$ [8,32]], [120],
  [E88], [H $in$ [64,160], n $in$ \{16,32,48,64\}, depth $in$ [12,40]], [240],
  [Transformer], [H $in$ [8,32], expansion $in$ [2,6], depth $in$ [12,36]], [120],
  [MinGRU/MinLSTM], [expansion $in$ [1,4], depth $in$ [12,40]], [120 each],
)

Model dimension was auto-calculated to hit the target parameter count given other hyperparameters.
