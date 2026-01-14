// Sparse and Hard Attention for Memory-Augmented Neural Networks
// A Research Survey for E23 Dual-Memory RNN Architecture

#set document(
  title: "Sparse and Hard Attention for Memory-Augmented Neural Networks",
  author: "Research Survey",
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
)

#set heading(numbering: "1.1")

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Sparse and Hard Attention for Memory-Augmented Neural Networks
  ]

  #v(0.5em)

  #text(size: 12pt)[
    A Research Survey for E23 Dual-Memory RNN Architecture
  ]

  #v(1em)

  #text(size: 10pt, style: "italic")[
    January 2026
  ]
]

#v(2em)

= Introduction

Memory-augmented neural networks (MANNs) such as Neural Turing Machines (NTMs) [graves2014ntm] and Differentiable Neural Computers (DNCs) [graves2016dnc] have demonstrated remarkable capabilities in learning algorithmic solutions to complex tasks. At the heart of these architectures lies the attention mechanism, which determines how the network reads from and writes to external memory.

The standard approach uses *soft attention* based on the softmax function, which produces a dense probability distribution over all memory locations. While this is fully differentiable and amenable to gradient-based optimization, it suffers from fundamental limitations when applied to memory systems:

- *Information smearing*: Every read operation retrieves a weighted combination of all memory slots, causing interference between stored items
- *Capacity degradation*: As memory size grows, the attention distribution becomes increasingly diffuse
- *Non-compositional operations*: True Turing machine semantics require addressing exactly one tape cell, not a weighted blend

This survey examines the landscape of techniques for achieving *sparse* and *hard* attention---attention mechanisms that can focus on one or few memory locations while remaining trainable. We specifically consider their applicability to the E23 architecture, a dual-memory RNN combining a Turing machine-style tape with working memory.

= The Problem: Why Soft Attention Fails for Memory

== Softmax and Dense Attention

The standard softmax function transforms a vector of scores $bold(z) in RR^K$ into a probability distribution:

$ "softmax"_i (bold(z)) = exp(z_i) / (sum_(j=1)^K exp(z_j)) $

This function has several desirable properties:
- Outputs are always positive and sum to 1
- Differentiable everywhere
- Smooth gradients for backpropagation

However, softmax *never produces exact zeros*. For any finite input, all output probabilities are strictly positive. This is problematic for memory systems.

== Interference and Capacity Loss

When a memory-augmented network reads from memory using soft attention, it computes:

$ bold(r) = sum_(i=1)^N w_i bold(M)_i $

where $bold(w)$ is the attention distribution and $bold(M)_i$ are memory rows. Even when the network "intends" to read from location $j$, it retrieves a contaminated signal mixing in content from all other locations [sukhbaatar2015memn2n].

This creates several problems:

1. *Catastrophic interference*: Similar to catastrophic forgetting in neural networks [kirkpatrick2017ewc], stored patterns interfere with each other. As the number of patterns approaches memory capacity, signal-to-noise ratio degrades exponentially with gradient descent approaches.

2. *Attention dilution*: With large memories, even a "peaked" softmax distribution spreads probability mass across many slots. For memory size $N$, the maximum attention weight on any single slot is bounded.

3. *Write interference*: Soft writes similarly corrupt neighboring memory locations, degrading previously stored information.

== The Discrete-Continuous Dilemma

The fundamental tension is:
- *Hard attention* (argmax): Selects exactly one location, but has zero gradient almost everywhere
- *Soft attention* (softmax): Fully differentiable, but never truly sparse

Algorithmically, we want Turing machine semantics---read or write exactly one tape cell. But mathematically, we need gradients to flow for learning.

= Sparsemax: Euclidean Projection onto the Simplex

== Origins and Formulation

Martins and Astudillo [martins2016sparsemax] introduced sparsemax as a sparse alternative to softmax. While softmax can be viewed as maximizing entropy subject to moment constraints, sparsemax solves a different optimization problem---Euclidean projection onto the probability simplex:

$ "sparsemax"(bold(z)) = arg min_(bold(p) in Delta^(K-1)) ||bold(p) - bold(z)||_2^2 $

where $Delta^(K-1) = {bold(p) in RR^K : bold(p) >= 0, sum_i p_i = 1}$ is the $(K-1)$-simplex.

== Closed-Form Solution

The projection has an elegant closed-form solution:

$ "sparsemax"_i (bold(z)) = [z_i - tau(bold(z))]_+ $

where $[t]_+ = max(0, t)$ and $tau(bold(z))$ is a threshold determined by the constraint that outputs sum to 1. The threshold can be found in $O(K log K)$ time via sorting or $O(K)$ time with median-finding algorithms [duchi2008projection].

== Sparsity Property

The key property of sparsemax is that it produces *exact zeros* for low-scoring inputs. When the projection hits the boundary of the simplex, some coordinates are clamped to zero. This is in stark contrast to softmax, which assigns positive probability to all outcomes.

For attention mechanisms, this means sparsemax can completely ignore irrelevant memory locations, assigning them exactly zero weight.

== Jacobian and Gradients

Martins et al. derived the Jacobian of sparsemax in closed form. Let $S(bold(z)) = "supp"("sparsemax"(bold(z)))$ be the set of indices with non-zero output. Then:

$ (partial "sparsemax"(bold(z))) / (partial bold(z)) = bold(I)_S - 1/(|S|) bold(1)_S bold(1)_S^T $

where $bold(I)_S$ is the identity restricted to $S$ and $bold(1)_S$ is the indicator vector for $S$. This is sparse when the output is sparse, enabling efficient backpropagation.

== Sparsemax Loss

For training with sparsemax outputs, a corresponding loss function is needed. The sparsemax loss is:

$ L_"sparsemax"(bold(z), y) = -z_y + 1/2 sum_(j in S(bold(z))) (z_j^2 - tau^2) + 1/2 $

This loss is smooth and convex, and reveals an unexpected connection to the Huber classification loss [martins2016sparsemax].

== Empirical Results

Sparsemax has shown promising results in:
- *Natural language inference*: Comparable accuracy to softmax with more interpretable, focused attention
- *Multi-label classification*: Slight advantage over softmax and logistic losses in several benchmarks
- *Attention mechanisms*: Sparse attention maps that highlight key elements without accuracy loss

However, sparsemax can be "too aggressive" in some cases, assigning zero probability to items that should receive small but non-zero weight.

= Entmax: The Alpha-Entmax Family

== Generalization via Tsallis Entropy

The $alpha$-entmax family [peters2019sparse] [correia2019adaptively] generalizes both softmax and sparsemax through the lens of Tsallis entropy. The transformation is defined as:

$ alpha"-entmax"(bold(z)) = arg max_(bold(p) in Delta^(K-1)) [bold(p)^T bold(z) + H_alpha^T (bold(p))] $

where $H_alpha^T$ is the Tsallis $alpha$-entropy:

$ H_alpha^T (bold(p)) = cases(
  -sum_j p_j log p_j & "if" alpha = 1,
  1/(alpha(alpha - 1)) sum_j (p_j - p_j^alpha) & "if" alpha != 1
) $

== Special Cases

The entmax family spans a continuum:
- $alpha = 1$: Recovers *softmax* (Shannon entropy regularization)
- $alpha = 2$: Recovers *sparsemax* (Euclidean projection)
- $alpha = 1.5$: A popular middle ground, "1.5-entmax"

For any $alpha > 1$, entmax can produce sparse outputs with exact zeros. The larger $alpha$, the sparser the output tends to be.

== The 1.5-Entmax Sweet Spot

The transformation $1.5$-entmax has emerged as a practical default:

$ 1.5"-entmax"_i (bold(z)) = [(alpha - 1) z_i - tau]_+^(1/(alpha-1)) $

where $alpha = 1.5$ gives an exponent of 2. This provides:
- More graceful sparsity than sparsemax
- Better gradient flow than sparsemax
- Still achieves exact zeros for low-scoring inputs

== Learnable Alpha

A powerful extension allows the $alpha$ parameter itself to be learned [correia2019adaptively]. Different attention heads can learn different sparsity levels:
- Some heads may prefer soft, diffuse attention ($alpha$ close to 1)
- Others may learn sharp, sparse attention ($alpha$ close to 2 or higher)

This adaptivity has shown improvements in machine translation and other sequence tasks.

== Computational Considerations

The computational overhead of entmax variants is moderate:
- 1.5-entmax runs at approximately 90% the speed of softmax
- Full $alpha$-entmax (with arbitrary $alpha$) runs at approximately 75% of softmax speed

Implementations are available in the `entmax` library (deep-spin/entmax on GitHub).

= Gumbel-Softmax: Differentiable Discrete Sampling

== The Reparameterization Problem

Sampling from discrete distributions is inherently non-differentiable. Given class probabilities $bold(pi)$, drawing a one-hot sample $bold(y)$ via $y_i = 1$ if $i = arg max_j [log pi_j + g_j]$ (where $g_j$ are i.i.d. Gumbel noise) has zero gradient with respect to $bold(pi)$.

Jang et al. [jang2017gumbel] and Maddison et al. [maddison2017concrete] independently discovered a continuous relaxation that enables gradient-based learning.

== The Gumbel-Softmax Distribution

Replace the hard argmax with a temperature-scaled softmax:

$ y_i = exp((log pi_i + g_i) / tau) / (sum_(j=1)^K exp((log pi_j + g_j) / tau)) $

where $g_i tilde.op "Gumbel"(0, 1)$ and $tau > 0$ is the temperature parameter.

This defines the *Gumbel-Softmax* (or *Concrete*) distribution, which:
- Is reparameterizable: samples can be computed as a differentiable function of parameters
- Approaches one-hot categorical as $tau -> 0$
- Becomes uniform as $tau -> infinity$

== Temperature Control

The temperature $tau$ controls the discreteness-smoothness tradeoff:
- *Low $tau$* (e.g., 0.1): Samples are nearly one-hot, but gradients have high variance
- *High $tau$* (e.g., 1.0+): Samples are smooth, gradients are stable, but poor approximation to discrete

In practice, temperatures in the range $[0.5, 1.0]$ often work well for training.

== Straight-Through Gumbel-Softmax

A common variant uses the "straight-through" trick:
- *Forward pass*: Use hard argmax to get true one-hot samples
- *Backward pass*: Use Gumbel-Softmax gradients

This gives the best of both worlds---discrete behavior during forward computation, but gradient flow during backpropagation. However, it introduces bias in the gradient estimates.

== Temperature Annealing Schedules

Many practitioners anneal temperature during training:
- Start with high $tau$ (e.g., 1.0 or higher) for stable early training
- Gradually reduce to low $tau$ (e.g., 0.1-0.5) for sharper final behavior

Common schedules include:
- *Linear*: $tau_t = tau_0 - (tau_0 - tau_min) dot t/T$
- *Exponential*: $tau_t = max(tau_min, tau_0 dot exp(-r dot t))$

Reported successful configurations include annealing from $tau = 30$ to $tau = 1$ over the first 10 epochs, or from $tau = 5$ to $tau = 0.05$ over 5 epochs.

== Comparison to REINFORCE

Gumbel-Softmax offers several advantages over REINFORCE-style gradient estimators [williams1992reinforce]:
- *Lower variance*: Reparameterization provides much lower variance than score function estimators
- *Simpler implementation*: No need for baselines or control variates
- *Faster convergence*: Empirically trains more stably

The tradeoff is bias---Gumbel-Softmax gradients are biased estimates of the true discrete gradient, while REINFORCE is unbiased but high-variance.

= Straight-Through Estimator (STE)

== Concept and Origins

The Straight-Through Estimator [bengio2013ste] is a simple but effective heuristic for training networks with discrete operations:
- *Forward pass*: Apply the discrete operation (e.g., argmax, thresholding, quantization)
- *Backward pass*: Pretend the operation was the identity (or some smooth proxy)

This "passes gradients straight through" the non-differentiable operation.

== Mathematical Formulation

For a discrete operation $f$ and smooth proxy $g$:

$ y = f(x) " (forward)" $
$ (partial L)/(partial x) = (partial L)/(partial y) dot g'(x) " (backward)" $

The simplest case uses $g(x) = x$, so $g'(x) = 1$, meaning gradients pass through unchanged.

== Bias-Variance Tradeoff

STE is a *biased* gradient estimator:
- It does not compute the true gradient of the loss with respect to parameters
- The bias can be significant when the discrete and smooth behaviors differ substantially

However, STE has *low variance*:
- No stochastic sampling is involved
- Gradients are deterministic given the forward computation

In practice, this bias-variance tradeoff often favors STE, especially in large networks where high-variance gradients are problematic.

== Theoretical Justification

Recent work [yin2019ste] has provided theoretical justification for STE:
- If the STE is properly chosen, the expected "coarse gradient" correlates positively with the true population gradient
- The negative coarse gradient is a descent direction for minimizing population loss
- Coarse gradient descent converges to critical points of the loss

== Applications

STE has found success in:
- *Binary neural networks*: Training networks with binary weights/activations
- *Quantization*: Training quantized neural networks for deployment
- *Hard attention*: Using argmax attention during training
- *Discrete latent variables*: VAEs with categorical latent codes

= Top-k and Local Attention

== Top-k Attention

Top-k attention restricts each query to attend to only the $k$ highest-scoring keys:

$ w_i = cases(
  "softmax"_i (bold(z))_S & "if" i in S,
  0 & "otherwise"
) $

where $S$ is the set of indices with the $k$ largest scores.

This provides:
- *Guaranteed sparsity*: Exactly $K - k$ entries are zero
- *Bounded computation*: Only $k$ memory locations are accessed
- *Controllable focus*: $k$ is a hyperparameter trading off precision and recall

== Differentiable Top-k

Pure top-k selection is non-differentiable. Several approaches enable gradient flow:

1. *Straight-through*: Use hard top-k forward, softmax gradients backward
2. *Relaxed top-k*: Use smooth approximations like entmax or Gumbel-Softmax
3. *SparseK attention* [sparsekattention]: A differentiable top-k mask operator

== Local/Window Attention

An alternative to content-based sparsity is *position-based* sparsity:
- Each position attends only to nearby positions (sliding window)
- Reduces complexity from $O(N^2)$ to $O(N dot W)$ for window size $W$

Architectures like BigBird [bigbird] combine:
- *Local attention*: Sliding window around each position
- *Global attention*: A few positions attend to all others
- *Random attention*: Sparse random connections

== Block-Sparse Attention

For efficiency, attention can be computed at the *block* level:
- Divide sequences into blocks of size $B$ (e.g., 16x16)
- Select top-k blocks rather than individual positions
- Enables efficient GPU implementations

Block-sparse attention has shown that 100x or more sparsity is achievable with minimal performance degradation on many tasks.

= Temperature Annealing for Attention

== Softmax Temperature

The softmax function can be parameterized with temperature:

$ "softmax"_i (bold(z); tau) = exp(z_i / tau) / (sum_(j=1)^K exp(z_j / tau)) $

Temperature effects:
- $tau = 1$: Standard softmax
- $tau < 1$: Sharper, more peaked distribution
- $tau > 1$: Softer, more uniform distribution
- $tau -> 0$: Approaches hard argmax
- $tau -> infinity$: Approaches uniform distribution

== Annealing Strategies

Temperature annealing gradually sharpens attention during training:

1. *Start warm* ($tau$ high): Encourages exploration, stable gradients
2. *End cool* ($tau$ low): Sharp, focused attention for final model

Common schedules:

$ tau_t = tau_"min" + (tau_"max" - tau_"min") dot (1 - t/T) " (linear)" $

$ tau_t = tau_"min" + (tau_"max" - tau_"min") dot exp(-r dot t) " (exponential)" $

$ tau_t = tau_"min" + (tau_"max" - tau_"min") / (1 + exp(s dot (t - T/2))) " (sigmoid)" $

== Empirical Findings

Research has found:
- Large temperature in early epochs is important for stable training
- Annealing from $tau = 30$ to $tau = 1$ over 10 epochs works well for some tasks
- For Gumbel-Softmax, schedules like $tau = max(0.5, exp(-r dot t))$ are common
- Final temperature should be non-zero to maintain gradient flow

== Learnable Temperature

Rather than a fixed schedule, temperature can be learned:
- Per-head learnable temperature in multi-head attention
- Allows different heads to specialize in sharp vs. diffuse attention
- Small but consistent improvements reported in transformer models [learnable_temp]

= Empirical Findings from Memory Networks Literature

== Neural Turing Machines (NTM)

Graves et al. [graves2014ntm] introduced NTMs with soft attention over external memory. Key findings:

- Soft attention enables end-to-end training via backpropagation
- Content-based + location-based addressing provides flexibility
- NTMs learn simple algorithms (copying, sorting, recall) from examples
- Training can be unstable; careful initialization is important

== Dynamic NTM with Hard Attention

Gulcehre et al. [dntm] extended NTMs with both soft and hard attention:
- *Hard attention*: Discrete read/write to single locations
- *Soft attention*: Traditional weighted combination

Key findings on Facebook bAbI tasks:
- Hard attention with GRU controller often *converges faster* than soft attention
- Hard attention models *outperform* soft attention on several tasks
- Discrete attention provides more interpretable memory access patterns

== Differentiable Neural Computer (DNC)

The DNC [graves2016dnc] extended NTMs with:
- Dynamic memory allocation (tracking usage)
- Temporal memory linkage (recording write order)
- Three forms of differentiable attention (content, allocation, temporal)

Sparse addressing refinements:
- Sparse memory addressing reduces time and space complexity by 1000x+
- Sparse Access Memory (SAM) [sam2016] achieves $O(N log N)$ time with $O(N)$ memory
- SAM reads from only $K = 4$ locations per head with minimal performance loss

== End-to-End Memory Networks (MemN2N)

Sukhbaatar et al. [sukhbaatar2015memn2n] introduced MemN2N:
- Soft attention over memory, trained end-to-end
- Multiple "hops" over memory for multi-step reasoning
- Key insight: soft attention enables learning without explicit supervision of attention

Limitations:
- Soft attention causes interference between memories
- Performance degrades as memory size increases
- Hard attention would improve but requires REINFORCE

== Practical Recommendations from Literature

Based on the surveyed literature:

1. *Soft attention is easier to train* but suffers from interference
2. *Hard attention can outperform soft attention* when successfully trained
3. *Sparse Access Memory* achieves best of both worlds for large memories
4. *Temperature annealing* helps transition from soft to hard-like attention
5. *Entmax provides a principled middle ground* between soft and hard

= Recommendations for RNN Memory Systems

== Analysis for E23 Architecture

The E23 architecture combines:
- A Turing machine-style tape (requires discrete addressing)
- Working memory (may benefit from softer attention)
- RNN controller (needs stable gradient flow)

This creates specific requirements:
1. Tape operations should be as discrete as possible (TM semantics)
2. Training must remain stable with gradient-based optimization
3. Memory capacity should not degrade with tape size

== Recommended Approaches

Based on our survey, we recommend a tiered approach:

=== Tier 1: Start with Entmax (Recommended Default)

$alpha$-entmax with $alpha in [1.5, 2.0]$ provides:
- Exact zeros for truly irrelevant locations
- Smooth gradients for stable training
- Tunable sparsity via $alpha$ parameter
- Well-tested in sequence models

*Implementation*: Use the `entmax` library. Start with $alpha = 1.5$, tune based on task.

=== Tier 2: Gumbel-Softmax with Straight-Through

If more discrete behavior is needed:
- Use Straight-Through Gumbel-Softmax
- Anneal temperature from 1.0 to 0.5 over training
- Forward pass uses hard argmax, backward uses soft gradients

*Tradeoff*: More discrete than entmax, but biased gradients.

=== Tier 3: Learnable Sparsity

For maximum flexibility:
- Use learnable $alpha$-entmax
- Different memory operations can learn different sparsity
- Read operations might prefer sharper attention than write

=== For Tape vs Working Memory

Consider different attention mechanisms for different memory types:
- *Tape*: Sharper attention ($alpha = 2.0$ or Gumbel-ST) for TM-like semantics
- *Working memory*: Softer attention ($alpha = 1.5$) for pattern completion

== Implementation Checklist

1. Replace softmax with entmax in attention computations
2. Start with $alpha = 1.5$, increase if more sparsity needed
3. Monitor attention entropy during training
4. Consider temperature annealing if using Gumbel-Softmax
5. Use sparse implementations for large memories (SAM-style)
6. Evaluate both training stability and final task performance

== Expected Benefits

Switching from softmax to sparse attention should provide:
- *Reduced interference*: Cleaner memory operations
- *Better scaling*: Memory capacity preserved with size
- *Improved interpretability*: Discrete attention patterns
- *Maintained trainability*: Gradient-based optimization still works

= Conclusion

The tension between discrete memory operations and differentiable training is fundamental to memory-augmented neural networks. This survey has examined the landscape of techniques for bridging this gap:

- *Sparsemax* provides Euclidean projection to the simplex with exact zeros
- *Entmax* generalizes this to a tunable family with learnable sparsity
- *Gumbel-Softmax* enables differentiable sampling from discrete distributions
- *Straight-Through Estimators* offer simple but effective gradient approximation
- *Top-k and local attention* provide structural sparsity constraints
- *Temperature annealing* allows transition from soft to hard during training

For the E23 dual-memory architecture, we recommend starting with $alpha$-entmax as a principled middle ground that provides sparse attention while maintaining stable training. More aggressive discretization via Gumbel-Softmax with straight-through can be explored if TM-like semantics are paramount.

The empirical literature suggests that sparse and hard attention mechanisms can match or exceed soft attention performance when successfully trained, with the additional benefits of improved interpretability and computational efficiency. The key is choosing the right technique and hyperparameters for the specific memory architecture and task requirements.

#pagebreak()

= References

// #bibliography(
  // title: none,
  // style: "ieee",
)

// Note: In actual Typst compilation, you would have a .bib file.
// Below are the reference entries for documentation purposes:

/*
References:

@article{graves2014ntm,
  title={Neural Turing Machines},
  author={Graves, Alex and Wayne, Greg and Danihelka, Ivo},
  journal={arXiv preprint arXiv:1410.5401},
  year={2014},
  url={https://arxiv.org/abs/1410.5401}
}

@article{graves2016dnc,
  title={Hybrid computing using a neural network with dynamic external memory},
  author={Graves, Alex and Wayne, Greg and Reynolds, Malcolm and others},
  journal={Nature},
  volume={538},
  pages={471--476},
  year={2016},
  url={https://www.nature.com/articles/nature20101}
}

@inproceedings{martins2016sparsemax,
  title={From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification},
  author={Martins, Andr{\'e} F. T. and Astudillo, Ram{\'o}n Fernandez},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={1614--1623},
  year={2016},
  url={https://arxiv.org/abs/1602.02068}
}

@inproceedings{peters2019sparse,
  title={Sparse Sequence-to-Sequence Models},
  author={Peters, Ben and Niculae, Vlad and Martins, Andr{\'e} F. T.},
  booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2019},
  url={https://arxiv.org/abs/1905.05702}
}

@inproceedings{correia2019adaptively,
  title={Adaptively Sparse Transformers},
  author={Correia, Gon{\c{c}}alo M. and Niculae, Vlad and Martins, Andr{\'e} F. T.},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2019},
  url={https://arxiv.org/abs/1909.00015}
}

@inproceedings{jang2017gumbel,
  title={Categorical Reparameterization with Gumbel-Softmax},
  author={Jang, Eric and Gu, Shixiang and Poole, Ben},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017},
  url={https://arxiv.org/abs/1611.01144}
}

@inproceedings{maddison2017concrete,
  title={The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables},
  author={Maddison, Chris J. and Mnih, Andriy and Teh, Yee Whye},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017},
  url={https://arxiv.org/abs/1611.00712}
}

@article{bengio2013ste,
  title={Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation},
  author={Bengio, Yoshua and L{\'e}onard, Nicholas and Courville, Aaron},
  journal={arXiv preprint arXiv:1308.3432},
  year={2013},
  url={https://arxiv.org/abs/1308.3432}
}

@inproceedings{yin2019ste,
  title={Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets},
  author={Yin, Penghang and Lyu, Jiancheng and Zhang, Shuai and Osher, Stanley and Qi, Yingyong and Xin, Jack},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019},
  url={https://arxiv.org/abs/1903.05662}
}

@article{williams1992reinforce,
  title={Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning},
  author={Williams, Ronald J.},
  journal={Machine Learning},
  volume={8},
  pages={229--256},
  year={1992}
}

@inproceedings{sukhbaatar2015memn2n,
  title={End-To-End Memory Networks},
  author={Sukhbaatar, Sainbayar and Szlam, Arthur and Weston, Jason and Fergus, Rob},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2015},
  url={https://arxiv.org/abs/1503.08895}
}

@inproceedings{dntm,
  title={Dynamic Neural Turing Machine with Soft and Hard Addressing Schemes},
  author={Gulcehre, Caglar and Chandar, Sarath and Cho, Kyunghyun and Bengio, Yoshua},
  booktitle={arXiv preprint},
  year={2016},
  url={https://arxiv.org/abs/1607.00036}
}

@inproceedings{sam2016,
  title={Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes},
  author={Rae, Jack and Hunt, Jonathan J. and Danihelka, Ivo and Harley, Timothy and Senior, Andrew W. and Wayne, Gregory and Graves, Alex and Lillicrap, Timothy P.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2016},
  url={https://arxiv.org/abs/1610.09027}
}

@inproceedings{kirkpatrick2017ewc,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and others},
  booktitle={Proceedings of the National Academy of Sciences (PNAS)},
  year={2017},
  url={https://www.pnas.org/doi/10.1073/pnas.1611835114}
}

@inproceedings{duchi2008projection,
  title={Efficient Projections onto the L1-Ball for Learning in High Dimensions},
  author={Duchi, John and Shalev-Shwartz, Shai and Singer, Yoram and Chandra, Tushar},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2008}
}

@article{bigbird,
  title={Big Bird: Transformers for Longer Sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Avinava and others},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@article{learnable_temp,
  title={Introducing a learnable temperature value into the softmax self-attention scores},
  author={Ryan, Nick},
  year={2024},
  url={https://nickcdryan.com/2024/08/02/introducing-a-learnable-temperature-value-into-the-self-attention-scores/}
}

@article{sparsekattention,
  title={Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers},
  year={2024},
  url={https://arxiv.org/abs/2406.16747}
}
*/
