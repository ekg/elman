# Elman Ablation Ladder: Comprehensive Literature Review

This document provides a comprehensive background literature review for the Elman Ablation Ladder research framework, covering foundational concepts, state-of-the-art architectures, implementation techniques, and optimization strategies.

---

## Table of Contents

1. [Foundational RNN Architectures](#1-foundational-rnn-architectures)
2. [State Space Models and Modern RNN Variants](#2-state-space-models-and-modern-rnn-variants)
3. [Gating Mechanisms](#3-gating-mechanisms)
4. [Log-Space Computation and Numerical Stability](#4-log-space-computation-and-numerical-stability)
5. [Spectral Normalization and RNN Stability](#5-spectral-normalization-and-rnn-stability)
6. [CUDA Kernel Optimization](#6-cuda-kernel-optimization)
7. [Training Techniques](#7-training-techniques)
8. [Architecture Design Patterns](#8-architecture-design-patterns)
9. [Benchmarks and Evaluation](#9-benchmarks-and-evaluation)
10. [Implementation Details](#10-implementation-details)

---

## 1. Foundational RNN Architectures

### 1.1 Elman Networks (Simple Recurrent Networks)

**Foundational Paper**: Elman, J.L. (1990). "Finding Structure in Time." *Cognitive Science*, 14, 179-211.
- [Wiley Online Library](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1)
- [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/036402139090002E)
- [Semantic Scholar PDF](https://www.semanticscholar.org/paper/Finding-Structure-in-Time-Elman/668087f0ae7ce1de6e0bd0965dbb480c08103260)

The Elman Network was the first successful recurrent network trained with backpropagation. Key innovations:
- **Context units**: Hidden unit patterns are fed back to themselves
- **Implicit time representation**: Time is represented by its effects on processing rather than explicitly
- **Dynamic memory**: Internal representations reflect task demands in context of prior internal states

Originally trained with truncated BPTT considering only two time-steps (t and t-1).

**Historical Context**:
- Jordan Networks (1986): Similar architecture but context units fed from output layer
- Hopfield Networks (1982): Earlier associative memory networks

**Tutorial Resources**:
- [Pablo Insente: The Recurrent Neural Network](https://pabloinsente.github.io/the-recurrent-net)
- [Computational Cognitive Book](https://com-cog-book.github.io/com-cog-book/features/recurrent-net.html)

### 1.2 The Vanishing/Exploding Gradient Problem

**Key Papers**:
- Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis.
- Bengio, Y., Simard, P., & Frasconi, P. (1994). "Learning long-term dependencies with gradient descent is difficult."
- [NeurIPS 2024: Recurrent neural networks: vanishing and exploding](https://proceedings.neurips.cc/paper_files/paper/2024/file/fbb07254ef01868967dc891ea3fa6c13-Paper-Conference.pdf)

**Problem**: In RNNs, the same weights are used at each time step. During BPTT, if weight products are less than 1, gradients shrink exponentially; if greater than 1, they explode.

**Solutions**:
1. **Architectural**: LSTM, GRU, skip connections
2. **Initialization**: Xavier, He, Orthogonal
3. **Normalization**: LayerNorm, RMSNorm, Spectral Norm
4. **Gradient Clipping**: Limit gradient magnitude

**Resources**:
- [GeeksforGeeks: Vanishing and Exploding Gradients](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/)
- [Neptune.ai: Debugging Vanishing Gradients](https://neptune.ai/blog/vanishing-and-exploding-gradients-debugging-monitoring-fixing)
- [Orthogonal Initialization Explained](https://smerity.com/articles/2016/orthogonal_init.html)

---

## 2. State Space Models and Modern RNN Variants

### 2.1 Structured State Space Models (S4)

**Foundational Paper**: Gu, A., Goel, K., & Ré, C. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces."
- [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)

S4 addresses two shortcomings of discrete SSMs:
1. Inefficiency of training
2. Inability to model long sequences

Key innovations:
- HiPPO matrix for long-range dependencies
- Efficient computation via structured parameterization
- Linear/near-linear complexity

### 2.2 Diagonal State Space Models (S4D, S5)

**S4D Paper**: Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). "On the Parameterization and Initialization of Diagonal State Space Models."
- [arXiv:2206.11893](https://arxiv.org/abs/2206.11893)

S4D simplifies S4 by using diagonal state matrices:
- Kernel computation requires just 2 lines of code
- State-of-the-art on image, audio, medical time-series
- 85% average on Long Range Arena

**S5 Paper**: Smith, J.T.H., Warrington, A., & Linderman, S.W. (2023). "Simplified State Space Layers for Sequence Modeling."
- [OpenReview](https://openreview.net/forum?id=Ai8Hw3AXqks)
- [arXiv PDF](https://arxiv.org/pdf/2208.04933)

S5 innovations:
- Multi-input multi-output (MIMO) structure
- Diagonalized parameterization
- Fully recurrent with linear-time complexity
- Highest score on Path-X task

### 2.3 Mamba: Selective State Spaces

**Paper**: Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
- [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- [GitHub](https://github.com/state-spaces/mamba)

Key innovation: **Input-dependent (selective) parameterization**
- Matrices B, C, and step size Δ become functions of input
- Overcomes inability of LTI SSMs to perform content-based reasoning
- Connection to RNN gating: Δ plays role of generalized gating

Performance:
- 5× higher throughput than Transformers
- Linear scaling in sequence length
- Mamba-3B outperforms Transformers of same size

**Mamba-2**: "Transformers are SSMs" (ICML 2024) - reveals duality between SSMs and attention.

**Survey**: [From S4 to Mamba: A Comprehensive Survey](https://arxiv.org/abs/2503.18970)

### 2.4 RWKV Architecture

**Paper**: Peng, B. et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era."
- [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)
- [GitHub](https://github.com/BlinkDL/RWKV-LM)
- [RWKV Wiki](https://wiki.rwkv.com/)

Key features:
- Combines Transformer parallelizable training with RNN efficient inference
- Linear attention mechanism with WKV (Weighted Key Value)
- Token shift via linear interpolation
- O(n) complexity vs O(n²) for Transformers

Evolution:
- **RWKV-4**: Original (EMNLP 2023)
- **RWKV-5/6 (Eagle/Finch)**: Multi-headed matrix-valued states
- **RWKV-7 (Goose)**: Dynamic State Evolution, Generalized Delta Rule

Scale: Up to 14B parameters - largest dense RNN ever trained.

### 2.5 minGRU/minLSTM (2024)

**Paper**: Feng, L. et al. (2024). "Were RNNs All We Needed?"
- [arXiv:2410.01201](https://arxiv.org/abs/2410.01201)
- [Hugging Face](https://huggingface.co/papers/2410.01201)
- [RBC Borealis Blog](https://rbcborealis.com/research-blogs/minimal-lstms-and-grus-simple-efficient-and-fully-parallelizable/)

Key insight: Remove hidden state dependencies from gates → fully parallelizable

Simplifications:
- **minGRU**: Remove reset gate, remove h dependency from update gate
- **minLSTM**: Remove h dependency from forget/input gates

Results:
- 175-1324× speedup over traditional GRU
- 85-87% parameter reduction
- Comparable to Mamba on Selective Copying task
- Similar test loss to Transformers on language modeling

### 2.6 Griffin and Hawk (Google DeepMind, 2024)

**Paper**: De, S. et al. (2024). "Griffin: Mixing Gated Linear Recurrences with Local Attention."
- [arXiv:2402.19427](https://arxiv.org/abs/2402.19427)

- **Hawk**: RNN with gated linear recurrences
- **Griffin**: Hybrid mixing gated recurrences with local attention

Results:
- Hawk exceeds Mamba on downstream tasks
- Griffin matches Llama-2 with 6× fewer training tokens
- Constant-size state with long-range extrapolation

### 2.7 Linear Attention and RNN Equivalence

**Foundational Paper**: Katharopoulos, A. et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention."
- [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)

Key insight: Linear attention allows O(N) complexity and iterative (RNN) implementation.

**Unified Framework**: [Explaining Modern Gated-Linear RNNs via Unified Implicit Attention](https://arxiv.org/html/2405.16504v2)
- Unifies Mamba, RWKV, Griffin, RetNet under common framework
- Shows implicit attention matrices of these models resemble Transformers

---

## 3. Gating Mechanisms

### 3.1 LSTM

**Paper**: Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory."
- [Neural Computation](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)

Three gates:
- **Forget gate (f)**: Controls information retention
- **Input gate (i)**: Controls new information addition
- **Output gate (o)**: Controls output from cell state

### 3.2 GRU

**Paper**: Cho, K. et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder."
- [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)

Simplified to two gates:
- **Update gate (z)**: How much of previous state to retain
- **Reset gate (r)**: How much past information to forget

Key difference: Merges cell state and hidden state.

**Resources**:
- [Wikipedia: Gated Recurrent Unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit)
- [Dive into Deep Learning: GRU](https://d2l.ai/chapter_recurrent-modern/gru.html)
- [Simplified Gating Paper](https://arxiv.org/pdf/1701.03441)

### 3.3 Delta/Interpolation Gates

The gating pattern used in this project:
```
h_new = (1 - δ) * h_prev + δ * candidate
```

This is equivalent to:
- Exponential moving average when δ is constant
- Selective update when δ is input-dependent
- Classical RNN gating as instance of SSM selection mechanism

**Connection to SSMs**: In Mamba, Δ (discretization step) plays the role of a generalized gate.

### 3.4 Minimal Gated Units (MGU)

Research on reducing gates:
- **MGU**: Single gate variant matching GRU performance
- **minGRU**: Parallelizable via removing state dependencies
- **minLSTM**: Same approach for LSTM

---

## 4. Log-Space Computation and Numerical Stability

### 4.1 The Log-Sum-Exp Trick

**Resources**:
- [Gregory Gundersen: The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- [Wikipedia: LogSumExp](https://en.wikipedia.org/wiki/LogSumExp)
- [Oxford Academic: Accurately computing log-sum-exp and softmax](https://academic.oup.com/imajna/article/41/4/2311/5893596)
- [Lei Mao: LogSumExp and Numerical Stability](https://leimao.github.io/blog/LogSumExp/)

**The Problem**: Direct computation of log(Σexp(xᵢ)) causes overflow/underflow.

**The Solution**:
```
log(Σexp(xᵢ)) = c + log(Σexp(xᵢ - c))
```
where c = max(xᵢ). This ensures exp(xᵢ - c) ∈ [0, 1].

**Key Property**: Gradient of LogSumExp is the softmax function.

### 4.2 Signed Log-Space Representation

For handling negative values in log-space:
- Store (log|x|, sign(x)) tuples
- Enables log-space computation with mixed signs
- Critical for RNN hidden states which can be negative

**Application in this project**:
```python
log_h, sign_h = to_log_space(h)  # h can be positive or negative
h_linear = from_log_space(log_h, sign_h)  # recover original value
```

### 4.3 Log-Space for RNNs

Benefits:
1. **Numerical stability**: Prevents overflow in long sequences
2. **Gradient stability**: More stable gradients through many time steps
3. **Hidden Markov Models**: Log-space is standard for HMM forward/backward algorithms

**HMM Application**: [Understanding Log-Sum-Exp in HMMs](https://chengyuan-zhang.github.io/posts/logsumexp/)

---

## 5. Spectral Normalization and RNN Stability

### 5.1 Spectral Normalization

**Original Paper**: Miyato, T. et al. (2018). "Spectral Normalization for Generative Adversarial Networks."
- [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)

Divides weight matrix by its spectral norm (largest singular value):
```
W_SN = W / σ(W)
```

**Application to RNNs**: Constrains spectral radius of recurrence matrix to prevent exploding gradients.

**Resources**:
- [Atom-101: Spectral Normalization Explained](https://atom-101.github.io/blog/posts/2022-03-18-spectral-norm.html)
- [Daniel Rapp: Spectral Radius in RNNs](http://danielrapp.github.io/rnn-spectral-radius/)

### 5.2 Eigenvalue Normalized RNNs (ENRNN)

**Paper**: Kerg, G. et al. (2019). "Eigenvalue Normalized Recurrent Neural Networks."
- [AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5831)
- [arXiv:1911.07964](https://arxiv.org/pdf/1911.07964)

Key innovation: Construct recurrence matrix with spectral radius < 1 via normalization.

### 5.3 Orthogonal/Unitary RNNs

**Concept**: Orthogonal matrices have all eigenvalues on unit circle → no vanishing/exploding.

**Trade-off**: Orthogonal constraints limit expressivity.

**This project's approach**: Three modes for R_h constraint:
1. `'free'`: No constraint (can be unstable)
2. `'spectral_norm'`: Power iteration to constrain spectral radius to 0.99
3. `'scaled_orthogonal'`: sigmoid(scale) × orthogonal_base (stable by construction)

### 5.4 Power Iteration for Spectral Norm

**Algorithm**:
```python
for _ in range(n_iterations):
    v = R_h.T @ u
    v = v / v.norm()
    u = R_h @ v
    u = u / u.norm()
sigma = (u @ R_h @ v).abs()  # spectral norm estimate
```

**Resources**:
- [Wikipedia: Power Iteration](https://en.wikipedia.org/wiki/Power_iteration)
- [Cornell: Power Iteration Lecture](https://www.cs.cornell.edu/~bindel/class/cs6210-f16/lec/2016-10-17.pdf)
- [Python Numerical Methods](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter15.02-The-Power-Method.html)

### 5.5 Non-Normal RNNs

**Paper**: Kerg, G. et al. (2019). "Non-normal Recurrent Neural Network (nnRNN)."
- [arXiv:1905.12080](https://arxiv.org/abs/1905.12080)

Key insight: Non-normal matrices can provide transient amplification (short-term memory boost) while maintaining long-term stability with spectral radius < 1.

---

## 6. CUDA Kernel Optimization

### 6.1 Haste Library (LMNT)

**Repository**: [github.com/lmnt-com/haste](https://github.com/lmnt-com/haste)

Haste provides CUDA implementations of fused RNN layers with:
- Built-in DropConnect and Zoneout regularization
- GRU, IndRNN, LSTM, LayerNormGRU, LayerNormLSTM
- C++ and Python APIs

**Key benefit**: Fused kernels avoid memory roundtrips for intermediate values.

### 6.2 Kernel Fusion Benefits

**Resources**:
- [Kevin Zhang: Building Fused CUDA Kernels for RNNs](https://medium.com/@wancongzhang/building-fused-cuda-kernels-for-rnns-d0419ce54261)
- [NVIDIA: Optimizing RNNs in cuDNN 5](https://developer.nvidia.com/blog/optimizing-recurrent-neural-networks-cudnn-5/)
- [PyTorch: Optimizing CUDA RNN with TorchScript](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/)

Benefits:
- Combine GEMM operations
- Fuse pointwise operations
- Pre-transpose weight matrices
- Stream GEMMs
- Up to 10× speedup

### 6.3 cuBLAS GEMM Optimization

**Resource**: [Simon Boehm: CUDA Matmul Optimization](https://siboehm.com/articles/22/CUDA-MMM)

This project replaces custom LogSpaceMatVecKernel with cuBLAS GEMM for:
- Better hardware utilization
- Optimized memory access patterns
- Tensor core acceleration on modern GPUs

### 6.4 Workspace Allocation Patterns

From recent commits, the project uses PyTorch's caching allocator for workspace:
```cpp
auto workspace = at::empty({workspace_size}, options);
```

Benefits:
- Avoids repeated allocation/deallocation
- Leverages PyTorch's memory pool
- Reduces fragmentation

### 6.5 FlashAttention Design Principles

**Paper**: Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."
- [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Blog](https://crfm.stanford.edu/2023/07/17/flash2.html)

Key principles applicable to RNN kernels:
- **IO-awareness**: Optimize for memory bandwidth, not just FLOPs
- **Tiling**: Load blocks from HBM to SRAM, compute, write back
- **Recomputation**: Trade compute for memory access
- GPU SRAM is 10× faster than HBM but much smaller

### 6.6 Parallel Scan for Linear Recurrence

**Paper**: Martin, E. & Cundy, C. (2017). "Parallelizing Linear Recurrent Neural Nets Over Sequence Length."
- [arXiv:1709.04057](https://arxiv.org/abs/1709.04057)
- [GitHub](https://github.com/eamartin/parallelizing_linear_rnns)

Key insight: Linear recurrences can be parallelized via prefix sum (scan) algorithm.
- O(T/p + log(p)) complexity
- Up to 9× speedup
- Backpropagation also parallelizable

**ParaRNN (2025)**: Extends to nonlinear RNNs with 665× speedup, 7B parameter training.
- [arXiv:2510.21450](https://arxiv.org/pdf/2510.21450)

---

## 7. Training Techniques

### 7.1 Truncated Backpropagation Through Time (TBPTT)

**Resources**:
- [PyTorch Lightning: TBPTT](https://lightning.ai/docs/pytorch/stable/common/tbptt.html)
- [Dive into Deep Learning: BPTT](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html)
- [MachineLearningMastery: BPTT Introduction](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)

**The Problem**: Full BPTT is memory-intensive for long sequences.

**TBPTT Solution**:
- Process sequence in chunks of size k
- Backpropagate only within each chunk
- Pass hidden state across chunks (detached from graph)

**Trade-off**: Loses long-term dependencies spanning more than k steps.

**This project's implementation**:
```python
if (args.tbptt or use_subword) and next_hidden is not None:
    hidden_state = [detach_hidden(h) for h in next_hidden]
```

### 7.2 Schedule-Free Optimization

**Paper**: Defazio, A. et al. (2024). "The Road Less Scheduled."
- [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)
- [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html)
- [GitHub](https://github.com/facebookresearch/schedule_free)

Key innovation: Eliminates need for learning rate schedules via momentum-based method.

Results:
- 98.4% on CIFAR-10 (beats cosine schedule by 0.2%)
- Top position in MLCommons AlgoPerf Challenge
- Matches or beats SOTA schedules

Usage notes:
- Learning rates 1-10× larger than scheduled approaches
- No warmup required (though project still uses warmup)

### 7.3 Mixed Precision Training (BFloat16)

**Resources**:
- [PyTorch: Mixed Precision Training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- [Lightning: Save Memory with Mixed Precision](https://lightning.ai/docs/fabric/stable/fundamentals/precision.html)
- [RunPod: FP16/BF16/FP8 Guide](https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training)

**BFloat16 Advantages**:
- 8-bit exponent (same range as FP32: ~3.4e38)
- 7-bit mantissa (lower precision than FP16)
- No loss scaling needed (unlike FP16)
- Native support on A100/H100

**This project**: Uses BF16 by default:
```python
dtype = torch.bfloat16 if args.bf16 else torch.float32
```

### 7.4 Gradient Accumulation

**Resources**:
- [Aman.ai: Gradient Accumulation and Checkpointing](https://aman.ai/primers/ai/grad-accum-checkpoint/)
- [DailyDoseOfDS: Gradient Accumulation](https://blog.dailydoseofds.com/p/gradient-accumulation-increase-batch)
- [Hugging Face Forum: Batch Size vs Gradient Accumulation](https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260)

**Concept**: Accumulate gradients over N mini-batches before updating.

**Effective batch size** = batch_size × world_size × grad_accum

**This project**:
```python
loss_scaled = loss / args.grad_accum
loss_scaled.backward()
if (step + 1) % args.grad_accum == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 7.5 Gradient Clipping

Standard technique to prevent exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

### 7.6 Distributed Data Parallel (DDP)

**Resources**:
- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/notes/ddp.html)
- [HuggingFace: Training on Larger Batches](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)

This project supports multi-GPU training via torchrun:
```bash
torchrun --nproc_per_node=8 train_ladder.py --ddp
```

---

## 8. Architecture Design Patterns

### 8.1 Pre-Norm vs Post-Norm

**Resources**:
- [Sebastian Raschka: LayerNorm Variants](https://magazine.sebastianraschka.com/p/why-the-original-transformer-figure)
- [EmergentMind: Pre-Norm Residual Connections](https://www.emergentmind.com/topics/pre-norm-residual-connections-prenorm)
- [Peri-LN Paper](https://arxiv.org/html/2502.02732v1)

**Post-LN**: Normalize after residual addition
- Original Transformer paper figure (but not code!)
- Can degrade gradient flow in deep networks

**Pre-LN**: Normalize before sublayer
- Default in modern LLMs (GPT, LLaMA, ViT)
- Smoother training, stable gradients
- Warmup can be reduced or removed

**This project uses Pre-Norm (Mamba-style)**:
```python
residual = x
x_norm = ln(x)
x_out, h_final = layer(x_norm, prev_hiddens[i])
x = residual + x_out
```

### 8.2 LayerNorm vs RMSNorm

**Resources**:
- [Sushant Kumar: Normalization in LLMs](https://sushant-kumar.com/blog/normalization-in-transformer-based-llms)
- [NeurIPS 2023: Pre-RMSNorm Transformers](https://proceedings.neurips.cc/paper_files/paper/2023/file/8f1bacee31caf990a4f08d84f0ccb322-Paper-Conference.pdf)

**RMSNorm**: Removes mean-centering step from LayerNorm
- 1-10% speedup
- For Pre-Norm, arithmetically equivalent to LayerNorm (assuming zero-mean)
- Used in Gopher, LLaMA, and many modern LLMs

### 8.3 Residual Connections for Deep Networks

Critical for training deep recurrent networks:
```python
x = residual + x_out  # KEY for deep networks!
```

### 8.4 Weight Tying

**Paper**: Press, O. & Wolf, L. (2016). "Using the Output Embedding to Improve Language Models."
- [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)
- [PapersWithCode: Weight Tying](https://paperswithcode.com/method/weight-tying)

Shares parameters between input embedding and output projection:
```python
self.lm_head.weight = self.embedding.weight
```

Benefits:
- Reduces parameters significantly
- Improves perplexity
- Standard in modern LLMs (NanoGPT, etc.)

### 8.5 SiLU/Swish and Gated Activations

**Resources**:
- [PyTorch: SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
- [Ultralytics: SiLU Explained](https://www.ultralytics.com/glossary/silu-sigmoid-linear-unit)
- [SwiGLU Explanation](https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it)

**SiLU (Swish)**: f(x) = x · σ(x)
- Non-monotonic (can capture negative information)
- Self-gating mechanism
- Standard in modern architectures

**This project uses compete × silu for selective output**:
```python
compete = F.softmax(h_reshaped, dim=-1).view(B, D)
silu_out = F.silu(w_out_h)
output = compete * silu_out
```

### 8.6 Grouped Softmax (Competitive Learning)

**Resources**:
- [Wikipedia: Mixture of Experts](https://en.wikipedia.org/wiki/Mixture_of_experts)
- [HuggingFace: Mixture of Experts Explained](https://huggingface.co/blog/moe)
- [Google: Expert Choice Routing](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/)

The project uses grouped softmax for selective attention within hidden state:
```python
h_reshaped = h.view(B, n_groups, group_size)
compete = F.softmax(h_reshaped, dim=-1)
```

This implements local competition within groups, similar to mixture-of-experts sparse gating.

---

## 9. Benchmarks and Evaluation

### 9.1 Long Range Arena (LRA)

**Paper**: Tay, Y. et al. (2020). "Long Range Arena: A Benchmark for Efficient Transformers."
- [arXiv:2011.04006](https://arxiv.org/abs/2011.04006)
- [OpenReview](https://openreview.net/forum?id=qVyeW-grC2k)

Tasks with 1K-16K sequences:
- ListOps, Text, Retrieval, Image, Pathfinder, Path-X

**Results**: SSMs (S4, Mamba) significantly outperform Transformers on LRA.

**Caveat**: Recent work shows LRA may not fully capture real-world long-range modeling (locality bias). [arXiv:2501.14850](https://arxiv.org/html/2501.14850v1)

### 9.2 TinyStories Dataset

**Paper**: Eldan, R. & Li, Y. (2023). "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
- [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)

Synthetic dataset of children's stories (3-4 year old vocabulary):
- Train models <10M parameters
- Single-day training on single GPU
- GPT-4 based evaluation framework

**Novel contribution**: Multi-dimensional evaluation (grammar, creativity, consistency).

### 9.3 FineWeb Dataset

**Paper**: Penedo, G. et al. (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale."
- [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- [HuggingFace Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

15 trillion tokens from 96 CommonCrawl dumps (2013-2024):
- Largest public clean LLM pretraining dataset
- FineWeb-Edu: 1.3T educational tokens
- FineWeb-2: 1000+ languages

### 9.4 Perplexity Evaluation

Standard language modeling metric:
```python
ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
```

---

## 10. Implementation Details

### 10.1 Custom PyTorch Autograd Functions

**Resources**:
- [PyTorch: Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html)
- [PyTorch Tutorial: Custom Autograd Functions](https://docs.pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html)

Pattern used in this project:
```python
class LogComputeFullFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, training, x, log_h0, ...):
        # Call CUDA kernel
        results = hasty_pytorch_lib.log_compute_full_forward(...)
        if training:
            ctx.save_for_backward(x, W_x, R_h, ...)
        return output, log_h, sign_h, None

    @staticmethod
    def backward(ctx, d_output, ...):
        # Call CUDA backward kernel
        return hasty_pytorch_lib.log_compute_full_backward(...)
```

### 10.2 Tokenization

**Tiktoken (OpenAI)**:
- [GitHub](https://github.com/openai/tiktoken)
- Fast Rust-backed BPE implementation
- p50k_base: ~50,257 tokens (GPT-2 compatible)

**Byte-level BPE**:
- 256 base tokens (all bytes)
- Lossless (no unknown tokens)
- Standard for modern LLMs

### 10.3 Data Streaming

The project uses streaming datasets for efficient training:
- `DocumentStreamDataset`: Single-stream, no hidden state persistence
- `BatchedStreamDataset`: Per-batch persistent streams for TBPTT
- `TokenizedStreamDataset`: Streaming subword tokenization

---

## Key Citations Summary

### Foundational Papers
1. Elman, J.L. (1990). "Finding Structure in Time." *Cognitive Science*.
2. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory."
3. Cho, K. et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder."

### State Space Models
4. Gu, A. et al. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces." (S4)
5. Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
6. Gu, A. et al. (2022). "On the Parameterization and Initialization of Diagonal State Space Models." (S4D)

### Modern RNN Variants
7. Peng, B. et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era."
8. Feng, L. et al. (2024). "Were RNNs All We Needed?" (minGRU/minLSTM)
9. De, S. et al. (2024). "Griffin: Mixing Gated Linear Recurrences with Local Attention."

### Optimization & Training
10. Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention."
11. Defazio, A. et al. (2024). "The Road Less Scheduled." (Schedule-Free)
12. Martin, E. & Cundy, C. (2017). "Parallelizing Linear Recurrent Neural Nets Over Sequence Length."

### Stability & Normalization
13. Miyato, T. et al. (2018). "Spectral Normalization for GANs."
14. Kerg, G. et al. (2019). "Eigenvalue Normalized Recurrent Neural Networks."

### Datasets & Benchmarks
15. Tay, Y. et al. (2020). "Long Range Arena: A Benchmark for Efficient Transformers."
16. Eldan, R. & Li, Y. (2023). "TinyStories: How Small Can Language Models Be?"
17. Penedo, G. et al. (2024). "The FineWeb Datasets."

---

## 11. Advanced State Space Model Theory

### 11.1 Mamba-2 and Structured State Space Duality (SSD)

**Paper**: Dao, T. & Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality."
- [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- [Tri Dao's Blog Part 1](https://tridao.me/blog/2024/mamba2-part1-model/)
- [Tri Dao's Blog Part 3](https://tridao.me/blog/2024/mamba2-part3-algorithm/)

Key innovations:
- Reveals theoretical equivalence between SSMs and variants of attention via semiseparable matrices
- SSD constrains diagonal A to scalar × identity for efficiency
- Multi-channel processing (e.g., 64 channels) with common recurrence
- 2-8× faster than Mamba-1
- Utilizes tensor cores via matmul-based computation

### 11.2 HiPPO (High-order Polynomial Projection Operators)

**Paper**: Gu, A. et al. (2020). "HiPPO: Recurrent Memory with Optimal Polynomial Projections."
- [arXiv:2008.07669](https://arxiv.org/abs/2008.07669)
- [Hazy Research Blog](https://hazyresearch.stanford.edu/blog/2020-12-05-hippo)
- [NeurIPS 2020 Paper](https://proceedings.neurips.cc/paper/2020/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf)

HiPPO provides the theoretical foundation for S4's long-range capabilities:
- Optimal online function approximation via polynomial projection
- **HiPPO-LegS**: Scaled Legendre measure, timescale-robust
- O(N) updates via special matrix structure
- Avoids vanishing/exploding gradients analytically

The HiPPO matrix initialization is empirically critical for S4's performance on long-range tasks.

### 11.3 Hyena and H3 Architectures

**Papers**:
- Poli, M. et al. (2023). "Hyena Hierarchy: Towards Larger Convolutional Language Models." ICML 2023.
  - [arXiv:2302.10866](https://arxiv.org/abs/2302.10866)
  - [Hazy Research Blog](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)
- Dao, T. et al. (2022). "Hungry Hungry Hippos (H3): Towards Language Modeling with State Space Models." ICLR 2023.

Hyena features:
- Subquadratic attention replacement via implicit long convolutions + gating
- 2× faster than attention at 8K, 100× faster at 64K sequence length
- 20% training compute reduction at 2K sequence length
- Matches attention on ImageNet (Vision Transformer replacement)

---

## 12. Continuous-Time and Hybrid Architectures

### 12.1 Neural ODEs and Discretization

**Paper**: Chen, R.T.Q. et al. (2018). "Neural Ordinary Differential Equations." NeurIPS.
- [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- [Awesome Neural ODE Collection](https://github.com/Zymrael/awesome-neural-ode)
- [Tutorial: Neural ODEs Chapter](http://implicit-layers-tutorial.org/neural_odes/)

Key insight: ResNet updates are Euler discretizations of continuous dynamics.

**Discretization methods**:
- **Euler**: Simplest, used in ResNets
- **ZOH (Zero-Order Hold)**: Exact for staircase inputs, used in S4/Mamba
- **Bilinear (Tustin)**: Best frequency-domain match
- **Runge-Kutta**: Higher accuracy ODE solvers

### 12.2 Liquid Neural Networks and CfC

**Paper**: Hasani, R. et al. (2022). "Closed-form continuous-time neural networks." *Nature Machine Intelligence*.
- [Nature](https://www.nature.com/articles/s42256-022-00556-7)
- [arXiv:2106.13898](https://arxiv.org/abs/2106.13898)
- [GitHub](https://github.com/raminmh/CfC)
- [MIT News](https://news.mit.edu/2022/solving-brain-dynamics-gives-rise-flexible-machine-learning-models-1115)

CfC (Closed-form Continuous-time) models:
- Solve the 1907 neuron differential equation in closed form
- 1-5 orders of magnitude faster than ODE-based continuous networks
- Explicit time dependence enables irregular time step handling
- 19 CfC neurons can drive a car (autonomous driving demo)

### 12.3 MEGA (Moving Average Equipped Gated Attention)

**Paper**: Ma, X. et al. (2022). "Mega: Moving Average Equipped Gated Attention."
- [arXiv:2209.10655](https://arxiv.org/abs/2209.10655)
- [OpenReview/ICLR](https://openreview.net/forum?id=qNLe3iq2El)
- [HuggingFace](https://huggingface.co/docs/transformers/main/model_doc/mega)

MEGA combines:
- Multi-head exponential moving average (EMA) for positional bias
- Single-head gated attention
- GRU/GAU backbone architecture

Results: Outperforms S4, Transformers, and efficient variants on all LRA tasks.

### 12.4 Jamba: Hybrid Mamba-Transformer

**Paper**: AI21 Labs (2024). "Jamba: A Hybrid Transformer-Mamba Language Model."
- [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
- [AI21 Blog](https://www.ai21.com/blog/announcing-jamba/)
- [HuggingFace](https://huggingface.co/ai21labs/Jamba-v0.1)

Architecture:
- Interleaves Transformer and Mamba layers (1:7 ratio)
- Mixture-of-Experts (MoE) every two blocks
- 52B total / 12B active parameters
- 256K context length
- Fits 140K tokens on single 80GB GPU

Jamba 1.5 scales to 398B total / 94B active parameters.

---

## 13. Diagonal and Alternative Recurrence Structures

### 13.1 IndRNN (Independently Recurrent Neural Networks)

**Paper**: Li, S. et al. (2018). "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN." CVPR.
- [arXiv:1803.04831](https://arxiv.org/abs/1803.04831)
- [GitHub PyTorch](https://github.com/Sunnydreamrain/IndRNN_pytorch)

IndRNN uses element-wise (diagonal) recurrence:
```
h_t = act(W * x_t + u ⊙ h_{t-1} + b)
```
where ⊙ is Hadamard product (each neuron has single scalar recurrent weight).

Benefits:
- Can use ReLU activation (non-saturating)
- Processes sequences >5000 steps
- 21+ layers successfully trained
- 10× faster than LSTM
- Cross-neuron connections via stacking layers

### 13.2 Quasi-Recurrent Neural Networks (QRNN)

**Paper**: Bradbury, J. et al. (2016). "Quasi-Recurrent Neural Networks."
- [arXiv:1611.01576](https://arxiv.org/abs/1611.01576)
- [GitHub](https://github.com/salesforce/pytorch-qrnn)

QRNNs alternate:
1. Parallel convolutions across timesteps
2. Minimal recurrent pooling across channels

Results: Up to 16× faster than cuDNN LSTM, better accuracy than stacked LSTMs.

### 13.3 Temporal Convolutional Networks (TCN)

**Paper**: Bai, S., Kolter, J.Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling."
- [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)

TCN = 1D FCN + causal convolutions + dilated convolutions:
- Exponentially large receptive field via dilation
- Parallelizable training
- Competitive with RNNs on many sequence tasks

---

## 14. Memory and Associative Networks

### 14.1 Modern Hopfield Networks

**Paper**: Ramsauer, H. et al. (2020). "Hopfield Networks is All You Need."
- [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)

**NeurIPS 2024**: "Provably Optimal Memory Capacity for Modern Hopfield Models"
- [arXiv:2410.23126](https://arxiv.org/abs/2410.23126)

Key advances:
- Exponential memory capacity (vs. linear in classical Hopfield)
- Connection to Transformer attention mechanisms
- Kernelized Hopfield Models (KHMs) as dense associative memories
- Optimal capacity via spherical codes theory

### 14.2 Memory Capacity Comparisons

Modern recurrent architectures can be unified as associative memory modules:
- RNNs: Fixed-size state, requires memory management
- Transformers: Growing KV-cache
- SSMs: Fixed state, content-based selection

---

## 15. Scaling Laws and Compute-Optimal Training

### 15.1 Chinchilla Scaling Laws

**Paper**: Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models."
- [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- [LifeArchitect Explanation](https://lifearchitect.ai/chinchilla/)

Key findings:
- Optimal: ~20 tokens per parameter
- GPT-3 was undertrained (1.7 tokens/parameter)
- Chinchilla (70B, 4× more data) outperforms Gopher (280B)

**Beyond Chinchilla**:
- LLaMA 3: ~200 tokens/parameter (10× Chinchilla)
- Phi-3: ~870 tokens/parameter (45× Chinchilla)
- Motivation: Smaller models cheaper to serve

### 15.2 Ablation Study Methodology

Best practices for architecture comparison:
- Control compute budget, not just parameter count
- Progressive complexity (ladder approach)
- Statistical significance across seeds
- Scaling behavior analysis

---

## 16. Positional Encoding and Length Generalization

### 16.1 Length Extrapolation Challenge

Transformers struggle to generalize to longer sequences than training.
- [Survey: Length Extrapolation](https://arxiv.org/html/2312.17044v5)
- [Impact of Positional Encoding](https://arxiv.org/pdf/2305.19466)

### 16.2 ALiBi (Attention with Linear Biases)

**Paper**: Press, O. et al. (2021). "Train Short, Test Long: Attention with Linear Biases."
- [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- [SambaNova Analysis](https://sambanova.ai/blog/alibi-interpolation-vs-extrapolation)

ALiBi adds position-proportional bias to attention scores:
- No positional embeddings
- Trains 11% faster, 11% less memory
- Extrapolates from 1K to 2K+ sequences
- No learnable parameters

### 16.3 RoPE (Rotary Position Embedding)

RoPE rotates keys/queries based on position:
- De facto standard in LLaMA, GPT-NeoX
- Good interpolation, poor extrapolation
- Complex distance-attention functions via Fourier basis

### 16.4 RNNs and Natural Length Handling

RNNs/SSMs inherently handle variable lengths:
- No positional encoding needed
- State-based memory naturally sequential
- Constant memory regardless of length

---

## 17. Activation Functions and FFN Design

### 17.1 Tanh in RNNs

Why tanh persists:
- Bounded output [-1, 1] prevents explosion
- Zero-centered (unlike sigmoid)
- Standard in LSTM cell states

Challenges:
- Saturation causes vanishing gradients
- Contractive mapping attenuates information

### 17.2 Alternatives to Tanh

- **Softsign**: f(x) = x/(1+|x|), larger unsaturation region
- **TanhLU**: Combines tanh + linear unit
- **Non-Saturating Recurrent Unit (NRU)**: No saturating gates
- **ReLU in RNNs**: Works with IndRNN diagonal structure

Resources:
- [Enhancement with TanhLU](https://www.sciencedirect.com/science/article/abs/pii/S0957417422005681)
- [Non-saturating RNNs](https://arxiv.org/pdf/1902.06704)

### 17.3 FFN Expansion Ratios

**Standard Transformer**: 4× expansion (d_ff = 4 × d_model)

**SwiGLU adjustment**: 8/3 ≈ 2.67× to maintain parameter parity:
- Three weight matrices (W, V, W₂) vs two
- LLaMA uses ~2.7× with SwiGLU
- FFN accounts for 2/3 of GPT-3 parameters

Resources:
- [FFN Architecture Guide](https://mbrenndoerfer.com/writing/transformer-feed-forward-networks)
- [SwiGLU Performance](https://dev.to/mshojaei77/swiglu-the-ffn-upgrade-i-use-to-get-free-performance-33jc)

### 17.4 Sparsemax and Entmax

Sparse alternatives to softmax:
- [GitHub: entmax](https://github.com/deep-spin/entmax)

α parameter controls sparsity:
- α=1: Softmax (dense)
- α=1.5: Entmax15 (moderate sparsity)
- α=2: Sparsemax (sparse)

Benefits: Zero out irrelevant dimensions while maintaining differentiability.

---

## 18. Deep Recurrent Architectures

### 18.1 Stacked RNNs

- [Dive into Deep Learning: Deep RNNs](https://d2l.ai/chapter_recurrent-modern/deep-rnn.html)

Standard approach: Stack RNN layers, output of layer n → input of layer n+1.

Challenge: Credit assignment across both space and time.

### 18.2 Recurrent Highway Networks

**Paper**: Zilly, J.G. et al. (2016). "Recurrent Highway Networks."
- [arXiv:1607.03474](https://arxiv.org/abs/1607.03474)

RHNs incorporate Highway layers inside recurrent transitions:
- Superior to simple stacking
- Long credit assignment paths in both time and space
- Gates for dynamic remembering/forgetting/transforming

### 18.3 Hierarchical Recurrent Highway Networks

- [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167865518302708)

Highways within both hierarchical and temporal structure for unimpeded gradient flow.

---

## 19. Document-Level Training

### 19.1 In-Context Pretraining

**Paper**: Shi, W. et al. (2023). "In-Context Pretraining: Language Modeling Beyond Document Boundaries."
- [arXiv:2310.10638](https://arxiv.org/abs/2310.10638)

Key innovation: Sort documents by contextual similarity during pretraining.
- 8% average improvement on in-context learning
- 15% average gain on complex reasoning tasks

### 19.2 Cross-Document Language Modeling (CDLM)

**Paper**: Caciularu, A. et al. (2021). "CDLM: Cross-Document Language Modeling."
- [arXiv:2101.00406](https://arxiv.org/abs/2101.00406)
- [ACL Anthology](https://aclanthology.org/2021.findings-emnlp.225/)

Based on Longformer with dynamic global attention across related documents.

---

## 20. Tropical Algebra and Log-Space Semirings

### 20.1 Tropical Geometry in Deep Learning

- [Tropical Geometry of DNNs](https://www.stat.uchicago.edu/~lekheng/work/tropical.pdf)
- [IEEE: Tropical Geometry and ML](https://ieeexplore.ieee.org/document/9394420/)

Tropical semiring: (ℝ ∪ {-∞}, max, +)
- ReLU networks ↔ tropical rational functions
- Piecewise-linear geometry preservation

### 20.2 Semiring Neural Networks

**Paper**: Smets, B. et al. (2024). "Semiring Activation in Neural Networks."
- [arXiv:2405.18805](https://bmnsmets.com/files/smets2024semiring.pdf)

Log-semiring and tropical operations in neural architectures:
- Require special initialization (not Xavier/Kaiming)
- "Fair" backpropagation for tropical operators
- Applications in combinatorial optimization

---

## 21. Learnable and Polynomial Activations

### 21.1 Adaptive Activation Functions

**Survey**: "Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark"
- [arXiv:2109.14545](https://arxiv.org/pdf/2109.14545)

Key approaches:
- **SLAF (Self-Learnable AF)**: Linear regression trains activation per neuron
- **LEAF**: Combines squashing and rectifier properties
- **Locally Adaptive**: Layer-wise and neuron-wise adaptation
- **Kolmogorov-Arnold Networks (KANs)**: Learnable basis functions + trainable scaling

### 21.2 Learning Polynomial Activations

**Paper**: "Learning Polynomial Activation Functions for Deep Neural Networks"
- [arXiv:2510.03682](https://arxiv.org/html/2510.03682)

Formulates training as polynomial optimization, solvable by Moment-SOS relaxations.

### 21.3 Physics-Informed Neural Networks (PINNs)

- [Adaptive Activations in PINNs](https://www.sciencedirect.com/science/article/abs/pii/S0010465525002553)
- [Royal Society Paper](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0334)

Scalable hyper-parameters in activation dynamically change loss topology.

---

## 22. Second-Order and Multiplicative RNNs

### 22.1 Second-Order RNNs (2RNNs)

**Paper**: "A Tensor Decomposition Perspective on Second-order RNNs" (ICML 2024)
- [arXiv:2406.05045](https://arxiv.org/abs/2406.05045)

Second-order RNNs use x_t ⊗ h_{t-1} tensor products:
- More expressive than first-order RNNs
- Connections to formal language theory
- CP decomposition (CPRNN) reduces parameter count

### 22.2 Multiplicative Interactions

**Paper**: "Theory of Gating in Recurrent Neural Networks" (Phys. Rev. X, 2022)
- [Physical Review X](https://link.aps.org/doi/10.1103/PhysRevX.12.011011)

Gating is multiplicative interaction controlling information flow:
- Present in biological neurons
- Enables learning significantly more complex tasks
- Key to LSTM/GRU success

### 22.3 Tensor Decomposition for RNNs

- **MIRNN**: Limits interaction types
- **Tucker decomposition**: Compresses bilinear interactions
- **Tensor Train (TT)**: High compression ratios
- **LSTMRNTN/GRURNTN**: LSTM/GRU with tensor products

---

## 23. RetNet and Retention Mechanisms

### 23.1 Retentive Network (RetNet)

**Paper**: Sun, Y. et al. (2023). "Retentive Network: A Successor to Transformer for Large Language Models."
- [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
- [HuggingFace Paper](https://huggingface.co/papers/2307.08621)
- [Survey of RetNet](https://arxiv.org/html/2506.06708)

Key innovation: Multi-scale retention with exponential decay:
```
Retention(X) = (QK^T ⊙ D) V
```
where D is a causal decay matrix.

### 23.2 Three Computation Paradigms

1. **Parallel**: Training parallelism (like Transformers)
2. **Recurrent**: O(1) inference (like RNNs)
3. **Chunkwise**: Linear complexity for long sequences

### 23.3 Multi-Scale Decay

Different decay rates per head → multi-scale modeling:
- Head 1: Fast decay (local patterns)
- Head N: Slow decay (long-range dependencies)

### 23.4 Performance

For 7B model, 8K sequence:
- 8.4× faster decoding
- 70% memory savings
- 25-50% training memory reduction
- 7× training acceleration

---

## 24. GPU Kernel Development

### 24.1 Triton Language

**OpenAI Triton**: Python-like GPU programming language
- [OpenAI Blog](https://openai.com/index/triton/)
- [PyTorch XLA Docs](https://docs.pytorch.org/xla/master/features/triton.html)
- [Red Hat Guide](https://next.redhat.com/2024/11/07/democratizing-ai-accelerators-and-gpu-kernel-programming-using-triton/)

Key benefits:
- Python-like syntax, near-CUDA performance
- Abstracts thread block complexity (coalescing, shared memory, tensor cores)
- Powers torch.compile / torch.inductor

### 24.2 Kernel Fusion

- [Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Custom Kernels Guide](https://alexdremov.me/speed-up-pytorch-with-custom-kernels-but-it-gets-progressively-darker/)

Fusing operations into single kernel:
- Naive softmax: 5MN + 2M reads, 3MN + 2M writes
- Fused: MN reads, MN writes
- Theoretical 4× speedup

### 24.3 PyTorch 2.0 Integration

torch.compile automatically:
- Decomposes eager PyTorch
- Generates optimized Triton kernels
- Applies fusion optimizations

FP16 matrix multiplication matching cuBLAS in <25 lines of Triton code.

---

## Extended Citations Summary

### Mamba Family
1. Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
2. Dao, T. & Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." (Mamba-2)

### Hybrid Architectures
3. AI21 Labs (2024). "Jamba: A Hybrid Transformer-Mamba Language Model."
4. Ma, X. et al. (2022). "Mega: Moving Average Equipped Gated Attention."
5. De, S. et al. (2024). "Griffin: Mixing Gated Linear Recurrences with Local Attention."

### Continuous-Time Models
6. Hasani, R. et al. (2022). "Closed-form continuous-time neural networks." *Nature Machine Intelligence*.
7. Chen, R.T.Q. et al. (2018). "Neural Ordinary Differential Equations."

### Foundational Theory
8. Gu, A. et al. (2020). "HiPPO: Recurrent Memory with Optimal Polynomial Projections."
9. Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla)

### Alternative Architectures
10. Li, S. et al. (2018). "Independently Recurrent Neural Network (IndRNN)."
11. Bradbury, J. et al. (2016). "Quasi-Recurrent Neural Networks."
12. Poli, M. et al. (2023). "Hyena Hierarchy: Towards Larger Convolutional Language Models."

### Positional Encoding
13. Press, O. et al. (2021). "Train Short, Test Long: Attention with Linear Biases." (ALiBi)
14. Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." (RoPE)

### Memory and Associative Networks
15. Ramsauer, H. et al. (2020). "Hopfield Networks is All You Need."

---

*Document generated: December 2024*
*Expanded edition with 100+ citations*
*For the Elman Ablation Ladder Research Framework*
