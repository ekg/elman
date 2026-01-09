# E23 Related Work: Literature Review

## Overview

E23 is a dual-memory Elman RNN with:
- **Working memory**: D-dimensional, nonlinear (tanh)
- **Tape memory**: N×D slots, linear storage
- **Interface**: Attention-based read/write, replacement write mechanism

This document surveys related architectures and compares them to E23.

---

## 1. Memory-Augmented Neural Networks

### Neural Turing Machine (NTM)
**Graves, Wayne, Danihelka (2014)** - [arXiv:1410.5401](https://arxiv.org/abs/1410.5401)

- **Architecture**: Controller network + external memory matrix
- **Read/Write**: Content-based + location-based addressing via soft attention
- **Write mechanism**: Additive (erase + add gates)
- **E23 comparison**:
  - NTM uses *additive* write with separate erase/add
  - E23 uses *replacement* write (convex combination) - simpler, bounded
  - NTM has explicit location-based addressing; E23 is purely content-based

### Differentiable Neural Computer (DNC)
**Graves et al. (2016)** - [Nature](https://www.nature.com/articles/nature20101)

- **Architecture**: Controller + memory matrix + temporal linking
- **Read/Write**: Content-based + temporal linking + allocation mechanism
- **Key features**: Dynamic allocation, memory usage tracking
- **E23 comparison**:
  - DNC has complex allocation/deallocation; E23 just overwrites via attention
  - DNC tracks temporal order; E23 relies on working memory for sequencing
  - DNC more powerful but ~10x more complex

---

## 2. State Space Models

### S4 - Structured State Spaces
**Gu, Goel, Re (2021)** - [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)

- **State update**: x' = Ax + Bu (continuous-time, discretized)
- **Key innovation**: HiPPO initialization of A for long-range dependencies
- **Complexity**: O(N log N) via FFT
- **E23 comparison**:
  - S4 state is *implicit* (no content-based access)
  - E23 tape is *explicit* with attention-based addressing
  - S4 state is fixed-size; E23 can scale N independently

### Mamba / Mamba2
**Gu & Dao (2023)** - [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

- **State update**: Selective SSM - parameters depend on input
- **Key innovation**: Input-dependent A, B, C matrices (selectivity)
- **Complexity**: O(N) with hardware-aware implementation
- **E23 comparison**:
  - Mamba state is 262K elements (N x D x heads), updated all at once
  - E23 tape is N x D with *selective* read/write via attention
  - Mamba: all state decays; E23: only attended slots change
  - Mamba: linear state update; E23: nonlinear working memory

### RWKV
**Peng et al. (2023)** - [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)

- **Architecture**: Linear attention RNN (no softmax)
- **Key innovation**: Parallelizable training, RNN inference
- **State**: Fixed-size key-value state
- **E23 comparison**:
  - RWKV has no explicit memory addressing
  - E23 uses softmax attention for explicit slot selection
  - RWKV scales to 14B params; E23 untested at scale

---

## 3. Dual/Fast Memory Architectures

### Fast Weights
**Ba, Hinton et al. (2016)** - [arXiv:1610.06258](https://arxiv.org/abs/1610.06258)

- **Architecture**: Slow weights (learned) + fast weights (per-sequence)
- **Fast weight update**: Hebbian outer product A += eta * (h ⊗ h)
- **Memory capacity**: Quadratic in hidden dimension (vs linear for RNN)
- **E23 comparison**:
  - Fast Weights: implicit associative memory (auto-associative)
  - E23: explicit slots with attention addressing (hetero-associative)
  - Fast Weights: additive update; E23: replacement write
  - Fast Weights: O(D^2) memory; E23: O(N x D) memory (independent scaling)

### Modern Hopfield Networks
**Ramsauer et al. (2020)** - "Hopfield Networks is All You Need" (ICLR 2021)

- **Key insight**: Transformer attention = Hopfield network update
- **Storage capacity**: Exponential in dimension (vs linear classical)
- **Update rule**: softmax(beta * X^T * q) - same as attention!
- **E23 comparison**:
  - Hopfield: attention as memory retrieval
  - E23: attention for both retrieval AND storage
  - E23's replacement write is like Hopfield update with write-back

---

## 4. Comparison Table

| Feature | NTM/DNC | Mamba | Fast Weights | E23 |
|---------|---------|-------|--------------|-----|
| Memory type | External matrix | Implicit state | Outer product | External tape |
| Addressing | Content + location | None (all state) | Auto-associative | Content-based |
| Write mechanism | Erase + Add | Decay + add | Additive Hebbian | Replacement |
| Bounded state | No (needs clipping) | Yes (decay) | No | Yes (convex) |
| Working memory | Controller | None explicit | Hidden state | Explicit D-dim |
| Complexity | O(N x D) | O(N x D) | O(D^2) | O(D^2 + N x D) |

---

## 5. What E23 Borrows

From **NTM/DNC**:
- Content-based addressing via attention
- Explicit external memory matrix
- Differentiable read/write operations

From **Fast Weights**:
- Separation of working memory and storage
- Idea of fast-changing memory for recent context

From **Hopfield Networks**:
- Attention as associative memory retrieval
- Content-addressed storage

From **State Space Models**:
- Efficient recurrent computation
- Comparison point for state efficiency

---

## 6. What E23 Simplifies

Compared to **NTM**:
- No separate erase and add gates (replacement write handles both)
- No location-based addressing (content-only is simpler)
- No sharpening/interpolation parameters

Compared to **DNC**:
- No temporal linking mechanism
- No dynamic allocation/freeing
- No usage tracking

Compared to **Fast Weights**:
- Explicit slots instead of implicit outer product
- Bounded writes (no explosion risk)

---

## 7. What E23 Adds

**Replacement write**: `new = (1 - attn) * old + attn * value`
- Provably bounded (convex combination)
- Self-normalizing (no need for separate erase gate)
- TM-like semantics (slot replacement, not accumulation)

**Explicit working memory**:
- Clear separation: working memory computes, tape stores
- Only working memory has nonlinearity
- Clean interface via attention

**Formal guarantees** (from Lean proofs):
- Bounded state maintenance
- TM semantics in hard-attention limit
- E1 as special case (N=0)

---

## 8. Open Questions from Literature

1. **Location-based addressing**: NTM uses both content and location. Should E23 add positional bias to attention scores?

2. **Learned temperature**: Hopfield work suggests learnable attention temperature. Could help E23 learn when to be sharp vs diffuse.

3. **Multi-head tape**: DNC uses multiple read heads. Should E23 have multiple read/write heads per tape?

4. **Hardware optimization**: Mamba achieves 5x transformer throughput via custom CUDA. E23 needs similar optimization.

5. **Allocation signals**: DNC tracks memory usage. Should E23 have a mechanism to prefer "empty" slots?

---

## 9. Key References

### Memory-Augmented Networks
- Graves, Wayne, Danihelka. "Neural Turing Machines." arXiv:1410.5401, 2014.
- Graves et al. "Hybrid computing using a neural network with dynamic external memory." Nature 538, 2016.
- Weston, Chopra, Bordes. "Memory Networks." ICLR 2015.

### State Space Models
- Gu, Goel, Re. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022.
- Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.
- Peng et al. "RWKV: Reinventing RNNs for the Transformer Era." EMNLP 2023.

### Associative Memory
- Ba, Hinton et al. "Using Fast Weights to Attend to the Recent Past." NeurIPS 2016.
- Ramsauer et al. "Hopfield Networks is All You Need." ICLR 2021.

### Foundational
- Hochreiter, Schmidhuber. "Long Short-Term Memory." Neural Computation, 1997.
- Vaswani et al. "Attention Is All You Need." NeurIPS 2017.

---

## 10. Summary

E23 occupies a unique position in the design space:

```
         Implicit State              Explicit Memory
         (SSMs, RWKV)                (NTM, DNC)
              |                           |
              |                           |
    Fast/Simple -------- E23 -------- Slow/Powerful
              |                           |
              |                           |
         Fast Weights               Full TM Simulation
```

E23 aims for the sweet spot:
- **Simpler than NTM/DNC**: No allocation, no temporal linking, no erase gates
- **More explicit than SSMs**: Content-addressed slots, selective updates
- **More bounded than Fast Weights**: Replacement write prevents explosion
- **More powerful than E1**: External tape enables longer-range storage

The key bet: **replacement write + attention routing** provides enough capability for practical tasks while maintaining simplicity and formal guarantees.
