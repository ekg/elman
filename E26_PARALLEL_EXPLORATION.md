# E26: Parallel Dual-Memory Exploration

## The Problem

E23 achieves ~25K tokens/sec vs E1's ~150-200K tokens/sec. The bottleneck is **sequential data dependency**:

```
read[t] = attn(tape[t-1], h_work[t-1])   -- depends on h_work
h_work[t] = f(h_work[t-1], x[t], read[t]) -- depends on read
tape[t] = write(tape[t-1], h_work[t])     -- depends on h_work
```

Every timestep depends on the previous. Cannot batch across time.

## Prior Attempts

| Attempt | Result | Why |
|---------|--------|-----|
| E23c (chunked, no read feedback) | 1.5x faster, worse loss | Breaks read→h_work loop |
| E23c_v2 (chunked + read feedback) | 4x SLOWER | Re-introduces sequentiality |
| cuBLAS batched attention | 2x faster attention | Doesn't solve time dependency |

## Candidate Reformulations

### Option 1: Separate "What" from "Where" (RECOMMENDED → E26)

**Key insight**: GEMMs compute *content*, attention computes *routing*. Decouple them.

```python
# PARALLEL PHASE: Batch all projections across time (BIG GEMM)
x_proj[0:T] = x[0:T] @ W_x.T           # [T, B, D]
h_proj[0:T] = h_init @ W_h.T           # or cumulative

# SEQUENTIAL PHASE: Only routing (cheap, no GEMM)
for t in range(T):
    read_scores = h_work @ tape.T      # [B, N] - just dots
    read = softmax(read_scores) @ tape # [B, D] - weighted sum
    h_work = tanh(x_proj[t] + W_h @ h_work + read + b)
    write_scores = h_work @ tape.T
    tape = replacement_write(tape, softmax(write_scores), h_work)
```

**Cost analysis**:
- Parallel: O(T) GEMMs batched into ONE big GEMM
- Sequential: O(T × N × D) dot products + O(T × N) softmax

For N=8, D=512: sequential part is ~4K ops/step vs ~500K for GEMM. **100x cheaper**.

**Preserves**: TM semantics, entmax compatibility, full read feedback.

---

### Option 2: Linear Attention (State Space Style)

Replace softmax with linear attention:
```
read = (h_work @ W_k) @ (tape @ W_v).T  -- no softmax
```

Enables **parallel scan** (like Mamba):
```
state[t] = A × state[t-1] + B × input[t]  -- associative!
```

**Pros**: Full parallelism via scan, O(T) complexity
**Cons**: Loses sharp addressing, blurs slot boundaries

---

### Option 3: Chunked with Iterative Refinement

E23c loses read feedback. Add it back via iteration:

```python
for chunk in chunks(T, K):
    # Pass 1: h_work without reads (parallel)
    h_approx = batch_rnn(x[chunk])

    # Pass 2: reads with approximate h_work (parallel)
    reads = batch_attention(tape, h_approx)

    # Pass 3: refine h_work (parallel)
    h_refined = batch_rnn_with_reads(x[chunk], reads)

    # Iterate until convergence...
```

**Pros**: Each pass is parallel, converges to true answer
**Cons**: Multiple passes, convergence not guaranteed

---

### Option 4: Transformer-RNN Hybrid

Compute tape in parallel from input history:

```python
# PARALLEL: Build tape via self-attention
tape = SelfAttention(x[0:T])  # [T, B, N, D] or pooled

# SEQUENTIAL: RNN reads from pre-computed tape
for t in range(T):
    read = attention(tape[0:t], h_work)  # causal
    h_work = f(h_work, x[t], read)
```

**Pros**: Tape construction is O(T²) but parallel
**Cons**: Changes tape semantics (no recurrent write)

---

### Option 5: Fixed-Point / Equilibrium Model

Formulate as finding consistent state:

```python
def equilibrium(x, tape_init):
    tape = tape_init
    for _ in range(max_iter):
        h_work = parallel_forward(x, tape)  # given tape, compute h_work
        tape_new = parallel_write(h_work)   # given h_work, compute tape
        if converged(tape, tape_new): break
        tape = tape_new
    return tape, h_work
```

**Pros**: Each iteration is parallel
**Cons**: Convergence issues, implicit differentiation needed

---

## Recommendation: E26 = Option 1

**Why Option 1**:
1. Preserves full TM semantics (sharp attention, slot isolation)
2. Works with entmax (sparse routing)
3. Minimal architecture change from E23/E25
4. Clear separation: parallel GEMMs + sequential routing
5. For small N (8-64), routing is negligible cost

**E26 Architecture**:
```
PARALLEL (batched cuBLAS):
  x_proj[0:T] = BatchGEMM(x[0:T], W_x)

SEQUENTIAL (cheap routing):
  for t in range(T):
    read = sparse_attention(tape, h_work)      # O(N×D) dots
    h_work = tanh(x_proj[t] + W_h @ h_work + read)  # O(D²) but sequential
    tape = sparse_write(tape, h_work)          # O(N×D) dots
```

**Expected performance**:
- Parallel phase: ~1 big GEMM ≈ 50-100μs for T=512
- Sequential phase: ~20μs/step × 512 = ~10ms (dominated by W_h @ h_work)

**Key optimization**: The W_h @ h_work is still sequential. Could potentially:
- Use smaller D for working memory
- Approximate with diagonal + low-rank
- Accept the cost since it's the fundamental RNN recurrence

---

## Next Steps

1. **Formalize E26 in Lean** - prove equivalence to E25 under certain conditions
2. **Implement E26 CUDA kernel** - parallel projection + sequential routing
3. **Benchmark against E1, E23, E25** - measure actual speedup
4. **Explore Option 2 (linear attention)** if E26 insufficient

---

## Files

- `ElmanProofs/Architectures/E26_ParallelDualMemory.lean` - Formalization
- `elman/E26_IMPLEMENTATION.md` - Implementation spec (TODO)
- `elman/cuda/lib/e26_parallel_gpu.cu.cc` - CUDA kernel (TODO)
