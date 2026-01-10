# E23-Fast: Single GEMM Implementation

## Overview

E23-Fast is a variant of E23 that achieves **1 GEMM per step** (vs 2 in standard E23) by computing write_value from the previous working memory state.

**Key insight**: Fuse `[W_h; W_write] @ h_work` into a single GEMM by computing write_value *before* the working memory update.

---

## Semantic Difference

| Aspect | E23 (Standard) | E23-Fast |
|--------|----------------|----------|
| Write content | `W_write @ h_work_new` | `W_write @ h_work` |
| Write routing | `attn(tape, h_work_new)` | `attn(tape, h_work_new)` |
| GEMMs per step | 2 | 1 |
| Speed | ~24us | ~12us (expected) |

**Interpretation**: E23-Fast writes "what you were thinking" to "where you now think it belongs".

- Content: from previous step (before current input processed)
- Location: from current step (after current input processed)

---

## Architecture

```
Standard E23:
  1. read = attention(tape, h_work)
  2. h_work_new = tanh(W_h @ h_work + W_x @ x + read + b)  [GEMM 1]
  3. write_val = W_write @ h_work_new                      [GEMM 2]
  4. tape_new = replacement_write(tape, attn(h_work_new), write_val)

E23-Fast (fused):
  1. combined = [W_h; W_write] @ h_work                    [GEMM 1 - FUSED]
     h_update = combined[:D]
     write_val = combined[D:]
  2. read = attention(tape, h_work)
  3. h_work_new = tanh(h_update + W_x @ x + read + b)
  4. tape_new = replacement_write(tape, attn(h_work_new), write_val)
```

---

## Parameters

```python
# Fused weight matrix: [2D, D]
W_combined: [2*D, D]     # = cat([W_h, W_write], dim=0)

# Separate (can't fuse because x has different dim)
W_x: [D, D_in]           # input projection
b_h: [D]                 # bias

# Output (unchanged)
W_out: [D_out, D]
b_out: [D_out]
```

**Parameter count** (D=1024, D_in=D_out=D):
- W_combined: 2M (was W_h: 1M + W_write: 1M)
- W_x: 1M
- W_out: 1M
- biases: ~2K
- **Total: ~4M** (same as E23, just organized differently)

---

## Forward Pass

```python
def e23_fast_forward(x, h_tape, h_work, params):
    """
    E23-Fast: Single GEMM variant.

    Args:
        x: [B, D_in] - input
        h_tape: [B, N, D] - tape state
        h_work: [B, D] - working memory (from previous step)
        params: E23FastParameters

    Returns:
        y: [B, D_out] - output
        h_tape_new: [B, N, D] - updated tape
        h_work_new: [B, D] - updated working memory
    """
    B, N, D = h_tape.shape
    scale = 1.0 / math.sqrt(D)

    # ============================================
    # STEP 0: FUSED GEMM (the key optimization!)
    # ============================================
    # One GEMM producing both h_update and write_val
    combined = h_work @ params.W_combined.T  # [B, 2D]
    h_update = combined[:, :D]               # [B, D] - for working memory
    write_val = combined[:, D:]              # [B, D] - for tape write

    # ============================================
    # STEP 1: READ FROM TAPE
    # ============================================
    read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1) * scale
    read_attn = F.softmax(read_scores, dim=-1)
    read = (read_attn[:, :, None] * h_tape).sum(dim=1)

    # ============================================
    # STEP 2: WORKING MEMORY UPDATE
    # ============================================
    pre_act = h_update + x @ params.W_x.T + read + params.b_h
    h_work_new = torch.tanh(pre_act)

    # ============================================
    # STEP 3: WRITE TO TAPE (routing uses h_work_new!)
    # ============================================
    write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1) * scale
    write_attn = F.softmax(write_scores, dim=-1)

    # write_val was computed from h_work, but routing uses h_work_new
    h_tape_new = (
        (1 - write_attn[:, :, None]) * h_tape +
        write_attn[:, :, None] * write_val[:, None, :]
    )

    # ============================================
    # STEP 4: OUTPUT
    # ============================================
    y = h_work_new @ params.W_out.T + params.b_out

    return y, h_tape_new, h_work_new
```

---

## Initialization

```python
def init_e23_fast_params(D, N, D_in, D_out):
    params = E23FastParameters()

    # Fused matrix: [W_h; W_write]
    W_h = orthogonal((D, D)) * 0.9
    W_write = xavier_uniform((D, D))
    params.W_combined = cat([W_h, W_write], dim=0)  # [2D, D]

    # Input projection
    params.W_x = xavier_uniform((D, D_in))
    params.b_h = zeros(D)

    # Output
    params.W_out = xavier_uniform((D_out, D))
    params.b_out = zeros(D_out)

    return params
```

---

## CUDA Kernel Structure

```cpp
// Single fused kernel per timestep

__global__ void e23_fast_step_kernel(...) {
    // Phase 1: Fused GEMM via cuBLAS
    // W_combined @ h_work -> [h_update, write_val]
    cublasSgemm(..., W_combined, h_work, combined);

    // Phase 2: Custom kernel for rest
    // - Read attention + weighted sum
    // - h_work_new = tanh(h_update + W_x @ x + read + b)
    // - Write attention + replacement write

    // Note: W_x @ x could be a second small GEMM, or fused if D_in == D
}
```

**Expected performance**: ~12us/step (matching E1's single-GEMM speed)

---

## Theoretical Analysis

### Why This Works

1. **Routing is current**: The attention for write uses `h_work_new`, which sees the current input. The *where* is up-to-date.

2. **Content is smooth**: `h_work` and `h_work_new` are related by:
   ```
   h_work_new = tanh(W_h @ h_work + W_x @ x + read + b)
   ```
   For stable training, this is typically a small perturbation of `h_work`.

3. **1-step delay is bounded**: The information "lag" is exactly 1 step. Over T steps, each piece of information is written with at most 1-step delay. This doesn't reduce computational class.

### What We Might Lose

1. **Input-dependent writes**: In E23, what you write depends on the current input. In E23-Fast, it depends on the previous state.

2. **Read-dependent writes**: In E23, `write_val` can incorporate what you just read. In E23-Fast, it can't.

Example scenario:
```
E23: Read "cat" from slot 3, immediately write "animal" based on this
E23-Fast: Read "cat", next step write what you were thinking before reading "cat"
```

### Mitigation

The network can learn to compensate:
- Use working memory to "buffer" what should be written next step
- The routing still uses current information, so location is correct
- Over training, the network adapts to the 1-step delay

---

## When to Use E23-Fast

**Use E23-Fast when:**
- Training throughput is critical
- Task doesn't require immediate read→write loops
- Willing to trade some expressivity for 2x speed

**Use standard E23 when:**
- Maximum expressivity needed
- Task requires tight read-compute-write cycles
- Throughput is less important than capability

---

## Comparison Summary

| Metric | E23 | E23-Fast |
|--------|-----|----------|
| GEMMs/step | 2 | 1 |
| Expected speed | ~24us | ~12us |
| Write content | Current | 1-step delayed |
| Write routing | Current | Current |
| Parameters | 4M | 4M |
| Bounded state | Yes | Yes |
| UTM class | Yes | Yes |

---

## Files

```
models/
  e23_dual_memory.py       # Standard E23
  e23_fast.py              # E23-Fast variant

cuda/
  e23_fast_kernel.cu       # Fused CUDA kernel
```
