# E24: True Single-GEMM Dual Memory

## Overview

E24 is the fully-fused variant of E23 that achieves **exactly 1 GEMM per step** by concatenating input and hidden state before a single dense matrix multiply.

**Key insight**: Use a dense [2D, 2D] weight matrix on concatenated [h_work; x] to produce both h_update and write_val in one operation.

---

## Architecture Comparison

| Variant | GEMMs/step | Weight Structure | write_val sees |
|---------|------------|------------------|----------------|
| E23 (Standard) | 3 | W_h, W_x, W_write separate | h_work_new only |
| E23-Fast | 2 | [W_h; W_write] fused, W_x separate | h_work only |
| **E24** | **1** | **W_all [2D, 2D] dense** | **h_work AND x** |

**E24 advantage**: write_val is computed from BOTH the previous state AND current input, making it more expressive than E23-Fast while being faster.

---

## The Single GEMM

```python
# Input: h_work [B, D], x [B, D]
# Concatenate: input [B, 2D]
# Weight: W_all [2D, 2D]
# Output: [B, 2D] -> split to h_update [B, D], write_val [B, D]

input = torch.cat([h_work, x], dim=-1)          # [B, 2D]
output = input @ W_all.T                         # [B, 2D] - THE SINGLE GEMM
h_update = output[:, :D]                         # [B, D]
write_val = output[:, D:]                        # [B, D]
```

---

## Weight Matrix Structure

The [2D, 2D] matrix W_all can be viewed as four D×D blocks:

```
W_all = | W_hh  W_hx |    [2D, 2D]
        | W_wh  W_wx |

where:
  W_hh [D, D]: h_work → h_update (like E23's W_h)
  W_hx [D, D]: x → h_update (like E23's W_x)
  W_wh [D, D]: h_work → write_val
  W_wx [D, D]: x → write_val (NEW! E23 doesn't have this)
```

**Key difference from E23**: The `W_wx` block lets write_val depend directly on the current input x. This is MORE expressive than standard E23, not less.

---

## Constraint: D_in = D

E24 requires input dimension equals hidden dimension:
- `D_in = D` (required for concatenation and square weight matrix)

This is typically satisfied by having an embedding layer or projection before E24.

---

## Parameters

```python
# E24 Parameters (D=1024)
W_all: [2D, 2D] = [2048, 2048]    # 4M params (was 3M in E23)
b_h: [D]                          # 1K params

# Output projection (unchanged)
W_out: [D_out, D]                 # 1M params
b_out: [D_out]                    # 1K params

# Total: ~5M per layer (vs ~4M for E23)
```

**Tradeoff**: 33% more parameters but 3x fewer GEMMs.

---

## Forward Pass

```python
def e24_step(x, h_tape, h_work, params):
    """
    E24: True single-GEMM dual memory.

    Args:
        x: [B, D] - input (must be D-dimensional)
        h_tape: [B, N, D] - tape state
        h_work: [B, D] - working memory
        params: E24Parameters

    Returns:
        y: [B, D_out] - output
        h_tape_new: [B, N, D] - updated tape
        h_work_new: [B, D] - updated working memory
    """
    B, N, D = h_tape.shape
    scale = 1.0 / math.sqrt(D)

    # ============================================
    # STEP 0: THE SINGLE GEMM (the key optimization!)
    # ============================================
    input_concat = torch.cat([h_work, x], dim=-1)  # [B, 2D]
    output = input_concat @ params.W_all.T          # [B, 2D]
    h_update = output[:, :D]                        # [B, D]
    write_val = output[:, D:]                       # [B, D]

    # ============================================
    # STEP 1: READ FROM TAPE
    # ============================================
    read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1) * scale
    read_attn = F.softmax(read_scores, dim=-1)      # [B, N]
    read = (read_attn[:, :, None] * h_tape).sum(dim=1)  # [B, D]

    # ============================================
    # STEP 2: WORKING MEMORY UPDATE
    # ============================================
    h_work_new = torch.tanh(h_update + read + params.b_h)

    # ============================================
    # STEP 3: WRITE TO TAPE (routing uses h_work_new)
    # ============================================
    write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1) * scale
    write_attn = F.softmax(write_scores, dim=-1)    # [B, N]

    # Replacement write (convex combination)
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
def init_e24_params(D, N, D_out):
    """Initialize E24 parameters."""
    params = E24Parameters()

    # Initialize W_all as 4 blocks
    W_hh = orthogonal((D, D)) * 0.9     # Stable recurrence
    W_hx = xavier_uniform((D, D))        # Input projection
    W_wh = xavier_uniform((D, D))        # Write from hidden
    W_wx = xavier_uniform((D, D))        # Write from input

    # Assemble [2D, 2D] matrix
    params.W_all = torch.cat([
        torch.cat([W_hh, W_hx], dim=1),  # Top row: [W_hh | W_hx]
        torch.cat([W_wh, W_wx], dim=1),  # Bottom row: [W_wh | W_wx]
    ], dim=0)  # Result: [2D, 2D]

    params.b_h = torch.zeros(D)

    # Output projection
    params.W_out = xavier_uniform((D_out, D))
    params.b_out = torch.zeros(D_out)

    return params
```

---

## CUDA Kernel Structure

```cpp
// E24: Single fused kernel per timestep

__global__ void e24_step_kernel(...) {
    // Phase 0: Single GEMM via cuBLAS
    // Concatenate h_work and x, then W_all @ [h_work; x]
    cublasSgemm(..., W_all, input_concat, output);

    // Phase 1: Split output, read attention, update
    // h_update = output[:D]
    // write_val = output[D:]
    // read = attention(tape, h_work)
    // h_work_new = tanh(h_update + read + b)

    // Phase 2: Write attention + replacement write
    // write_attn = softmax(tape @ h_work_new)
    // tape_new = (1 - attn) * tape + attn * write_val
}
```

**Expected performance**: ~12us/step (matching E1's single-GEMM speed)

---

## Why This Works

1. **More expressive than E23-Fast**: write_val sees current input x, not just delayed h_work.

2. **Routing is current**: Write attention uses h_work_new (after processing current input).

3. **Same computational class**: Still has nonlinearity (tanh) + routing (attention) = UTM-complete.

4. **Bounded state**: Replacement write maintains convex combination property.

---

## Comparison Summary

| Metric | E23 | E23-Fast | E24 |
|--------|-----|----------|-----|
| GEMMs/step | 3 | 2 | **1** |
| Expected speed | ~36us | ~24us | **~12us** |
| Parameters | 3D² | 3D² | 4D² |
| Write sees x | Via h_work_new | No (delayed) | **Yes (direct)** |
| Write sees h_work | Via h_work_new | Yes | Yes |
| Constraint | None | None | D_in = D |

---

## When to Use E24

**Use E24 when:**
- Maximum throughput is critical
- Input dimension can match hidden dimension
- Willing to trade 33% more parameters for 3x speed

**Use E23 when:**
- D_in ≠ D and can't add projection layer
- Want minimal parameter count
- Need per-layer configurability of write transform

---

## Files

```
# Formalization
elman-proofs/ElmanProofs/Architectures/E23_DualMemory.lean  # Contains E24 formalization

# Implementation (to create)
elman/models/e24_single_gemm.py      # E24 PyTorch model
elman/cuda/lib/e24_kernel.cu.cc      # E24 CUDA kernel
```

---

## Implementation Notes

1. **Input projection**: If your input x has dimension != D, add a learned projection:
   ```python
   x_proj = x @ W_embed.T  # [B, D_in] -> [B, D]
   ```
   This adds a second GEMM but keeps E24's main loop at 1 GEMM.

2. **Memory layout**: Keep W_all contiguous for efficient GEMM. Don't store as 4 separate blocks.

3. **Fused concatenation**: In CUDA, avoid explicit concatenation - just set up GEMM strides to read from h_work and x directly.

4. **Gradient checkpointing**: Same tape history issue as E23. Consider recomputing instead of storing.
