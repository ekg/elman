# E27: Tape-Gated Output Implementation

## Overview

E27 extends E25 (dual-memory with entmax) by making the output gate depend on the **tape read**, not just the input projection.

**Key insight**: The tape is long-term memory, but in E23/E25/E26 it doesn't directly control what gets output. E27 fixes this.

```python
# E23/E25/E26 (tape not in gate):
output = h_work * silu(z)           # z from input only

# E27b (tape in gate):
output = h_work * silu(z + read)    # read from tape!
```

**One line change, zero new parameters, direct tape influence on output.**

---

## Architecture

### Data Flow

```
Input x
   │
   ├──────────────────────┐
   │                      │
   ▼                      ▼
┌──────────┐         ┌─────────┐
│ x_proj   │         │    z    │  (input split)
└────┬─────┘         └────┬────┘
     │                    │
     ▼                    │
┌─────────────────────────┼───────────────────┐
│         TAPE            │                   │
│  ┌───┬───┬───┬───┐      │                   │
│  │ 0 │ 1 │ 2 │...│      │                   │
│  └───┴───┴───┴───┘      │                   │
│         │               │                   │
│    entmax read          │                   │
│         │               │                   │
│         ▼               │                   │
│      [read]─────────────┼──► z + read ──┐   │
│                         │               │   │
└─────────────────────────┼───────────────┼───┘
                          │               │
     ┌────────────────────┘               │
     │                                    │
     ▼                                    ▼
┌──────────┐                         ┌─────────┐
│ h_work   │                         │  silu   │
│ update   │                         │ (gate)  │
└────┬─────┘                         └────┬────┘
     │                                    │
     └──────────► h_work * gate ◄─────────┘
                       │
                       ▼
                   [output]
```

### Key Difference from E25

| Aspect | E25 | E27b |
|--------|-----|------|
| Output gate | `silu(z)` | `silu(z + read)` |
| Tape → output path | Indirect (via h_work) | **Direct** (via gate) |
| Extra parameters | 0 | 0 |
| Extra compute | 0 | 1 addition per dim |

---

## Forward Pass

```python
def e27b_step(x, h_tape, h_work, params):
    """
    E27b: Tape-gated output with additive combination.

    Args:
        x: [B, D_in] - input
        h_tape: [B, N, D] - tape state
        h_work: [B, D] - working memory
        params: E27Parameters

    Returns:
        y: [B, D_out] - output
        h_tape_new: [B, N, D] - updated tape
        h_work_new: [B, D] - updated working memory
    """
    B, N, D = h_tape.shape
    scale = 1.0 / math.sqrt(D)

    # ============================================
    # STEP 1: INPUT PROJECTION (split into x_proj and z)
    # ============================================
    xz = x @ params.W_xz.T                    # [B, 2*D]
    x_proj = xz[:, :D]                        # [B, D] - for h_work update
    z = xz[:, D:]                             # [B, D] - for output gate

    # ============================================
    # STEP 2: READ FROM TAPE (sparse attention)
    # ============================================
    read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1) * scale  # [B, N]
    read_attn = entmax_1_5(read_scores, dim=-1)                       # [B, N] sparse!
    read = (read_attn[:, :, None] * h_tape).sum(dim=1)               # [B, D]

    # ============================================
    # STEP 3: WORKING MEMORY UPDATE
    # ============================================
    h_recur = h_work @ params.W_h.T                                  # [B, D]
    h_work_new = torch.tanh(x_proj + h_recur + read + params.b_h)    # [B, D]

    # ============================================
    # STEP 4: WRITE TO TAPE (sparse attention)
    # ============================================
    write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1) * scale
    write_attn = entmax_1_5(write_scores, dim=-1)                    # [B, N] sparse!
    write_val = h_work_new @ params.W_write.T                        # [B, D]

    # Replacement write
    h_tape_new = (
        (1 - write_attn[:, :, None]) * h_tape +
        write_attn[:, :, None] * write_val[:, None, :]
    )

    # ============================================
    # STEP 5: OUTPUT WITH TAPE-GATED SELECTIVITY
    # ============================================
    # THE KEY DIFFERENCE: gate sees tape read!
    gate = F.silu(z + read)                   # [B, D] - tape influences gate!
    y = h_work_new * gate                     # [B, D]
    y = y @ params.W_out.T + params.b_out     # [B, D_out]

    return y, h_tape_new, h_work_new
```

---

## Parameters

```python
# E27b Parameters (same count as E25!)
W_xz: [2*D, D_in]      # Input projection (split to x_proj and z)
W_h: [D, D]            # Recurrence
W_write: [D, D]        # Write value projection
W_out: [D_out, D]      # Output projection
b_h: [D]               # Hidden bias
b_out: [D_out]         # Output bias

# Total: same as E25 (~4M for D=1024)
```

**No additional parameters over E25.** The tape-gated output comes for free.

---

## Comparison: E25 vs E27b

```python
# E25 output (tape not in gate):
gate = F.silu(z)
y = h_work_new * gate

# E27b output (tape in gate):
gate = F.silu(z + read)      # <-- ONE LINE CHANGE
y = h_work_new * gate
```

---

## Why This Works

### Information Flow

**E25:**
```
tape → read → h_work → output
              ↑
       z → silu(z) = gate
```
The tape affects h_work, but the gate only sees z (input).

**E27b:**
```
tape → read ──┬──→ h_work → output
              │         ↑
       z ─────┴→ silu(z + read) = gate
```
The tape DIRECTLY influences the output gate via read.

### Semantic Meaning

- **z** encodes "what the input wants to output"
- **read** encodes "what the tape (memory) says is relevant"
- **z + read** combines both: output when input AND memory agree

When the tape read is zero (no relevant content found), E27b falls back to E25 behavior.

---

## Entmax Synergy

E27b works especially well with entmax:

1. **Sparse read** → `read` has many zeros
2. Where `read = 0`, gate = `silu(z)` (E25 behavior)
3. Where `read ≠ 0`, gate = `silu(z + read)` (tape-influenced)

This means:
- Tape only affects output gate where it has relevant content
- Irrelevant tape slots don't interfere with gating
- Clean separation: tape influences gate only when attention is non-zero

---

## CUDA Kernel Changes

The change is minimal. In the output computation phase:

```cpp
// E25 (current):
// gate = silu(z)
for (int d = tid; d < DIM; d += blockDim.x) {
    float g = z_sh[d];
    gate_sh[d] = g / (1.0f + expf(-g));  // silu
}

// E27b (new):
// gate = silu(z + read)
for (int d = tid; d < DIM; d += blockDim.x) {
    float g = z_sh[d] + read_sh[d];      // <-- ONE ADDITION
    gate_sh[d] = g / (1.0f + expf(-g));  // silu
}
```

**Performance impact**: One addition per dimension per step. Negligible.

---

## Initialization

Same as E25. No special initialization needed for E27b.

```python
def init_e27b_params(D, N, D_in, D_out):
    params = E27Parameters()

    # Input projection (split to x_proj and z)
    params.W_xz = xavier_uniform((2*D, D_in))

    # Recurrence (orthogonal for stability)
    params.W_h = orthogonal((D, D)) * 0.9

    # Write projection
    params.W_write = xavier_uniform((D, D))

    # Output projection
    params.W_out = xavier_uniform((D_out, D))

    # Biases
    params.b_h = zeros(D)
    params.b_out = zeros(D_out)

    return params
```

---

## Variants (for reference)

| Variant | Gate Formula | Extra Params | Notes |
|---------|--------------|--------------|-------|
| **E27a** | `silu(read)` | 0 | Ignores input z, not recommended |
| **E27b** | `silu(z + read)` | 0 | **RECOMMENDED** |
| **E27c** | `silu(W_z@z + W_r@read)` | 2D² | Most expressive |
| **E27d** | `silu(z) * silu(read)` | 0 | Double gating, strong selectivity |

---

## Expected Behavior

1. **When tape is empty/uniform**: `read ≈ 0`, so `gate ≈ silu(z)` → E25 behavior

2. **When tape has relevant content**: `read ≠ 0`, tape influences gate
   - Positive read values boost the gate
   - Negative read values suppress the gate

3. **With entmax**: Sparse attention means only a few slots contribute to read
   - Gate is influenced by specific tape content, not blurred average

---

## Testing Checklist

- [ ] Verify E27b output differs from E25 when tape has content
- [ ] Verify E27b ≈ E25 when tape is zero-initialized
- [ ] Check gradient flow through new path (tape → read → gate → output)
- [ ] Benchmark: confirm negligible performance overhead
- [ ] Compare loss curves: E27b vs E25 on same tasks

---

## Files

```
# Formalization
elman-proofs/ElmanProofs/Architectures/E27_TapeGatedOutput.lean

# Implementation (to create)
elman/models/e27_tape_gated.py
elman/cuda/lib/e27_tape_gated_gpu.cu.cc
```

---

## Summary

**E27b = E25 + one line change:**

```python
# Before (E25):
gate = F.silu(z)

# After (E27b):
gate = F.silu(z + read)
```

- Zero extra parameters
- Negligible extra compute (one add per dim)
- Direct tape → output influence
- Works especially well with entmax (sparse read = selective gating)
