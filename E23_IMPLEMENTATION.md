# E23 Baseline Implementation Specification

## Overview

E23 is a dual-memory Elman RNN with:
- **Tape**: Large linear storage (N slots × D dimensions)
- **Working Memory**: Small nonlinear compute (D dimensions)

Key design: **No decay, replacement write**. The tape persists until explicitly overwritten.

---

## Architecture Summary

```
State:
  h_tape: [B, N, D]  -- tape memory (N=64 slots)
  h_work: [B, D]     -- working memory (D=1024)

Per step:
  1. Input writes to tape (additive outer product)
  2. Working memory reads from tape (attention)
  3. Working memory updates (Elman + read)
  4. Working memory writes to tape (replacement)
  5. Output from working memory
```

---

## Hyperparameters

| Name | Symbol | Default | Notes |
|------|--------|---------|-------|
| Working dimension | D | 1024 | Same as E1 |
| Tape slots | N | 64 | State expansion knob |
| Input dimension | D_in | D | Usually same as D |
| Output dimension | D_out | D | Usually same as D |
| Attention scale | scale | 1/sqrt(D) | Standard scaling |

---

## Parameters (Learnable)

```python
# Tape input projection
W_k: [N, D_in]      # input → tape key (which slots to write)
W_v: [D, D_in]      # input → tape value (what to write)

# Working memory
W_h: [D, D]         # recurrence matrix
W_x: [D, D_in]      # input projection
b_h: [D]            # bias

# Write projection
W_write: [D, D]     # working → write value

# Output
W_out: [D_out, D]   # working → output
b_out: [D_out]      # output bias
```

**Parameter count** (D=1024, N=64, D_in=D_out=D):
- W_k: 64K
- W_v: 1M
- W_h: 1M
- W_x: 1M
- W_write: 1M
- W_out: 1M
- biases: ~2K
- **Total: ~5M per layer**

---

## Forward Pass (Exact Equations)

```python
def e23_forward(x, h_tape, h_work, params):
    """
    Args:
        x: [B, D_in] - input at this timestep
        h_tape: [B, N, D] - tape state
        h_work: [B, D] - working memory state
        params: E23Parameters

    Returns:
        y: [B, D_out] - output
        h_tape_new: [B, N, D] - updated tape
        h_work_new: [B, D] - updated working memory
    """
    B, N, D = h_tape.shape
    scale = 1.0 / math.sqrt(D)

    # ============================================
    # STEP 1: INPUT WRITES TO TAPE (additive)
    # ============================================
    # Project input to key (which slots) and value (what to write)
    key = x @ params.W_k.T          # [B, N]
    value = x @ params.W_v.T        # [B, D]

    # Outer product update: tape += key ⊗ value
    # key[:, :, None] is [B, N, 1]
    # value[:, None, :] is [B, 1, D]
    h_tape = h_tape + key[:, :, None] * value[:, None, :]  # [B, N, D]

    # ============================================
    # STEP 2: WORKING MEMORY READS FROM TAPE
    # ============================================
    # Attention scores: working memory queries tape
    # h_work[:, None, :] is [B, 1, D]
    # h_tape is [B, N, D]
    read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1)  # [B, N]
    read_scores = read_scores * scale
    read_attn = F.softmax(read_scores, dim=-1)  # [B, N]

    # Weighted sum of tape slots
    # read_attn[:, :, None] is [B, N, 1]
    read = (read_attn[:, :, None] * h_tape).sum(dim=1)  # [B, D]

    # ============================================
    # STEP 3: WORKING MEMORY UPDATE (Elman + read)
    # ============================================
    pre_act = (
        h_work @ params.W_h.T +     # [B, D] @ [D, D].T = [B, D]
        x @ params.W_x.T +          # [B, D_in] @ [D, D_in].T = [B, D]
        read +                       # [B, D]
        params.b_h                   # [D] broadcasts
    )
    h_work_new = torch.tanh(pre_act)  # [B, D]

    # ============================================
    # STEP 4: WORKING MEMORY WRITES TO TAPE (replacement)
    # ============================================
    # Project working memory to write value
    write_value = h_work_new @ params.W_write.T  # [B, D]

    # Attention scores for write (same mechanism as read)
    write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1)  # [B, N]
    write_scores = write_scores * scale
    write_attn = F.softmax(write_scores, dim=-1)  # [B, N]

    # REPLACEMENT write (NOT additive!)
    # h_tape = (1 - attn) * h_tape + attn * new_value
    # write_attn[:, :, None] is [B, N, 1]
    # write_value[:, None, :] is [B, 1, D]
    h_tape_new = (
        (1 - write_attn[:, :, None]) * h_tape +
        write_attn[:, :, None] * write_value[:, None, :]
    )

    # ============================================
    # STEP 5: OUTPUT
    # ============================================
    y = h_work_new @ params.W_out.T + params.b_out  # [B, D_out]

    return y, h_tape_new, h_work_new
```

---

## Initialization

```python
def init_e23_params(D, N, D_in, D_out):
    params = E23Parameters()

    # Tape projections: Xavier
    params.W_k = xavier_uniform((N, D_in))
    params.W_v = xavier_uniform((D, D_in))

    # Working memory: orthogonal W_h (scaled down)
    params.W_h = orthogonal((D, D)) * 0.9
    params.W_x = xavier_uniform((D, D_in))
    params.b_h = zeros(D)

    # Write projection: Xavier
    params.W_write = xavier_uniform((D, D))

    # Output: Xavier
    params.W_out = xavier_uniform((D_out, D))
    params.b_out = zeros(D_out)

    return params

def init_e23_state(B, N, D):
    """Initialize hidden state to zeros."""
    h_tape = torch.zeros(B, N, D)
    h_work = torch.zeros(B, D)
    return h_tape, h_work
```

---

## Sequence Processing

```python
def e23_sequence(x_seq, params):
    """
    Process a sequence.

    Args:
        x_seq: [B, T, D_in] - input sequence
        params: E23Parameters

    Returns:
        y_seq: [B, T, D_out] - output sequence
    """
    B, T, D_in = x_seq.shape
    D = params.W_h.shape[0]
    N = params.W_k.shape[0]
    D_out = params.W_out.shape[0]

    # Initialize state
    h_tape, h_work = init_e23_state(B, N, D)
    h_tape = h_tape.to(x_seq.device)
    h_work = h_work.to(x_seq.device)

    outputs = []
    for t in range(T):
        y, h_tape, h_work = e23_forward(x_seq[:, t], h_tape, h_work, params)
        outputs.append(y)

    return torch.stack(outputs, dim=1)  # [B, T, D_out]
```

---

## CRITICAL Implementation Notes

### DO:
1. **Use replacement write for working→tape**: `(1-attn)*old + attn*new`
2. **Use additive write for input→tape**: `old + outer_product`
3. **Scale attention by 1/sqrt(D)** before softmax
4. **Initialize W_h orthogonal and scaled to 0.9**
5. **Initialize tape and working memory to zeros**

### DO NOT:
1. **DO NOT add decay to tape** - no `alpha * h_tape` anywhere
2. **DO NOT use additive write for working→tape** - this causes explosion
3. **DO NOT forget the scale factor** - attention will be too sharp/soft
4. **DO NOT use learned attention projections** - keep it simple (dot product)
5. **DO NOT skip the W_write projection** - working memory needs transformation before writing

---

## Testing Checklist

Before training, verify:

1. **Shape check**: Run one forward pass, verify all tensor shapes
2. **Gradient check**: Verify gradients flow to all parameters
3. **Boundedness check**: Run 1000 steps with random input, verify h_work stays in [-1, 1]
4. **Tape persistence**: With zero input after step 100, verify tape doesn't decay
5. **Attention distribution**: Print attention weights, verify they're valid distributions (sum to 1)

```python
def test_e23():
    B, T, D, N = 2, 100, 64, 16  # Small for testing
    params = init_e23_params(D, N, D, D)
    x = torch.randn(B, T, D)

    # Forward pass
    y = e23_sequence(x, params)
    assert y.shape == (B, T, D), f"Output shape wrong: {y.shape}"

    # Gradient check
    loss = y.sum()
    loss.backward()
    for name, p in params.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    # Boundedness check
    h_tape, h_work = init_e23_state(B, N, D)
    for t in range(1000):
        x_t = torch.randn(B, D)
        _, h_tape, h_work = e23_forward(x_t, h_tape, h_work, params)
    assert (h_work.abs() <= 1.0).all(), "Working memory not bounded!"

    print("All tests passed!")
```

---

## Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Loss vs E1 | -0.02 to -0.05 nats | Tape should help |
| Loss vs E18-A | -0.01 to -0.03 nats | Similar or better |
| Throughput vs E1 | 0.4-0.6× | Tape ops add overhead |
| Memory vs E1 | ~1.5× | Tape state is large but cheap |

---

## File Structure

```
models/
  e23_dual_memory.py    # Main implementation

tests/
  test_e23.py           # Unit tests

experiments/
  e23_baseline.py       # Training script
```

---

## Summary

E23 = Elman + Tape Memory

1. **Tape** stores information persistently (no decay)
2. **Working memory** does computation (tanh nonlinearity)
3. **Attention** routes information between them
4. **Replacement write** keeps values bounded

The key insight: separate storage (tape) from computation (working memory).
