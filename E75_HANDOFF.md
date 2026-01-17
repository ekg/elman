# E74v2 & E75 Implementation Handoff

This document covers two related implementations:
1. **E74v2**: Extended E74 full matrix kernel with multiple update types and gate types
2. **E75**: New gated delta model combining E74's delta rule with E61/E68's forget gate

---

# E74 Full Matrix V2

## Overview

E74v2 extends the E74 full matrix kernel with ablation support for different update rules and gating mechanisms. This enables systematic comparison of different memory update strategies.

## Update Types (update_type parameter)

| ID | Name | Update Rule | Description |
|----|------|-------------|-------------|
| 0 | **DELTA** | `S = tanh(S + outer(v - S@k, k))` | Delta rule: error-correcting writes |
| 1 | **RESIDUAL** | `S = tanh(scale * S + outer(v, k))` | Learned per-row residual scaling |
| 2 | **NTM** | `S = tanh(S * (1-erase) + outer(write, k))` | Neural Turing Machine style erase/write |
| 3 | **RETRIEVED_GATE** | `S = tanh(S + gate * outer(v - S@k, k))` | Gated delta (gate controls write strength) |
| 4 | **EMA** | `S = tanh(alpha * S + (1-alpha) * outer(v, k))` | Exponential moving average |

## Gate Types (gate_type parameter)

| ID | Name | Output Gating | Description |
|----|------|---------------|-------------|
| 0 | **OUTPUT** | `out = Sq * silu(Sq)` | Self-gating (E68/E73 style) |
| 1 | **INPUT** | `out = Sq * silu(z_gate)` | E1-style input gating |

## Files

### CUDA Kernel
- **`elman/cuda/lib/e74_full_matrix_v2_gpu.cu.cc`** (~1500 lines)
  - Forward kernel with all update/gate type combinations
  - Backward kernel with gradient checkpointing
  - Utility kernels for sigmoid, bias, reductions

### Python Model
- **`elman/models/e74_ablations.py`** - Contains E74v2 model classes

### PyTorch Bindings
- `e74_full_matrix_forward_v2()` in `elman/cuda/pytorch/elman_ladder.cc`
- `e74_full_matrix_backward_v2()` in `elman/cuda/pytorch/elman_ladder.cc`

## Parameters by Update Type

### DELTA (update_type=0)
No extra parameters - uses basic k, v, q projections.

### RESIDUAL (update_type=1)
| Parameter | Shape | Description |
|-----------|-------|-------------|
| residual_scale | [n_state] | Per-row scaling factor |

### NTM (update_type=2)
| Parameter | Shape | Description |
|-----------|-------|-------------|
| W_erase | [n_state, dim] | Erase gate projection |
| b_erase | [n_state] | Erase gate bias |
| W_write | [n_state, dim] | Write value projection |
| b_write | [n_state] | Write value bias |

### RETRIEVED_GATE (update_type=3)
| Parameter | Shape | Description |
|-----------|-------|-------------|
| W_gate | [n_state, dim] | Write gate projection |
| b_gate | [n_state] | Write gate bias |

### EMA (update_type=4)
| Parameter | Shape | Description |
|-----------|-------|-------------|
| W_alpha | [n_state, dim] | Decay rate projection |
| b_alpha | [n_state] | Decay rate bias (init positive for ~0.9 alpha) |

### INPUT Gate (gate_type=1)
| Parameter | Shape | Description |
|-----------|-------|-------------|
| W_z_gate | [n_state, dim] | Input gate projection |
| b_z_gate | [n_state] | Input gate bias |

## Python Interface

```python
import hasty_pytorch_lib

# Forward
results = hasty_pytorch_lib.e74_full_matrix_forward_v2(
    training,        # bool
    x,               # [T, B, dim]
    S0,              # [B, n_state, n_state]
    proj_type,       # 0=tied_kvq, 1=tied_kq, 2=no_z
    use_tanh,        # bool
    update_type,     # 0=delta, 1=residual, 2=ntm, 3=retrieved_gate, 4=ema
    gate_type,       # 0=output, 1=input
    W_kvq, W_k, W_v, W_q,  # Projections
    residual_scale,  # For residual
    W_erase, b_erase, W_write, b_write,  # For NTM
    W_gate, b_gate,  # For retrieved_gate
    W_alpha, b_alpha,  # For EMA
    W_z_gate, b_z_gate,  # For input gate
)
# Returns: [S, output, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache]

# Backward
grads = hasty_pytorch_lib.e74_full_matrix_backward_v2(
    x, S_checkpoints, Sq_cache,
    k_cache, v_cache, q_cache, d_output,
    proj_type, use_tanh, update_type, gate_type,
    W_kvq, W_k, W_v, W_q,
    residual_scale, erase_cache, write_cache, gate_cache, alpha_cache,
    W_erase, W_write, W_gate, W_alpha,
    z_gate_cache, W_z_gate,
)
# Returns: [dx, dW_kvq, dW_k, dW_v, dW_q, d_residual_scale,
#           dW_erase, db_erase, dW_write, db_write,
#           dW_gate, db_gate, dW_alpha, db_alpha, dW_z_gate, db_z_gate]
```

## Gradient Validation Scripts

Located in `elman/cuda/`:
- `validate_e74_v2_delta_gradients.py` - DELTA update type
- `test_e74_residual_gradients.py` - RESIDUAL update type
- `test_e74_ntm_gradients.py` - NTM update type
- `validate_e74_retrieved_gate.py` - RETRIEVED_GATE update type
- `test_e74_ema_gradients.py` - EMA update type
- `validate_e74_input_gate.py` - INPUT gate type

## Bug Fixes Applied

The following bugs were identified and fixed in this session:

1. **EMA forward**: Added missing `sigmoid(W_alpha @ x + b_alpha)`
2. **INPUT gate forward**: Added missing `b_z_gate` bias
3. **RESIDUAL backward**: Implemented `d_residual_scale` reduction
4. **NTM backward**: Added `W_erase`, `W_write` contributions to `dx`
5. **RETRIEVED_GATE forward**: Added missing `sigmoid()` and `b_gate`
6. **RETRIEVED_GATE backward**: Added `W_gate` contribution to `dx`
7. **INPUT gate backward**: Added `W_z_gate` contribution to `dx`, `db_z_gate` reduction

---

# E75 Gated Delta

## Overview

E75 is a new Elman variant that combines E74's delta rule associative memory with E61/E68's active forgetting mechanism. The key insight: winners in the E-series ladder (E61, E68) all have input-dependent decay/forget gates.

## Architecture

```
k = W_k @ x                           # [B, n_state]
v = W_v @ x                           # [B, n_state]
q = W_q @ x                           # [B, n_state]
beta = sigmoid(W_beta @ x + b_beta)   # [B, n_state] per-row forget gate

k_norm = k / ||k||                    # L2 normalize key
retrieved = S @ k_norm                # [B, n_state] read from memory
delta = v - retrieved                 # What to write (delta rule)
S = tanh(beta * S + outer(delta, k_norm))  # Gated update with tanh

Sq = S @ q                            # [B, n_state] query memory
out = Sq * silu(Sq)                   # Self-gated output
```

## Files Implemented

### CUDA Kernel
- **`elman/cuda/lib/e75_gated_delta_gpu.cu.cc`** (~750 lines)
  - `E75GatedDeltaForwardKernel_BF16<N_STATE>` - Forward pass
  - `E75GatedDeltaBackwardKernel_BF16<N_STATE>` - Backward with checkpoint recomputation
  - Gradient checkpointing every 16 steps
  - Extended shared memory for n_state >= 48 (uses cudaFuncSetAttribute)

### PyTorch Bindings
- **`elman/cuda/pytorch/elman_ladder.cc`** (added ~150 lines)
  - `e75_gated_delta_forward()` - Python-callable forward
  - `e75_gated_delta_backward()` - Python-callable backward
  - Registered in `elman_ladder_init()` module

### Python Model
- **`elman/models/e75_gated_delta.py`** (~400 lines)
  - `E75CUDAFunction` - Autograd function wrapping CUDA
  - `E75GatedDeltaCell` - Core cell with projections (W_k, W_v, W_q, W_beta, b_beta)
  - `E75GatedDelta` - Full layer with in_proj, cell, out_proj

### Header
- **`elman/cuda/lib/hasty/elman_ladder.h`** (added E75 declarations)
  - `E75GatedDeltaForward<T>` struct
  - `E75GatedDeltaBackward<T>` struct
  - WorkspaceSize methods

### Makefile
- **`elman/cuda/Makefile`** - Added `lib/e75_gated_delta_gpu.o`

## Supported Dimensions

### n_state (state matrix size)
- **Supported**: 16, 24, 32, 48, 64, 96, 128
- State matrix shape: `[B, n_state, n_state]`
- For n_state >= 48, uses extended shared memory (>48KB)

### Other Dimensions
- **dim**: Input dimension (any, typically 256-1024)
- **expansion**: Inner expansion factor (default 1.0)
- **T**: Sequence length (any)
- **B**: Batch size (any)

## Python Interface

```python
from elman.models.e75_gated_delta import E75GatedDelta, E75GatedDeltaCell

# Full layer (with in/out projections)
model = E75GatedDelta(
    dim=512,           # Input dimension
    expansion=2.0,     # Inner dim = dim * expansion
    n_state=64,        # State matrix is [B, 64, 64]
    dropout=0.0,       # Optional dropout
    use_conv=False,    # Optional 1D conv before RNN
    d_conv=4,          # Conv kernel size if use_conv=True
    init_beta_bias=2.0,# Bias toward preserving (sigmoid(2)~0.88)
    use_cuda=True,     # Use CUDA kernel (False for PyTorch fallback)
)

# Input: [B, T, dim], Output: [B, T, dim]
x = torch.randn(batch, seq_len, 512, device='cuda', dtype=torch.bfloat16)
output, final_state = model(x)

# Or just the cell (no in/out projections)
cell = E75GatedDeltaCell(
    dim=1024,          # Cell input dimension
    n_state=64,
    init_beta_bias=2.0,
    use_cuda=True,
)
# Cell input: [T, B, dim], Output: [T, B, n_state]
```

## Testing

### Quick Test
```bash
cd /home/erikg/elman
python -m elman.models.e75_gated_delta
```

### Expected Output
```
Testing E75 (Gated Delta Matrix)...
============================================================
Device: cuda
CUDA kernel available: True

--- PyTorch Fallback ---
Output: torch.Size([2, 32, 512]), State: torch.Size([2, 64, 64])
Backward passed!

--- CUDA Kernel ---
Output: torch.Size([2, 32, 512]), State: torch.Size([2, 64, 64])
CUDA Backward passed!

============================================================
Gradient correctness test (CUDA vs PyTorch)
============================================================
Output relative error: 0.0082
dx relative error: 0.0087
dW_k relative error: 0.0067
dW_beta relative error: 0.0148
PASSED: Gradients match within 5% relative tolerance!
```

### Gradient Verification
```python
import torch
from elman.models.e75_gated_delta import E75GatedDeltaCell

cell_pt = E75GatedDeltaCell(dim=64, n_state=32, use_cuda=False).cuda().bfloat16()
cell_cuda = E75GatedDeltaCell(dim=64, n_state=32, use_cuda=True).cuda().bfloat16()
cell_cuda.load_state_dict(cell_pt.state_dict())

x = torch.randn(8, 4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)

# Compare outputs
out_pt, _ = cell_pt(x)
out_cuda, _ = cell_cuda(x)
print(f"Forward match: {(out_pt - out_cuda).abs().max().item():.6f}")

# Compare gradients
out_pt.sum().backward()
dx_pt = x.grad.clone()
x.grad = None
out_cuda.sum().backward()
dx_cuda = x.grad
print(f"Backward match: {(dx_pt - dx_cuda).abs().max().item():.6f}")
```

## Training Benchmark

To run a 100M parameter, 10-minute benchmark:
```bash
cd /home/erikg/elman
python train.py \
    --model e75 \
    --n_state 64 \
    --dim 512 \
    --expansion 2.0 \
    --batch_size 32 \
    --seq_len 512 \
    --dtype bfloat16
```

## Known Limitations

1. **bfloat16 only**: CUDA kernel only supports bfloat16 (not float32 or float16)
2. **Gradient tolerance**: ~1-2% relative error due to checkpoint recomputation in bfloat16
3. **Memory**: Large n_state (96, 128) requires extended shared memory support on GPU

## Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| W_k | [n_state, dim] | Key projection |
| W_v | [n_state, dim] | Value projection |
| W_q | [n_state, dim] | Query projection |
| W_beta | [n_state, dim] | Forget gate projection |
| b_beta | [n_state] | Forget gate bias (init: 2.0 for ~88% preserve) |

## Comparison to Related Models

| Model | State | Update Rule | Forget Gate |
|-------|-------|-------------|-------------|
| E70 | Matrix | decay * S + outer(v,k) | Per-element decay (learned) |
| E71 | Matrix | decay * S + outer(v,k) | Input-gated decay |
| E73 | Matrix | tanh(z*S + outer(v,k)) | Multiplicative z |
| E74 | Matrix | tanh(S + outer(delta,k)) | None (delta rule only) |
| **E75** | Matrix | tanh(beta*S + outer(delta,k)) | **Per-row beta gate** |

E75 uniquely combines delta rule (error-correcting writes) with per-row forget gates.
