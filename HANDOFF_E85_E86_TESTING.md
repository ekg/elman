# E85 & E86 Testing Handoff

## Summary

Two new architectures have been implemented with CUDA kernels:

1. **E85 (Input-as-Matrix)**: Simple matrix recurrence where input IS the state update matrix
2. **E86 (Input-Matrix Delta Rule)**: Combines E85's input-as-matrix with E75's delta rule, multi-head support

## Architecture Details

### E85: Input-as-Matrix
```
Input: x [B, T, n_state^2] -> reshaped to A [B, n_state, n_state]
Update: M = M + scale * (A @ M)
Output: Sq * silu(Sq) where Sq = S @ q, q = A.mean(dim=rows)
```
- Single learnable parameter: `scale` (scalar)
- cell_dim = n_state^2

### E86: Input-Matrix Delta Rule (Multi-Head)
```
Per head h:
  A_h = input reshaped to [n_state, n_state]
  k = A_h.mean(dim=rows)     # key from row means
  v = A_h.mean(dim=cols)     # value from col means
  beta = sigmoid(scale * A_h.mean() + bias)

  k_norm = k / ||k||
  retrieved = S_h @ k_norm
  delta = v - retrieved
  S_h = tanh(beta * S_h + outer(delta, k_norm))

  out_h = Sq * silu(Sq) where Sq = S_h @ k_norm
```
- Learnable parameters: `scale`, `bias` (shared across heads)
- cell_dim = n_heads * n_state^2
- out_dim = n_heads * n_state

## Registered Variants

### E85 Variants
| Level | n_state | cell_dim | out_dim |
|-------|---------|----------|---------|
| 85    | 32      | 1024     | 32      |
| 85n16 | 16      | 256      | 16      |
| 85n24 | 24      | 576      | 24      |
| 85n32 | 32      | 1024     | 32      |
| 85n48 | 48      | 2304     | 48      |

### E86 Variants (Single Head)
| Level | n_state | cell_dim | out_dim |
|-------|---------|----------|---------|
| 86    | 32      | 1024     | 32      |
| 86n16 | 16      | 256      | 16      |
| 86n24 | 24      | 576      | 24      |
| 86n32 | 32      | 1024     | 32      |
| 86n48 | 48      | 2304     | 48      |

### E86 Multi-Head Variants
| Level   | n_heads | n_state | cell_dim | out_dim |
|---------|---------|---------|----------|---------|
| 86h2    | 2       | 32      | 2048     | 64      |
| 86h4    | 4       | 32      | 4096     | 128     |
| 86h2n24 | 2       | 24      | 1152     | 48      |
| 86h4n24 | 4       | 24      | 2304     | 96      |
| 86h4n16 | 4       | 16      | 1024     | 64      |
| 86h8n16 | 8       | 16      | 2048     | 128     |

## Testing Commands

### Quick Validation (Python vs CUDA)
```bash
cd /home/erikg/elman
export LD_LIBRARY_PATH=/home/erikg/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Test E85
python3 elman/models/e85_input_as_matrix.py

# Test E86
python3 elman/models/e86_input_matrix_delta.py
```

### Training Benchmarks (10 min each)

#### E85 variants
```bash
# Single head variants
python3 train.py --model ladder --level 85 --n_params 100m --max_time 600
python3 train.py --model ladder --level 85n24 --n_params 100m --max_time 600
python3 train.py --model ladder --level 85n48 --n_params 100m --max_time 600
```

#### E86 single head variants
```bash
python3 train.py --model ladder --level 86 --n_params 100m --max_time 600
python3 train.py --model ladder --level 86n24 --n_params 100m --max_time 600
python3 train.py --model ladder --level 86n48 --n_params 100m --max_time 600
```

#### E86 multi-head variants (compare capacity scaling)
```bash
# Compare different head configurations at similar cell_dim
python3 train.py --model ladder --level 86h2 --n_params 100m --max_time 600    # 2 heads, cell_dim=2048
python3 train.py --model ladder --level 86h4 --n_params 100m --max_time 600    # 4 heads, cell_dim=4096
python3 train.py --model ladder --level 86h4n24 --n_params 100m --max_time 600 # 4 heads, cell_dim=2304
python3 train.py --model ladder --level 86h8n16 --n_params 100m --max_time 600 # 8 heads, cell_dim=2048
```

## Key Comparisons to Make

1. **E85 vs E86**: Does adding delta rule semantics help?
2. **E86 single vs multi-head**: Does multi-head improve capacity?
3. **E86 vs E75n48**: Compare input-as-matrix delta rule vs learned-projection delta rule
4. **E86h4 vs E86h8n16**: Same-ish cell_dim, different head configurations

## Files Changed

- `elman/models/e85_input_as_matrix.py` - E85 Python implementation
- `elman/models/e86_input_matrix_delta.py` - E86 Python implementation
- `elman/cuda/lib/e85_input_as_matrix_gpu.cu.cc` - E85 CUDA kernel
- `elman/cuda/lib/e86_input_matrix_delta_gpu.cu.cc` - E86 CUDA kernel
- `elman/cuda/lib/hasty/elman_ladder.h` - Header declarations
- `elman/cuda/pytorch/elman_ladder.cc` - Python bindings
- `elman/models/ladder_lm.py` - Variant registration

## CUDA Build

If CUDA extension needs rebuild:
```bash
cd /home/erikg/elman/elman/cuda
export PATH=/usr/local/cuda-12.8/bin:$PATH
python3 setup.py build_ext --inplace
```

## Notes

- E86 CUDA kernel uses gradient checkpointing (saves state every 16 steps)
- Multi-head E86 runs one CUDA block per (batch, head) pair
- Python implementation matches CUDA output within ~1e-4 (bfloat16 precision)
- Both E85 and E86 have minimal learnable parameters in the recurrence (no learned projections)
