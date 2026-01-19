# E83 Circular Tower Benchmarks Handoff

## Completed Work

The E83 Circular Tower CUDA kernel now has full input-bias support:

- **Input-bias mode**: Per-timestep biases computed from input via W_b projection
- **K values**: 2, 3, 4, 8 heads supported
- **n_state values**: 16, 24, 32, 48 supported in shared memory
- **Three bias modes**: fixed bias (default), no-bias (`use_bias=False`), input-bias (`input_bias=True`)

Registered variants in `ladder_lm.py`:
- `83k2`, `83k3`, `83k4`, `83k8` - fixed bias variants
- `83k2nb`, `83k3nb`, `83k4nb`, `83k8nb` - no-bias variants
- `83k2ib`, `83k3ib`, `83k4ib`, `83k8ib` - input-bias variants

## Tasks for Next Agent

### 1. Fix Tokenizer (CRITICAL)

The current setup uses tiktoken which has ~100k vocab. Training showed loss of ~10.4 which is impossible with byte-level modeling (256 vocab).

**Fix needed**: Update `train_ladder.py` to use byte-level tokenizer with pile.txt:
- vocab_size should be 256
- Each byte is a token
- No need for tiktoken - just convert bytes to integers

Look for how tokenizer is set up and replace with simple byte encoding:
```python
# Simple byte-level tokenization
def tokenize_bytes(text):
    return list(text.encode('utf-8'))
```

### 2. Run E83 Benchmark Suite

Run 8-way or 16-way comparison of E83 variants:

**Parameters:**
- Dataset: `pile.txt` with byte-level tokenizer
- Duration: 10 minutes per run
- Seed: 42
- Expansion: 2
- Model size: 100M parameters
- Variants to test:

**8-way comparison (recommended first):**
1. `83k2` - 2 heads, fixed bias
2. `83k4` - 4 heads, fixed bias
3. `83k8` - 8 heads, fixed bias (many small heads hypothesis)
4. `83k2ib` - 2 heads, input-bias
5. `83k4ib` - 4 heads, input-bias
6. `83k8ib` - 8 heads, input-bias
7. `llama` - baseline transformer
8. `mamba2` - baseline SSM

**16-way comparison (if time permits):**
Add no-bias variants (`83k2nb`, `83k4nb`, `83k8nb`) and additional baselines.

### 3. Key Hypotheses to Test

1. **Many small heads**: K=8 with n_state=24 might outperform K=2 with n_state=48 at same parameter count
2. **Input-bias vs fixed-bias**: Does dynamic bias from input help?
3. **Circular gating**: Does mutual gating between K matrices help vs independent matrices?

### 4. Expected Command Structure

```bash
# Single E83 run example
python train_ladder.py \
    --level 83k4ib \
    --data pile.txt \
    --total_params 100000000 \
    --expansion 2 \
    --seed 42 \
    --max_time 600 \
    --vocab_size 256  # byte-level

# Or use run_benchmark.py if it exists
```

### 5. Notes

- E83 hidden state is a list of K matrices `[M_0, M_1, ..., M_{K-1}]`
- Each M_i is shape `[batch, n_state, n_state]`
- The `detach_hidden` and `reset_hidden` functions in train_ladder.py now handle this
- CUDA kernel requires bfloat16 dtype

## Files Modified in This Session

- `elman/cuda/lib/e83_circular_tower_gpu.cu.cc` - Added ~1200 lines for input-bias kernels
- `elman/cuda/lib/hasty/elman_ladder.h` - Added wrapper class declarations
- `elman/cuda/pytorch/elman_ladder.cc` - Added Python bindings
- `elman/models/e83_circular_tower.py` - Added autograd function for input-bias CUDA
- `elman/models/ladder_lm.py` - Registered all E83 variants
- `train_ladder.py` - Fixed parse_level and hidden state handling
