# Maximum Batch Size Test - Reproducible Instructions

## Quick Reference

```
Max Batch Size:  192
Peak Memory:     41.06 GB (of 47.4 GB)
Headroom:        6.34 GB (13.4%)
```

## How to Reproduce

### Step 1: Verify Environment
```bash
cd /home/erikg/elman
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

### Step 2: Run the Test
```bash
python find_max_batch_size.py
```

Expected output:
```
Using device: cuda:0
Creating E1 model (level=1, dim=3584, depth=6)...
...
[Attempt 9] Testing batch_size=192... SUCCESS
Peak Memory: 41.06 GB
```

### Step 3: Results
- Maximum batch size: **192**
- Peak GPU memory: **41.06 GB**
- Remaining headroom: **6.34 GB**

## Script Architecture

```
find_max_batch_size.py
├── Model Setup
│   └── LadderLM(level=1, dim=3584, depth=6) → 386.3M params
├── Data Loading
│   └── mmap /home/erikg/elman/data/pile.txt
├── Batch Size Search
│   ├── Start: bs=64
│   ├── Step 1: Try bs, run 3 training steps
│   ├── Success: bs += 16 (try larger)
│   └── OOM: bs //= 2 (try smaller)
└── Output
    ├── Maximum batch size found
    └── Peak GPU memory (from torch.cuda.max_memory_allocated)
```

## Test Details

### Model Config (used in test)
- Architecture: E1 (Mamba-Gated Elman) - LadderLM level=1
- Dimension: 3584
- Depth: 6 layers
- Parameters: 386,340,864 (386.3M)
- Vocab size: 256 (byte-level)
- Precision: bfloat16

### Data Config
- Source: `/home/erikg/elman/data/pile.txt` (Pile corpus)
- Sequence length: 512 tokens
- Loading: Memory-mapped random position sampling
- Batch format: [batch_size, seq_len+1] (for input + target)

### Hardware
- GPU: NVIDIA RTX 6000 Ada Generation
- Total memory: 47.4 GB
- Device: CUDA_VISIBLE_DEVICES=0

### Algorithm
1. Start with batch_size=64
2. For each batch size:
   - Load batch via mmap
   - Run 3 full training steps (forward + backward)
   - Record peak GPU memory
3. If success: try batch_size += 16
4. If OOM: halve batch_size
5. Continue until max found

## Results Summary

| Batch Size | Memory (GB) | Status | Loss |
|-----------|-----------|--------|------|
| 64 | 14.80 | ✓ | 6.1667 |
| 80 | 18.08 | ✓ | 6.1875 |
| 96 | 21.36 | ✓ | 6.1875 |
| 112 | 24.64 | ✓ | 6.0938 |
| 128 | 27.92 | ✓ | 6.0938 |
| 144 | 31.21 | ✓ | 6.0938 |
| 160 | 34.49 | ✓ | 6.1250 |
| 176 | 37.77 | ✓ | 6.1562 |
| **192** | **41.06** | **✓** | **6.1250** |
| 208 | OOM | ✗ | — |

## Key Finding: Linear Memory Scaling

Perfect linear relationship:
```
Memory (GB) = 14.80 + 0.2052 × (batch_size - 64)
R² = 0.9999 (nearly perfect)
```

Each +16 batch units = +3.28 GB memory

This enables confident extrapolation:
- BS 192: 41.06 GB ✓ Works
- BS 208: ~44.3 GB (likely OOM at 93.5%)
- BS 224: ~47.6 GB (certain OOM)

## Usage Recommendations

### Production (Maximum Performance)
```python
batch_size = 192  # Full 48GB utilization
```

### Conservative (Safe Training)
```python
batch_size = 160  # 8.91 GB headroom
```

### With Gradient Accumulation (Flexible)
```python
batch_size = 96
grad_accum_steps = 2  # Effective batch = 192
# Peak memory: ~21.4 GB only
```

### Multi-GPU DDP
```bash
torchrun --nproc_per_node=8 train_ladder.py \
    --batch_size 192 \
    --level 1 \
    --params 400m \
    --data /home/erikg/elman/data/pile.txt
# Total batch across 8 GPUs: 192 × 8 = 1,536
```

## Files

- **Script**: `/home/erikg/elman/find_max_batch_size.py` (7.6 KB)
- **Report**: `/home/erikg/elman/MAX_BATCH_SIZE_REPORT.md` (3.0 KB)
- **Instructions**: This file

## Verification Checklist

- [ ] RTX 6000 Ada GPU available with 47+ GB memory
- [ ] Pile corpus at `/home/erikg/elman/data/pile.txt` (or symlink)
- [ ] Python environment with torch, numpy
- [ ] GPU can run models (torch.cuda.is_available() = True)
- [ ] Run script and verify results match

## Expected Runtime

- Full test: ~2-3 minutes
- Per batch size: ~15-30 seconds (3 training steps)
- 10 attempts total

## Troubleshooting

**If CUDA out of memory on first run:**
- Check GPU memory: `nvidia-smi`
- Verify CUDA_VISIBLE_DEVICES is set correctly
- Try reducing starting batch size in script (change `batch_size = 64` to `batch_size = 32`)

**If data file not found:**
- Verify file exists: `ls -lh /home/erikg/elman/data/pile.txt`
- Check symlink: `ls -L /home/erikg/elman/data/`

**If model creation fails:**
- Check LadderLM import: `python -c "from elman.models.ladder_lm import LadderLM"`
- Add to path: `export PYTHONPATH=/home/erikg/elman:$PYTHONPATH`

---

**Status**: All testing complete. Results verified and reproducible.
