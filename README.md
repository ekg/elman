# Elman Ladder Research

A research framework for exploring non-linear, non-associative recurrent architectures that can compete with linear state-space models like Mamba2.

## Research Objective

**Core hypothesis**: Non-linearity and non-associativity in recurrent models can provide more expressivity than linear SSMs like Mamba2, but we need numerically stable implementations to fairly test this.

Modern SSMs like Mamba2 achieve stability through:
1. Log-space computation preventing overflow/underflow
2. Bounded gradients via softmax weights from logsumexp
3. Linear state updates enabling parallel scan

The Elman architecture uses `tanh(W_h @ h)` which is:
- **Non-linear**: tanh squashes values
- **Non-associative**: `tanh(A + B) != tanh(A) + tanh(B)`

This should provide more expressivity, but naively implementing it causes:
- Gradient vanishing through depth (tanh saturation)
- Numerical instability in long sequences
- No parallel scan (sequential by nature)

This repository implements the "Elman Ladder" - a 7-level ablation study that progressively adds complexity to measure the cost/benefit of each architectural choice.

## The Elman Ladder

| Level | Name | Key Change | Associative? | Parallel? |
|-------|------|------------|--------------|-----------|
| 0 | Stock Elman | Pure `h = tanh(W_x @ x + W_h @ h + b)` | No | No |
| 1 | Gated Elman | Add discretization: `h = (1-δ)*h + δ*cand` | No | No |
| 2 | Selective Elman | Add compete softmax output | No | No |
| 3 | Diagonal Selective | Diagonal `r_h` instead of full `W_h` | No | No |
| 4 | Log-Storage | Store h as (log\|h\|, sign(h)) | No | No |
| 5 | Log-Compute | Log-space matrix multiply | No | No |
| 6 | Log-Space Triple R | Full log-space with polynomial activation | Partial | Partial |

See [docs/ladder.md](docs/ladder.md) for detailed documentation.

## Comparison with Mamba2

| Aspect | Mamba2 | Elman (Level 0-3) | Elman (Level 4-6) |
|--------|--------|-------------------|-------------------|
| State update | Linear | Non-linear (tanh) | Non-linear |
| Associative | Yes | No | No |
| Parallel scan | Yes | No | No (sequential) |
| Numerical stability | Excellent (log-space) | Moderate | Good (log-space) |
| Gradient flow | Bounded (softmax) | Vanishes at depth | Research frontier |

## Installation

Uses the existing `mingru` micromamba environment:

```bash
micromamba activate mingru
pip install -r requirements.txt
```

## Quick Start

### Training (without TBPTT - recommended for comparison)

```bash
# Single GPU
python train.py --data /path/to/data.txt --level 3 --params 100m --bf16

# Multi-GPU DDP
torchrun --nproc_per_node=8 train_ladder.py --level 3 --params 500m --data /path/to/data.txt --ddp --bf16
```

### Training (with TBPTT)

TBPTT is **opt-in** via `--tbptt` flag. Only use when needed:

```bash
python train.py --data /path/to/data.txt --level 3 --params 100m --bf16 --tbptt
```

**Note**: TBPTT requires `BatchedStreamDataset` which maintains persistent per-batch-element streams. Without TBPTT, hidden states are not carried across chunks.

### Data Format

Raw bytes with `0x1e` (ASCII record separator) as document delimiter.

## Repository Structure

```
elman/
├── elman/
│   ├── models/     # Ladder implementations (Levels 0-6)
│   ├── kernels/    # CUDA/Triton kernels
│   └── data/       # Data loading utilities
├── scripts/        # Training job scripts
├── tests/          # Comparative tests
├── docs/           # Documentation
└── logs/           # Training logs
```

## Key Files

**Models (Levels 0-3 implemented, 4-6 documented):**
- `elman/models/stock_elman.py` - Level 0: Pure Elman
- `elman/models/gated_elman.py` - Level 1: Discretized Elman
- `elman/models/selective_elman.py` - Level 2: With compete softmax
- `elman/models/diagonal_selective.py` - Level 3: Diagonal hidden weight
- `elman/models/ladder_lm.py` - Language model wrapper for all levels
- `elman/models/mamba2_baseline.py` - Mamba2 baseline for comparison

**Data:**
- `elman/data/dataset.py` - Document-aware streaming datasets
  - `DocumentStreamDataset` - Single stream (no TBPTT)
  - `BatchedStreamDataset` - Per-batch-element streams (for TBPTT)
- `elman/data/tokenizers.py` - Byte-level and tiktoken tokenizers

**Training:**
- `train.py` - Single GPU training
- `train_ladder.py` - DDP multi-GPU training

**CUDA Kernels:**
- `elman/cuda/` - Haste-based CUDA kernels (build with `make && pip install -e .`)

## Current Status

### Working (Linear-space)
- Levels 0-3: Fully functional with CUDA kernels
- Training verified up to 1B parameters

### Research Frontier (Log-space)
- Level 4: Forward works, backward has gradient issues
- Level 5-6: Theoretical framework, implementation in progress

See [docs/logspace.md](docs/logspace.md) for the log-space research frontier.

## Related Work

- [gruboros](../gruboros) - Original research codebase
- [Mamba2](https://arxiv.org/abs/2405.21060) - State-space model with log-space computation
- [FlashRNN](https://github.com/NX-AI/flashrnn) - Fast RNN kernels
