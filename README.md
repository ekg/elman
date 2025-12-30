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

### Linear-Space Levels (0-3)
| Level | Name | Key Change | Associative? | Parallel? |
|-------|------|------------|--------------|-----------|
| 0 | Stock Elman | Pure `h = tanh(W_x @ x + W_h @ h + b)` | No | No |
| 1 | Gated Elman | Add discretization: `h = (1-δ)*h + δ*cand` | No | No |
| 2 | Selective Elman | Add compete softmax output | No | No |
| 3 | Diagonal Selective | Diagonal `r_h` instead of full `W_h` | No | No |

### Log-Space Polynomial Levels (log_0, log_1, log_2) - **RECOMMENDED**
| Level | Name | Key Change | Gradient Stability |
|-------|------|------------|-------------------|
| log_0 | Log-Space Polynomial | Polynomial activation `α*log\|v\|` with input-dependent α | Bounded |
| log_1 | Log-Space Selective | + compete×silu output | Bounded |
| log_2 | Log-Space Diagonal Selective | + diagonal r_h in log-space | Bounded |

### Experimental Levels (4-6)
| Level | Name | Key Change |
|-------|------|------------|
| 4 | Log-Storage | Store h as (log\|h\|, sign(h)) |
| 5 | Log-Compute | Log-space matrix multiply |
| 6 | Log-Space Triple R | Full log-space with Triple R |

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

### Training (8 GPU DDP with tiktoken - default)

```bash
# Set library path
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Train log-space polynomial (recommended)
torchrun --nproc_per_node=8 train_ladder.py --level log_0 --params 100m --data /path/to/data.txt

# Train stock Elman baseline
torchrun --nproc_per_node=8 train_ladder.py --level 0 --params 100m --data /path/to/data.txt
```

**Default settings**: DDP, CUDA, bf16, tiktoken (p50k_base ~50k vocab), AdamWScheduleFree

### Single GPU (for debugging)

```bash
python train_ladder.py --level log_0 --params 10m --data /path/to/data.txt --no-ddp
```

### Training with TBPTT

TBPTT is **opt-in** via `--tbptt` flag:

```bash
torchrun --nproc_per_node=8 train_ladder.py --level log_0 --params 100m --data /path/to/data.txt --tbptt
```

### Data Format

Raw text file. Documents separated by `0x1e` (ASCII record separator).

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

**Models:**
- `elman/models/stock_elman.py` - Level 0: Pure Elman
- `elman/models/gated_elman.py` - Level 1: Discretized Elman
- `elman/models/selective_elman.py` - Level 2: With compete softmax
- `elman/models/diagonal_selective.py` - Level 3: Diagonal hidden weight
- `elman/models/logspace_polynomial.py` - Level log_0: Log-space polynomial **(RECOMMENDED)**
- `elman/models/logspace_selective.py` - Level log_1: + compete×silu
- `elman/models/logspace_diagonal_selective.py` - Level log_2: + diagonal r_h
- `elman/models/ladder_lm.py` - Language model wrapper for all levels

**Data:**
- `elman/data/dataset.py` - Document-aware streaming datasets
- `elman/data/tokenizers.py` - Byte-level and tiktoken tokenizers

**Training:**
- `train_ladder.py` - Main training script (DDP, schedule-free, tiktoken defaults)

**CUDA Kernels:**
- `elman/cuda/` - Haste-based CUDA kernels

Build CUDA kernels:
```bash
cd elman/cuda
make
pip install -e .
```

## Current Status

### Production Ready
- **Levels 0-3**: Linear-space, CUDA kernels, verified to 1B params
- **Levels log_0, log_1, log_2**: Log-space polynomial with CUDA kernels, bounded gradients

### Experimental
- Levels 4-6: Theoretical framework, implementation in progress

See [docs/logspace.md](docs/logspace.md) for log-space implementation details.

## Related Work

- [gruboros](../gruboros) - Original research codebase
- [Mamba2](https://arxiv.org/abs/2405.21060) - State-space model with log-space computation
- [FlashRNN](https://github.com/NX-AI/flashrnn) - Fast RNN kernels
