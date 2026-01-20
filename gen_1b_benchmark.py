#!/usr/bin/env python3
"""
Generate 1B parameter benchmark for all models.
Dims are pre-computed to hit ~1B params at depth=20, expansion=2.0.

Calculation method: Scale from 100M benchmark dims by sqrt(10) ≈ 3.16
Then round to nearest 128-aligned value.
"""

import sys
from datetime import datetime

# Standard args for all models
COMMON = (
    "--data data/pile.txt "
    "--depth 20 "
    "--batch_size 16 "  # Smaller batch for 1B models
    "--chunk_size 512 "
    "--lr 3e-4 "
    "--warmup_steps 100 "
    "--seed 42 "
    "--bf16 "
    "--train_minutes 10"
)

# Models with their 1B-param dims at depth=20, expansion=2.0
# Dims calculated by scaling from 100M benchmark dims
# Format: (level, dim, extra_args)
MODELS = [
    # === BASELINES ===
    # mamba2: 100M@dim=896 -> 1B@dim=2816 (scale ~3.14x)
    ('mamba2', 2816, ''),
    # fla-gdn: 100M@dim=768 -> 1B@dim=2432 (scale ~3.16x)
    ('fla-gdn', 2432, '--expansion 2.0'),
    # llama: 100M@dim=640 -> 1B@dim=2048 (scale ~3.2x, includes attention overhead)
    ('llama', 2048, '--expansion 2.0'),

    # === RNN BASELINES (GRU/LSTM have ~4x params per dim due to 4 gates) ===
    # GRU: ~4*dim^2 per layer, so scale dim by sqrt(10)/2 ≈ 1.58x from 100M
    ('gru', 1280, '--expansion 2.0'),
    ('lstm', 1152, '--expansion 2.0'),  # LSTM has even more params
    ('mingru', 1280, '--expansion 2.0'),
    ('minlstm', 1152, '--expansion 2.0'),
    ('cudagru', 1280, '--expansion 2.0'),
    ('cudalstm', 1152, '--expansion 2.0'),

    # === BEST ELMAN VARIANTS ===
    # E0: Stock Elman - 100M@dim=640 -> 1B@dim=1792
    (0, 1792, '--expansion 2.0'),
    # E1: Mamba-gated - 100M@dim=640 -> 1B@dim=1920
    (1, 1920, '--expansion 2.0'),
    # E32: No pre-silu - same as E1
    (32, 1920, '--expansion 2.0'),
    # E36: Linear recurrence - fewer params, 100M@dim=640 -> 1B@dim=2048
    (36, 2048, '--expansion 2.0'),
    # E38: No W_x - fewer params
    (38, 2304, '--expansion 2.0'),
    # E42: Linear + tied - 100M@dim=768 -> 1B@dim=2432
    (42, 2432, '--expansion 2.0'),
    # E52: Quadratic gate
    (52, 2304, '--expansion 2.0'),
    # E61: Decay-gated - 100M@dim=640 -> 1B@dim=2048
    (61, 2048, '--expansion 2.0'),
    # E68: Self-gating h-dep
    (68, 2048, '--expansion 2.0'),

    # === E75 MULTI-HEAD ===
    # E75 needs larger dims due to smaller n_state overhead
    ('E75h8n32', 3200, '--expansion 2.0 --n_state 32'),
    ('E75h8n24', 3456, '--expansion 2.0 --n_state 24'),
]

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'benchmark_results/1b_10min_{timestamp}'

    print(f"# 1B Parameter Benchmark", file=sys.stderr)
    print(f"# Models: {len(MODELS)}", file=sys.stderr)
    print(f"# Output: {output_dir}", file=sys.stderr)

    for level, dim, extra in MODELS:
        cmd = f"python train.py --level {level} --dim {dim} {extra} {COMMON} --output {output_dir}/{level}"
        print(cmd)


if __name__ == '__main__':
    main()
