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
# Dims recalibrated based on actual param counts from test run
# Format: (level, dim, extra_args)
MODELS = [
    # === BASELINES ===
    # mamba2: 973M@dim=2816 -> 1B needs slight increase
    ('mamba2', 2944, ''),
    # fla-gdn: 950M@dim=2432 -> 1B needs slight increase
    ('fla-gdn', 2560, '--expansion 2.0'),
    # llama: 1028M@dim=2048 -> already at 1B
    ('llama', 2048, '--expansion 2.0'),

    # === RNN BASELINES ===
    # GRU/LSTM: 918M@dim=1280 -> close to 1B, keep same
    # NOTE: Standard GRU/LSTM fail to learn at 1B scale, using minGRU/minLSTM
    ('mingru', 2816, '--expansion 2.0'),   # 197M@1280 -> need ~5x params -> dim*2.2
    ('minlstm', 2560, '--expansion 2.0'),  # 213M@1152 -> need ~4.7x params -> dim*2.2

    # === BEST ELMAN VARIANTS ===
    # E0: 1028M@dim=1792 -> at target
    (0, 1792, '--expansion 2.0'),
    # E1: 1033M@dim=1920 -> at target
    (1, 1920, '--expansion 2.0'),
    # E32: 1033M@dim=1920 -> at target
    (32, 1920, '--expansion 2.0'),
    # E36: 1007M@dim=2048 -> at target
    (36, 2048, '--expansion 2.0'),
    # E38: 850M@dim=2304 -> need increase
    (38, 2560, '--expansion 2.0'),
    # E42: 947M@dim=2432 -> slight increase
    (42, 2560, '--expansion 2.0'),
    # E52: failed at 850M, try larger
    (52, 2560, '--expansion 2.0'),
    # E61: 1007M@dim=2048 -> at target
    (61, 2048, '--expansion 2.0'),
    # E68: 1008M@dim=2048 -> at target
    (68, 2048, '--expansion 2.0'),

    # === E75 MULTI-HEAD ===
    # E75h8n32: 558M@dim=3200 -> need ~1.8x params -> dim*1.34
    ('E75h8n32', 4352, '--expansion 2.0 --n_state 32'),
    # E75h8n24: 598M@dim=3456 -> need ~1.67x params -> dim*1.29
    ('E75h8n24', 4480, '--expansion 2.0 --n_state 24'),
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
