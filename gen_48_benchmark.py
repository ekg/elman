#!/usr/bin/env python3
"""
Generate 48-model Elman benchmark with expansion=2.0

Covers the full range of high-performing Elman variants.
All models target ~100M params at depth=20 with expansion=2.0.
"""

import sys
from datetime import datetime

# Standard args for all models
COMMON = (
    "--data data/pile.txt "
    "--depth 20 "
    "--batch_size 32 "
    "--chunk_size 512 "
    "--lr 3e-4 "
    "--warmup_steps 100 "
    "--seed 42 "
    "--bf16 "
    "--train_minutes 10"
)

# Models with their 100M-param dims at expansion=2.0
# Format: (name, dim, extra_args, description)
# Dims calculated for ~100M params at depth=20, expansion=2.0
MODELS = [
    # Baselines
    ('mamba2', 896, '', 'Mamba2 SSM'),
    ('fla-gdn', 768, '--expansion 2.0', 'FLA GatedDeltaNet'),

    # Classic Elman variants (E0-E6)
    ('0', 640, '--expansion 2.0', 'Stock Elman'),
    ('1', 640, '--expansion 2.0', 'Mamba-Gated Elman'),
    ('6', 640, '--expansion 2.0', 'Diagonal Elman'),

    # Self-gating variants (E33-E42)
    ('33', 640, '--expansion 2.0', 'Self-gate'),
    ('34', 640, '--expansion 2.0', 'Diagonal W_h'),
    ('36', 640, '--expansion 2.0', 'Linear recurrence'),
    ('37v2', 640, '--expansion 2.0', 'Tied weights batched'),
    ('42', 768, '--expansion 2.0', 'Linear + tied'),

    # Concat and highway (E56-E60)
    ('56', 640, '--expansion 2.0', 'Concat Elman'),
    ('59', 640, '--expansion 2.0', 'Highway Elman'),
    ('59b', 640, '--expansion 2.0', 'Gated highway'),
    ('60', 640, '--expansion 2.0', 'Residual nonlinear'),
    ('60b', 640, '--expansion 2.0', 'Gated residual'),

    # Decay and selective (E61-E63)
    ('61', 640, '--expansion 2.0', 'Decay-gated'),
    ('61b', 640, '--expansion 2.0', 'Additive decay'),
    ('62', 640, '--expansion 2.0', 'Selective write'),
    ('63', 512, '--expansion 2.0', 'Nonlinear delta'),
    ('63a', 512, '--expansion 2.0', 'Complementary gates'),

    # H-dependent (E64-E68)
    ('64', 640, '--expansion 2.0', 'Additive H'),
    ('65', 640, '--expansion 2.0', 'Diagonal H'),
    ('66', 640, '--expansion 2.0', 'Low-rank H'),
    ('67', 640, '--expansion 2.0', 'H-gated alpha'),
    ('68', 640, '--expansion 2.0', 'Self-gating H-dep'),

    # Matrix state models (E70-E73) - need n_state param
    ('70', 1408, '--expansion 2.0 --n_state 96', 'Linear matrix'),
    ('71', 1408, '--expansion 2.0 --n_state 96', 'S-dependent gate'),
    ('72', 1408, '--expansion 2.0 --n_state 96', 'Memory-gated value'),
    ('73', 1408, '--expansion 2.0 --n_state 96', 'Nonlinear delta rule'),

    # E75 Multi-head variants (best from previous benchmarks)
    ('E75h4n16', 1408, '--expansion 2.0 --n_state 16', '4 heads n=16'),
    ('E75h4n24', 1408, '--expansion 2.0 --n_state 24', '4 heads n=24'),
    ('E75h4n32', 1280, '--expansion 2.0 --n_state 32', '4 heads n=32'),
    ('E75h6n24', 1280, '--expansion 2.0 --n_state 24', '6 heads n=24'),
    ('E75h6n32', 1152, '--expansion 2.0 --n_state 32', '6 heads n=32'),
    ('E75h8n16', 1280, '--expansion 2.0 --n_state 16', '8 heads n=16'),
    ('E75h8n24', 1152, '--expansion 2.0 --n_state 24', '8 heads n=24'),
    ('E75h8n32', 1024, '--expansion 2.0 --n_state 32', '8 heads n=32'),

    # Sparse/special variants
    ('30', 640, '--expansion 2.0', 'Diagonal gated'),
    ('31', 640, '--expansion 2.0', 'Sparse gated'),
    ('31a', 640, '--expansion 2.0', 'ReLU gating'),
    ('32', 640, '--expansion 2.0', 'No pre-silu'),
    ('35', 640, '--expansion 2.0', 'Cubic gate'),

    # Additional exploration
    ('38', 640, '--expansion 2.0', 'No W_x'),
    ('39', 640, '--expansion 2.0', 'No bias'),
    ('40', 640, '--expansion 2.0', 'No pre-silu v2'),
    ('51', 640, '--expansion 2.0', 'No self-gate'),
    ('52', 640, '--expansion 2.0', 'Quadratic gate'),
    ('53', 640, '--expansion 2.0', 'Sigmoid gate only'),
]

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'benchmark_results/elman48_{timestamp}'

    print(f"# 48-Model Elman Benchmark", file=sys.stderr)
    print(f"# Models: {len(MODELS)}", file=sys.stderr)
    print(f"# Output: {output_dir}", file=sys.stderr)

    for name, dim, extra, desc in MODELS:
        cmd = f"python train.py --level {name} --dim {dim} {extra} {COMMON} --output {output_dir}/{name}"
        print(cmd)


if __name__ == '__main__':
    main()
