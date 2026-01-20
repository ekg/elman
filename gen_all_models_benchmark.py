#!/usr/bin/env python3
"""
Generate comprehensive benchmark for ALL available models.
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

# ALL models with their 100M-param dims at expansion=2.0
# Format: (name, dim, extra_args)
# Dims must be 128-aligned
MODELS = [
    # === BASELINES ===
    ('mamba2', 896, ''),  # Mamba2 SSM (hardcoded expand=2)
    ('fla-gdn', 768, '--expansion 2.0'),  # FLA GatedDeltaNet
    ('llama', 640, '--expansion 2.0'),  # Transformer baseline

    # === CLASSIC ELMAN (E0-E6) ===
    ('0', 640, '--expansion 2.0'),  # Stock Elman
    ('1', 640, '--expansion 2.0'),  # Mamba-Gated Elman (IMPORTANT)
    ('2', 640, '--expansion 2.0'),  # Slot Elman
    ('3', 640, '--expansion 2.0'),  # Low-Rank Slot Elman
    ('4', 640, '--expansion 2.0'),  # Low-Rank Elman
    ('5', 640, '--expansion 2.0'),  # Pure Low-Rank Elman
    ('6', 640, '--expansion 2.0'),  # Diagonal Elman

    # === SCALED/HYBRID (E8-E12) ===
    ('8', 640, '--expansion 2.0'),  # Scaled Low-Rank Elman
    ('9', 640, '--expansion 2.0'),  # Hybrid Elman
    ('10', 640, '--expansion 2.0'),  # Multi-Scale Elman
    ('11', 640, '--expansion 2.0'),  # Selective Elman
    ('12', 640, '--expansion 2.0'),  # Selective Gated Elman

    # === MATRIX STATE (E14-E17) ===
    ('14', 640, '--expansion 2.0'),  # Matrix State Elman
    ('15', 640, '--expansion 2.0'),  # Softsign Elman
    ('16', 640, '--expansion 2.0'),  # Diagonal State Elman
    ('17', 640, '--expansion 2.0'),  # Selective Wh Elman

    # === H-AWARE GATE (E18) ===
    ('18a', 640, '--expansion 2.0'),  # gate = z + h
    ('18b', 640, '--expansion 2.0'),  # gate = z + Rh
    ('18e', 640, '--expansion 2.0'),  # no gate

    # === SIMPLIFIED GATE (E19) ===
    ('19a', 640, '--expansion 2.0'),  # gate = Wx + h
    ('19b', 640, '--expansion 2.0'),  # gate = h-only
    ('19d', 640, '--expansion 2.0'),  # residual + z
    ('19e', 640, '--expansion 2.0'),  # residual + Wx + h

    # === MAMBA2-INFORMED (E20-E21) ===
    ('20', 640, '--expansion 2.0'),  # Mamba2-style matrix
    ('21', 640, '--expansion 2.0'),  # Structured (MIMO)

    # === DIAGONAL/SPARSE GATED (E30-E35) ===
    ('30', 640, '--expansion 2.0'),  # Diagonal gated
    ('31', 640, '--expansion 2.0'),  # Sparse gated
    ('31a', 640, '--expansion 2.0'),  # ReLU gating
    ('31b', 640, '--expansion 2.0'),  # Softplus gating
    ('32', 640, '--expansion 2.0'),  # No pre-silu
    ('33', 640, '--expansion 2.0'),  # Self-gate
    ('34', 640, '--expansion 2.0'),  # Diagonal W_h
    ('35', 640, '--expansion 2.0'),  # Cubic gate

    # === LINEAR/TIED (E36-E42) ===
    ('36', 640, '--expansion 2.0'),  # Linear recurrence
    ('37', 640, '--expansion 2.0'),  # Tied weights
    ('38', 640, '--expansion 2.0'),  # No W_x
    ('39', 640, '--expansion 2.0'),  # No bias
    ('40', 640, '--expansion 2.0'),  # No pre-silu v2
    ('41', 640, '--expansion 2.0'),  # Diagonal W_x
    ('42', 768, '--expansion 2.0'),  # Linear + tied (IMPORTANT)

    # === MINIMAL (E43-E48) ===
    ('43', 640, '--expansion 2.0'),  # Scalar decay
    ('44', 640, '--expansion 2.0'),  # Diagonal W
    ('45', 640, '--expansion 2.0'),  # Pure accumulation
    ('46', 640, '--expansion 2.0'),  # No in_proj
    ('48', 640, '--expansion 2.0'),  # No projections

    # === GATING VARIANTS (E51-E55) ===
    ('51', 640, '--expansion 2.0'),  # No self-gate
    ('52', 640, '--expansion 2.0'),  # Quadratic gate
    ('53', 640, '--expansion 2.0'),  # Sigmoid gate only
    ('54', 640, '--expansion 2.0'),  # Diagonal no proj
    ('55', 640, '--expansion 2.0'),  # Scalar no proj

    # === CONCAT/HIGHWAY (E56-E60) ===
    ('56', 640, '--expansion 2.0'),  # Concat Elman
    ('57', 640, '--expansion 2.0'),  # Learned radius
    ('58', 640, '--expansion 2.0'),  # Learned radii
    ('59', 640, '--expansion 2.0'),  # Highway
    ('60', 640, '--expansion 2.0'),  # Residual nonlinear

    # === DECAY/SELECTIVE (E61-E68) ===
    ('61', 640, '--expansion 2.0'),  # Decay-gated
    ('62', 640, '--expansion 2.0'),  # Selective write
    ('63', 512, '--expansion 2.0'),  # Nonlinear delta
    ('64', 640, '--expansion 2.0'),  # Additive H
    ('65', 640, '--expansion 2.0'),  # Diagonal H
    ('66', 640, '--expansion 2.0'),  # Low-rank H
    ('66r16', 640, '--expansion 2.0'),  # Low-rank H r=16
    ('66r64', 640, '--expansion 2.0'),  # Low-rank H r=64
    ('66r128', 640, '--expansion 2.0'),  # Low-rank H r=128
    ('67', 640, '--expansion 2.0'),  # H-gated alpha
    ('68', 640, '--expansion 2.0'),  # Self-gating h-dep

    # === MATRIX STATE MODELS (E70-E73) - need n_state ===
    ('70', 1408, '--expansion 2.0 --n_state 96'),  # Linear matrix
    ('70n32', 1408, '--expansion 2.0 --n_state 32'),
    ('70n128', 1280, '--expansion 2.0 --n_state 128'),
    ('71', 1408, '--expansion 2.0 --n_state 96'),  # S-dependent gate
    ('71n32', 1408, '--expansion 2.0 --n_state 32'),
    ('71n128', 1280, '--expansion 2.0 --n_state 128'),
    ('72', 1408, '--expansion 2.0 --n_state 96'),  # Memory-gated value
    ('73', 1408, '--expansion 2.0 --n_state 96'),  # Nonlinear delta

    # === E75 GATED DELTA (need n_state) ===
    ('75', 1408, '--expansion 2.0 --n_state 96'),  # Default
    ('75n32', 1408, '--expansion 2.0 --n_state 32'),
    ('75n48', 1408, '--expansion 2.0 --n_state 48'),
    ('75n64', 1408, '--expansion 2.0 --n_state 64'),
    ('75n96', 1408, '--expansion 2.0 --n_state 96'),

    # === E75 MULTI-HEAD (best configs) ===
    ('E75h4n16', 1408, '--expansion 2.0 --n_state 16'),
    ('E75h4n24', 1408, '--expansion 2.0 --n_state 24'),
    ('E75h4n32', 1280, '--expansion 2.0 --n_state 32'),
    ('E75h6n24', 1280, '--expansion 2.0 --n_state 24'),
    ('E75h6n32', 1152, '--expansion 2.0 --n_state 32'),
    ('E75h8n16', 1280, '--expansion 2.0 --n_state 16'),
    ('E75h8n24', 1152, '--expansion 2.0 --n_state 24'),
    ('E75h8n32', 1024, '--expansion 2.0 --n_state 32'),
]

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'benchmark_results/all_models_{timestamp}'

    print(f"# Comprehensive Model Benchmark", file=sys.stderr)
    print(f"# Models: {len(MODELS)}", file=sys.stderr)
    print(f"# Output: {output_dir}", file=sys.stderr)

    for name, dim, extra in MODELS:
        cmd = f"python train.py --level {name} --dim {dim} {extra} {COMMON} --output {output_dir}/{name}"
        print(cmd)


if __name__ == '__main__':
    main()
