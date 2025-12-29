#!/bin/bash
# Elman Ablation Ladder Experiments
#
# Run all 7 levels to find the simplest level that matches Mamba2 (~3.92 loss).
# Each level adds one feature on top of the previous.
#
# Level 0: Stock Elman     - Basic tanh recurrence
# Level 1: Gated Elman     - + Input-dependent delta gate
# Level 2: Selective Elman - + compete×silu output
# Level 3: Diagonal Select - + Diagonal r_h (like Mamba2's diagonal A)
# Level 4: Log-Storage     - + Signed log storage for hidden state
# Level 5: Log-Compute     - + Full R via logsumexp
# Level 6: Triple R        - + R_delta modulation
#
# Usage:
#   ./train.elman_ladder_ablation.sh [level]
#   ./train.elman_ladder_ablation.sh        # Run level 0
#   ./train.elman_ladder_ablation.sh 3      # Run level 3
#   ./train.elman_ladder_ablation.sh all    # Run all levels sequentially

set -e

# Configuration
DATA="/home/erikg/data/pile/train_00.txt"
PARAMS="500m"
CHUNK_SIZE=512
BATCH_SIZE=32
GRAD_ACCUM=4
LR=3e-4
MAX_STEPS=50000
OUTPUT_BASE="outputs/elman_ladder"

# Parse arguments
LEVEL="${1:-0}"

run_level() {
    local lvl=$1
    echo "========================================"
    echo "Training Level $lvl"
    echo "========================================"

    python train_ladder.py \
        --level $lvl \
        --params $PARAMS \
        --data $DATA \
        --chunk_size $CHUNK_SIZE \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --lr $LR \
        --max_steps $MAX_STEPS \
        --output "${OUTPUT_BASE}/level${lvl}" \
        --cuda \
        --bf16 \
        --log_interval 10 \
        --save_interval 2000

    echo "Level $lvl complete!"
    echo ""
}

if [ "$LEVEL" = "all" ]; then
    echo "Running all 7 ablation levels..."
    for lvl in 0 1 2 3 4 5 6; do
        run_level $lvl
    done
    echo "All levels complete!"
else
    run_level $LEVEL
fi
