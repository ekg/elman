#!/bin/bash
# Elman Ablation Ladder - 1000 Step Quick Tests (8 GPU DDP)
#
# Run all 7 levels for 1000 steps each to compare training dynamics.
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
#   ./train.elman_ladder_1k.sh [level]
#   ./train.elman_ladder_1k.sh        # Run level 0
#   ./train.elman_ladder_1k.sh 3      # Run level 3
#   ./train.elman_ladder_1k.sh all    # Run all levels sequentially

set -e

# Configuration for 1k tests
DATA="/mnt/nvme2n1/erikg/pile.txt"
PARAMS="500m"
CHUNK_SIZE=512
BATCH_SIZE=32
GRAD_ACCUM=4
LR=3e-4
MAX_STEPS=1000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_elman_ladder_1k"

# DDP settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
NUM_GPUS=8

mkdir -p "${OUTPUT_BASE}"
mkdir -p logs

# Parse arguments
LEVEL="${1:-0}"

run_level() {
    local lvl=$1
    echo "========================================"
    echo "Training Level $lvl - 1000 steps (8 GPUs)"
    echo "========================================"

    /home/erikg/micromamba/envs/mingru/bin/torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_ladder.py \
        --level $lvl \
        --params $PARAMS \
        --data $DATA \
        --chunk_size $CHUNK_SIZE \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --lr $LR \
        --max_steps $MAX_STEPS \
        --output "${OUTPUT_BASE}/level${lvl}" \
        --ddp \
        --bf16 \
        --log_interval 10 \
        --save_interval 500 \
        2>&1 | tee "logs/elman_ladder_level${lvl}_${TIMESTAMP}.log"

    echo "Level $lvl complete!"
    echo ""
}

if [ "$LEVEL" = "all" ]; then
    echo "Running all 7 ablation levels (1000 steps each, 8 GPUs)..."
    echo "Output: ${OUTPUT_BASE}"
    echo ""
    for lvl in 0 1 2 3 4 5 6; do
        run_level $lvl
    done
    echo "All levels complete!"
    echo ""
    echo "Summary of final losses:"
    for lvl in 0 1 2 3 4 5 6; do
        LOGFILE="logs/elman_ladder_level${lvl}_${TIMESTAMP}.log"
        if [ -f "$LOGFILE" ]; then
            FINAL_LOSS=$(grep "Step  1000" "$LOGFILE" | grep -oP 'Loss \K[0-9.]+' | tail -1)
            echo "Level $lvl: $FINAL_LOSS"
        fi
    done
else
    run_level $LEVEL
fi
