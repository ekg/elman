#!/bin/bash
# Train all 13 ladder levels at 500M scale
# 8-way DDP, 3k steps, batch 16, chunk 512

set -e

# Require CUDA kernels - no fallback to slow PyTorch
export ELMAN_REQUIRE_CUDA=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA="${1:-/mnt/nvme2n1/erikg/pile.txt}"
OUTPUT_BASE="${2:-/mnt/nvme2n1/erikg/minlms/elman_500m_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${3:-3000}"
BATCH_SIZE="${4:-2}"  # Per GPU, effective = 2*8 = 16
CHUNK_SIZE="${5:-512}"

# All 13 levels
LEVELS=(0 1 2 3 4 5 6 log_0 log_1 log_2 log_3 log_4 log_5)

echo "=============================================="
echo "Training 500M models on all 13 ladder levels"
echo "=============================================="
echo "Data: $DATA"
echo "Output: $OUTPUT_BASE"
echo "Steps: $MAX_STEPS"
echo "Batch: $BATCH_SIZE x 8 GPUs = $((BATCH_SIZE * 8))"
echo "Chunk: $CHUNK_SIZE"
echo ""

# Check data exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: Data file not found: $DATA"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

for LEVEL in "${LEVELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Training Level $LEVEL"
    echo "=============================================="

    OUTPUT_DIR="$OUTPUT_BASE/level_$LEVEL"

    # Skip if already completed
    if [ -f "$OUTPUT_DIR/DONE" ]; then
        echo "Skipping (already done): $OUTPUT_DIR"
        continue
    fi

    torchrun --nproc_per_node=8 train_ladder.py \
        --level "$LEVEL" \
        --params 500m \
        --data "$DATA" \
        --output "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --chunk_size "$CHUNK_SIZE" \
        --max_steps "$MAX_STEPS" \
        --warmup_steps 100 \
        --lr 3e-4 \
        --log_interval 10 \
        --save_interval 1000 \
        --ddp \
        --bf16 \
        --tbptt

    # Mark as done
    touch "$OUTPUT_DIR/DONE"

    echo "Level $LEVEL complete: $OUTPUT_DIR"
done

echo ""
echo "=============================================="
echo "All training complete!"
echo "=============================================="
echo "Results in: $OUTPUT_BASE"
