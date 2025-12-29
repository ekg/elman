#!/bin/bash
# Compare StockElman (level 0) vs LogSpacePolynomial (log_0) on The Pile
# 500M params, tiktoken p50k_base (50k vocab), 8-way DDP, 16 batch/GPU

set -e

DATA="/mnt/nvme2n1/erikg/pile.txt"
OUTPUT_BASE="outputs/comparison_$(date +%Y%m%d_%H%M%S)"
MAX_STEPS=3000
BATCH_SIZE=16
CHUNK_SIZE=512

# Common args
COMMON_ARGS="--data $DATA \
    --params 500m \
    --tokenizer tiktoken \
    --tokenizer_name p50k_base \
    --batch_size $BATCH_SIZE \
    --chunk_size $CHUNK_SIZE \
    --max_steps $MAX_STEPS \
    --warmup_steps 200 \
    --lr 3e-4 \
    --log_interval 10 \
    --save_interval 500 \
    --bf16 \
    --ddp \
    --tbptt"

echo "=============================================="
echo "Elman Ladder Comparison: Stock vs Log-Space Polynomial"
echo "=============================================="
echo "Data: $DATA"
echo "Params: 500M"
echo "Vocab: tiktoken p50k_base (~50k)"
echo "Batch/GPU: $BATCH_SIZE, 8 GPUs = $(($BATCH_SIZE * 8)) effective"
echo "Chunk size: $CHUNK_SIZE"
echo "Max steps: $MAX_STEPS"
echo "Output: $OUTPUT_BASE"
echo "=============================================="

# Run Level 0 (Stock Elman)
echo ""
echo ">>> Starting Level 0 (Stock Elman)..."
torchrun --nproc_per_node=8 train_ladder.py \
    --level 0 \
    --output ${OUTPUT_BASE}/level0_stock \
    $COMMON_ARGS

# Run Log Level 0 (Log-Space Polynomial)
echo ""
echo ">>> Starting Log Level 0 (Log-Space Polynomial)..."
torchrun --nproc_per_node=8 train_ladder.py \
    --level log_0 \
    --output ${OUTPUT_BASE}/log_0_polynomial \
    $COMMON_ARGS

echo ""
echo "=============================================="
echo "Comparison complete!"
echo "Results in: $OUTPUT_BASE"
echo "=============================================="
