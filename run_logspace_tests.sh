#!/bin/bash
# Test all log-space levels: log_0, log_1, log_2
# 220M params, tiktoken p50k_base, 8-way DDP, 16 batch/GPU

set -e

DATA="/mnt/nvme2n1/erikg/pile.txt"
OUTPUT_BASE="outputs/logspace_test_$(date +%Y%m%d_%H%M%S)"
MAX_STEPS=1000  # Quick test runs
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
    --warmup_steps 100 \
    --lr 3e-4 \
    --log_interval 10 \
    --save_interval 500 \
    --bf16 \
    --ddp \
    --tbptt"

echo "=============================================="
echo "Log-Space Ladder Tests: log_0, log_1, log_2"
echo "=============================================="
echo "Data: $DATA"
echo "Params: 500M"
echo "Vocab: tiktoken p50k_base (~50k)"
echo "Batch/GPU: $BATCH_SIZE, 8 GPUs = $((BATCH_SIZE * 8)) effective"
echo "Max steps: $MAX_STEPS"
echo "Output: $OUTPUT_BASE"
echo "=============================================="

echo ""
echo ">>> Starting log_0 (Log-Space Polynomial)..."
echo ">>> $(date)"
torchrun --nproc_per_node=8 train_ladder.py \
    --level log_0 \
    --output ${OUTPUT_BASE}/log_0_polynomial \
    $COMMON_ARGS

echo ""
echo ">>> Starting log_1 (Log-Space Selective)..."
echo ">>> $(date)"
torchrun --nproc_per_node=8 train_ladder.py \
    --level log_1 \
    --output ${OUTPUT_BASE}/log_1_selective \
    $COMMON_ARGS

echo ""
echo ">>> Starting log_2 (Log-Space Diagonal Selective)..."
echo ">>> $(date)"
torchrun --nproc_per_node=8 train_ladder.py \
    --level log_2 \
    --output ${OUTPUT_BASE}/log_2_diag_selective \
    $COMMON_ARGS

echo ""
echo "=============================================="
echo "All log-space tests complete!"
echo ">>> $(date)"
echo "Results in: $OUTPUT_BASE"
echo "=============================================="
