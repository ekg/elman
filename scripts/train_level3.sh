#!/bin/bash
# Train Level 3 (Diagonal Selective) model
#
# Usage: ./scripts/train_level3.sh /path/to/data.txt

DATA_PATH="${1:-/path/to/training/data.txt}"

python train.py \
    --data "$DATA_PATH" \
    --level 3 \
    --params 100m \
    --batch_size 16 \
    --chunk_size 512 \
    --lr 1e-4 \
    --grad_accum 4 \
    --steps 50000 \
    --bf16 \
    --output ./output
