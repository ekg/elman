#!/bin/bash
# Elman Ladder Level 0 (Stock Elman) - 1000 step test
# ~500M params: dim=2048, depth=30

set -e -x

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/nvme2n1/erikg/minlms/${TIMESTAMP}_ladder_level0_1k"
DATA_PATH="/mnt/nvme2n1/erikg/pile.txt"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "======================================================================="
echo "=== Elman Ladder Level 0 (Stock Elman) - 1000 steps ==="
echo "=== dim=2048, depth=30, ~500M params ==="
echo "======================================================================="

/home/erikg/micromamba/envs/mingru/bin/torchrun --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py --data "$DATA_PATH" --output "$OUTPUT_DIR" --tokenizer tiktoken --tiktoken_encoding p50k_base --dim 2048 --depth 30 --expansion_factor 1.0 --ff_mult 0.0 --dropout 0.0 --elman_ladder_level 0 --compete_n_groups 32 --no-tbptt --chunk_size 512 --batch_size 16 --grad_accum 1 --train_steps 1000 --lr 0.0006 --weight_decay 0.1 --grad_clip 1.0 --save_every 500 --keep_checkpoints 3 --ddp --ddp-find-unused --cuda --bf16 2>&1 | tee "logs/ladder_level0_1k_${TIMESTAMP}.log"

echo "Level 0 complete!"
