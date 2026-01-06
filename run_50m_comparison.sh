#!/bin/bash
# 50M byte-level comparison: E0, E1, E2, E3, Mamba2 on pile.txt

mkdir -p output/comparison_50m

# E0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python train_ladder.py \
    --level 0 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 2000 --log_interval 100 \
    --output output/comparison_50m/e0 --no-ddp 2>&1 | tee output/comparison_50m/e0.log &

# E1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python train_ladder.py \
    --level 1 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 2000 --log_interval 100 \
    --output output/comparison_50m/e1 --no-ddp 2>&1 | tee output/comparison_50m/e1.log &

# E2 on GPU 2
CUDA_VISIBLE_DEVICES=2 python train_ladder.py \
    --level 2 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 2000 --log_interval 100 \
    --output output/comparison_50m/e2 --no-ddp 2>&1 | tee output/comparison_50m/e2.log &

# E3 on GPU 3
CUDA_VISIBLE_DEVICES=3 python train_ladder.py \
    --level 3 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 2000 --log_interval 100 \
    --output output/comparison_50m/e3 --no-ddp 2>&1 | tee output/comparison_50m/e3.log &

# Mamba2 on GPU 4
CUDA_VISIBLE_DEVICES=4 python train_ladder.py \
    --level mamba2 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 2000 --log_interval 100 \
    --output output/comparison_50m/mamba2 --no-ddp 2>&1 | tee output/comparison_50m/mamba2.log &

wait
echo "All done!"
