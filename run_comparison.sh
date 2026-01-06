#!/bin/bash
# Run E0-E3 + Mamba2 training comparison in parallel

mkdir -p output/comparison_50m

echo "Starting training comparison at $(date)"

# E0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python train_ladder.py \
    --level 0 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 1000 --log_interval 100 \
    --output output/comparison_50m/e0 --no-ddp \
    > output/comparison_50m/e0.log 2>&1 &

# E1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python train_ladder.py \
    --level 1 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 1000 --log_interval 100 \
    --output output/comparison_50m/e1 --no-ddp \
    > output/comparison_50m/e1.log 2>&1 &

# E2 on GPU 2
CUDA_VISIBLE_DEVICES=2 python train_ladder.py \
    --level 2 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 1000 --log_interval 100 \
    --output output/comparison_50m/e2 --no-ddp \
    > output/comparison_50m/e2.log 2>&1 &

# E3 on GPU 3
CUDA_VISIBLE_DEVICES=3 python train_ladder.py \
    --level 3 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 1000 --log_interval 100 \
    --output output/comparison_50m/e3 --no-ddp \
    > output/comparison_50m/e3.log 2>&1 &

echo "Started E0-E3, waiting 5s before starting Mamba2..."
sleep 5

# Mamba2 on GPU 4
CUDA_VISIBLE_DEVICES=4 python train_ladder.py \
    --level mamba2 --params 50m --data data/pile.txt --chunk_size 512 \
    --tokenizer byte --batch_size 32 --max_steps 1000 --log_interval 100 \
    --output output/comparison_50m/mamba2 --no-ddp \
    > output/comparison_50m/mamba2.log 2>&1 &

echo "All jobs started. Monitoring progress..."
wait

echo ""
echo "=== Final Results ==="
for model in e0 e1 e2 e3 mamba2; do
    echo "--- $model ---"
    tail -10 output/comparison_50m/$model.log 2>/dev/null | grep -E "Step|Complete|Final|Error"
done
