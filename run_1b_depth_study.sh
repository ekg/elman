#!/bin/bash
# 1B parameter depth study: depth=6 vs depth=24 for E1, E33, E42, Mamba2
# 8 jobs on 8 GPUs, 1 hour each
# NO spectral norm (r_h_mode=none is now default)

cd /home/erikg/elman

# Kill existing training
pkill -9 -f "train.py" 2>/dev/null
sleep 2

echo "Launching 8 1B training runs (1 hour each)..."
echo "Depth 6: E1, E33, E42, Mamba2 on GPUs 0-3"
echo "Depth 24: E1, E33, E42, Mamba2 on GPUs 4-7"
echo "All using r_h_mode=none (no spectral norm)"

# Use python -u for unbuffered output
# Depth 6 (from scaling_config.py)
CUDA_VISIBLE_DEVICES=0 python -u train.py --data data/pile.txt --level 1 --dim 5888 --depth 6 --batch_size 48 --chunk_size 512 --train_minutes 60 --seed 42 --bf16 --log_every 50 --output /tmp/1b_E1_d6 > /tmp/1b_E1_d6.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u train.py --data data/pile.txt --level 33 --dim 6400 --depth 6 --batch_size 48 --chunk_size 512 --train_minutes 60 --seed 42 --bf16 --log_every 50 --output /tmp/1b_E33_d6 > /tmp/1b_E33_d6.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u train.py --data data/pile.txt --level 42 --dim 7680 --depth 6 --batch_size 48 --chunk_size 512 --train_minutes 60 --seed 42 --bf16 --log_every 50 --output /tmp/1b_E42_d6 > /tmp/1b_E42_d6.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u train.py --data data/pile.txt --level mamba2 --dim 5248 --depth 6 --batch_size 48 --chunk_size 512 --train_minutes 60 --seed 42 --bf16 --log_every 50 --output /tmp/1b_Mamba2_d6 > /tmp/1b_Mamba2_d6.log 2>&1 &

# Depth 24 (dims ~1/2 to maintain ~1B params)
CUDA_VISIBLE_DEVICES=4 python -u train.py --data data/pile.txt --level 1 --dim 2944 --depth 24 --batch_size 32 --chunk_size 512 --train_minutes 60 --seed 42 --bf16 --log_every 50 --output /tmp/1b_E1_d24 > /tmp/1b_E1_d24.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u train.py --data data/pile.txt --level 33 --dim 3200 --depth 24 --batch_size 32 --chunk_size 512 --train_minutes 60 --seed 42 --bf16 --log_every 50 --output /tmp/1b_E33_d24 > /tmp/1b_E33_d24.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -u train.py --data data/pile.txt --level 42 --dim 3840 --depth 24 --batch_size 32 --chunk_size 512 --train_minutes 60 --seed 42 --bf16 --log_every 50 --output /tmp/1b_E42_d24 > /tmp/1b_E42_d24.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -u train.py --data data/pile.txt --level mamba2 --dim 2624 --depth 24 --batch_size 32 --chunk_size 512 --train_minutes 60 --seed 42 --bf16 --log_every 50 --output /tmp/1b_Mamba2_d24 > /tmp/1b_Mamba2_d24.log 2>&1 &

sleep 5
echo ""
echo "Jobs launched:"
ps aux | grep "train.py" | grep -v grep | wc -l
echo "Check progress: tail -f /tmp/1b_E1_d6.log"
