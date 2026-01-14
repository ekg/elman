#!/bin/bash
# Spectral radius benchmark: E57 (learned), E33+specnorm, E42+specnorm
# Compare against E1+specnorm baseline

DATA=/home/erikg/elman/data/pile.txt
OUTDIR=/home/erikg/elman/benchmark_results/specradius_100m
mkdir -p $OUTDIR

# GPU 0: E1 + spectral_norm (baseline from previous run, for reference)
CUDA_VISIBLE_DEVICES=0 python train.py --level 1 --dim 900 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E1_specnorm > $OUTDIR/E1_specnorm.log 2>&1 &

# GPU 1: E57 (learned spectral radius)
CUDA_VISIBLE_DEVICES=1 python train.py --level 57 --dim 900 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E57_learned > $OUTDIR/E57_learned.log 2>&1 &

# GPU 2: E33 + spectral_norm (self-gate + specnorm)
CUDA_VISIBLE_DEVICES=2 python train.py --level 33 --dim 1000 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E33_specnorm > $OUTDIR/E33_specnorm.log 2>&1 &

# GPU 3: E42 + spectral_norm (linear tied + specnorm)
CUDA_VISIBLE_DEVICES=3 python train.py --level 42 --dim 1150 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E42_specnorm > $OUTDIR/E42_specnorm.log 2>&1 &

# GPU 4: E33 without spectral_norm (for comparison)
CUDA_VISIBLE_DEVICES=4 python train.py --level 33 --dim 1000 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E33_no_specnorm > $OUTDIR/E33_no_specnorm.log 2>&1 &

# GPU 5: E42 without spectral_norm (for comparison)
CUDA_VISIBLE_DEVICES=5 python train.py --level 42 --dim 1150 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E42_no_specnorm > $OUTDIR/E42_no_specnorm.log 2>&1 &

wait
echo "All spectral radius benchmarks complete!"
