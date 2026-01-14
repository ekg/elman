#!/bin/bash

# Run E43-E48 benchmarks in parallel with CUDA integration

DATA=/home/erikg/elman/data/pile.txt
OUTDIR=/home/erikg/elman/benchmark_results/100m_cuda

mkdir -p $OUTDIR

echo "Starting E43-E48 benchmarks at $(date)"

# E43 (dim 768, depth 84 for ~99M params)
CUDA_VISIBLE_DEVICES=0 python train.py --level 43 --dim 768 --depth 84 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E43_scalar_decay 2>&1 | tee $OUTDIR/E43_scalar_decay.log &
PID1=$!

# E44 (dim 768, depth 84 for ~99M params)
CUDA_VISIBLE_DEVICES=1 python train.py --level 44 --dim 768 --depth 84 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E44_diagonal_w 2>&1 | tee $OUTDIR/E44_diagonal_w.log &
PID2=$!

# E45 (dim 768, depth 85 for ~100M params)
CUDA_VISIBLE_DEVICES=2 python train.py --level 45 --dim 768 --depth 85 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E45_pure_accumulation 2>&1 | tee $OUTDIR/E45_pure_accumulation.log &
PID3=$!

# E46 (dim 768, depth 84 for ~99M params)
CUDA_VISIBLE_DEVICES=3 python train.py --level 46 --dim 768 --depth 84 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E46_no_in_proj 2>&1 | tee $OUTDIR/E46_no_in_proj.log &
PID4=$!

# E48 (dim 1024, depth 95 for ~100M params)
CUDA_VISIBLE_DEVICES=4 python train.py --level 48 --dim 1024 --depth 95 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E48_no_projections 2>&1 | tee $OUTDIR/E48_no_projections.log &
PID5=$!

echo "Running PIDs: $PID1 $PID2 $PID3 $PID4 $PID5"
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo "All E43-E48 benchmarks complete at $(date)"
