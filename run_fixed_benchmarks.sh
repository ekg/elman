#!/bin/bash

# Run E51, E52, E53, E56 benchmarks in parallel with CUDA integration

DATA=/home/erikg/elman/data/pile.txt
OUTDIR=/home/erikg/elman/benchmark_results/100m_cuda

mkdir -p $OUTDIR

echo "Starting benchmarks at $(date)"

# E51
CUDA_VISIBLE_DEVICES=0 python train.py --level 51 --params 100m --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E51_no_self_gate 2>&1 | tee $OUTDIR/E51_no_self_gate.log &
PID1=$!

# E52
CUDA_VISIBLE_DEVICES=1 python train.py --level 52 --params 100m --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E52_quadratic_gate 2>&1 | tee $OUTDIR/E52_quadratic_gate.log &
PID2=$!

# E53
CUDA_VISIBLE_DEVICES=2 python train.py --level 53 --params 100m --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E53_sigmoid_gate 2>&1 | tee $OUTDIR/E53_sigmoid_gate.log &
PID3=$!

# E56
CUDA_VISIBLE_DEVICES=3 python train.py --level 56 --params 100m --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E56_concat_elman 2>&1 | tee $OUTDIR/E56_concat_elman.log &
PID4=$!

echo "Running PIDs: $PID1 $PID2 $PID3 $PID4"
wait $PID1 $PID2 $PID3 $PID4

echo "All benchmarks complete at $(date)"
