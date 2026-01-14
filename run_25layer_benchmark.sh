#!/bin/bash
# 25-layer benchmark: ~4M params/layer, 100M total
# Models with simpler recurrence need larger hidden dims

DATA=/home/erikg/elman/data/pile.txt
OUTDIR=/home/erikg/elman/benchmark_results/25layer_100m
mkdir -p $OUTDIR

# GPU 0: E1 (gated elman) - dim=900, depth=25, ~101.5M
CUDA_VISIBLE_DEVICES=0 python train.py --level 1 --dim 900 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E1_gated 2>&1 | tee $OUTDIR/E1_gated.log &

# GPU 1: E44 (diagonal W) - dim=1410, depth=25, ~99.9M
CUDA_VISIBLE_DEVICES=1 python train.py --level 44 --dim 1410 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E44_diagonal_w 2>&1 | tee $OUTDIR/E44_diagonal_w.log &

# GPU 2: E45 (pure accumulation) - dim=1410, depth=25, ~99.8M
CUDA_VISIBLE_DEVICES=2 python train.py --level 45 --dim 1410 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E45_pure_accum 2>&1 | tee $OUTDIR/E45_pure_accum.log &

# GPU 3: E46 (no in proj) - dim=1410, depth=25, ~99.8M
CUDA_VISIBLE_DEVICES=3 python train.py --level 46 --dim 1410 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E46_no_in_proj 2>&1 | tee $OUTDIR/E46_no_in_proj.log &

# GPU 4: E48 (no projections) - dim=1990, depth=25, ~99.6M
CUDA_VISIBLE_DEVICES=4 python train.py --level 48 --dim 1990 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E48_no_proj 2>&1 | tee $OUTDIR/E48_no_proj.log &

# GPU 5: E51 (no self-gate) - dim=1150, depth=25, ~99.5M
CUDA_VISIBLE_DEVICES=5 python train.py --level 51 --dim 1150 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E51_no_self_gate 2>&1 | tee $OUTDIR/E51_no_self_gate.log &

# GPU 6: E52 (quadratic gate) - dim=1150, depth=25, ~99.5M
CUDA_VISIBLE_DEVICES=6 python train.py --level 52 --dim 1150 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E52_quadratic 2>&1 | tee $OUTDIR/E52_quadratic.log &

# GPU 7: E53 (sigmoid gate) - dim=1150, depth=25, ~99.5M
CUDA_VISIBLE_DEVICES=7 python train.py --level 53 --dim 1150 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E53_sigmoid 2>&1 | tee $OUTDIR/E53_sigmoid.log &

wait
echo "First batch complete. Starting second batch..."

# GPU 0: E43 (scalar decay) - dim=1410, depth=25, ~99.8M
CUDA_VISIBLE_DEVICES=0 python train.py --level 43 --dim 1410 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E43_scalar_decay 2>&1 | tee $OUTDIR/E43_scalar_decay.log &

# GPU 1: E56 (concat elman) - dim=890, depth=25, ~99.3M
CUDA_VISIBLE_DEVICES=1 python train.py --level 56 --dim 890 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E56_concat 2>&1 | tee $OUTDIR/E56_concat.log &

# GPU 2: Mamba2 baseline - dim=800, depth=25, ~99.6M
CUDA_VISIBLE_DEVICES=2 python train.py --model mamba2 --dim 800 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/Mamba2 2>&1 | tee $OUTDIR/Mamba2.log &

wait
echo "All benchmarks complete!"
