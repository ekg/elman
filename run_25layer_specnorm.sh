#!/bin/bash
# 25-layer benchmark WITH spectral normalization enabled
# Same configs as before but with --r_h_mode spectral_norm

DATA=/home/erikg/elman/data/pile.txt
OUTDIR=/home/erikg/elman/benchmark_results/25layer_specnorm
mkdir -p $OUTDIR

# GPU 0: E1 (gated elman) - dim=900, depth=25
CUDA_VISIBLE_DEVICES=0 python train.py --level 1 --dim 900 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E1_gated > $OUTDIR/E1_gated.log 2>&1 &

# GPU 1: E44 (diagonal W) - dim=1410, depth=25
CUDA_VISIBLE_DEVICES=1 python train.py --level 44 --dim 1410 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E44_diagonal_w > $OUTDIR/E44_diagonal_w.log 2>&1 &

# GPU 2: E45 (pure accumulation) - dim=1410, depth=25
CUDA_VISIBLE_DEVICES=2 python train.py --level 45 --dim 1410 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E45_pure_accum > $OUTDIR/E45_pure_accum.log 2>&1 &

# GPU 3: E46 (no in proj) - dim=1410, depth=25
CUDA_VISIBLE_DEVICES=3 python train.py --level 46 --dim 1410 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E46_no_in_proj > $OUTDIR/E46_no_in_proj.log 2>&1 &

# GPU 4: E48 (no projections) - dim=1990, depth=25
CUDA_VISIBLE_DEVICES=4 python train.py --level 48 --dim 1990 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E48_no_proj > $OUTDIR/E48_no_proj.log 2>&1 &

# GPU 5: E51 (no self-gate) - dim=1150, depth=25
CUDA_VISIBLE_DEVICES=5 python train.py --level 51 --dim 1150 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E51_no_self_gate > $OUTDIR/E51_no_self_gate.log 2>&1 &

# GPU 6: E52 (quadratic gate) - dim=1150, depth=25
CUDA_VISIBLE_DEVICES=6 python train.py --level 52 --dim 1150 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E52_quadratic > $OUTDIR/E52_quadratic.log 2>&1 &

# GPU 7: E53 (sigmoid gate) - dim=1150, depth=25
CUDA_VISIBLE_DEVICES=7 python train.py --level 53 --dim 1150 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E53_sigmoid > $OUTDIR/E53_sigmoid.log 2>&1 &

wait
echo "First batch complete (8 models). Starting second batch..."

# GPU 0: E43 (scalar decay) - dim=1410, depth=25
CUDA_VISIBLE_DEVICES=0 python train.py --level 43 --dim 1410 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E43_scalar_decay > $OUTDIR/E43_scalar_decay.log 2>&1 &

# GPU 1: E56 (concat elman) - dim=890, depth=25
CUDA_VISIBLE_DEVICES=1 python train.py --level 56 --dim 890 --depth 25 --data $DATA \
    --r_h_mode spectral_norm \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E56_concat > $OUTDIR/E56_concat.log 2>&1 &

wait
echo "All spectral norm benchmarks complete!"
