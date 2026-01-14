#!/bin/bash
# Learned Radii Benchmark
# Compare E58 (per-dim learned) vs E57 (scalar learned) vs E1 (fixed)
# Also test with auto r_h_mode (which now defaults to spectral_norm for these)

DATA=/home/erikg/elman/data/pile.txt
OUTDIR=/home/erikg/elman/benchmark_results/learned_radii
mkdir -p $OUTDIR

# GPU 0: E1 with auto (will use spectral_norm)
CUDA_VISIBLE_DEVICES=0 python train.py --level 1 --dim 900 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E1_auto > $OUTDIR/E1_auto.log 2>&1 &

# GPU 1: E57 (scalar learned radius)
CUDA_VISIBLE_DEVICES=1 python train.py --level 57 --dim 900 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E57_scalar > $OUTDIR/E57_scalar.log 2>&1 &

# GPU 2: E58 (per-dimension learned radii)
CUDA_VISIBLE_DEVICES=2 python train.py --level 58 --dim 900 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E58_perdim > $OUTDIR/E58_perdim.log 2>&1 &

# GPU 3: E33 with auto (will use spectral_norm)
CUDA_VISIBLE_DEVICES=3 python train.py --level 33 --dim 1000 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E33_auto > $OUTDIR/E33_auto.log 2>&1 &

# GPU 4: E42 with auto (will use spectral_norm)
CUDA_VISIBLE_DEVICES=4 python train.py --level 42 --dim 1150 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E42_auto > $OUTDIR/E42_auto.log 2>&1 &

# GPU 5: E44 with auto (will use none - diagonal W)
CUDA_VISIBLE_DEVICES=5 python train.py --level 44 --dim 1410 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/E44_auto > $OUTDIR/E44_auto.log 2>&1 &

# GPU 6: Mamba2 baseline
CUDA_VISIBLE_DEVICES=6 python train.py --level mamba2 --dim 800 --depth 25 --data $DATA \
    --batch_size 16 --chunk_size 512 --lr 1e-4 --warmup_steps 1000 \
    --train_minutes 10 --log_every 50 --seed 42 \
    --output $OUTDIR/Mamba2 > $OUTDIR/Mamba2.log 2>&1 &

wait
echo "All learned radii benchmarks complete!"
