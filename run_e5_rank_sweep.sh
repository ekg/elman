#!/bin/bash
# E5 rank/dim sweep - 50M params, 1000 steps, different configurations
# All use FUSED kernel (cuBLAS)

DATA="/home/erikg/elman/data/fineweb_100mb.txt"
STEPS=1000
BATCH=32
CHUNK=512
LR=3e-4
LOG_INT=50
OUT_DIR="benchmark_results/e5_rank_sweep_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUT_DIR

echo "=== E5 Rank/Dim Sweep ==="
echo "Data: $DATA"
echo "Steps: $STEPS"
echo "Batch: $BATCH x Chunk: $CHUNK = $((BATCH * CHUNK)) tokens/step"
echo "Output: $OUT_DIR"
echo ""

# Config format: dim,rank,depth (all ~50M params)
# dim=512, rank=256, depth=63 -> 49.7M
# dim=768, rank=192, depth=56 -> 49.8M
# dim=1024, rank=192, depth=42 -> 49.9M
# dim=1536, rank=256, depth=21 -> 50.0M

# Run on GPUs 0-3 in parallel
echo "Starting E5 variants..."

# Config 1: dim=512, rank=256, depth=63
CUDA_VISIBLE_DEVICES=0 python -u benchmark_e5_backends.py \
    --data $DATA --dim 512 --rank 256 --depth 63 \
    --batch_size $BATCH --chunk_size $CHUNK --lr $LR \
    --steps $STEPS --log_every $LOG_INT --backends fused \
    > $OUT_DIR/e5_d512_r256.log 2>&1 &
P0=$!

# Config 2: dim=768, rank=192, depth=56
CUDA_VISIBLE_DEVICES=1 python -u benchmark_e5_backends.py \
    --data $DATA --dim 768 --rank 192 --depth 56 \
    --batch_size $BATCH --chunk_size $CHUNK --lr $LR \
    --steps $STEPS --log_every $LOG_INT --backends fused \
    > $OUT_DIR/e5_d768_r192.log 2>&1 &
P1=$!

# Config 3: dim=1024, rank=192, depth=42
CUDA_VISIBLE_DEVICES=2 python -u benchmark_e5_backends.py \
    --data $DATA --dim 1024 --rank 192 --depth 42 \
    --batch_size $BATCH --chunk_size $CHUNK --lr $LR \
    --steps $STEPS --log_every $LOG_INT --backends fused \
    > $OUT_DIR/e5_d1024_r192.log 2>&1 &
P2=$!

# Config 4: dim=1536, rank=256, depth=21
CUDA_VISIBLE_DEVICES=3 python -u benchmark_e5_backends.py \
    --data $DATA --dim 1536 --rank 256 --depth 21 \
    --batch_size $BATCH --chunk_size $CHUNK --lr $LR \
    --steps $STEPS --log_every $LOG_INT --backends fused \
    > $OUT_DIR/e5_d1536_r256.log 2>&1 &
P3=$!

echo "PIDs: $P0 $P1 $P2 $P3"
echo "Waiting for completion..."

wait $P0 $P1 $P2 $P3

echo ""
echo "=== RESULTS ==="
echo ""
printf "%-20s %12s %12s %12s\n" "Config" "Params" "Final Loss" "tok/s"
echo "------------------------------------------------------------"
for f in $OUT_DIR/*.log; do
    config=$(basename $f .log)
    params=$(grep "Parameters:" $f | grep -oP '[\d,]+' | head -1)
    loss=$(grep "Final loss:" $f | grep -oP '[\d.]+' | head -1)
    toks=$(grep "Avg tok/s:" $f | grep -oP '[\d,]+' | head -1)
    printf "%-20s %12s %12s %12s\n" "$config" "$params" "$loss" "$toks"
done

echo ""
echo "Done! Results in $OUT_DIR"
