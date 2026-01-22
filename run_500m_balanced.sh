#!/bin/bash
# Run 500M Balanced E88 Benchmark
# Uses all 8 GPUs in parallel

set -e

OUTPUT_DIR="benchmark_results/500m_balanced"
mkdir -p "$OUTPUT_DIR"

echo "Starting 500M Balanced Benchmark at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Round 1: Baselines + first 6 E88 configs (8 jobs on 8 GPUs)
echo "=== Round 1: Baselines + E88 configs ==="

CUDA_VISIBLE_DEVICES=0 python train.py --level mamba2 --dim 1600 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/mamba2 2>&1 | tee $OUTPUT_DIR/mamba2.log &

CUDA_VISIBLE_DEVICES=1 python train.py --level fla-gdn --dim 2304 --depth 20 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/fla-gdn 2>&1 | tee $OUTPUT_DIR/fla-gdn.log &

CUDA_VISIBLE_DEVICES=2 python train.py --level E88_b56n32 --dim 2176 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/E88_b56n32 2>&1 | tee $OUTPUT_DIR/E88_b56n32.log &

CUDA_VISIBLE_DEVICES=3 python train.py --level E88_b60n32 --dim 2048 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/E88_b60n32 2>&1 | tee $OUTPUT_DIR/E88_b60n32.log &

CUDA_VISIBLE_DEVICES=4 python train.py --level E88_b64n32 --dim 1920 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/E88_b64n32 2>&1 | tee $OUTPUT_DIR/E88_b64n32.log &

CUDA_VISIBLE_DEVICES=5 python train.py --level E88_b40n48 --dim 2048 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/E88_b40n48 2>&1 | tee $OUTPUT_DIR/E88_b40n48.log &

CUDA_VISIBLE_DEVICES=6 python train.py --level E88_b44n48 --dim 1792 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/E88_b44n48 2>&1 | tee $OUTPUT_DIR/E88_b44n48.log &

CUDA_VISIBLE_DEVICES=7 python train.py --level E88_b28n64 --dim 2176 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/E88_b28n64 2>&1 | tee $OUTPUT_DIR/E88_b28n64.log &

echo "Waiting for round 1 to complete..."
wait

# Round 2: Last E88 config
echo ""
echo "=== Round 2: Remaining E88 config ==="

CUDA_VISIBLE_DEVICES=0 python train.py --level E88_b32n64 --dim 1920 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/E88_b32n64 2>&1 | tee $OUTPUT_DIR/E88_b32n64.log &

wait

echo ""
echo "=== Benchmark Complete at $(date) ==="
echo ""

# Extract results
echo "Results summary:"
echo "================"
for log in $OUTPUT_DIR/*.log; do
    model=$(basename $log .log)
    # Get last loss value
    last_loss=$(grep "loss" $log 2>/dev/null | tail -1 | sed 's/.*loss \([0-9.]*\).*/\1/' || echo "N/A")
    # Get throughput
    throughput=$(grep "tok/s" $log 2>/dev/null | tail -1 | sed 's/.*tok\/s \([0-9]*\).*/\1/' || echo "N/A")
    # Get step count
    steps=$(grep "step" $log 2>/dev/null | tail -1 | sed 's/.*step *\([0-9]*\).*/\1/' || echo "N/A")
    echo "$model: loss=$last_loss, tok/s=$throughput, steps=$steps"
done
