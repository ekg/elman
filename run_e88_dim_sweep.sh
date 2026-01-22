#!/bin/bash
# E88 Dimension Sweep: Does wider dimension help at ~500M params?
# Tests dim from 1536 to 3072 with adjusted depth to keep params constant

set -e

OUTPUT_DIR="benchmark_results/e88_scaling"
mkdir -p "$OUTPUT_DIR"

echo "Starting E88 Dimension Sweep at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run all 7 dimension configs in parallel on 7 GPUs
echo "=== Running dimension sweep (7 configs on 7 GPUs) ==="

CUDA_VISIBLE_DEVICES=0 python train.py --level E88_dim1536 --dim 1536 --depth 50 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/dim1536 2>&1 | tee $OUTPUT_DIR/dim1536.log &

CUDA_VISIBLE_DEVICES=1 python train.py --level E88_dim1792 --dim 1792 --depth 38 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/dim1792 2>&1 | tee $OUTPUT_DIR/dim1792.log &

CUDA_VISIBLE_DEVICES=2 python train.py --level E88_dim2048 --dim 2048 --depth 28 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/dim2048 2>&1 | tee $OUTPUT_DIR/dim2048.log &

CUDA_VISIBLE_DEVICES=3 python train.py --level E88_dim2304 --dim 2304 --depth 22 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/dim2304 2>&1 | tee $OUTPUT_DIR/dim2304.log &

CUDA_VISIBLE_DEVICES=4 python train.py --level E88_dim2560 --dim 2560 --depth 18 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/dim2560 2>&1 | tee $OUTPUT_DIR/dim2560.log &

CUDA_VISIBLE_DEVICES=5 python train.py --level E88_dim2816 --dim 2816 --depth 16 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/dim2816 2>&1 | tee $OUTPUT_DIR/dim2816.log &

CUDA_VISIBLE_DEVICES=6 python train.py --level E88_dim3072 --dim 3072 --depth 14 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/dim3072 2>&1 | tee $OUTPUT_DIR/dim3072.log &

echo "Waiting for dimension sweep to complete..."
wait

echo ""
echo "=== Dimension Sweep Complete at $(date) ==="
echo ""

# Extract results
echo "Results (sorted by loss):"
echo "========================="
printf "%-12s %6s %6s %8s %10s\n" "Config" "dim" "depth" "loss" "tok/s"
echo "----------------------------------------------"

for dir in $OUTPUT_DIR/dim*/level*; do
    if [ -d "$dir" ]; then
        config=$(echo $dir | sed 's|.*/\(dim[0-9]*\)/level.*|\1|')
        dim=$(echo $config | sed 's/dim//')
        # Get depth from args.json
        depth=$(grep '"depth"' $dir/args.json 2>/dev/null | grep -oE '[0-9]+' || echo "?")
        latest=$(ls -t $dir/checkpoint_*.pt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            loss=$(basename $latest | sed 's/.*loss_\([0-9.]*\)\.pt/\1/')
            log="$OUTPUT_DIR/${config}.log"
            tps=$(grep 'tok/s' $log 2>/dev/null | tail -1 | grep -oE 'tok/s [0-9]+' | grep -oE '[0-9]+' || echo "N/A")
            printf "%-12s %6s %6s %8s %10s\n" "$config" "$dim" "$depth" "$loss" "$tps"
        fi
    fi
done | sort -t' ' -k4 -n
