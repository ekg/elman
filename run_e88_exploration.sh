#!/bin/bash
# E88 Comprehensive Exploration
# Testing: convolutions, deeper models, expansion factors
# Using GPUs 3-7 (avoiding 0-2 which may be occupied)

set -e

OUTPUT_DIR="benchmark_results/e88_exploration"
mkdir -p "$OUTPUT_DIR"

echo "Starting E88 Exploration at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Run all experiments in parallel on GPUs 1-7 (avoiding GPU 0)
echo ""
echo "=== Running all experiments in parallel on GPUs 1-7 ==="

# GPU 1: Baseline (E88 ungated, no conv - our best)
CUDA_VISIBLE_DEVICES=1 python train.py --level E88 --dim 1792 --depth 38 --n_heads 56 --n_state 32 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/baseline 2>&1 | tee $OUTPUT_DIR/baseline.log &

# GPU 2: With short convolutions (FLA-GDN style)
CUDA_VISIBLE_DEVICES=2 python train.py --level E88_conv_silu --dim 1792 --depth 38 --n_heads 56 --n_state 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/with_conv 2>&1 | tee $OUTPUT_DIR/with_conv.log &

# GPU 3: depth=48, dim=1536 (~500M)
CUDA_VISIBLE_DEVICES=3 python train.py --level E88 --dim 1536 --depth 48 --n_heads 48 --n_state 32 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/d48_dim1536 2>&1 | tee $OUTPUT_DIR/d48_dim1536.log &

# GPU 4: depth=56, dim=1408 (~500M)
CUDA_VISIBLE_DEVICES=4 python train.py --level E88 --dim 1408 --depth 56 --n_heads 44 --n_state 32 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/d56_dim1408 2>&1 | tee $OUTPUT_DIR/d56_dim1408.log &

# GPU 5: depth=64, dim=1280 (~500M)
CUDA_VISIBLE_DEVICES=5 python train.py --level E88 --dim 1280 --depth 64 --n_heads 40 --n_state 32 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/d64_dim1280 2>&1 | tee $OUTPUT_DIR/d64_dim1280.log &

# GPU 6: n_state=24 at depth=48 (more heads, smaller state)
CUDA_VISIBLE_DEVICES=6 python train.py --level E88 --dim 1664 --depth 48 --n_heads 69 --n_state 24 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n24_d48 2>&1 | tee $OUTPUT_DIR/n24_d48.log &

# GPU 7: n_state=48 at depth=28 (fewer heads, larger state)
CUDA_VISIBLE_DEVICES=7 python train.py --level E88 --dim 1920 --depth 28 --n_heads 40 --n_state 48 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n48_d28 2>&1 | tee $OUTPUT_DIR/n48_d28.log &

echo "Waiting for all experiments to complete..."
wait

echo ""
echo "=== All Rounds Complete at $(date) ==="
echo ""

# Extract results
echo "Results (sorted by loss):"
echo "========================="
printf "%-20s %8s %10s %s\n" "Config" "loss" "tok/s" "notes"
echo "------------------------------------------------------------"

for dir in $OUTPUT_DIR/*/level*; do
    if [ -d "$dir" ]; then
        config=$(dirname "$dir" | xargs basename)
        latest=$(ls -t "$dir"/checkpoint_*.pt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            loss=$(basename $latest | sed 's/.*loss_\([0-9.]*\)\.pt/\1/')
            log="$OUTPUT_DIR/${config}.log"
            tps=$(grep 'tok/s' $log 2>/dev/null | tail -1 | grep -oE 'tok/s [0-9]+' | grep -oE '[0-9]+' || echo "N/A")
            printf "%-20s %8s %10s\n" "$config" "$loss" "$tps"
        fi
    fi
done | sort -t' ' -k2 -n

echo ""
echo "Results saved to $OUTPUT_DIR/"
