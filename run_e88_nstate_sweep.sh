#!/bin/bash
# E88 n_state sweep at optimal dim=1792, depth=38
# Finding: n_state=64 >> n_state=32, now testing around 64

set -e

OUTPUT_DIR="benchmark_results/e88_nstate_sweep"
mkdir -p "$OUTPUT_DIR"

echo "Starting E88 n_state sweep at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Testing n_state values: 48, 56, 72, 80, 96, 112, 128"
echo "All at dim=1792, depth=38 (~500M params)"
echo ""

# GPU 1: n_state=48 (n_heads = 1792/48 ≈ 37)
CUDA_VISIBLE_DEVICES=1 python train.py --level E88 --dim 1792 --depth 38 --n_heads 37 --n_state 48 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n48 2>&1 | tee $OUTPUT_DIR/n48.log &

# GPU 2: n_state=56 (n_heads = 1792/56 = 32)
CUDA_VISIBLE_DEVICES=2 python train.py --level E88 --dim 1792 --depth 38 --n_heads 32 --n_state 56 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n56 2>&1 | tee $OUTPUT_DIR/n56.log &

# GPU 3: n_state=72 (n_heads = 1792/72 ≈ 25)
CUDA_VISIBLE_DEVICES=3 python train.py --level E88 --dim 1792 --depth 38 --n_heads 25 --n_state 72 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n72 2>&1 | tee $OUTPUT_DIR/n72.log &

# GPU 4: n_state=80 (n_heads = 1792/80 ≈ 22)
CUDA_VISIBLE_DEVICES=4 python train.py --level E88 --dim 1792 --depth 38 --n_heads 22 --n_state 80 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n80 2>&1 | tee $OUTPUT_DIR/n80.log &

# GPU 5: n_state=96 (n_heads = 1792/96 ≈ 19)
CUDA_VISIBLE_DEVICES=5 python train.py --level E88 --dim 1792 --depth 38 --n_heads 19 --n_state 96 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n96 2>&1 | tee $OUTPUT_DIR/n96.log &

# GPU 6: n_state=112 (n_heads = 1792/112 = 16)
CUDA_VISIBLE_DEVICES=6 python train.py --level E88 --dim 1792 --depth 38 --n_heads 16 --n_state 112 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n112 2>&1 | tee $OUTPUT_DIR/n112.log &

# GPU 7: n_state=128 (n_heads = 1792/128 = 14)
CUDA_VISIBLE_DEVICES=7 python train.py --level E88 --dim 1792 --depth 38 --n_heads 14 --n_state 128 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/n128 2>&1 | tee $OUTPUT_DIR/n128.log &

echo "Waiting for all experiments to complete..."
wait

echo ""
echo "=== n_state sweep complete at $(date) ==="
echo ""

# Extract results
echo "Results (sorted by loss):"
echo "========================="
printf "%-10s %8s %8s %8s %10s\n" "n_state" "n_heads" "ratio" "loss" "tok/s"
echo "------------------------------------------------------------"

for log in $OUTPUT_DIR/n*.log; do
    config=$(basename $log .log)
    n_state=$(echo $config | sed 's/n//')

    # Extract n_heads from log
    n_heads=$(grep "n_heads" $log 2>/dev/null | head -1 | grep -oE 'n_heads=[0-9]+' | cut -d= -f2 || echo "N/A")

    # Extract final loss (last step logged)
    loss=$(grep "^step" $log 2>/dev/null | tail -1 | grep -oE 'loss [0-9.]+' | cut -d' ' -f2 || echo "N/A")

    # Extract throughput
    tps=$(grep "^step" $log 2>/dev/null | tail -1 | grep -oE 'tok/s [0-9]+' | cut -d' ' -f2 || echo "N/A")

    # Calculate ratio
    if [ "$n_heads" != "N/A" ] && [ "$n_state" != "N/A" ]; then
        ratio=$(echo "scale=2; $n_heads * $n_state / 1792" | bc)
    else
        ratio="N/A"
    fi

    printf "%-10s %8s %8s %8s %10s\n" "$n_state" "$n_heads" "$ratio" "$loss" "$tps"
done | sort -t' ' -k4 -n

echo ""
echo "Reference: n_state=64 achieved 1.44 loss"
echo "Target: FLA-GDN achieved 1.40 loss"
