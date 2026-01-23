#!/bin/bash
# E88 Gating Ablation: Compare ungated vs sigmoid vs SiLU gating
# Best config: dim=1792, depth=38, n_heads=56, n_state=32
# Using GPUs 3-7 (avoiding 0-2 which are occupied)

set -e

OUTPUT_DIR="benchmark_results/e88_gating_ablation"
mkdir -p "$OUTPUT_DIR"

echo "Starting E88 Gating Ablation at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run 5 configs in parallel on GPUs 3-7
echo "=== Running E88 gating ablation (5 configs on GPUs 3-7) ==="

# 1. E88 ungated (baseline - achieved 1.44 loss in dim sweep)
CUDA_VISIBLE_DEVICES=3 python train.py --level E88_dim1792 --dim 1792 --depth 38 --n_heads 56 --n_state 32 --use_gate 0 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/e88_ungated 2>&1 | tee $OUTPUT_DIR/e88_ungated.log &

# 2. E88 with sigmoid gating (explicitly set n_heads, n_state, use_gate, gate_activation)
CUDA_VISIBLE_DEVICES=4 python train.py --level E88_gated_sigmoid --dim 1792 --depth 38 --n_heads 56 --n_state 32 --use_gate 1 --gate_activation sigmoid --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/e88_sigmoid 2>&1 | tee $OUTPUT_DIR/e88_sigmoid.log &

# 3. E88 with SiLU gating (explicitly set gate_activation silu)
CUDA_VISIBLE_DEVICES=5 python train.py --level E88_gated_silu --dim 1792 --depth 38 --n_heads 56 --n_state 32 --use_gate 1 --gate_activation silu --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/e88_silu 2>&1 | tee $OUTPUT_DIR/e88_silu.log &

# 4. FLA-GDN baseline
CUDA_VISIBLE_DEVICES=6 python train.py --level fla-gdn --dim 2304 --depth 20 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/fla-gdn 2>&1 | tee $OUTPUT_DIR/fla-gdn.log &

# 5. Mamba2 baseline
CUDA_VISIBLE_DEVICES=7 python train.py --level mamba2 --dim 1600 --depth 32 --data data/pile.txt --batch_size 32 --chunk_size 512 --lr 3e-4 --warmup_steps 100 --seed 42 --bf16 --train_minutes 10 --output $OUTPUT_DIR/mamba2 2>&1 | tee $OUTPUT_DIR/mamba2.log &

echo "Waiting for all jobs to complete..."
wait

echo ""
echo "=== Gating Ablation Complete at $(date) ==="
echo ""

# Extract results
echo "Results (sorted by loss):"
echo "========================="
printf "%-20s %8s %10s %10s\n" "Config" "loss" "tok/s" "params"
echo "-----------------------------------------------------"

for dir in $OUTPUT_DIR/*/level*; do
    if [ -d "$dir" ]; then
        config=$(echo $dir | sed 's|.*/\([^/]*\)/level.*|\1|')
        latest=$(ls -t $dir/checkpoint_*.pt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            loss=$(basename $latest | sed 's/.*loss_\([0-9.]*\)\.pt/\1/')
            log="$OUTPUT_DIR/${config}.log"
            tps=$(grep 'tok/s' $log 2>/dev/null | tail -1 | grep -oE 'tok/s [0-9]+' | grep -oE '[0-9]+' || echo "N/A")
            params=$(grep 'Parameters:' $log 2>/dev/null | head -1 | grep -oE '[0-9.]+M' || echo "N/A")
            printf "%-20s %8s %10s %10s\n" "$config" "$loss" "$tps" "$params"
        fi
    fi
done | sort -t' ' -k2 -n

echo ""
echo "Results saved to $OUTPUT_DIR/"
