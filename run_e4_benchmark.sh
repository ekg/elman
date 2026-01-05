#!/bin/bash
# E4 vs E1 vs Mamba2 Benchmark
# 1k steps, pile.txt, byte tokens, ~50M params

set -e

OUTPUT_BASE="output/e4_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

DATA="data/pile.txt"
STEPS=1000
PARAMS="50m"
BATCH=32
CHUNK=512
TOKENIZER="byte"

echo "=============================================="
echo "E4 vs E1 vs Mamba2 Benchmark"
echo "=============================================="
echo "Output: $OUTPUT_BASE"
echo "Steps: $STEPS, Params: $PARAMS, Batch: $BATCH, Chunk: $CHUNK"
echo ""

# Define experiments: GPU MODEL_ARGS OUTPUT_NAME
# E4 with larger hidden states trades depth for width
declare -a EXPERIMENTS=(
    # Baselines
    "0|--level 0|e0_stock"                           # E0: 19 layers, d_inner=768
    "1|--level 1|e1_baseline"                        # E1: 21 layers, d_inner=768

    # E4 with increasing hidden state sizes
    "2|--level 4 --expansion 1.5|e4_1.5x"            # d_inner=768, deeper
    "3|--level 4 --expansion 2.0|e4_2x"              # d_inner=1024, 17 layers
    "4|--level 4 --expansion 3.0|e4_3x"              # d_inner=1536, 10 layers
    "5|--level 4 --expansion 4.0|e4_4x"              # d_inner=2048, 7 layers

    # Mamba2
    "6|--level mamba2|mamba2"                        # Reference: dim=672, 18 layers
)

# Launch all experiments in parallel
PIDS=()
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r GPU ARGS NAME <<< "$exp"

    LOG="$OUTPUT_BASE/${NAME}.log"

    echo "Launching $NAME on GPU $GPU..."

    CUDA_VISIBLE_DEVICES=$GPU python train_ladder.py \
        $ARGS \
        --params $PARAMS \
        --data "$DATA" \
        --tokenizer $TOKENIZER \
        --batch_size $BATCH \
        --chunk_size $CHUNK \
        --max_steps $STEPS \
        --output "$OUTPUT_BASE/$NAME" \
        --log_interval 100 \
        --no-ddp \
        > "$LOG" 2>&1 &

    PIDS+=($!)
    sleep 0.5
done

echo ""
echo "All ${#EXPERIMENTS[@]} experiments launched!"
echo "Waiting for completion..."
echo ""

# Wait and collect results
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    IFS='|' read -r GPU ARGS NAME <<< "${EXPERIMENTS[$i]}"

    if wait $PID; then
        echo "✓ $NAME completed"
    else
        echo "✗ $NAME FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
echo "Results Summary"
echo "=============================================="

# Parse results from logs
printf "%-15s %8s %8s %10s %8s %8s %12s\n" "Model" "Loss" "PPL" "Tok/s" "Layers" "d_inner" "Params"
printf "%-15s %8s %8s %10s %8s %8s %12s\n" "-----" "----" "---" "-----" "------" "-------" "------"

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r GPU ARGS NAME <<< "$exp"
    LOG="$OUTPUT_BASE/${NAME}.log"

    if [ -f "$LOG" ]; then
        # Extract final metrics from log
        FINAL_LINE=$(grep "Step.*1000" "$LOG" | tail -1)
        LOSS=$(echo "$FINAL_LINE" | grep -oP 'Loss \K[0-9.]+' || echo "N/A")
        PPL=$(echo "$FINAL_LINE" | grep -oP 'PPL \K[0-9.]+' || echo "N/A")
        TOKS=$(echo "$FINAL_LINE" | grep -oP 'Tok/s \K[0-9,]+' || echo "N/A")

        # Get model info
        LAYERS=$(grep -oP 'depth=\K[0-9]+' "$LOG" | head -1 || echo "-")
        PARAMS_ACTUAL=$(grep "Model Parameters" "$LOG" | grep -oP '[0-9,]+' | head -1 || echo "-")

        # Get d_inner from layer info
        D_INNER=$(grep -oP 'd_inner=\K[0-9]+' "$LOG" | head -1 || echo "-")
        if [ "$D_INNER" = "-" ]; then
            D_INNER=$(grep "Layer 0:" "$LOG" | grep -oP '[0-9]+(?= params)' | head -1 || echo "-")
        fi

        printf "%-15s %8s %8s %10s %8s %8s %12s\n" "$NAME" "$LOSS" "$PPL" "$TOKS" "$LAYERS" "$D_INNER" "$PARAMS_ACTUAL"
    else
        printf "%-15s %8s %8s %10s %8s %8s %12s\n" "$NAME" "ERROR" "-" "-" "-" "-" "-"
    fi
done

echo ""
echo "Logs saved to: $OUTPUT_BASE/"
echo "Failed: $FAILED/${#EXPERIMENTS[@]}"

# Copy results to a summary file
{
    echo "E4 Benchmark Results"
    echo "===================="
    echo "Date: $(date)"
    echo "Steps: $STEPS, Batch: $BATCH, Chunk: $CHUNK"
    echo ""
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r GPU ARGS NAME <<< "$exp"
        LOG="$OUTPUT_BASE/${NAME}.log"
        if [ -f "$LOG" ]; then
            echo "=== $NAME ==="
            head -20 "$LOG"
            echo ""
            tail -10 "$LOG"
            echo ""
        fi
    done
} > "$OUTPUT_BASE/summary.txt"

echo "Summary written to: $OUTPUT_BASE/summary.txt"
