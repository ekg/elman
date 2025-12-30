#!/bin/bash
#
# Run 500M parameter comparison across all Elman Ladder levels
#
# Each model trains for 3k steps with ~100k tokens per step
# Total: 300M tokens per model
#
# Usage:
#   ./run_500m_comparison.sh /path/to/training_data.txt
#
# Output:
#   outputs/500m_comparison/
#     level_0/steps.jsonl
#     level_1/steps.jsonl
#     ...
#     log_2/steps.jsonl
#     summary.json

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <data_path>"
    echo "Example: $0 /data/fineweb-edu/sample-10BT.txt"
    exit 1
fi

DATA_PATH="$1"
OUTPUT_BASE="outputs/500m_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

# Training configuration
PARAMS="500m"
MAX_STEPS=3000
BATCH_SIZE=32           # Per GPU
CHUNK_SIZE=512          # Sequence length
GRAD_ACCUM=4            # Accumulation steps
LOG_INTERVAL=10
SAVE_INTERVAL=500

# Tokens per step = batch_size * chunk_size * grad_accum * world_size
# With batch_size=32, chunk_size=512, grad_accum=4 = 32*512*4 = 65536 tokens/step
# 3000 steps * 65536 = ~196M tokens

# All levels to test
LEVELS=(
    "0"       # StockElman
    "1"       # GatedElman
    "2"       # SelectiveElman
    "3"       # DiagonalSelective
    "log_0"   # LogSpacePolynomial
    "log_1"   # LogSpaceSelective
    "log_2"   # LogSpaceDiagonalSelective
)

# Level names for display
declare -A LEVEL_NAMES
LEVEL_NAMES["0"]="StockElman"
LEVEL_NAMES["1"]="GatedElman"
LEVEL_NAMES["2"]="SelectiveElman"
LEVEL_NAMES["3"]="DiagonalSelective"
LEVEL_NAMES["log_0"]="LogSpacePolynomial"
LEVEL_NAMES["log_1"]="LogSpaceSelective"
LEVEL_NAMES["log_2"]="LogSpaceDiagonalSelective"

echo "=============================================================="
echo "500M Parameter Elman Ladder Comparison"
echo "=============================================================="
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Steps: $MAX_STEPS"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM accum = $(($BATCH_SIZE * $GRAD_ACCUM)) effective"
echo "Tokens per step: $(($BATCH_SIZE * $CHUNK_SIZE * $GRAD_ACCUM))"
echo "Total tokens: $(($BATCH_SIZE * $CHUNK_SIZE * $GRAD_ACCUM * $MAX_STEPS))"
echo ""
echo "Levels to test:"
for level in "${LEVELS[@]}"; do
    echo "  - $level: ${LEVEL_NAMES[$level]}"
done
echo "=============================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save run config
cat > "$OUTPUT_DIR/run_config.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "data_path": "$DATA_PATH",
    "params": "$PARAMS",
    "max_steps": $MAX_STEPS,
    "batch_size": $BATCH_SIZE,
    "chunk_size": $CHUNK_SIZE,
    "grad_accum": $GRAD_ACCUM,
    "tokens_per_step": $(($BATCH_SIZE * $CHUNK_SIZE * $GRAD_ACCUM)),
    "total_tokens": $(($BATCH_SIZE * $CHUNK_SIZE * $GRAD_ACCUM * $MAX_STEPS)),
    "levels": $(printf '%s\n' "${LEVELS[@]}" | jq -R . | jq -s .)
}
EOF

# Set library path for CUDA
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Run each level
for level in "${LEVELS[@]}"; do
    level_name="${LEVEL_NAMES[$level]}"
    level_dir="$OUTPUT_DIR/$level"

    echo ""
    echo "=============================================================="
    echo "Training Level $level: $level_name"
    echo "=============================================================="
    echo "Output: $level_dir"
    echo ""

    # Run training (single GPU, no DDP for simplicity)
    python train_ladder.py \
        --level "$level" \
        --params "$PARAMS" \
        --data "$DATA_PATH" \
        --output "$level_dir" \
        --batch_size "$BATCH_SIZE" \
        --chunk_size "$CHUNK_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --max_steps "$MAX_STEPS" \
        --log_interval "$LOG_INTERVAL" \
        --save_interval "$SAVE_INTERVAL" \
        --no-ddp \
        --tbptt \
        2>&1 | tee "$level_dir/training.log"

    echo ""
    echo "Completed Level $level: $level_name"
    echo ""
done

echo ""
echo "=============================================================="
echo "All training complete!"
echo "=============================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results:"
echo "  python analyze_comparison.py $OUTPUT_DIR"
echo ""
