#!/bin/bash
# Re-run E23 after --n_slots fix
# Run after cmaes_v9_catchup completes

set -e

export OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/cmaes_v9_e23}"
mkdir -p "$OUTPUT_DIR"

# Settings (Option B - same as v9)
TRAIN_MINS="${TRAIN_MINS:-10}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PARAMS="${PARAMS:-480M}"
LHS_SAMPLES="${LHS_SAMPLES:-64}"

LOG_FILE="$OUTPUT_DIR/e23_search.log"

echo "======================================================================" | tee "$LOG_FILE"
echo "CMA-ES E23 (DualMemoryElman) - Started $(date)" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

python -u cmaes_search_v2.py \
    --model "e23" \
    --phase "both" \
    --train_minutes "$TRAIN_MINS" \
    --gpus "$GPUS" \
    --params "$PARAMS" \
    --output "$OUTPUT_DIR" \
    --lhs_samples "$LHS_SAMPLES" \
    --sigma 0.35 \
    --min_generations 4 \
    --converge 0.01 \
    --consecutive 2 \
    --cmaes_refinements 2 \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
