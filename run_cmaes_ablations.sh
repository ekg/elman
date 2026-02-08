#!/bin/bash
# Run CMA-ES ablation search for E88 variants
# Tests the contribution of tanh and gating to E88 performance
#
# Ablations:
#   e88         - baseline (use_gate=1, linear_state=0) [runs in v9]
#   e88-linear  - remove tanh (use_gate=1, linear_state=1)
#   e88-nogate  - remove gating (use_gate=0, linear_state=0)
#   e88-minimal - remove both (use_gate=0, linear_state=1)
#
# Usage:
#   ./run_cmaes_ablations.sh           # Run all ablations
#   ./run_cmaes_ablations.sh e88-linear  # Run specific ablation

set -e

# Output directory - sibling to cmaes_v9, same settings
export OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/cmaes_v9_ablations}"
mkdir -p "$OUTPUT_DIR"

# Settings (Option B - fast)
TRAIN_MINS="${TRAIN_MINS:-10}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PARAMS="${PARAMS:-480M}"
LHS_SAMPLES="${LHS_SAMPLES:-64}"

# Ablation models (e88 baseline runs in v9, not here)
if [ -n "$1" ]; then
    ABLATIONS="$1"
else
    ABLATIONS="e88-linear e88-nogate e88-minimal"
fi

# Completion tracking
COMPLETED_FILE="$OUTPUT_DIR/.completed_ablations"
touch "$COMPLETED_FILE"

# Log file
ALL_LOG="$OUTPUT_DIR/run_ablations.log"

echo "======================================================================" | tee -a "$ALL_LOG"
echo "E88 Ablation Search - Started $(date)" | tee -a "$ALL_LOG"
echo "Output: $OUTPUT_DIR" | tee -a "$ALL_LOG"
echo "Ablations: $ABLATIONS" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"

TOTAL=0
SUCCEEDED=0
FAILED=0

for MODEL in $ABLATIONS; do
    TOTAL=$((TOTAL + 1))

    # Skip if already completed
    if grep -qw "$MODEL" "$COMPLETED_FILE"; then
        echo "Skipping $MODEL (already completed)" | tee -a "$ALL_LOG"
        SUCCEEDED=$((SUCCEEDED + 1))
        continue
    fi

    echo "" | tee -a "$ALL_LOG"
    echo "[$TOTAL] Starting $MODEL at $(date)" | tee -a "$ALL_LOG"

    LOG_FILE="$OUTPUT_DIR/${MODEL}_search.log"

    echo "======================================================================" | tee "$LOG_FILE"
    echo "CMA-ES Ablation Search: $MODEL" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"
    echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"

    if python -u cmaes_search_v2.py \
        --model "$MODEL" \
        --phase "sweep" \
        --sweep_param n_state \
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
        2>&1 | tee -a "$LOG_FILE"; then

        echo "$MODEL" >> "$COMPLETED_FILE"
        echo "[$TOTAL] $MODEL SUCCEEDED at $(date)" | tee -a "$ALL_LOG"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "[$TOTAL] $MODEL FAILED at $(date)" | tee -a "$ALL_LOG"
        FAILED=$((FAILED + 1))
        echo "Continuing to next ablation..." | tee -a "$ALL_LOG"
    fi
done

echo "" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"
echo "E88 Ablation Search - Completed $(date)" | tee -a "$ALL_LOG"
echo "Results: $SUCCEEDED succeeded, $FAILED failed out of $TOTAL total" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
