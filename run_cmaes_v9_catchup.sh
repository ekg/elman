#!/bin/bash
# CMA-ES v9 Catchup: E23 + E88 Ablations
#
# Runs models that were missed or added after v9 started:
#   - e23: DualMemoryElman (was missing from cmaes_search_v2.py)
#   - e88-linear: E88 without tanh (linear state)
#   - e88-nogate: E88 without output gating
#   - e88-minimal: E88 without both tanh and gating
#
# Uses same Option B settings as v9 for consistency.
#
# Usage:
#   ./run_cmaes_v9_catchup.sh              # Run all
#   ./run_cmaes_v9_catchup.sh e23          # Run specific model
#   ./run_cmaes_v9_catchup.sh e88-linear   # Run specific ablation

set -e

# Output directory - sibling to cmaes_v9
export OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/cmaes_v9_catchup}"
mkdir -p "$OUTPUT_DIR"

# Settings (Option B - same as v9)
TRAIN_MINS="${TRAIN_MINS:-10}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PARAMS="${PARAMS:-480M}"
LHS_SAMPLES="${LHS_SAMPLES:-64}"

# Models to run
if [ -n "$1" ]; then
    MODELS="$1"
else
    MODELS="e23 e88-linear e88-nogate e88-minimal"
fi

# Completion tracking
COMPLETED_FILE="$OUTPUT_DIR/.completed_models"
touch "$COMPLETED_FILE"

# Log file
ALL_LOG="$OUTPUT_DIR/run_catchup.log"

echo "======================================================================" | tee -a "$ALL_LOG"
echo "CMA-ES v9 Catchup - Started $(date)" | tee -a "$ALL_LOG"
echo "Output: $OUTPUT_DIR" | tee -a "$ALL_LOG"
echo "Models: $MODELS" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"

TOTAL=0
SUCCEEDED=0
FAILED=0

for MODEL in $MODELS; do
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
    echo "CMA-ES v9 Catchup: $MODEL" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"

    # Determine phase based on model type
    if [[ "$MODEL" == e88* ]]; then
        # E88 variants use sweep over n_state
        PHASE="sweep"
        EXTRA_ARGS="--sweep_param n_state"
    else
        # Other models use both phases
        PHASE="both"
        EXTRA_ARGS=""
    fi

    if python -u cmaes_search_v2.py \
        --model "$MODEL" \
        --phase "$PHASE" \
        $EXTRA_ARGS \
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
        echo "Continuing to next model..." | tee -a "$ALL_LOG"
    fi
done

echo "" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"
echo "CMA-ES v9 Catchup - Completed $(date)" | tee -a "$ALL_LOG"
echo "Results: $SUCCEEDED succeeded, $FAILED failed out of $TOTAL total" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
