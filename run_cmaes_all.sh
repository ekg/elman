#!/bin/bash
# Run CMA-ES search for all models, one at a time
# Can be restarted - skips completed models
#
# Usage:
#   ./run_cmaes_all.sh                    # Run all models
#   ./run_cmaes_all.sh --from mamba2      # Start from mamba2 (skip earlier)
#   SKIP_MODELS="e88 fla-gdn" ./run_cmaes_all.sh  # Skip specific models

set -e

# All models in order
MODELS="e88 fla-gdn mamba2 transformer e1 e23 e42 mingru minlstm"

# Output directory
export OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/cmaes_v8}"
mkdir -p "$OUTPUT_DIR"

# Completion tracking
COMPLETED_FILE="$OUTPUT_DIR/.completed_models"
touch "$COMPLETED_FILE"

# Parse --from argument
START_FROM=""
if [ "$1" == "--from" ]; then
    START_FROM="$2"
    echo "Starting from: $START_FROM"
fi

# Log file
ALL_LOG="$OUTPUT_DIR/run_all.log"

echo "======================================================================" | tee -a "$ALL_LOG"
echo "CMA-ES Full Search - Started $(date)" | tee -a "$ALL_LOG"
echo "Output: $OUTPUT_DIR" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"

FOUND_START=false
if [ -z "$START_FROM" ]; then
    FOUND_START=true
fi

TOTAL=0
SUCCEEDED=0
FAILED=0

for MODEL in $MODELS; do
    TOTAL=$((TOTAL + 1))

    # Skip until we find the starting model
    if [ "$FOUND_START" = false ]; then
        if [ "$MODEL" == "$START_FROM" ]; then
            FOUND_START=true
        else
            echo "Skipping $MODEL (before --from)" | tee -a "$ALL_LOG"
            continue
        fi
    fi

    # Skip if in SKIP_MODELS
    if echo "$SKIP_MODELS" | grep -qw "$MODEL"; then
        echo "Skipping $MODEL (in SKIP_MODELS)" | tee -a "$ALL_LOG"
        continue
    fi

    # Skip if already completed
    if grep -qw "$MODEL" "$COMPLETED_FILE"; then
        echo "Skipping $MODEL (already completed)" | tee -a "$ALL_LOG"
        SUCCEEDED=$((SUCCEEDED + 1))
        continue
    fi

    echo "" | tee -a "$ALL_LOG"
    echo "[$TOTAL] Starting $MODEL at $(date)" | tee -a "$ALL_LOG"

    if ./run_cmaes_model.sh "$MODEL"; then
        echo "$MODEL" >> "$COMPLETED_FILE"
        echo "[$TOTAL] $MODEL SUCCEEDED at $(date)" | tee -a "$ALL_LOG"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "[$TOTAL] $MODEL FAILED at $(date)" | tee -a "$ALL_LOG"
        FAILED=$((FAILED + 1))

        # Continue to next model on failure (don't abort entire run)
        echo "Continuing to next model..." | tee -a "$ALL_LOG"
    fi
done

echo "" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"
echo "CMA-ES Full Search - Completed $(date)" | tee -a "$ALL_LOG"
echo "Results: $SUCCEEDED succeeded, $FAILED failed out of $TOTAL total" | tee -a "$ALL_LOG"
echo "======================================================================" | tee -a "$ALL_LOG"

# Exit with failure if any model failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
