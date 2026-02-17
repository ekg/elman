#!/bin/bash
# Run CMA-ES search for a specific model
# Usage: ./run_cmaes_model.sh <model> [options]
#
# Examples:
#   ./run_cmaes_model.sh e88                    # E88 with sweep over n_state
#   ./run_cmaes_model.sh fla-gdn               # FLA-GDN with both LHS+CMAES
#   ./run_cmaes_model.sh mamba2                # Mamba2
#   ./run_cmaes_model.sh e88 --phase lhs       # Just LHS phase
#
# To run all models in sequence (restartable):
#   for model in e88 fla-gdn mamba2 transformer e1 e23 e42 mingru minlstm; do
#       ./run_cmaes_model.sh $model || echo "$model FAILED"
#   done

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model> [extra args...]"
    echo ""
    echo "Models: e88, fla-gdn, mamba2, transformer, e1, e23, e42, mingru, minlstm"
    echo ""
    echo "Extra args are passed directly to cmaes_search_v2.py"
    echo "  --phase {lhs,cmaes,both,sweep}  Phase to run (default: sweep for e88, both for others)"
    echo "  --train_minutes N               Training time per config (default: 10)"
    echo "  --gpus 0,1,2,3,4,5,6,7          GPUs to use"
    echo "  --params 480M                   Target parameter count"
    echo "  --lhs_samples N                 Number of LHS samples (default: 64)"
    echo ""
    echo "torch.compile is ENABLED by default (+17% throughput)"
    exit 1
fi

MODEL=$1
shift  # Remove model from args, rest are passed through

# Output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results/cmaes_v9}"

# Default settings - Option B (fast)
TRAIN_MINS="${TRAIN_MINS:-10}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PARAMS="${PARAMS:-480M}"
LHS_SAMPLES="${LHS_SAMPLES:-64}"

# Model-specific phase
case $MODEL in
    e88)
        DEFAULT_PHASE="sweep"
        EXTRA_ARGS="--sweep_param n_state"
        ;;
    *)
        DEFAULT_PHASE="both"
        EXTRA_ARGS=""
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file for this model
LOG_FILE="$OUTPUT_DIR/${MODEL}_search.log"

echo "======================================================================" | tee "$LOG_FILE"
echo "CMA-ES Search: $MODEL" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# Run the search with torch.compile enabled (+17% throughput)
python -u cmaes_search_v2.py \
    --model "$MODEL" \
    --phase "$DEFAULT_PHASE" \
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
    --compile \
    $EXTRA_ARGS \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "SUCCESS: $MODEL completed at $(date)" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "FAILED: $MODEL exited with code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE
