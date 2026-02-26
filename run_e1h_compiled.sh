#!/bin/bash
# E1H compiled CMA-ES sweep — matches v10 compiled run settings
# Seeds from non-compiled E1H sweep (Feb 24 2026, best: 1.3358)

OUTPUT_DIR="benchmark_results/cmaes_v10_compiled"
GPUS="0,1,2,3,4,5,6,7"
PARAMS="480M"
TRAIN_MINUTES=10
LHS_SAMPLES=64

log_msg() {
    echo "[$(date)] $1"
}

wait_for_gpus() {
    log_msg "Waiting for all 8 GPUs to be free..."
    while true; do
        FREE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '$1 < 10' | wc -l)
        if [ "$FREE" -ge 8 ]; then
            log_msg "All 8 GPUs free"
            return
        fi
        sleep 30
    done
}

log_msg "E1H compiled CMA-ES sweep starting"

wait_for_gpus
python -u cmaes_search_v2.py \
    --model e1h \
    --phase sweep \
    --train_minutes $TRAIN_MINUTES \
    --gpus $GPUS \
    --params $PARAMS \
    --output $OUTPUT_DIR \
    --lhs_samples $LHS_SAMPLES \
    --compile \
    2>&1 | tee ${OUTPUT_DIR}/e1h_compiled.log

log_msg "E1H compiled CMA-ES sweep completed"
