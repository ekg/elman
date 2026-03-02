#!/bin/bash
# CMA-ES Architecture Search at Multiple Scales with Progressive Training
# Four scales: 512 (baseline), 512→8K, 512→32K, 512→128K
# 6 model configs per scale: FLA-GDN, Mamba2, E88 n16, E88 n32, E1H n16, E1H n32
#
# All seeded with KNOWN_GOOD_CONFIGS (accumulated from prior rounds).
# Batch sizes auto-scaled per model and chunk size via PHASE2_BATCH_SIZES_BY_SCALE.
# Phase 1/2 stdout saved to eval dirs for post-hoc analysis.

set -e

GPUS="0,1,2,3,4,5,6,7"
LHS=64

MODELS="fla-gdn mamba2"
E88_STATES="16 32"
E1H_STATES="16 32"

run_scale() {
    local SCALE=$1
    local P1_MIN=$2
    local P2_MIN=$3
    local OUTDIR="benchmark_results/cmaes_${SCALE}"

    if [ "$SCALE" = "512" ]; then
        # No progressive — just standard search at 512
        COMMON="--train_minutes $P1_MIN --gpus $GPUS --phase both --lhs_samples $LHS --output $OUTDIR"

        echo "=== CMA-ES @ 512 (baseline, ${P1_MIN} min) ==="
        for model in $MODELS; do
            echo ">>> $model @ 512..."
            python cmaes_search_v2.py --model $model $COMMON 2>&1 | tee ${OUTDIR}_${model}.log
        done
        for ns in $E88_STATES; do
            echo ">>> E88 n${ns} @ 512..."
            python cmaes_search_v2.py --model e88 --fixed_n_state $ns $COMMON 2>&1 | tee ${OUTDIR}_e88_n${ns}.log
        done
        for ns in $E1H_STATES; do
            echo ">>> E1H n${ns} @ 512..."
            python cmaes_search_v2.py --model e1h --fixed_n_state $ns $COMMON 2>&1 | tee ${OUTDIR}_e1h_n${ns}.log
        done
    else
        # Progressive: Phase 1 @ 512 → Phase 2 @ SCALE
        COMMON="--progressive --phase1_minutes $P1_MIN --phase2_minutes $P2_MIN --phase2_chunk_size $SCALE --gpus $GPUS --phase both --lhs_samples $LHS --output $OUTDIR"

        echo "=== CMA-ES 512→${SCALE} (${P1_MIN}+${P2_MIN} min) ==="
        for model in $MODELS; do
            echo ">>> $model @ ${SCALE}..."
            python cmaes_search_v2.py --model $model $COMMON 2>&1 | tee ${OUTDIR}_${model}.log
        done
        for ns in $E88_STATES; do
            echo ">>> E88 n${ns} @ ${SCALE}..."
            python cmaes_search_v2.py --model e88 --fixed_n_state $ns $COMMON 2>&1 | tee ${OUTDIR}_e88_n${ns}.log
        done
        for ns in $E1H_STATES; do
            echo ">>> E1H n${ns} @ ${SCALE}..."
            python cmaes_search_v2.py --model e1h --fixed_n_state $ns $COMMON 2>&1 | tee ${OUTDIR}_e1h_n${ns}.log
        done
    fi

    echo "=== Scale ${SCALE} complete ==="
    echo ""
}

# Scale 1: 512 baseline (10 min)
run_scale 512 10 0

# Scale 2: 512→8K (10+10 min)
run_scale 8192 10 10

# Scale 3: 512→32K (10+10 min)
run_scale 32768 10 10

# Scale 4: 512→128K (10+10 min)
run_scale 131072 10 10

echo "=== All scales complete ==="
