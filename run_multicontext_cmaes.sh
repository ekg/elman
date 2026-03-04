#!/bin/bash
# Multi-context CMA-ES search: 512, 8K, 32K, 128K
# 6 models × 4 context lengths = 24 searches, queued one at a time
# Each search gets all 8 GPUs for maximum parallelism + backfill efficiency
# Phase 1: 10 min @ 512, Phase 2: 20 min @ target context

set -e

PHASE1=10
PHASE2=20
LHS=128
GPUS="0,1,2,3,4,5,6,7"
OUTPUT_BASE="benchmark_results/cmaes_multicontext"

mkdir -p "$OUTPUT_BASE"

SEARCHES=(
    # Round 1: 512→512
    "512 e88 32 e88_n32_512"
    "512 e88 16 e88_n16_512"
    "512 e1h 32 e1h_n32_512"
    "512 e1h 16 e1h_n16_512"
    "512 fla-gdn _ fla-gdn_512"
    "512 mamba2 _ mamba2_512"
    # Round 2: 512→8K
    "8192 e88 32 e88_n32_8k"
    "8192 e88 16 e88_n16_8k"
    "8192 e1h 32 e1h_n32_8k"
    "8192 e1h 16 e1h_n16_8k"
    "8192 fla-gdn _ fla-gdn_8k"
    "8192 mamba2 _ mamba2_8k"
    # Round 3: 512→32K
    "32768 e88 32 e88_n32_32k"
    "32768 e88 16 e88_n16_32k"
    "32768 e1h 32 e1h_n32_32k"
    "32768 e1h 16 e1h_n16_32k"
    "32768 fla-gdn _ fla-gdn_32k"
    "32768 mamba2 _ mamba2_32k"
    # Round 4: 512→128K
    "131072 e88 32 e88_n32_128k"
    "131072 e88 16 e88_n16_128k"
    "131072 e1h 32 e1h_n32_128k"
    "131072 e1h 16 e1h_n16_128k"
    "131072 fla-gdn _ fla-gdn_128k"
    "131072 mamba2 _ mamba2_128k"
)

TOTAL=${#SEARCHES[@]}
IDX=0

for entry in "${SEARCHES[@]}"; do
    read -r CTX MODEL NS LABEL <<< "$entry"
    IDX=$((IDX + 1))

    NS_FLAG=""
    if [ "$NS" != "_" ]; then
        NS_FLAG="--fixed_n_state $NS"
    fi

    echo "============================================================"
    echo "[$IDX/$TOTAL] $LABEL — model=$MODEL ctx=512→$CTX — all 8 GPUs"
    echo "============================================================"

    python cmaes_search_v2.py --model "$MODEL" --phase both --lhs_samples $LHS \
        --progressive --phase1_minutes $PHASE1 --phase2_minutes $PHASE2 --phase2_chunk_size $CTX \
        $NS_FLAG --gpus $GPUS --output "$OUTPUT_BASE/$LABEL" \
        2>&1 | tee "$OUTPUT_BASE/${LABEL}.log"

    echo "[$IDX/$TOTAL] $LABEL COMPLETE"
    echo ""
done

echo "============================================================"
echo "ALL $TOTAL SEARCHES COMPLETE"
echo "============================================================"
