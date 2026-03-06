#!/bin/bash
# Multi-context CMA-ES search: 512, 8K, 32K, 128K
# 6 models × 4 context lengths = 24 searches, queued one at a time
# Each search gets all 8 GPUs for maximum parallelism + backfill efficiency
# Phase 1: 10 min @ 512, Phase 2: 20 min @ target context

set -e

PHASE1=10
PHASE2=20
LHS=128
POPSIZE=16
GPUS="0,1,2,3,4,5,6"
GPU_FILE="benchmark_results/cmaes_multicontext/gpus.txt"
OUTPUT_BASE="benchmark_results/cmaes_multicontext"

mkdir -p "$OUTPUT_BASE"

# Write initial GPU list (edit this file while running to add/remove GPUs)
echo "$GPUS" > "$GPU_FILE"
echo "GPU file: $GPU_FILE (edit to dynamically add/remove GPUs)"
echo "Current GPUs: $GPUS"
echo ""

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

    # Skip completed searches (results.json exists)
    RESULTS=$(find "$OUTPUT_BASE/$LABEL" -name "results.json" 2>/dev/null | head -1)
    if [ -n "$RESULTS" ]; then
        echo "[$IDX/$TOTAL] SKIP $LABEL — already complete ($(cat "$RESULTS" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"best={d[\"best_loss\"]:.4f}, evals={d[\"total_evals\"]}")' 2>/dev/null || echo 'results.json exists'))"
        echo ""
        continue
    fi

    NS_FLAG=""
    if [ "$NS" != "_" ]; then
        NS_FLAG="--fixed_n_state $NS"
    fi

    echo "============================================================"
    echo "[$IDX/$TOTAL] $LABEL — model=$MODEL ctx=512→$CTX — GPUs from $GPU_FILE"
    echo "============================================================"

    python cmaes_search_v2.py --model "$MODEL" --phase both --lhs_samples $LHS \
        --progressive --phase1_minutes $PHASE1 --phase2_minutes $PHASE2 --phase2_chunk_size $CTX \
        --popsize $POPSIZE $NS_FLAG --gpus $GPUS --gpu_file "$GPU_FILE" \
        --output "$OUTPUT_BASE/$LABEL" --resume \
        2>&1 | tee -a "$OUTPUT_BASE/${LABEL}.log"

    echo "[$IDX/$TOTAL] $LABEL COMPLETE"
    echo ""
done

echo "============================================================"
echo "ALL $TOTAL SEARCHES COMPLETE"
echo "============================================================"
