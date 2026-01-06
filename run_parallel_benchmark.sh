#!/bin/bash
# Parallel benchmark of all Elman ladder levels - NO GRAD CLIPPING

export LD_LIBRARY_PATH=/home/erikg/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

DATA="/home/erikg/elman/data/fineweb_100mb.txt"
OUT="/tmp/benchmark_noclip"
STEPS=1000
BATCH=16
CHUNK=512
LR="1e-3"
LOG_INT=100

mkdir -p $OUT

cd /home/erikg/elman

run_model() {
    GPU=$1
    LEVEL=$2
    NAME=$3
    PORT=$((29700 + GPU))

    echo "[GPU $GPU] Starting $NAME (level=$LEVEL, NO CLIPPING)..."
    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
        train_ladder.py \
        --level $LEVEL \
        --params 100m \
        --data $DATA \
        --batch_size $BATCH \
        --chunk_size $CHUNK \
        --lr $LR \
        --grad_clip 0 \
        --max_steps $STEPS \
        --log_interval $LOG_INT \
        --output $OUT/$NAME \
        2>&1 > $OUT/${NAME}.log &
}

# Batch 1: GPUs 0,1,2,3,7
run_model 0 0 "level_0_stock"
run_model 1 1 "level_1_gated"
run_model 2 2 "level_2_selective"
run_model 3 3 "level_3_diagonal"
run_model 7 4 "level_4_full_recur"

echo "Batch 1 started (levels 0-4). Waiting..."
wait

# Batch 2: GPUs 0,1,2,3,7
run_model 0 5 "level_5_triple_r"
run_model 1 6 "level_6_polynomial"
run_model 2 log_0 "log_0_polynomial"
run_model 3 log_3 "log_3_diagonal"
run_model 7 log_4 "log_4_compute"

echo "Batch 2 started (levels 5,6,log_0,log_3,log_4). Waiting..."
wait

# Batch 3: GPUs 0,1,2
run_model 0 log_5 "log_5_triple_r"
run_model 1 log_1 "log_1_selective"
run_model 2 log_2 "log_2_diag_sel"

echo "Batch 3 started (log_1,log_2,log_5). Waiting..."
wait

echo ""
echo "========================================"
echo "ALL RUNS COMPLETE (NO GRAD CLIPPING)"
echo "========================================"

echo ""
echo "Results sorted by loss:"
echo "======================="
for f in $OUT/*.log; do
    name=$(basename $f .log)
    loss=$(grep "Final loss" $f 2>/dev/null | awk '{print $NF}')
    grad=$(grep "Step.*1000" $f 2>/dev/null | sed 's/.*Grad //' | awk '{print $1}')
    toks=$(grep "Step.*1000" $f 2>/dev/null | sed 's/.*Tok\/s //' | awk '{print $1}')
    if [ -n "$loss" ]; then
        echo "$loss|$name|grad=$grad|$toks tok/s"
    fi
done | sort -t'|' -k1 -n | while IFS='|' read loss name grad toks; do
    printf "  %-20s loss=%-8s %s  %s\n" "$name" "$loss" "$grad" "$toks"
done
