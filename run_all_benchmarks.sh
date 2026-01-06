#!/bin/bash
# Run all model benchmarks on GPUs 4-7

DATA="/home/erikg/elman/data/pile_1mb.txt"
PARAMS="50m"
STEPS=500
LOG_INT=50
OUT_DIR="/tmp/benchmark_all_models"

mkdir -p $OUT_DIR

echo "=== Running benchmarks on GPUs 4-7 ==="
echo "Data: $DATA"
echo "Params: $PARAMS"
echo "Steps: $STEPS"

# Batch 1: mamba2, log_0, log_1, log_2
echo "Batch 1: mamba2, log_0, log_1, log_2"
CUDA_VISIBLE_DEVICES=4 python benchmark_baselines.py --data $DATA --model mamba2 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/mamba2 > $OUT_DIR/mamba2.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python benchmark_baselines.py --data $DATA --model log_0 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/log_0 > $OUT_DIR/log_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python benchmark_baselines.py --data $DATA --model log_1 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/log_1 > $OUT_DIR/log_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python benchmark_baselines.py --data $DATA --model log_2 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/log_2 > $OUT_DIR/log_2.log 2>&1 &
wait
echo "Batch 1 done"

# Batch 2: log_3, log_5, level 0, level 1
echo "Batch 2: log_3, log_5, level 0, level 1"
CUDA_VISIBLE_DEVICES=4 python benchmark_baselines.py --data $DATA --model log_3 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/log_3 > $OUT_DIR/log_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python benchmark_baselines.py --data $DATA --model log_5 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/log_5 > $OUT_DIR/log_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python benchmark_baselines.py --data $DATA --model 0 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/level_0 > $OUT_DIR/level_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python benchmark_baselines.py --data $DATA --model 1 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/level_1 > $OUT_DIR/level_1.log 2>&1 &
wait
echo "Batch 2 done"

# Batch 3: level 2, level 3, level 4, level 5
echo "Batch 3: level 2, level 3, level 4, level 5"
CUDA_VISIBLE_DEVICES=4 python benchmark_baselines.py --data $DATA --model 2 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/level_2 > $OUT_DIR/level_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python benchmark_baselines.py --data $DATA --model 3 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/level_3 > $OUT_DIR/level_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python benchmark_baselines.py --data $DATA --model 4 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/level_4 > $OUT_DIR/level_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python benchmark_baselines.py --data $DATA --model 5 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/level_5 > $OUT_DIR/level_5.log 2>&1 &
wait
echo "Batch 3 done"

# Batch 4: level 6
echo "Batch 4: level 6"
CUDA_VISIBLE_DEVICES=4 python benchmark_baselines.py --data $DATA --model 6 --params $PARAMS --max_steps $STEPS --log_interval $LOG_INT --output $OUT_DIR/level_6 > $OUT_DIR/level_6.log 2>&1 &
wait
echo "Batch 4 done"

echo ""
echo "=== ALL RESULTS ==="
echo ""
printf "%-12s %12s %12s %12s %10s\n" "Model" "Params" "Loss" "Grad" "tok/s"
echo "------------------------------------------------------------"
for f in $OUT_DIR/*.log; do
    model=$(basename $f .log)
    # Extract final line with results
    final=$(grep "Final:" $f | tail -1)
    if [ -n "$final" ]; then
        loss=$(echo $final | grep -oP 'loss=\K[\d.]+')
        grad=$(echo $final | grep -oP 'grad=\K[\d.]+')
        # Get params and speed from earlier in log
        params=$(grep "Parameters:" $f | grep -oP '[\d.]+[MBK]' | head -1)
        speed=$(grep "step  500" $f | grep -oP '\d+ tok/s' | head -1 | cut -d' ' -f1)
        printf "%-12s %12s %12s %12s %10s\n" "$model" "$params" "$loss" "$grad" "$speed"
    fi
done | sort -t'|' -k3 -n
