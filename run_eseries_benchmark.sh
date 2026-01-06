#!/bin/bash
# Run E-series benchmarks in parallel on different GPUs

DATA="data/fineweb_100mb.txt"
PARAMS="50m"
BATCH=64
STEPS=1000
LOG_INT=100

mkdir -p benchmark_results

echo "Starting 5 parallel benchmarks..."

CUDA_VISIBLE_DEVICES=0 python benchmark_baselines.py --data $DATA --model 0 --params $PARAMS --batch_size $BATCH --max_steps $STEPS --log_interval $LOG_INT > benchmark_results/e0_spec.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python benchmark_baselines.py --data $DATA --model 0 --params $PARAMS --batch_size $BATCH --max_steps $STEPS --log_interval $LOG_INT --no_spectral_norm > benchmark_results/e0_nospec.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python benchmark_baselines.py --data $DATA --model 1 --params $PARAMS --batch_size $BATCH --max_steps $STEPS --log_interval $LOG_INT > benchmark_results/e1_spec.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python benchmark_baselines.py --data $DATA --model 1 --params $PARAMS --batch_size $BATCH --max_steps $STEPS --log_interval $LOG_INT --no_spectral_norm > benchmark_results/e1_nospec.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=4 python benchmark_baselines.py --data $DATA --model mamba2 --params $PARAMS --batch_size $BATCH --max_steps $STEPS --log_interval $LOG_INT > benchmark_results/mamba2_50m.log 2>&1 &
PID4=$!

echo "PIDs: e0=$PID0 e0-nospec=$PID1 e1=$PID2 e1-nospec=$PID3 mamba2=$PID4"
echo "Waiting for all to complete..."

wait $PID0 $PID1 $PID2 $PID3 $PID4

echo ""
echo "=== RESULTS ==="
echo ""
echo "e0 (spectral norm):"
tail -5 benchmark_results/e0_spec.log
echo ""
echo "e0 (no spectral norm):"
tail -5 benchmark_results/e0_nospec.log
echo ""
echo "e1 (spectral norm):"
tail -5 benchmark_results/e1_spec.log
echo ""
echo "e1 (no spectral norm):"
tail -5 benchmark_results/e1_nospec.log
echo ""
echo "mamba2:"
tail -5 benchmark_results/mamba2_50m.log
