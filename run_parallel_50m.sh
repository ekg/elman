#!/bin/bash
# Run 50M benchmarks in parallel across GPUs

DATA="data/fineweb_100mb.txt"
PARAMS="50m"
STEPS=1000
OUTDIR="benchmark_results/50m_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTDIR

# Run each model on a different GPU in background
CUDA_VISIBLE_DEVICES=0 python benchmark_baselines.py --data $DATA --params $PARAMS --max_steps $STEPS --model mamba2 --output $OUTDIR > $OUTDIR/mamba2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python benchmark_baselines.py --data $DATA --params $PARAMS --max_steps $STEPS --model 0 --output $OUTDIR > $OUTDIR/level0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python benchmark_baselines.py --data $DATA --params $PARAMS --max_steps $STEPS --model 1 --output $OUTDIR > $OUTDIR/level1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python benchmark_baselines.py --data $DATA --params $PARAMS --max_steps $STEPS --model 2 --output $OUTDIR > $OUTDIR/level2.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python benchmark_baselines.py --data $DATA --params $PARAMS --max_steps $STEPS --model 3 --output $OUTDIR > $OUTDIR/level3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python benchmark_baselines.py --data $DATA --params $PARAMS --max_steps $STEPS --model 5 --output $OUTDIR > $OUTDIR/level5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python benchmark_baselines.py --data $DATA --params $PARAMS --max_steps $STEPS --model 7 --output $OUTDIR > $OUTDIR/level7.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python benchmark_baselines.py --data $DATA --params $PARAMS --max_steps $STEPS --model gru --output $OUTDIR > $OUTDIR/gru.log 2>&1 &

echo "Started all benchmarks in parallel"
echo "Output directory: $OUTDIR"
echo "Waiting for all jobs to complete..."
wait
echo "All benchmarks complete!"

# Show summary from each log
echo ""
echo "========== RESULTS SUMMARY =========="
for f in $OUTDIR/*.log; do
    model=$(basename $f .log)
    final=$(grep -E "Final:" $f | tail -1)
    if [ -n "$final" ]; then
        echo "$model: $final"
    else
        echo "$model: (check log for errors)"
    fi
done
