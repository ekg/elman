#!/bin/bash
# Run CMA-ES searches for all models sequentially
# Each search: 15 generations × 10 min = ~2.5 hours

set -e
cd /home/erikg/elman

echo "=== Starting CMA-ES searches for all models ==="
echo "Estimated total time: ~12.5 hours (5 models × 2.5 hours)"
echo ""

# Transformer (already running, wait for it)
echo "Waiting for transformer search to complete..."
while pgrep -f "cmaes_search.py.*transformer" > /dev/null; do
    sleep 60
done
echo "Transformer search complete!"

# GRU
echo ""
echo "=== Starting GRU search ==="
python cmaes_search.py --model gru --generations 15 --train_minutes 10 \
    --gpus 0,1,2,3,4,5,6,7 --params 480M --tolerance 30M --start_from_best \
    --output benchmark_results/cmaes_gru_10min 2>&1 | tee benchmark_results/cmaes_gru_10min.log

# LSTM
echo ""
echo "=== Starting LSTM search ==="
python cmaes_search.py --model lstm --generations 15 --train_minutes 10 \
    --gpus 0,1,2,3,4,5,6,7 --params 480M --tolerance 30M --start_from_best \
    --output benchmark_results/cmaes_lstm_10min 2>&1 | tee benchmark_results/cmaes_lstm_10min.log

# minGRU
echo ""
echo "=== Starting minGRU search ==="
python cmaes_search.py --model mingru --generations 15 --train_minutes 10 \
    --gpus 0,1,2,3,4,5,6,7 --params 480M --tolerance 30M --start_from_best \
    --output benchmark_results/cmaes_mingru_10min 2>&1 | tee benchmark_results/cmaes_mingru_10min.log

# minLSTM
echo ""
echo "=== Starting minLSTM search ==="
python cmaes_search.py --model minlstm --generations 15 --train_minutes 10 \
    --gpus 0,1,2,3,4,5,6,7 --params 480M --tolerance 30M --start_from_best \
    --output benchmark_results/cmaes_minlstm_10min 2>&1 | tee benchmark_results/cmaes_minlstm_10min.log

echo ""
echo "=== All CMA-ES searches complete! ==="
echo "Results:"
echo "  benchmark_results/cmaes_transformer_10min.log"
echo "  benchmark_results/cmaes_gru_10min.log"
echo "  benchmark_results/cmaes_lstm_10min.log"
echo "  benchmark_results/cmaes_mingru_10min.log"
echo "  benchmark_results/cmaes_minlstm_10min.log"
