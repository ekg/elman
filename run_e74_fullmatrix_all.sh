#!/bin/bash
# E74 Full Matrix Benchmark - All 5 Batches
# Runs 36 models total, 8 GPUs at a time
#
# Usage:
#   ./run_e74_fullmatrix_all.sh           # Run all batches
#   ./run_e74_fullmatrix_all.sh --minutes 5  # Quick test with 5 min per model

set -e

MINUTES=${1:-10}
if [[ "$1" == "--minutes" ]]; then
    MINUTES=$2
fi

echo "========================================================================"
echo "E74 FULL MATRIX 100M BENCHMARK - ALL BATCHES"
echo "Training time: $MINUTES minutes per model"
echo "Total: 36 models across 5 batches"
echo "========================================================================"
echo ""

START_TIME=$(date +%s)

for batch in 1 2 3 4 5; do
    echo ""
    echo "========================================================================"
    echo "BATCH $batch of 5"
    echo "Started at: $(date)"
    echo "========================================================================"

    python run_e74_fullmatrix_benchmark.py --batch $batch --minutes $MINUTES

    echo ""
    echo "Batch $batch complete at $(date)"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================================================"
echo "ALL BATCHES COMPLETE"
echo "Total time: ${HOURS}h ${MINS}m"
echo "Results in: benchmark_results/e74_fullmatrix_100m/"
echo "========================================================================"

# Generate summary
echo ""
echo "Generating results summary..."
python -c "
import json
from pathlib import Path

output_dir = Path('benchmark_results/e74_fullmatrix_100m')
results = []

for log_file in sorted(output_dir.glob('*.log')):
    model_name = log_file.stem
    try:
        with open(log_file) as f:
            content = f.read()

        lines = [l for l in content.split('\n') if 'step' in l and 'loss' in l]
        if lines:
            recent_losses = []
            recent_toks = []
            for line in lines[-100:]:
                parts = line.split('|')
                for p in parts:
                    if 'loss' in p and 'loss/' not in p:
                        try:
                            recent_losses.append(float(p.split()[-1]))
                        except:
                            pass
                    if 'tok/s' in p:
                        try:
                            recent_toks.append(float(p.split()[-1]))
                        except:
                            pass

            if recent_losses:
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_toks = sum(recent_toks) / len(recent_toks) if recent_toks else 0
                results.append((model_name, avg_loss, avg_toks))
    except:
        pass

print()
print('=' * 70)
print('RESULTS SUMMARY (sorted by loss)')
print('=' * 70)
print(f'{\"Model\":<45} {\"Loss\":>10} {\"Tok/s\":>10}')
print('-' * 70)

results.sort(key=lambda x: x[1])
for name, loss, toks in results:
    print(f'{name:<45} {loss:>10.4f} {toks:>10.0f}')

print('-' * 70)
print(f'Total models: {len(results)}')
"
