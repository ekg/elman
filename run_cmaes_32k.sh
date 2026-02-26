#!/bin/bash
# CMA-ES Architecture Search at 32K with Progressive Training
# Each evaluation: 10 min @ 512 → 10 min @ 32K
# 6 searches: FLA-GDN, Mamba2, E88 n16, E88 n32, E1H n16, E1H n32

set -e

COMMON="--progressive --phase1_minutes 10 --phase2_minutes 10 \
  --gpus 0,1,2,3,4,5,6,7 --phase both --lhs_samples 64 \
  --output benchmark_results/cmaes_32k"

echo "=== CMA-ES 32K Progressive Search ==="
echo "Each eval: 10 min @ 512 + 10 min @ 32K = 20 min"
echo "8 GPUs, popsize=16, ~40 min/generation"
echo ""

# FLA-GDN (no n_state sweep, bs=1 at 32K)
echo ">>> Starting FLA-GDN search..."
python cmaes_search_v2.py --model fla-gdn $COMMON 2>&1 | tee benchmark_results/cmaes_32k_fla-gdn.log

# Mamba2 (no n_state sweep, bs=1 at 32K)
echo ">>> Starting Mamba2 search..."
python cmaes_search_v2.py --model mamba2 $COMMON 2>&1 | tee benchmark_results/cmaes_32k_mamba2.log

# E88 n_state=16 (bs=4 at 32K, grad ckpt + proj chunking)
echo ">>> Starting E88 n16 search..."
python cmaes_search_v2.py --model e88 --fixed_n_state 16 $COMMON 2>&1 | tee benchmark_results/cmaes_32k_e88_n16.log

# E88 n_state=32 (bs=4 at 32K, grad ckpt + proj chunking)
echo ">>> Starting E88 n32 search..."
python cmaes_search_v2.py --model e88 --fixed_n_state 32 $COMMON 2>&1 | tee benchmark_results/cmaes_32k_e88_n32.log

# E1H n_state=16 (bs=2 at 32K, grad ckpt)
echo ">>> Starting E1H n16 search..."
python cmaes_search_v2.py --model e1h --fixed_n_state 16 $COMMON 2>&1 | tee benchmark_results/cmaes_32k_e1h_n16.log

# E1H n_state=32 (bs=2 at 32K, grad ckpt)
echo ">>> Starting E1H n32 search..."
python cmaes_search_v2.py --model e1h --fixed_n_state 32 $COMMON 2>&1 | tee benchmark_results/cmaes_32k_e1h_n32.log

echo ""
echo "=== All 6 searches complete ==="
