#!/bin/bash
cd /home/erikg/elman

# 100 steps for quick test (~10 mins)
CUDA_VISIBLE_DEVICES=0 python -u benchmark_baselines.py --data data/fineweb_100mb.txt --model 0 --params 50m --batch_size 64 --max_steps 100 --log_interval 20 > benchmark_results/e0_spec.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u benchmark_baselines.py --data data/fineweb_100mb.txt --model 0 --params 50m --batch_size 64 --max_steps 100 --log_interval 20 --no_spectral_norm > benchmark_results/e0_nospec.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u benchmark_baselines.py --data data/fineweb_100mb.txt --model 1 --params 50m --batch_size 64 --max_steps 100 --log_interval 20 > benchmark_results/e1_spec.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u benchmark_baselines.py --data data/fineweb_100mb.txt --model 1 --params 50m --batch_size 64 --max_steps 100 --log_interval 20 --no_spectral_norm > benchmark_results/e1_nospec.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u benchmark_baselines.py --data data/fineweb_100mb.txt --model mamba2 --params 50m --batch_size 64 --max_steps 100 --log_interval 20 > benchmark_results/mamba2_50m.log 2>&1 &

echo "Launched 5 jobs on GPUs 0-4 (100 steps each)"
