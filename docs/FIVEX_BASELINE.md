# Phase 0 Baseline

Honest e2e measurements with dQ included. All at B=1, bf16.

| Config | T | CUDA fwd | CUDA bwd* | CUDA f+b | P | ADMM fwd | Iters | Par bwd | Hybrid | Speedup |
|--------|---|----------|-----------|----------|---|----------|-------|---------|--------|---------|
| E88-n16 480M | 4096 | 4.8 | 8.7 | 13.5 | 16 | 2.3 | 2 | 4.6 | 6.9 | **1.96×** |
| E88-n16 480M | 16384 | 17.9 | 35.9 | 53.8 | 16 | 8.2 | 2 | 19.2 | 27.4 | **1.96×** |
| E88-n16 480M | 32768 | 35.8 | 72.1 | 107.9 | 16 | 17.9 | 2 | 37.3 | 55.2 | **1.96×** |
| E88-n16 480M | 65536 | 71.7 | 144.4 | 216.0 | 16 | 34.1 | 2 | 75.0 | 109.1 | **1.98×** |
| E88-n32 480M | 4096 | 7.2 | 15.9 | 23.1 | 16 | 2.5 | 2 | 7.4 | 9.8 | **2.34×** |
| E88-n32 480M | 16384 | 28.6 | 65.1 | 93.8 | 16 | 8.6 | 2 | 29.4 | 38.0 | **2.47×** |
| E88-n32 480M | 32768 | 57.2 | 130.5 | 187.7 | 16 | 18.1 | 2 | 58.9 | 77.0 | **2.44×** |
| E88-n32 480M | 65536 | 114.4 | 261.2 | 375.6 | 16 | 36.5 | 2 | 119.1 | 155.6 | **2.41×** |

* CUDA bwd estimated as (CUDA f+b) - (CUDA fwd).

## Targets for subsequent phases

Phase 1 (warm-start ADMM): cut ADMM iters 2 → 1. Expected:
  - E88-n16 480M T=65K: ~ hybrid drops by ~half of ADMM fwd time
  - E88-n32 480M T=65K: same proportional reduction

Phase 3 (faster backward): target ~1.5× speedup on Par bwd column.