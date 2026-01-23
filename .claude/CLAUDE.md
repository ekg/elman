# Elman Project Guidelines

## CRITICAL: Loss Reporting

**ONLY TRUST LOSS VALUES AVERAGED OVER AT LEAST 100 STEPS.**

Single-step or short-window loss values are extremely noisy and misleading. When reporting benchmark results:

1. Always compute `last-100 average loss` (average of final 100 training steps)
2. Never cite a single step's loss as representative
3. If fewer than 100 steps completed, state this limitation explicitly
4. When comparing models, ensure both use the same averaging window

Example of WRONG reporting:
```
E88_b56n32: 1.55 loss  ← This was probably a single lucky step
```

Example of CORRECT reporting:
```
E88_b56n32: 2.09 last-100 avg loss (steps 321-420)
```

## Benchmark Standards

- Training time: 10 minutes minimum for quick comparisons
- Batch size: 32, chunk size: 512 (or document if different)
- Loss metric: last-100 step average
- Always report: params, throughput (tok/s), loss, config details

## Parallel Execution

For independent experiments, run them in parallel:
```bash
# Good: parallel execution
nohup python train.py --config A > a.log 2>&1 &
nohup python train.py --config B > b.log 2>&1 &
wait  # Wait for all to complete

# Bad: sequential when parallelism is possible
python train.py --config A
python train.py --config B
```

## E88 Linear vs Tanh Finding

**Tanh is NOT the scaling problem.** At 500M scale, linear and tanh E88 achieve identical loss:
- E88 TANH: 1.686
- E88 LINEAR: 1.684

The gap to Mamba2/FLA-GDN (~0.21-0.27) is due to architectural/optimization differences, not the nonlinearity.

## E88 Configuration

The balanced principle: `n_heads × n_state ≈ dim` (ratio 0.5-2.0)

Working n_state values in CUDA kernel:
- 4, 8, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 80, 96, 128

Square states (expansion=1.0) generally outperform rectangular states.

## E88 Gate Activation Finding

**Use SiLU (swish) gating, NOT sigmoid.** FLA-GDN uses silu/swish for output gating.

At 500M scale (10 min training):
- No gate: 2.25 loss
- SiLU gate: 2.08 loss (BETTER by 0.17)

The difference between FLA-GDN gating (silu) and E88's original gating (sigmoid):
- FLA-GDN: `output = RMSNorm(x) * silu(g)` = `RMSNorm(x) * g * sigmoid(g)`
- E88 original: `output = RMSNorm(x) * sigmoid(g)`

SiLU includes the raw gate value `g`, allowing negative gates and better gradient flow.
Use `--gate_activation silu` with `--use_gate 1`.
