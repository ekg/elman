# Length-extrapolation results

**Protocol:** Train at sequence length T=40, evaluate at T ∈ {40, 80, 160, 320, 500}.
A model that *learns the algorithm* extrapolates; a model that *memorizes the training-length distribution* does not.

**Config:** dim=384, depth=4, n_heads=32, n_state=32, sf-AdamW, 5K steps, batch_size=32, 3 seeds.

## modular_counter K=5

## Results (mean ± std over 3 seeds)

| Pattern        | T=40 (train)     | T=80             | T=160            | T=320            | T=500            |
|----------------|------------------|------------------|------------------|------------------|------------------|
| **pure_E88**   | **0.973 ± 0.018**| **0.794 ± 0.083**| **0.517 ± 0.048**| **0.350 ± 0.023**| **0.300 ± 0.020**|
| pure_FLA       | 0.477 ± 0.019    | 0.339 ± 0.010    | 0.268 ± 0.005    | 0.234 ± 0.002    | 0.225 ± 0.001    |
| hybrid_AABB    | 0.508 ± 0.151    | 0.365 ± 0.083    | 0.283 ± 0.042    | 0.240 ± 0.022    | 0.227 ± 0.001    |

Random baseline = 0.20 (5-class modular_counter).

## Read

**Pure E88 is the only pattern that learns the task and extrapolates.**

- At training length (T=40), E88 reaches 0.97 accuracy. FLA and hybrid don't get past 0.5 — they haven't learned the K=5 counter at this short length. (At T=128 in the canonical sweep, FLA reaches 0.65; the task seems to need more signal length to learn.)
- At 2× extrapolation (T=80), E88 still holds 0.79.
- At 4× extrapolation (T=160), E88 holds 0.52 — well above random (0.20).
- FLA and hybrid drop to random by T=160-320, regardless of training success.

This is the expected separation. E88's nonlinear-in-time matrix recurrence with bounded state can encode the K-counter as a periodic limit cycle in S, which extends naturally to longer sequences. Parallel-scan SSMs (FLA-GDN) lack the algorithmic structure to extrapolate counter dynamics they didn't see at training length.

**Hybrid does not help.** Stacking E88 with FLA degrades both training-length performance (0.51 vs E88's 0.97) and extrapolation. Consistent with the canonical sweep finding: hybrid is an ablation, not a result.

## Caveats

- 5K steps is short — longer training might let FLA reach the canonical-sweep accuracy of 0.65 at T=40. The extrapolation pattern is the more durable signal.
- Only modular_counter K=5 here. Parity also length-extrapolates well for E88 (0.81 at T=500 from earlier smoke). FSM tracking + Dyck-2 should be added.
- Sequence length of training matters: at T=40, FLA is well below grok threshold; at T=128 (canonical sweep) it's at 0.65. Length-extrap from a stronger base would be more informative.

## fsm_tracking K=4

| Pattern        | T=40 (train)     | T=80             | T=160            | T=320            | T=500            |
|----------------|------------------|------------------|------------------|------------------|------------------|
| **pure_E88**   | **1.000 ± 0.000**| **1.000 ± 0.001**| **0.903 ± 0.065**| **0.711 ± 0.103**| **0.591 ± 0.102**|
| pure_FLA       | 0.988 ± 0.006    | 0.924 ± 0.037    | 0.677 ± 0.081    | 0.473 ± 0.048    | 0.387 ± 0.034    |

Random baseline = 0.25 (4-state FSM).

**Read.** Both patterns grok at training length (E88 1.000, FLA 0.988). On extrapolation, E88 holds 0.59 at T=500 (12.5× training length, well above 0.25 random); FLA falls to 0.39 — closer to random. The gap widens monotonically with T: at T=160 E88 still has 0.90, FLA already at 0.68. This is the same pattern as modular_counter, but with a stronger FLA baseline (FLA actually learns FSM at T=40, just doesn't extrapolate).

## Joint take

Across two state-tracking tasks (modular_counter K=5 and fsm_tracking K=4), pure E88 extrapolates substantially better than pure FLA-GDN. On modular_counter the gap is dramatic (E88 doesn't degrade to random at T=500; FLA never groks at T=40). On fsm_tracking the gap is monotone and growing — both grok at T=40, but E88 retains 0.59 at T=500 vs FLA's 0.39.

Hybrid_AABB (modular_counter only — not run for fsm) tracks FLA, not E88. Stacking E88 with FLA does not preserve E88's extrapolation; the FLA layers degrade what pure E88 can do alone.

## Reproduce

```bash
bash experiments/expressivity_tasks/run_lenextrap_sweep.sh
```
or for a specific config:
```bash
python experiments/expressivity_tasks/train_hybrid.py \
  --task modular_counter --layer_pattern E88 \
  --dim 384 --depth 4 --n_heads 32 --n_state 32 \
  --steps 5000 --seq_len 40 --batch_size 32 --K 5 \
  --optimizer schedulefree \
  --eval_lengths 40 80 160 320 500 \
  --label myrun --output_dir results
```
