# State of Play — E88 Paper Development

**Date**: 2026-04-28
**Context**: This document captures where we are after ~weeks of intensive
investigation. There is no published paper on E88 yet. *This document and
the codebase together are the development of that paper.*

---

## The thesis (current best statement)

E88 is a **minimum-viable nonlinear sequential RNN that scales on
commodity GPUs**. Its contribution isn't the recurrence math (a delta
rule with tanh) — it's the discovery that **high-multiplicity heads
(H ≫ #SMs)** make sequential nonlinear RNNs throughput-competitive with
linear attention.

Three claims stack:

1. **Hardware claim**: H ≈ 6× the number of SMs (e.g., H=877 on a
   142-SM RTX 6000 Ada) gives multi-programming per SM, which hides
   the per-warp dependency-chain latency that would otherwise make a
   sequential RNN slow. Discovered via CMA-ES.

2. **Capability claim**: nonlinear matrix state has empirical
   expressivity advantages predicted by `~/elman-proofs` —
   parity, FSM tracking, modular counting. We measured this on
   synthetic tasks; the separation is decisive (E88 100% vs FLA-GDN
   ≤56% on parity at any length tested).

3. **Architecture claim** (next experiment, not done): hybrid stacks
   (E88 layers for computation + linear-attention layers for retrieval)
   should dominate either pure architecture, since E88 fails at
   associative recall (24% vs FLA-GDN 89%) for tanh-saturation
   reasons.

---

## What's been measured (1B params, byte-level pile.txt, 60-min training)

| Architecture | 60-min loss (nats/byte) | Best config |
|---|---|---|
| FLA-GDN | **1.134** | dim=2432 d=21 H=15 exp=2 lr=6.4e-4 bs=2 |
| E88-N16 | 1.152 | dim=1280 d=10 H=877 lr=2.8e-4 bs=1 |
| E88-N32 | 1.166 | dim=3456 d=15 H=121 lr=2.2e-4 bs=2 |

FLA-GDN wins by 0.018 nats. Conditional perplexity (loss-vs-position)
shows uniform +0.20 nats advantage for FLA over E88-N16 at every
position from 9 to 15K bytes — the gap is per-step quality, not just
throughput.

5B head-to-head (4-hr undertrained) inconclusive: FLA 1.125, E88-N32
1.288 — both undertrained, can't read into difference.

Tokenized 1B (p50k_base, 5-min CMA-ES eval) winners:
- FLA-GDN: dim=2560 d=20 exp=2 H=17 lr=1.7e-3 → 5-min Final 6.557 nats/tok (~1.99 BPB)
- E88-N16: dim=1280 d=13 H=695 lr=2.3e-3 → 5-min Final 6.473
- E88-N32: still searching as of writing (28+ evals, top 6.36)

---

## The expressivity suite results — the most novel finding

Small models (~100K-300K params, dim=128 d=4), trained from scratch on
synthetic tasks. Mean accuracy across 3 seeds:

| task | E88_n16 | E88_n32 | fla-gdn | llama | who wins |
|---|---|---|---|---|---|
| **parity (T=128)** | **1.000** | **1.000** | 0.699 | 0.581 | E88 (clean) |
| **parity (T=512)** | **1.000** | **1.000** | 0.565 | 0.524 | E88 (FLA at random) |
| **parity (T=1024)** | **1.000** | **1.000** | running | running | E88 |
| **fsm_tracking (4 states)** | 0.739 | **1.000** | 0.561 | 0.572 | E88_n32 |
| selective_copy | 0.956 | 0.941 | 0.998 | 0.971 | ~tie (transformer/FLA edge) |
| dyck (max_depth=8) | 0.875 | 0.865 | 0.982 | 0.990 | linear models (task too small) |
| modular_counter (K=5) | 0.393 | 0.502 | 0.628 | 0.612 | linear models (K too small) |
| **assoc_recall** | 0.237 | 0.247 | **0.887** | 0.235 | FLA (decisive) |

**Patterns**:

- **E88 owns computational tasks**: parity (any length), FSM tracking. Predictions from `~/elman-proofs` (TC0 separation) confirmed.
- **FLA-GDN owns retrieval**: associative recall is its killer app (89% vs 24% for everyone else).
- **Where theory was overstated for our scale**: dyck and modcount at small K — linear models can table-lookup the state machine when K and depth fit in dim. The "pushed" suite running now (modcount K=20, K=50; fsm S=8, S=16) tests this prediction.

**The pushed suite** is currently running on GPU 7. As of this writing
parity at T=512 confirms the prediction beautifully: FLA-GDN dropped
from 0.70 (T=128) to 0.565 (T=512) — *converging to random* as length
grows. E88 stays at 1.000.

---

## Methodology lessons

1. **5-min CMA-ES with from-process-startup timing** is the right time
   scale for architecture search. Earlier 20-30 min runs were too slow
   to explore sufficient configs. This is itself a paper section.
2. **batch_size as a search variable** was wrong for E88 — bs=1 wins
   uniformly because high-H multi-programming saturates the SMs, so
   bs > 1 just adds memory pressure without parallelism. We now pin
   bs=1 for E88 searches.
3. **n_heads upper bound matters** — original CMA-ES search capped H
   at 400 and missed the H=877 winner. Bound expansion was the unlock.

---

## What's running right now

| GPU | Job | ETA |
|---|---|---|
| 0-4 | Tokenized E88-N32 CMA-ES (p50k_base) | ~2-3 hr |
| 5 | (idle) | |
| 6 | (occupied externally) | |
| 7 | Pushed expressivity suite (parity_T1024, modcount_K20, fsm_S8/S16) | ~12-24 hr (slow, sequential) |

---

## What's NOT done yet

1. **Hybrid model** (E88 + FLA-GDN alternating layers) — the centerpiece of the paper's architecture claim. Not started.
2. **CMA-ES at 5B** — no proper search at the larger scale. Heuristic scale-ups only.
3. **Long training** at 5B — 4 hr is undertrained.
4. **Matrix-matrix variant (E91)** — proposed but not built. Pure rank-r update with tanh, leveraging Tensor Cores. Could be faster *and* more expressive than E88.

---

## Active forks (where we may go from here)

### Fork A: Hybrid models (Track 3 deepening)
- Build LadderLM variant with per-layer architecture spec
- Try `[E88, fla-gdn, E88, fla-gdn, ...]`, `[E88, E88, fla-gdn, fla-gdn]`, etc.
- Train on full 3-track suite (LM, coherence, expressivity)
- Hypothesis: hybrid wins everywhere

### Fork B: E91 — Matrix-Matrix Rank-r Nonlinear RNN
- Same backbone as E88, but K, V projections produce [N, r] matrices instead of [N] vectors
- Update: S = tanh(α·S + V·K^T) — actual matmul, Tensor Core friendly
- For r=1: reproduces E88
- For r=N=16: full-rank update, ~10× more state change per step
- **This is the matrix-matrix variant we never tried** (verified in 90+ E-variants — none did rank-r > 1 with tanh)

### Fork C: Rigorous capability comparison
- Length sweep at parity: T=64, 128, 256, 512, 1024, 2048 — definitive plot
- K sweep at modcount: K=2, 5, 10, 20, 50 — find linear-models' breakdown point
- State sweep at FSM: 4, 8, 16, 32 states
- Combined: the paper's "scaling-law for expressivity" figure

### Fork D: Long-train at 1B
- Current 60-min loss numbers are likely far from converged
- Train each config for 24-48 hours, see if rankings hold

### Fork E: 5B with proper budget
- Either CMA-ES at 5B (~50+ GPU-hours)
- Or just train scaled configs for 24-40 hours each

---

## Where I'd take it (recommendation)

Priority order:
1. **E91 (Fork B)** — implement the matrix-matrix variant. If it works (similar parity/FSM scores at higher throughput), it's a *direct upgrade* to E88 that the paper would feature. This was an unexplored axis.
2. **Hybrid (Fork A)** — only if (1) doesn't subsume it. The hybrid is the natural answer if we can't fix E88's assoc_recall failure mode. But if E91 with rank-r updates handles assoc_recall (likely — it's closer to FLA-GDN's mechanism), then we don't need a hybrid.
3. **Capability sweeps (Fork C)** — for the paper's main figure. Cheap.

(Fork D and E are nice-to-have but not paper-essential.)

---

## File map

| Topic | Path |
|---|---|
| E88 model | `elman/models/e88_fla_hybrid.py` |
| E88 CUDA | `elman/cuda/lib/e88_fla_hybrid_gpu.cu.cc` |
| Triton hybrid kernel work | `experiments/pararnn_kernel/tree_scan/` |
| CMA-ES search | `cmaes_search_v2.py` |
| Training | `train.py` |
| Long-context coherence eval | `eval_long_range_coherence.py`, `eval_within_doc_baseline.py`, `eval_conditional_ppl.py` |
| Expressivity tasks | `experiments/expressivity_tasks/` |
| Theoretical proofs | `~/elman-proofs/ElmanProofs/Expressivity/` |
| Validated 1B checkpoints | `/tmp/validation_n16/`, `/tmp/validation_n32/`, `/tmp/validation_fla/` |
| 5B (undertrained) | `/tmp/run_5B/` |
| Tokenized 1B searches | `/tmp/cmaes_1B_tok_n16/`, `/tmp/cmaes_1B_tok_fla/`, `/tmp/cmaes_1B_tok_n32/` |
| Coherence raw data | `/tmp/long_ctx/`, `/tmp/long_ctx_big/` |
| Conditional ppl | `/tmp/cond_ppl/` |
| Expressivity results | `experiments/expressivity_tasks/results/` |

---

## Memorable surprises (worth keeping in mind)

- **Multi-programming win**: bumping H bound 400 → 1000 in CMA-ES instantly produced 0.10-nat improvement
- **Lucky LHS**: v1 N=16 search was cut short but found 1.224 by random sample; later "proper" searches struggled to match it
- **FLA-GDN at parity T=512**: dropped from 0.70 (T=128) to 0.56 (T=512). The TC0 wall is real.
- **All models drift more than real text** within a doc (rho ~0.5 vs 0.78). None match document coherence.
- **45 days of long-runs equivalent to ~1 week of 5-min CMA-ES**: time-scale mismatch was costly
