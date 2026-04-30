# E93 paper-rigor plan (2026-04-30)

Working back from a paper claim. Earlier headline ("E93 no_decay beats FLA-GDN")
was wrong — based on noisy `last-100` metric + comparison against stale Pile-trained
FLA-GDN config.

## What we actually know

### Reliable facts (commapile, 60-min, multi-seed last-1000)
- E93 no_decay byte: 1.40 ± 0.01 (3 seeds)
- E93 no_decay tok:  5.20 ± 0.02 (3 seeds)
- FLA-GDN byte (commapile, bs=8, lr=3e-4, dim=1920 d=17):  1.17 (1 seed)
- FLA-GDN tok  (commapile, bs=8, lr=3e-4, dim=1920 d=17):  4.33 (1 seed)
→ FLA-GDN wins commapile by 0.23 byte / 0.87 tok.

### Pile historical results (different configs, single seed)
- E88 N=32:  5.44 tok last-1000  (dim=1408 d=11 h=380 n=32 bs=1)
- E88 N=16:  5.60 tok
- FLA-GDN:   5.69 tok  (dim=2560 d=20 exp=2 n=64 bs=1)
→ E88 N=32 beats FLA-GDN on Pile by 0.25 nats.

### Expressivity (small scale dim=128 d=4, fixed config — NOT rigorous)
- E88 N=32 wins parity (1.0), fsm_tracking (1.0), copy (0.94), strong on dyck (0.87).
- E93 vanilla wins assoc_recall (0.93).
- E93 no_decay loses on parity (0.57), fsm_tracking (0.33), dyck (0.63).

## Hypothesis

**E88 (N=32) may actually be the strongest architecture overall.** Heads + gating
+ decay give it parallel specialization that single-state E93 lacks.

The "best E93" question: does adding E88-style **silu output gate** to E93 close
the gap? We never tested that.

## Tonight's plan

### Phase A — finish current Pile CMA-ES (running, ~1.5 hours)
- E93 no_decay on Pile byte (GPUs 0-3) and tok (GPUs 4,5,7).
- Find Pile-optimal config for E93 no_decay (the one variant we have).

### Phase B — add silu output gate to E93 (1 hour, no GPUs needed)
1. Add `use_gate: bool = False` and `g_proj` to `E93Minimal.__init__`.
2. After `out = self.out_proj(out)`, multiply by `silu(g_proj(x))`.
3. Add `gate_activation: str = 'silu'` param (default silu, support sigmoid for completeness).
4. Test parity: existing tests still pass with use_gate=False.
5. Add Triton kernel flag if needed (probably just python wrapper applies gate
   after the kernel — kernel itself unchanged).
6. Wire into ladder_lm.py:
   - 'E93g' (E93 + silu gate)
   - 'E93g_no_decay' (no_decay + silu gate)

### Phase C — train.py timing fix (5 min, no GPUs)
- Change `--train_minutes` to count from first training step, not process start.
- Skips ~30-60s of model construction and memory probe overhead.

### Phase D — short CMA-ES for E93 + gate variants (3-min eval)
- Run CMA-ES for E93g and E93g_no_decay on Pile tokenized (the regime that matters
  for the headline claim).
- Use 3-min from-first-step training per eval.
- ~1-1.5 hours total per variant.

### Phase E — final 4-arch comparison (Pile, tokenized, multi-seed)
**Models:**
- E88 N=32 (Pile-tuned: dim=1408 d=11 h=380 n=32 bs=1 lr=1.5e-3)
- FLA-GDN (Pile-tuned: dim=2560 d=20 exp=2 n=64 bs=1 lr=1.7e-3)
- E93 best (whichever variant wins phase D ablation)
- Transformer (optional) — Pile-tuned best

**Setup:**
- Pile dataset (`/home/erikg/elman/data/pile.txt`)
- p50k_base tokenizer
- chunk_size 512
- 10-min training each (from first training step)
- 5 seeds: 42, 123, 456, 789, 1024
- last-1000 averaged loss for reporting

= 4 archs × 5 seeds = 20 runs at 10 min each
On 7 GPUs in parallel: 20 / 7 ≈ 3 rounds × ~12 min wall = ~40 min total

### Phase F (optional) — length scaling
After phase E settles, repeat ONCE per arch at chunk_size=2048. Mini scaling figure.

## Total compute estimate (after current CMA-ES finishes)
- B (no GPU): 1 hour
- C (no GPU): 5 min
- D: 1.5 hours (2 variants in parallel)
- E: 40 min
- F: 30 min
**= ~3.5 hours wall after current CMA-ES finishes.**

## What goes in the paper

Table 1: 4-arch × 5-seed Pile-tokenized comparison @ 10 min, last-1000 avg ± std.
Table 2: E93 ablation (vanilla, no_decay, +gate, no_decay+gate, linear, min_lin)
         at 1B, multi-seed, last-1000.
Table 3: Expressivity tasks with proper per-(task, model) CMA-ES and 3 seeds.
Table 4 (optional): chunk_size 512 vs 2048 scaling.

## Honest paper framing

NOT "we beat everyone with the simplest RNN."

CANDIDATE framings to evaluate after phase E:
1. **"Minimum viable nonlinear sequential matrix-state RNN"** — architecture-study,
   characterizes which ingredients are actually needed at scale.
2. **"E88 wins on Pile; here's why"** — pivot to the actual best architecture.
3. **"Long-context advantage"** — only if T=32K+ flips the result.

Decision criterion: pick the framing where our experimental matrix has the
strongest, most defensible claim.
