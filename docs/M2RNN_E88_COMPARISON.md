# M2RNN and E88 comparison plan

## Positioning

M2RNN is useful evidence for the same theoretical direction as E88: nonlinear
recurrent updates over matrix-valued state are a natural answer to state-capacity
limits in linear recurrent models. The clean framing is not that matrix state is
unique to E88. It is that M2RNN validates the premise, while E88 tests a deeper
claim: a pure nonlinear recurrent stack can be trained efficiently at 1.27B
parameters and can match the best linear recurrent baselines on language-model
quality.

The core empirical distinction to test is therefore:

- M2RNN: nonlinear matrix-state recurrence used primarily as a hybrid component.
- E88: pure nonlinear recurrent LM, optimized end to end with the Triton kernel.
- Expressivity axis: state-tracking tasks that are motivated by the theory but
  are not the central evaluation in M2RNN.

Avoid broad priority claims like "first nonlinear matrix-state RNN." The stronger
claim is narrower and more defensible: pure E88 can train at production scale,
does not need attention or linear-recurrent layers to work, and hybrids can
degrade the state-tracking behavior that nonlinear recurrence is meant to supply.

## Implemented harness support

Commit `5917a67` adds M2RNN support to the expressivity harness:

- `m2rnn`: tied-head M2RNN geometry used by the local search path.
- `m2rnn-paper`: grouped-head geometry matching the released M2RNN configs
  more closely: one q/k head, many v/f/g/W heads, K=64, V=`n_state`.
- Hybrid patterns can now include `m2rnn` and `m2rnn-paper` alongside `E88` and
  `fla-gdn`.

The harness disables bf16 autocast for M2RNN-containing expressivity runs because
the upstream XMA Triton path currently has a bf16-autocast compile edge case in
small synthetic-task shapes. This does not change the production Pile trainer.

## Current focused sweep

The first comparison should be small and decisive: use the existing canonical
paper protocol and run the state-tracking tasks M2RNN did not emphasize.

Command launched:

```bash
XMA_PATH=/tmp/m2rnn_xma PYTHONPATH=/tmp/m2rnn_xma \
python3 experiments/expressivity_tasks/run_canonical_sweep.py \
  --gpus 0 4 5 6 \
  --tasks modular_counter fsm_tracking parity \
  --patterns pure_M2RNN pure_M2RNN_paper \
             hybrid_GDN_M2RNN_single hybrid_GDN_E88_single \
  --seeds 42 123 456 \
  --output_dir /tmp/m2rnn_expressivity_canon
```

Protocol:

- dim 384
- depth 4
- n_heads 32
- n_state 32
- schedule-free AdamW
- 10K steps
- batch size 32

Output is intentionally under `/tmp` until the results are summarized:

```text
/tmp/m2rnn_expressivity_canon
```

## Readout

Compare against the existing canonical E88/FLA/hybrid table in
`experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`.

Important comparisons:

- `pure_M2RNN_paper` vs `pure_E88`
- `hybrid_GDN_M2RNN_single` vs `pure_M2RNN_paper`
- `hybrid_GDN_E88_single` vs `pure_E88`
- both hybrids vs `pure_FLA`
- E93 variants vs `pure_E88`, as a minimal single-matrix-state nonlinear
  recurrent control without the E88 head structure.

If M2RNN-paper is weak on modular counter or FSM tracking, that supports the
claim that the paper validated the matrix-state premise but did not optimize the
expressivity-relevant pure recurrent regime. If hybrids underperform pure E88,
that reinforces the existing result that hybridization is not the right default
for the computational-expressivity axis.

## E93 follow-up

E93 is the right minimal-control axis: one rectangular matrix state per layer,
delta update, learned row transform, and no explicit head abstraction. Recent
E93 ablations add optional output gating and bounded residual output, so it can
separate "matrix-state nonlinear recurrence is enough" from "E88's shaped
many-head design is necessary."

Useful quick sweep:

```bash
python3 experiments/expressivity_tasks/run_suite.py \
  --tasks modular_counter fsm_tracking parity \
  --models E93 E93_no_decay \
  --seeds 42 123 456 \
  --dim 384 --depth 4 \
  --output_dir /tmp/e93_expressivity_controls
```
