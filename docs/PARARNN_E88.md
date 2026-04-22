# E88 + ParaRNN: parallel nonlinear-RNN training

## The finding

E88's Jacobian is **block-diagonal with n blocks of size n×n**. Each row of
the matrix state S evolves independently given (k, v, decay). The `tanh` is
elementwise so it doesn't break the block structure.

This is *exactly* the structure `pararnn.RNNCellBlockDiagImpl` is designed for.

## Why block-diagonal?

```
delta[i] = v[i] − Σ_ℓ S[i,ℓ]·k[ℓ]      ← only row i of S
outer[i,j] = delta[i]·k[j]
S_new[i,j] = tanh(decay·S[i,j] + outer[i,j])
```

`S_new[i,j]` depends only on `S[i,:]`. No cross-row coupling inside one
recurrence step. Cross-row mixing happens only through the readout
`Sq = S @ q` which feeds the next layer's input, not the state update.

## Jacobian structure

Per row i, the n×n Jacobian block is:
```
J_i[j,b] = tanh'(pre[i,j]) · (decay·δ_{j,b} − k[b]·k[j])
        = diag(tanh') · (decay·I − k·kᵀ)
```

For the full flattened state [H·n·n]: **H·n independent n×n blocks**.

## Empirical verification

`experiments/pararnn_e88_proto.py` implements a minimal cell and shows:
- **Newton converges on E88's recurrence** (no proof needed, empirical).
- **Pure-PyTorch parallel mode matches sequential** within machine-precision
  × seq_len at small n with Dense impl.

Next step: custom CUDA kernel for num_blocks=n=32 block-diagonal case. ParaRNN
ships kernels only for 2×2 and 3×3; larger sizes are a "supported extension"
per the upstream README.

## Why this is a big deal

- Nonlinear sequential RNNs were considered un-parallelizable along the time
  axis. ParaRNN (Apple, Oct 2025) showed Newton's method changes this.
- Most ParaRNN demos are vector-state (diagonal Jacobian). E88 is the first
  published nonlinear matrix-state RNN compatible with ParaRNN's block-diag
  path.
- If we land a 32×32 block-diag CUDA kernel, E88 becomes **sequence-
  parallelizable on Frontier**, breaking the "sequential RNN can't use CP"
  constraint that was shaping our distributed training plan.

## Status of the implementation attempt

**What works:**
- `RNNCellDenseImpl` with `max_its=10`: numerically correct on E88's
  recurrence. Newton converges. `diff ≤ 1e-7` for float32 at
  seq_len=512, n=8.

**What doesn't:**
- `RNNCellBlockDiagImpl`: produces wrong answers. ParaRNN's source docstring
  reveals BlockDiag assumes **diagonal-within-block** structure (N blocks
  of N diagonal entries each). E88's blocks are dense — ParaRNN can't
  represent them with its current infrastructure.
- Template instantiation for `num_hidden_vars=32` in ParaRNN's existing
  CUDA kernel: NVCC compile hangs (>15 min) due to register blowup on the
  register-owned reduction at block_size=32.

**Why Dense is too slow for us:**
- Uses `torch.func.jacrev` to build full n²×n² Jacobian per timestep.
- Newton solve is O(n⁶). At seq=512, n=8: parallel is 17× *slower* than
  sequential. Worse at n=32.

## What we actually need — custom kernel for diag+rank-1 blocks

E88's per-row Jacobian has known structure:

```
J_i = diag(decay · tanh'(pre[i,:])) - (tanh'(pre[i,:]) ⊙ k) ⊗ k
    = D_i - u_i ⊗ k
```

This is **diagonal + rank-1**. Sherman-Morrison gives O(n) inversion
instead of O(n³). The parallel reduction step can be rewritten to
propagate `(D, u, v)` triples through the associative combine, preserving
structure without ever materializing dense blocks.

This is the right kernel to write. Non-trivial — probably a 2-3 day
engineering sprint plus numerical testing. Tracked as future work.

## Files

- `experiments/pararnn_e88_proto.py` — minimal PoC cell using Dense impl,
  verifies Newton convergence on E88's recurrence (correct, but slow).
- (TODO) custom CUDA kernel for diag+rank-1 block-diagonal parallel
  reduction.
- (TODO) `experiments/pararnn_e88_full.py` — full per-head + batch +
  multi-layer integration once the kernel is ready.

## Takeaway for the Frontier plan

**Main finding**: ParaRNN in its current form can't be dropped in for E88.
We have a mathematical foundation (block-diag confirmed, Newton converges,
rank-1 structure identified) but need custom kernel engineering to make
it practically fast. This changes the Frontier plan: don't budget a full
run for ParaRNN acceleration unless the kernel lands first.

The "E88 gets sequence-parallelism for free via ParaRNN" hope is *not*
ruled out, but also not a 1-week drop-in. It's a real engineering
project that should be scoped separately.
