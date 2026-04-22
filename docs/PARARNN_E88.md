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

## Files

- `experiments/pararnn_e88_proto.py` — minimal PoC: Dense impl, verifies
  Newton convergence on E88's recurrence.
- (TODO) `elman/cuda/lib/pararnn_block32_gpu.cu.cc` — custom 32×32 kernel.
- (TODO) `experiments/pararnn_e88_full.py` — full per-head + batch +
  multi-layer integration.
