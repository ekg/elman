# E26: Parallel Dual-Memory Elman

E26 = E23 with explicit parallel/sequential separation (softmax attention).

## Performance (D=512, N=32)

- E26: 893K tok/s
- E23c: 2317K tok/s (2.6x faster, uses chunked batching)
- E25: 433K tok/s (2x slower, uses entmax)

## Architecture

```
PARALLEL PHASE:
  x_proj[0:T] = x @ W_x.T   # One big GEMM

SEQUENTIAL PHASE:
  for t in range(T):
    read = softmax(h_work @ tape.T) @ tape
    h_work = tanh(x_proj[t] + W_h @ h_work + read + b)
    tape = write(tape, h_work)
```

E23c is faster because it batches attention across chunks.
E26 is conceptually cleaner but per-timestep attention is slower.
