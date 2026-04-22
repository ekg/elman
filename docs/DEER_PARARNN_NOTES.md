# DEER, ELK, ParaRNN — literature notes and our contribution

## Related work chain

1. **DEER** (Lim et al., arXiv:2309.12252, ICLR 2024)
   - Newton's method on full-T residual. Scan over dense D×D Jacobians.
   - O(T·D³) work, O(T·D²) memory per Newton iteration.
   - At D=64 speedup over sequential drops to ~25%.
   - JAX code: https://github.com/machine-discovery/deer

2. **Quasi-DEER / ELK** (Gonzalez et al., arXiv:2407.19115, 2024)
   - Approximates Jacobian with diagonal only. O(T·D) work/memory.
   - ~2× more Newton iterations than DEER, 2.5× faster wall-clock.
   - Proposition 1: states 1..i are exact after Newton iter i.
   - Appendix A.1: sub-block-diagonal of J can be arbitrary and still
     converges in ≤T iterations.
   - Explicit open question (§7): "structured approximations of the
     Jacobian that still admit fast parallelism but are more faithful
     approximations may allow for more accurate quasi-Newton steps".
   - JAX code: https://github.com/lindermanlab/elk

3. **ParaRNN** (Apple, arXiv:2510.21450, Oct 2025)
   - Per-cell structured Jacobians + custom CUDA. Diagonal only.
   - Our target library; infrastructure documented in `PARARNN_E88.md`.

## Our contribution (for E88)

E88's Jacobian per row:
```
J = diag(decay · tanh'(pre)) − (tanh'(pre) ⊙ k) · kᵀ
  = D − u · kᵀ                      (diagonal + rank-1)
```

**Empirical finding**: prefix products `P_t = J_t · J_{t-1} · ... · J_1`
compress to rank-1 in their off-diagonal part at **machine epsilon**
(tested float64 at T≤512, n∈{4,8,16,32}, multiple seeds). See
`experiments/pararnn_kernel/phase2_structured.py`.

This fills exactly the gap called out in Gonzalez et al. §7 — a
structured Jacobian approximation that is both *faithful* (exact to
machine precision) and admits *fast parallelism* (O(T·D) work, D=n).

### Why r=1 is enough (hypothesis)

Likely connection to Gonzalez et al. Appendix A.3: when the scaled
eigenvalues of the Jacobian are strongly damped (ours are, via the
decay + tanh contraction), rank-1 perturbation directions align
across timesteps. The theory doesn't explicitly predict r=1 sufficiency
but it's consistent with the stability analysis.

Worth writing up as a standalone contribution.

## Design wisdom we'll adopt

From DEER/ELK papers:

1. **Warm start across training steps** — cache the previous step's
   converged trajectory as the Newton initial guess. Dramatic
   convergence speedup after warmup (§3.1 of Lim et al.).

2. **Zero-reset for diverged states** — Proposition 1 guarantees
   `states[0..i]` are exact after Newton iter `i`. Check for NaN/Inf
   in `states[i+1..T]` and reset to zero. Correctness preserved, just
   slower convergence.

3. **Tolerance targets**:
   - fp64: 1e-7
   - fp32: 1e-4
   - bf16: ~1e-3 (compatible with bf16 accumulation noise)

4. **Quadratic convergence near solution** — typical iteration count
   5–20 with warm start. Observed in Phase 1 tests (5-10 iters at fp64).

5. **Backward pass is ONE additional parallel scan**, asymptotically
   cheaper than forward. Gonzalez et al. §3.1.1 Eq. 6–7.

6. **Levenberg-Marquardt damping** (ELK paper) for unstable eigenvalues.
   E88 is naturally contractive via decay; probably won't need.

## Caveats

- Our rank-1 result is empirical in the regime tested (T≤512, contractive
  decay). Unclear if it extends to T=128K training contexts without
  additional checks. Phase 3-6 tests at larger T will probe this.
- Quasi-DEER's paper warns Newton can diverge when Jacobian eigenvalues
  exceed unit magnitude. Our decay gating (σ < 1) provides a natural
  guard, but long sequences with weak decay could cause trouble.
