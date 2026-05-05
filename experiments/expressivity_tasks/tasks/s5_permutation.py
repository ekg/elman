"""S₅ permutation tracking — the canonical hardest state-tracking task.

S₅ is the symmetric group on 5 elements (120 permutations). It is the smallest
non-solvable group; provably outside what linear-scan SSMs can compute.
Reference: Merrill+Sabharwal "Illusion of State" (2024), DeltaProduct (2025).

Setup:
- Tokens 0..N-1 represent N generators of S₅ (small set of transpositions).
  We use 4 generators (sufficient to generate S₅): swap (1,2), (2,3), (3,4), (4,5).
- Model sees sequence of generators; must output the current permutation
  applied to a fixed starting permutation.

Targets: which permutation is the running composition? We encode permutation
as the index in [0, 120) lexicographically, so the model has 120-class output.

For practicality we predict only at the FINAL position (end-of-sequence eval)
or at every position (running track).
"""
import numpy as np
from itertools import permutations as _perm


def _all_perms_5():
    return list(_perm(range(5)))


_PERMS = _all_perms_5()
_PERM_TO_ID = {p: i for i, p in enumerate(_PERMS)}


def _apply(perm, swap):
    """Apply a transposition (swap=(i,j)) to a permutation tuple."""
    p = list(perm)
    i, j = swap
    p[i], p[j] = p[j], p[i]
    return tuple(p)


_GENERATORS = [(0, 1), (1, 2), (2, 3), (3, 4)]  # 4 adjacent transpositions of S₅


class S5PermutationTask:
    name = 's5_permutation'

    def __init__(self, mode: str = 'running', n_classes: int = 120):
        assert mode in ('running', 'final')
        self.mode = mode
        # vocab = N generators (4) + sentinel (5) = 5 input tokens
        # output classes = 120 permutations
        self.n_gen = len(_GENERATORS)
        self.n_classes = n_classes
        self.vocab_size = max(self.n_gen + 1, n_classes)

    def generate_batch(self, B: int, T: int, rng: np.random.Generator):
        gen_ids = rng.integers(0, self.n_gen, size=(B, T)).astype(np.int64)
        targets = np.zeros((B, T), dtype=np.int64)
        identity = tuple(range(5))

        for b in range(B):
            cur = identity
            for t in range(T):
                cur = _apply(cur, _GENERATORS[gen_ids[b, t]])
                targets[b, t] = _PERM_TO_ID[cur]

        inputs = gen_ids
        if self.mode == 'running':
            mask = np.ones((B, T), dtype=bool)
        else:
            mask = np.zeros((B, T), dtype=bool)
            mask[:, T-1] = True
        return inputs, targets, mask

    def random_baseline_acc(self):
        return 1.0 / self.n_classes
