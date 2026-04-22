"""Minimal E88-like cell in ParaRNN — correctness prototype.

State: S ∈ R^{n×n} (single head, flattened to n²).
Update: S = tanh(decay · S + outer(v − S·k, k))

x_t is packed as [k_t (n), v_t (n), decay_scalar (1)] = 2n+1 dims.
h_t is S flattened (n² dims).

Uses RNNCellDenseImpl (no CUDA accel; full n²×n² Jacobian).
"""

from dataclasses import dataclass, field
import typing as typ
import torch
import torch.nn as nn

from pararnn.rnn_cell.rnn_cell import BaseRNNCell
from pararnn.rnn_cell.rnn_cell_impl import RNNCellDenseImpl, RNNCellBlockDiagImpl
from pararnn.rnn_cell.rnn_cell_utils import SystemParameters, Config

T = typ.TypeVar('T')


@dataclass(frozen=True)
class E88ProtoTrait:
    pass


@dataclass
class E88ProtoConfig(Config[E88ProtoTrait]):
    n_state: int = 8  # small for proto; final n²=64


@dataclass
class E88ProtoParams(SystemParameters[E88ProtoTrait]):
    n_state: int  # carried so recurrence_step can unflatten

    def unpack(self):
        return (self.n_state,)

    @classmethod
    def repack(cls, pars):
        return E88ProtoParams(n_state=pars[0])


class E88ProtoImpl(RNNCellDenseImpl[E88ProtoParams]):
    """Dense impl — verifies Newton converges for E88's recurrence."""

    @classmethod
    def recurrence_step(cls, x, h, system_parameters):
        """
        Args:
          x: [..., 2n+1]  input — [k(n), v(n), decay(1)]
          h: [..., n²]    previous state, flattened
        Returns:
          h_new: [..., n²]
        """
        (n,) = system_parameters.unpack()

        k = x[..., :n]
        v = x[..., n:2 * n]
        decay = x[..., 2 * n:2 * n + 1]  # [..., 1]

        # Unflatten state [..., n², 1] -> [..., n, n]
        S = h.reshape(*h.shape[:-1], n, n)

        # retrieved = S @ k  (einsum: s[i,j] * k[j] -> [i])
        retrieved = torch.einsum('...ij,...j->...i', S, k)
        delta = v - retrieved  # [..., n]

        # outer[i,j] = delta[i] * k[j]
        outer = torch.einsum('...i,...j->...ij', delta, k)

        # pre = decay·S + outer, with decay broadcast
        decay_ = decay.unsqueeze(-1)  # [..., 1, 1]
        pre = decay_ * S + outer

        S_new = torch.tanh(pre)
        return S_new.reshape(*h.shape[:-1], n * n)


class E88ProtoCell(BaseRNNCell[E88ProtoConfig, E88ProtoParams, E88ProtoImpl]):

    def __init__(self, config):
        super().__init__(config)

    def _specific_init(self, config):
        self._n_state = config.n_state

    @property
    def _system_parameters(self):
        return E88ProtoParams(n_state=self._n_state)


def make_impl(n):
    """Make a fresh BlockDiagImpl class with num_blocks = n."""
    impl = type(f'E88ProtoImpl_n{n}', (E88ProtoImpl,), {'_BLOCKS': n})
    return impl


def run_test(seq_length=64, batch_size=2, n=4, device='cuda'):
    """Compare sequential vs parallel on a small E88-proto."""
    torch.manual_seed(0)

    # Make impl class matching this n (block count)
    impl_cls = make_impl(n)

    config = E88ProtoConfig(
        state_dim=n * n,
        input_dim=2 * n + 1,
        mode='sequential',
        device=torch.device(device),
        dtype=torch.float32,
        n_state=n,
    )
    cell = E88ProtoCell(config).to(device)
    cell.impl_type = impl_cls  # override generic-derived impl with per-n version

    # Inputs: k and v drawn from N(0, 0.3), decay in (0.9, 1.0)
    x = torch.empty(batch_size, seq_length, 2 * n + 1, device=device)
    x[..., :2 * n] = 0.3 * torch.randn_like(x[..., :2 * n])
    x[..., 2 * n:2 * n + 1] = 0.9 + 0.1 * torch.rand_like(x[..., 2 * n:2 * n + 1])

    import time

    # Sequential reference
    from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode
    cell.mode = RNNCellApplicationMode.SEQUENTIAL
    torch.cuda.synchronize()
    t0 = time.time()
    h_seq = cell(x)
    torch.cuda.synchronize()
    t_seq = time.time() - t0
    print(f"Sequential: {t_seq * 1000:.2f} ms, h_seq shape {h_seq.shape}")

    # Parallel (pure PyTorch)
    cell.mode = RNNCellApplicationMode.PARALLEL
    torch.cuda.synchronize()
    t0 = time.time()
    h_par = cell(x)
    torch.cuda.synchronize()
    t_par = time.time() - t0
    print(f"Parallel:   {t_par * 1000:.2f} ms ({t_seq / t_par:.2f}× speedup)")

    # Numerical comparison
    diff = (h_seq - h_par).abs().max().item()
    print(f"Max abs diff (sequential vs parallel): {diff:.2e}")
    print(f"  Expected ~ seq_len × machine_precision = {seq_length * 1e-7:.2e}")

    # Also compare the final state
    diff_last = (h_seq[:, -1] - h_par[:, -1]).abs().max().item()
    print(f"Max abs diff at final position: {diff_last:.2e}")


if __name__ == '__main__':
    print("--- seq_length=32, n=4 (state=16) ---")
    run_test(seq_length=32, n=4)
    print("\n--- seq_length=128, n=4 (state=16) ---")
    run_test(seq_length=128, n=4)
    print("\n--- seq_length=128, n=8 (state=64) ---")
    run_test(seq_length=128, n=8)
    print("\n--- seq_length=512, n=8 (state=64) ---")
    run_test(seq_length=512, n=8)
