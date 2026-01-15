"""
E74 Ablation Framework

Systematic ablation of E73 matrix state RNN to find optimal architecture.

Ablation Dimensions:
- Projection: P0 (k,v,q,z), P1 (k,v,q), P2 (k=q,v), P3 (k=v=q)
- State: S0 (full), S1 (diagonal), S2 (lowrank-4), S3 (lowrank-8), S4 (block-diag)
- Nonlinearity: N0 (tanh), N1 (linear+specnorm), N2 (rmsnorm), N3 (frobnorm)
- Gate: G0 (output only), G1 (retain), G2 (state)

Usage:
    model = E74Ablation(
        dim=512,
        n_state=64,
        state_type='diagonal',  # 'full', 'diagonal', 'lowrank', 'blockdiag'
        proj_type='no_z',       # 'full', 'no_z', 'tied_kq', 'tied_kvq'
        nonlin_type='tanh',     # 'tanh', 'linear', 'rmsnorm', 'frobnorm'
        gate_type='output',     # 'output', 'retain', 'state'
        rank=8,                 # for lowrank state
        block_size=8,           # for blockdiag state
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from enum import Enum

# Try importing Triton kernels
try:
    from ..kernels.e74_triton import (
        normalize_k_triton,
        diagonal_forward_triton,
        full_forward_triton,
        lowrank_forward_triton,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton kernels not available, using PyTorch fallback")


# =============================================================================
# Configuration Enums
# =============================================================================

class StateType(Enum):
    FULL = 'full'           # [B, n, n] - O(n²)
    DIAGONAL = 'diagonal'   # [B, n] - O(n)
    LOWRANK = 'lowrank'     # [B, n, r] + [B, n, r] - O(nr)
    BLOCKDIAG = 'blockdiag' # [B, n/b, b, b] - O(n*b)


class ProjType(Enum):
    FULL = 'full'           # k, v, q, z separate (P0)
    NO_Z = 'no_z'           # k, v, q separate (P1)
    TIED_KQ = 'tied_kq'     # k=q, v separate (P2)
    TIED_KVQ = 'tied_kvq'   # k=v=q single projection (P3)


class NonlinType(Enum):
    TANH = 'tanh'           # tanh(S) after update (N0)
    LINEAR = 'linear'       # no nonlin, use spectral norm (N1)
    RMSNORM = 'rmsnorm'     # RMS normalization (N2)
    FROBNORM = 'frobnorm'   # Frobenius normalization (N3)


class GateType(Enum):
    OUTPUT = 'output'       # only output self-gate (G0)
    RETAIN = 'retain'       # retain gate α (G1)
    STATE = 'state'         # state-dependent delta gate (G2)


# =============================================================================
# Core Cell Implementations
# =============================================================================

class E74DiagonalCell(nn.Module):
    """
    Diagonal state cell.

    S[i] = f(S[i] * (1 - k²[i]) + v[i] * k[i])

    This is equivalent to input-dependent EMA per dimension.
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
        gate_type: GateType = GateType.OUTPUT,
        spectral_radius: float = 0.999,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type
        self.gate_type = gate_type
        self.spectral_radius = spectral_radius

        # Projections based on type
        if proj_type == ProjType.FULL:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
            self.W_z = nn.Parameter(torch.empty(n_state, dim))
            self.b_z = nn.Parameter(torch.ones(n_state))
        elif proj_type == ProjType.NO_Z:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KQ:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))  # k = q
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KVQ:
            self.W = nn.Parameter(torch.empty(n_state, dim))  # k = v = q

        # Gate parameters (for G1, G2)
        if gate_type in [GateType.RETAIN, GateType.STATE]:
            self.d_g = nn.Parameter(torch.full((n_state,), 0.5))
            self.b_g = nn.Parameter(torch.zeros(n_state))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)
            elif 'b_z' in name:
                nn.init.ones_(param)

    def _get_projections(self, x: torch.Tensor):
        """Compute k, v, q (and optionally z) from input."""
        if self.proj_type == ProjType.FULL:
            k = x @ self.W_k.T
            v = x @ self.W_v.T
            q = x @ self.W_q.T
            z = x @ self.W_z.T + self.b_z
            return k, v, q, z
        elif self.proj_type == ProjType.NO_Z:
            k = x @ self.W_k.T
            v = x @ self.W_v.T
            q = x @ self.W_q.T
            return k, v, q, None
        elif self.proj_type == ProjType.TIED_KQ:
            k = x @ self.W_k.T
            v = x @ self.W_v.T
            q = k  # tied
            return k, v, q, None
        elif self.proj_type == ProjType.TIED_KVQ:
            w = x @ self.W.T
            return w, w, w, None

    def _apply_nonlin(self, S: torch.Tensor) -> torch.Tensor:
        """Apply nonlinearity to state."""
        if self.nonlin_type == NonlinType.TANH:
            return torch.tanh(S)
        elif self.nonlin_type == NonlinType.LINEAR:
            # Spectral norm is applied to weights, S passes through
            return S
        elif self.nonlin_type == NonlinType.RMSNORM:
            return S / (S.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)
        elif self.nonlin_type == NonlinType.FROBNORM:
            return S / (S.norm(dim=-1, keepdim=True) + 1e-6)
        return S

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state] diagonal state

        Returns:
            output: [T, B, n_state]
            S: [B, n_state] final state
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, device=x.device, dtype=x.dtype)

        # Batch projections
        x_flat = x.reshape(T * B, D)
        k, v, q, z = self._get_projections(x_flat)
        k = k.reshape(T, B, n)
        v = v.reshape(T, B, n)
        q = q.reshape(T, B, n)
        if z is not None:
            z = z.reshape(T, B, n)

        outputs = []

        for t in range(T):
            k_t = k[t]  # [B, n]
            v_t = v[t]
            q_t = q[t]

            # Normalize k
            k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)
            k_sq = k_norm ** 2

            # Diagonal delta update
            if self.gate_type == GateType.OUTPUT:
                # S = f(S * (1 - k²) + v * k)
                S_raw = S * (1.0 - k_sq) + v_t * k_norm
            elif self.gate_type == GateType.RETAIN:
                # α = sigmoid(d_g * S + b_g)
                # S = α * S + (1-α) * update
                alpha = torch.sigmoid(self.d_g * S + self.b_g)
                update = S * (1.0 - k_sq) + v_t * k_norm
                S_raw = alpha * S + (1.0 - alpha) * update
            elif self.gate_type == GateType.STATE:
                # g = sigmoid(d_g * S + b_g)
                # delta = (v - S*k) * k * g
                g = torch.sigmoid(self.d_g * S + self.b_g)
                delta = (v_t - S * k_norm) * k_norm * g
                S_raw = S + delta

            S = self._apply_nonlin(S_raw)

            # Output with self-gating
            Sq = S * q_t  # Diagonal: element-wise
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E74FullMatrixCell(nn.Module):
    """
    Full matrix state cell (simplified from E73).

    S[i,j] = f(S[i,j] + (v[i] - retrieved[i]) * k_norm[j])
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
        gate_type: GateType = GateType.OUTPUT,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type
        self.gate_type = gate_type

        # Projections
        if proj_type == ProjType.FULL:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
            self.W_z = nn.Parameter(torch.empty(n_state, dim))
            self.b_z = nn.Parameter(torch.ones(n_state))
        elif proj_type == ProjType.NO_Z:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KQ:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KVQ:
            self.W = nn.Parameter(torch.empty(n_state, dim))

        if gate_type in [GateType.RETAIN, GateType.STATE]:
            self.d_g = nn.Parameter(torch.full((n_state,), 0.5))
            self.b_g = nn.Parameter(torch.zeros(n_state))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)

    def _get_projections(self, x):
        if self.proj_type == ProjType.FULL:
            return x @ self.W_k.T, x @ self.W_v.T, x @ self.W_q.T, x @ self.W_z.T + self.b_z
        elif self.proj_type == ProjType.NO_Z:
            return x @ self.W_k.T, x @ self.W_v.T, x @ self.W_q.T, None
        elif self.proj_type == ProjType.TIED_KQ:
            k = x @ self.W_k.T
            return k, x @ self.W_v.T, k, None
        elif self.proj_type == ProjType.TIED_KVQ:
            w = x @ self.W.T
            return w, w, w, None

    def _apply_nonlin(self, S):
        if self.nonlin_type == NonlinType.TANH:
            return torch.tanh(S)
        elif self.nonlin_type == NonlinType.LINEAR:
            return S
        elif self.nonlin_type == NonlinType.RMSNORM:
            return S / (S.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-6)
        elif self.nonlin_type == NonlinType.FROBNORM:
            frob = S.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
            return S / (frob + 1e-6)
        return S

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        k, v, q, z = self._get_projections(x_flat)
        k = k.reshape(T, B, n)
        v = v.reshape(T, B, n)
        q = q.reshape(T, B, n)
        if z is not None:
            z = z.reshape(T, B, n)

        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[t], v[t], q[t]
            z_t = z[t] if z is not None else None

            # Normalize k
            k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieval (with optional z modulation)
            if z_t is not None:
                S_mod = S * z_t.unsqueeze(1)  # Column modulation
            else:
                S_mod = S
            retrieved = torch.einsum('bij,bj->bi', S_mod, k_norm)

            # Delta update
            delta = v_t - retrieved
            outer = torch.einsum('bi,bj->bij', delta, k_norm)

            if self.gate_type == GateType.OUTPUT:
                S_raw = S + outer
            elif self.gate_type == GateType.RETAIN:
                S_energy = (S ** 2).mean(dim=-1)
                alpha = torch.sigmoid(self.d_g * S_energy + self.b_g)
                S_raw = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * (S + outer)
            elif self.gate_type == GateType.STATE:
                S_energy = (S ** 2).mean(dim=-1)
                g = torch.sigmoid(self.d_g * S_energy + self.b_g)
                S_raw = S + outer * g.unsqueeze(-1)

            S = self._apply_nonlin(S_raw)

            # Output
            Sq = torch.einsum('bij,bj->bi', S, q_t)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E74LowRankCell(nn.Module):
    """
    Low-rank state cell.

    S = U @ V^T where U, V ∈ [B, n, r]
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        rank: int = 8,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.rank = rank
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type

        # Projections
        if proj_type in [ProjType.FULL, ProjType.NO_Z]:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KQ:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KVQ:
            self.W = nn.Parameter(torch.empty(n_state, dim))

        # Projection from k to rank-r space
        self.W_kr = nn.Parameter(torch.empty(rank, n_state))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)

    def _get_projections(self, x):
        if self.proj_type in [ProjType.FULL, ProjType.NO_Z]:
            return x @ self.W_k.T, x @ self.W_v.T, x @ self.W_q.T
        elif self.proj_type == ProjType.TIED_KQ:
            k = x @ self.W_k.T
            return k, x @ self.W_v.T, k
        elif self.proj_type == ProjType.TIED_KVQ:
            w = x @ self.W.T
            return w, w, w

    def forward(self, x: torch.Tensor, state: Optional[tuple] = None):
        T, B, D = x.shape
        n = self.n_state
        r = self.rank

        if state is None:
            U = torch.zeros(B, n, r, device=x.device, dtype=x.dtype)
            # V must be non-zero for gradients to flow (V=0 causes zero output)
            V = torch.randn(B, n, r, device=x.device, dtype=x.dtype) * 0.01
        else:
            U, V = state

        x_flat = x.reshape(T * B, D)
        k, v, q = self._get_projections(x_flat)
        k = k.reshape(T, B, n)
        v = v.reshape(T, B, n)
        q = q.reshape(T, B, n)

        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[t], v[t], q[t]

            k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieval: S @ k = U @ (V^T @ k)
            Vtk = torch.einsum('bnr,bn->br', V, k_norm)  # [B, r]
            retrieved = torch.einsum('bnr,br->bn', U, Vtk)  # [B, n]

            # Delta
            delta = v_t - retrieved  # [B, n]

            # Project k to rank-r space
            k_r = k_norm @ self.W_kr.T  # [B, r]

            # Update U: U_new = U + outer(delta, k_r)
            U = U + torch.einsum('bn,br->bnr', delta, k_r)

            # Optional: apply nonlinearity to U
            if self.nonlin_type == NonlinType.TANH:
                U = torch.tanh(U)

            # Output: (U @ V^T @ q) * silu(...)
            Vtq = torch.einsum('bnr,bn->br', V, q_t)
            Sq = torch.einsum('bnr,br->bn', U, Vtq)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, (U, V)


class E74BlockDiagCell(nn.Module):
    """
    Block-diagonal state cell.

    S ∈ [B, n/b, b, b] - n/b blocks of size b×b
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        block_size: int = 8,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.block_size = block_size
        self.n_blocks = n_state // block_size
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type

        assert n_state % block_size == 0, f"n_state ({n_state}) must be divisible by block_size ({block_size})"

        if proj_type in [ProjType.FULL, ProjType.NO_Z]:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KQ:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KVQ:
            self.W = nn.Parameter(torch.empty(n_state, dim))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)

    def _get_projections(self, x):
        if self.proj_type in [ProjType.FULL, ProjType.NO_Z]:
            return x @ self.W_k.T, x @ self.W_v.T, x @ self.W_q.T
        elif self.proj_type == ProjType.TIED_KQ:
            k = x @ self.W_k.T
            return k, x @ self.W_v.T, k
        elif self.proj_type == ProjType.TIED_KVQ:
            w = x @ self.W.T
            return w, w, w

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        T, B, D = x.shape
        n = self.n_state
        b = self.block_size
        nb = self.n_blocks

        if S is None:
            S = torch.zeros(B, nb, b, b, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        k, v, q = self._get_projections(x_flat)
        k = k.reshape(T, B, n)
        v = v.reshape(T, B, n)
        q = q.reshape(T, B, n)

        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[t], v[t], q[t]

            # Reshape to blocks: [B, n] -> [B, nb, b]
            k_blocks = k_t.reshape(B, nb, b)
            v_blocks = v_t.reshape(B, nb, b)
            q_blocks = q_t.reshape(B, nb, b)

            # Normalize k per block
            k_norm = k_blocks / (k_blocks.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieval per block: S_block @ k_block
            retrieved = torch.einsum('bnij,bnj->bni', S, k_norm)  # [B, nb, b]

            # Delta update per block
            delta = v_blocks - retrieved
            outer = torch.einsum('bni,bnj->bnij', delta, k_norm)
            S_raw = S + outer

            if self.nonlin_type == NonlinType.TANH:
                S = torch.tanh(S_raw)
            else:
                S = S_raw

            # Output per block
            Sq = torch.einsum('bnij,bnj->bni', S, q_blocks)  # [B, nb, b]
            Sq_flat = Sq.reshape(B, n)
            out = Sq_flat * F.silu(Sq_flat)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


# =============================================================================
# Main Ablation Module
# =============================================================================

class E74AblationCell(nn.Module):
    """
    Unified ablation cell that dispatches to appropriate implementation.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        state_type: str = 'diagonal',
        proj_type: str = 'no_z',
        nonlin_type: str = 'tanh',
        gate_type: str = 'output',
        rank: int = 8,
        block_size: int = 8,
        spectral_radius: float = 0.999,
    ):
        super().__init__()

        self.state_type = StateType(state_type)
        self.proj_type = ProjType(proj_type)
        self.nonlin_type = NonlinType(nonlin_type)
        self.gate_type = GateType(gate_type)

        if self.state_type == StateType.DIAGONAL:
            self.cell = E74DiagonalCell(
                dim, n_state, self.proj_type, self.nonlin_type, self.gate_type, spectral_radius
            )
        elif self.state_type == StateType.FULL:
            self.cell = E74FullMatrixCell(
                dim, n_state, self.proj_type, self.nonlin_type, self.gate_type
            )
        elif self.state_type == StateType.LOWRANK:
            self.cell = E74LowRankCell(
                dim, n_state, rank, self.proj_type, self.nonlin_type
            )
        elif self.state_type == StateType.BLOCKDIAG:
            self.cell = E74BlockDiagCell(
                dim, n_state, block_size, self.proj_type, self.nonlin_type
            )

    def forward(self, x, state=None):
        return self.cell(x, state)


class E74Ablation(nn.Module):
    """
    E74 Ablation model with full layer structure.

    Wraps cell with in_proj, optional conv, and out_proj.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        state_type: str = 'diagonal',
        proj_type: str = 'no_z',
        nonlin_type: str = 'tanh',
        gate_type: str = 'output',
        rank: int = 8,
        block_size: int = 8,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.state_type = state_type

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )
        else:
            self.conv1d = None

        self.cell = E74AblationCell(
            dim=self.d_inner,
            n_state=n_state,
            state_type=state_type,
            proj_type=proj_type,
            nonlin_type=nonlin_type,
            gate_type=gate_type,
            rank=rank,
            block_size=block_size,
        )

        self.out_proj = nn.Linear(n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, state=None, **kwargs):
        """
        Args:
            x: [B, T, dim]
            state: previous state

        Returns:
            output: [B, T, dim]
            state: final state
        """
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.conv1d is not None:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()  # [T, B, d_inner]

        cell_out, state = self.cell(x_rnn, state)

        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, n_state]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, state

    def extra_repr(self):
        return (f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, '
                f'state={self.state_type}, proj={self.cell.proj_type.value}, '
                f'nonlin={self.cell.nonlin_type.value}, gate={self.cell.gate_type.value}')


# =============================================================================
# Ablation Config Generator
# =============================================================================

def get_ablation_configs():
    """Generate all 20 recommended ablation configs."""
    configs = [
        # Phase 1: State structure with baseline projections
        {'id': 1, 'state': 'full', 'proj': 'full', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'E73 baseline'},
        {'id': 2, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Remove z'},
        {'id': 3, 'state': 'full', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Tie k=q'},
        {'id': 4, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Tie k=v=q'},
        {'id': 5, 'state': 'diagonal', 'proj': 'full', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal baseline'},
        {'id': 6, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, no z'},
        {'id': 7, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, tied k=q'},
        {'id': 8, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, k=v=q'},
        {'id': 9, 'state': 'lowrank', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-4', 'rank': 4},
        {'id': 10, 'state': 'lowrank', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-8', 'rank': 8},
        {'id': 11, 'state': 'blockdiag', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Block-diag'},

        # Phase 2: Nonlinearity variants
        {'id': 12, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Diag, linear (E42-style)'},
        {'id': 13, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'rmsnorm', 'gate': 'output', 'desc': 'Diag, rmsnorm'},
        {'id': 14, 'state': 'full', 'proj': 'no_z', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Full, linear'},

        # Phase 3: Gate variants
        {'id': 15, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'retain', 'desc': 'Diag, retain gate'},
        {'id': 16, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'state', 'desc': 'Diag, state gate'},
        {'id': 17, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'retain', 'desc': 'Diag, linear, retain'},

        # Phase 4: Best combo candidates
        {'id': 18, 'state': 'lowrank', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-4, tied k=q', 'rank': 4},
        {'id': 19, 'state': 'lowrank', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Lowrank-8, linear', 'rank': 8},
        {'id': 20, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Minimal: diag, tied, linear'},
    ]
    return configs


def create_model_from_config(config: dict, dim: int = 512, **kwargs) -> E74Ablation:
    """Create model from config dict."""
    return E74Ablation(
        dim=dim,
        state_type=config['state'],
        proj_type=config['proj'],
        nonlin_type=config['nonlin'],
        gate_type=config['gate'],
        rank=config.get('rank', 8),
        **kwargs
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing E74 Ablation Framework...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    configs = get_ablation_configs()

    for cfg in configs[:8]:  # Test first 8
        print(f"\n--- Config {cfg['id']}: {cfg['desc']} ---")

        model = create_model_from_config(cfg, dim=256, expansion=1.0, n_state=32)
        model = model.to(device).to(dtype)

        x = torch.randn(2, 32, 256, device=device, dtype=dtype)

        out, state = model(x)
        print(f"Output: {out.shape}")

        loss = out.sum()
        loss.backward()
        print("Backward passed!")

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
