"""
Level log_5: Log-Space Triple R Elman

Log-space storage with THREE R matrices for complete recurrence control:
- R_h: Hidden state recurrence
- R_x: Input transformation (replaces W_x for symmetry)
- R_delta: Delta gate modulation from hidden state

Architecture:
    # All projections
    v = R_x @ x + R_h @ h_prev + b
    delta_raw = W_delta @ x + R_delta @ h_prev + b_delta
    delta = sigmoid(delta_raw)
    h_new = (1-delta) * h_prev + delta * tanh(v)

    # Selective output
    compete = softmax(h.reshape(groups), dim=-1)
    output = compete * silu(W_out @ h)

Key features:
- Triple R matrices for symmetric input/hidden/gate recurrence
- R_delta allows hidden state to modulate gating (input-state interaction)
- Most expressive recurrence structure in the ladder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import log-space utilities
from .logspace_polynomial import (
    LOG_ZERO, LOG_EPS, to_log_space, from_log_space,
    signed_log_add
)

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    HASTE_LOG_TRIPLE_R_AVAILABLE = hasattr(hasty_pytorch_lib, 'logspace_triple_r_forward')
except ImportError:
    hasty_pytorch_lib = None
    HASTE_LOG_TRIPLE_R_AVAILABLE = False

LOG_SPACE_TRIPLE_R_AVAILABLE = True  # PyTorch fallback always available


class LogSpaceTripleRFunction(torch.autograd.Function):
    """Autograd function for Log-Space Triple R Elman (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, x, log_h0, sign_h0, R_h, R_x, R_delta,
                W_delta, W_out, b, b_delta, n_groups):
        results = hasty_pytorch_lib.logspace_triple_r_forward(
            training,
            x.contiguous(),
            log_h0.contiguous(),
            sign_h0.contiguous(),
            R_h.contiguous(),
            R_x.contiguous(),
            R_delta.contiguous(),
            W_delta.contiguous(),
            W_out.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            n_groups
        )
        log_h, sign_h, output, v_cache, delta_cache, compete_cache = results

        if training:
            ctx.save_for_backward(
                x, R_h, R_x, R_delta, W_delta, W_out,
                log_h, sign_h, v_cache, delta_cache, compete_cache
            )
            ctx.n_groups = n_groups

        # Return h_linear as None since it's not computed in CUDA kernel
        return output, log_h, sign_h, None

    @staticmethod
    def backward(ctx, d_output, d_log_h, d_sign_h, d_h_linear):
        (x, R_h, R_x, R_delta, W_delta, W_out,
         log_h, sign_h, v_cache, delta_cache, compete_cache) = ctx.saved_tensors

        dx, dR_h, dR_x, dR_delta, dW_delta, dW_out, db, db_delta = \
            hasty_pytorch_lib.logspace_triple_r_backward(
                R_h, R_x, R_delta, W_delta, W_out,
                x, log_h, sign_h, v_cache, delta_cache, compete_cache,
                d_output.contiguous(),
                ctx.n_groups
            )

        return (None, dx, None, None, dR_h, dR_x, dR_delta,
                dW_delta, dW_out, db, db_delta, None)


class LogSpaceTripleRCell(nn.Module):
    """
    Log-Space Triple R Elman cell - Level log_5.

    Log-space storage with three R matrices for complete
    recurrence control: R_h, R_x, R_delta.

    Args:
        dim: Hidden dimension
        n_groups: Number of groups for compete softmax
        delta_init: Initial bias for delta gate
        r_h_mode: Constraint mode for R_h/R_delta matrices
        r_h_init_gain: Initial gain for R_h initialization
    """

    def __init__(self, dim, n_groups=32, delta_init=-2.0,
                 r_h_mode='spectral_norm', r_h_init_gain=0.1, **kwargs):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.group_size = dim // n_groups
        self.r_h_mode = r_h_mode
        self.r_h_init_gain = r_h_init_gain

        assert dim % n_groups == 0, f"dim {dim} must be divisible by n_groups {n_groups}"

        # Triple R matrices with optional constraints
        if r_h_mode == 'scaled_orthogonal':
            self.register_buffer('R_h_base', torch.empty(dim, dim))
            self.register_buffer('R_delta_base', torch.empty(dim, dim))
            self.r_h_log_scale = nn.Parameter(torch.tensor(-2.2))  # sigmoid(-2.2) ≈ 0.1
            self.r_delta_log_scale = nn.Parameter(torch.tensor(-2.2))
        else:
            self.R_h = nn.Parameter(torch.empty(dim, dim))
            self.R_delta = nn.Parameter(torch.empty(dim, dim))

        # R_x is input-only, no constraint needed
        self.R_x = nn.Parameter(torch.empty(dim, dim))

        # Bias
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Delta gate also uses linear W_delta @ x
        self.W_delta = nn.Parameter(torch.empty(dim, dim))

        # Output projection
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        if self.r_h_mode == 'scaled_orthogonal':
            nn.init.orthogonal_(self.R_h_base)
            nn.init.orthogonal_(self.R_delta_base)
        else:
            nn.init.orthogonal_(self.R_h, gain=self.r_h_init_gain)
            nn.init.orthogonal_(self.R_delta, gain=0.1)
        nn.init.orthogonal_(self.R_x, gain=1.0)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)
        nn.init.xavier_uniform_(self.W_out)

    def _apply_spectral_norm(self, W, name):
        """Apply spectral normalization to a weight matrix."""
        target_radius = 0.99
        u = getattr(self, f'_spectral_u_{name}', None)
        if u is None or u.shape[0] != self.dim:
            u = torch.randn(self.dim, device=W.device, dtype=W.dtype)
            u = u / u.norm()
        with torch.no_grad():
            for _ in range(3):
                v = W.T @ u
                v = v / (v.norm() + 1e-8)
                u = W @ v
                u = u / (u.norm() + 1e-8)
            setattr(self, f'_spectral_u_{name}', u)
        sigma = (u @ W @ v).abs()
        return W * (target_radius / (sigma + 1e-8))

    def get_R_h(self):
        """Get the effective R_h matrix with constraints applied."""
        if self.r_h_mode == 'scaled_orthogonal':
            scale = torch.sigmoid(self.r_h_log_scale)
            return scale * self.R_h_base
        elif self.r_h_mode == 'spectral_norm':
            return self._apply_spectral_norm(self.R_h, 'R_h')
        else:
            return self.R_h

    def get_R_delta(self):
        """Get the effective R_delta matrix with constraints applied."""
        if self.r_h_mode == 'scaled_orthogonal':
            scale = torch.sigmoid(self.r_delta_log_scale)
            return scale * self.R_delta_base
        elif self.r_h_mode == 'spectral_norm':
            return self._apply_spectral_norm(self.R_delta, 'R_delta')
        else:
            return self.R_delta

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: tuple of (log_h0, sign_h0) each [B, dim], or None

        Returns:
            output: [T, B, dim] selective output
            log_h: [T+1, B, dim] log magnitudes
            sign_h: [T+1, B, dim] signs
            h_linear: [T, B, dim] linear hidden states (None for CUDA)
        """
        T, B, D = x.shape

        if h0 is None:
            log_h0 = torch.full((B, self.dim), LOG_ZERO, device=x.device, dtype=x.dtype)
            sign_h0 = torch.ones(B, self.dim, device=x.device, dtype=x.dtype)
        else:
            log_h0, sign_h0 = h0

        # Get constrained matrices
        R_h = self.get_R_h()
        R_delta = self.get_R_delta()

        # Use CUDA kernel when available
        if HASTE_LOG_TRIPLE_R_AVAILABLE and x.is_cuda:
            return LogSpaceTripleRFunction.apply(
                self.training, x, log_h0, sign_h0,
                R_h, self.R_x, R_delta,
                self.W_delta, self.W_out, self.b, self.b_delta, self.n_groups
            )
        else:
            return self._forward_pytorch(x, log_h0, sign_h0, R_h, R_delta)

    def _forward_pytorch(self, x, log_h0, sign_h0, R_h, R_delta):
        """Pure PyTorch implementation with triple R matrices."""
        T, B, D = x.shape

        log_h_list = [log_h0]
        sign_h_list = [sign_h0]
        output_list = []
        h_linear_list = []

        for t in range(T):
            log_h_prev = log_h_list[-1]
            sign_h_prev = sign_h_list[-1]
            x_t = x[t]

            # Convert h_prev from log to linear
            h_prev_linear = from_log_space(log_h_prev, sign_h_prev)

            # v = R_x @ x + R_h @ h_prev + b
            v = x_t @ self.R_x.T + h_prev_linear @ R_h.T + self.b
            candidate = torch.tanh(v)

            # Delta gate: sigmoid(W_delta @ x + R_delta @ h_prev + b_delta)
            Rdelta_h = h_prev_linear @ R_delta.T
            delta_raw = x_t @ self.W_delta.T + Rdelta_h + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Gated update
            h_new = (1 - delta) * h_prev_linear + delta * candidate

            # Convert to log space
            log_h_new, sign_h_new = to_log_space(h_new)

            # Selective output
            h_reshaped = h_new.view(B, self.n_groups, self.group_size)
            compete = F.softmax(h_reshaped, dim=-1)
            compete = compete.view(B, D)

            w_out_h = h_new @ self.W_out.T
            silu_out = F.silu(w_out_h)
            output = compete * silu_out

            log_h_list.append(log_h_new)
            sign_h_list.append(sign_h_new)
            output_list.append(output)
            h_linear_list.append(h_new)

        log_h = torch.stack(log_h_list, dim=0)
        sign_h = torch.stack(sign_h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        h_linear = torch.stack(h_linear_list, dim=0)

        return output, log_h, sign_h, h_linear


class LogSpaceTripleR(nn.Module):
    """
    Log-Space Triple R Elman layer for use in LadderLM.

    Log-space storage with triple R matrices.
    Most expressive recurrence in the ladder.

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor
        n_groups: Number of groups for compete softmax
        delta_init: Initial bias for delta gate
        dropout: Dropout rate
        r_h_mode: Constraint mode for R_h/R_delta matrices
        r_h_init_gain: Initial gain for R_h initialization
        **kwargs: Ignored (for API compatibility)
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0,
                 dropout=0.0, r_h_mode='spectral_norm', r_h_init_gain=0.1, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.r_h_mode = r_h_mode

        # Adjust n_groups for inner dimension
        while self.d_inner % n_groups != 0 and n_groups > 1:
            n_groups -= 1

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Log-space triple R cell
        self.cell = LogSpaceTripleRCell(
            self.d_inner, n_groups=n_groups, delta_init=delta_init,
            r_h_mode=r_h_mode, r_h_init_gain=r_h_init_gain
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: tuple of (log_h0, sign_h0) or None

        Returns:
            output: [B, T, dim] output for residual connection
            h_final: tuple of (log_h_final, sign_h_final) for TBPTT
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)

        # Transpose to [T, B, d_inner] for cell
        x_proj = x_proj.transpose(0, 1).contiguous()

        # Run cell
        output, log_h, sign_h, h_linear = self.cell(x_proj, h0)

        # Transpose back
        output = output.transpose(0, 1).contiguous()

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        # Final hidden state for TBPTT
        log_h_final = log_h[-1]
        sign_h_final = sign_h[-1]
        h_final = (log_h_final, sign_h_final)

        return output, h_final


__all__ = [
    'LogSpaceTripleRCell',
    'LogSpaceTripleR',
    'LOG_SPACE_TRIPLE_R_AVAILABLE',
    'HASTE_LOG_TRIPLE_R_AVAILABLE',
]
