"""
Level log_6: Log-Space Diagonal Triple R Elman

Log-space storage with DIAGONAL r_h and r_delta vectors:
- r_h: Element-wise recurrence multiplier (like Mamba2's diagonal A)
- r_delta: Element-wise delta gate modulation from hidden state

Architecture (diagonal version of log_5):
    v = r_h * h_prev + W_x @ x + b              -- DIAGONAL r_h (element-wise)
    delta_raw = W_delta @ x + r_delta * h_prev + b_delta  -- DIAGONAL r_delta
    delta = sigmoid(delta_raw)
    h_new = (1-delta) * h_prev + delta * tanh(v)

    # Selective output
    compete = softmax(h.reshape(groups), dim=-1)
    output = compete * silu(W_out @ h)

Key features:
- Diagonal r_h for efficient recurrence (O(d) vs O(d²))
- Diagonal r_delta for hidden-state modulated gating
- More efficient than full R matrices while retaining expressivity
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
    HASTE_LOG_DIAG_TRIPLE_R_AVAILABLE = hasattr(hasty_pytorch_lib, 'logspace_diag_triple_r_forward')
except ImportError:
    hasty_pytorch_lib = None
    HASTE_LOG_DIAG_TRIPLE_R_AVAILABLE = False

LOGSPACE_LEVEL_6_AVAILABLE = True  # PyTorch fallback always available


class LogSpaceDiagTripleRFunction(torch.autograd.Function):
    """Autograd function for Log-Space Diagonal Triple R Elman (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, n_groups, x, log_h0, sign_h0,
                W_x, log_r_h, sign_r_h, log_r_delta, sign_r_delta,
                W_delta, W_out, b, b_delta, log_gamma):
        results = hasty_pytorch_lib.logspace_diag_triple_r_forward(
            training,
            n_groups,
            x.contiguous(),
            log_h0.contiguous(),
            sign_h0.contiguous(),
            W_x.contiguous(),
            log_r_h.contiguous(),
            sign_r_h.contiguous(),
            log_r_delta.contiguous(),
            sign_r_delta.contiguous(),
            W_delta.contiguous(),
            W_out.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            log_gamma.contiguous()
        )

        (log_h, sign_h, output, h_linear_cache, log_v_cache, sign_v_cache,
         log_h_unbounded_cache, delta_cache, weight_rh_cache,
         rdelta_h_cache, compete_cache, log_rms_cache) = results

        if training:
            ctx.save_for_backward(
                W_x, log_r_h, sign_r_h, log_r_delta, sign_r_delta,
                W_delta, W_out, log_gamma,
                x, log_h, sign_h, log_v_cache, sign_v_cache,
                log_h_unbounded_cache, delta_cache, weight_rh_cache,
                rdelta_h_cache, h_linear_cache, compete_cache, log_rms_cache
            )
            ctx.n_groups = n_groups

        return output, log_h, sign_h, h_linear_cache

    @staticmethod
    def backward(ctx, d_output, d_log_h, d_sign_h, d_h_linear):
        (W_x, log_r_h, sign_r_h, log_r_delta, sign_r_delta,
         W_delta, W_out, log_gamma,
         x, log_h, sign_h, log_v_cache, sign_v_cache,
         log_h_unbounded_cache, delta_cache, weight_rh_cache,
         rdelta_h_cache, h_linear_cache, compete_cache, log_rms_cache) = ctx.saved_tensors

        grads = hasty_pytorch_lib.logspace_diag_triple_r_backward(
            ctx.n_groups,
            W_x, log_r_h, sign_r_h, log_r_delta, sign_r_delta,
            W_delta, W_out, log_gamma,
            x, log_h, sign_h, log_v_cache, sign_v_cache,
            log_h_unbounded_cache, delta_cache, weight_rh_cache,
            rdelta_h_cache, h_linear_cache, compete_cache, log_rms_cache,
            d_output.contiguous()
        )

        # Unpack: dx, dW_x, d_log_r_h, d_log_r_delta, dW_delta, dW_out, db, db_delta, d_log_gamma
        dx, dW_x, d_log_r_h, d_log_r_delta, dW_delta, dW_out, db, db_delta, d_log_gamma = grads

        # Return gradients in same order as forward inputs
        return (None, None, dx, None, None,
                dW_x, d_log_r_h, None, d_log_r_delta, None,
                dW_delta, dW_out, db, db_delta, d_log_gamma)


class LogSpaceDiagTripleRCell(nn.Module):
    """
    Log-Space Diagonal Triple R Elman cell - Level log_6.

    Diagonal recurrence with r_h and r_delta vectors.
    Efficient O(d) recurrence with hidden-state modulated gating.

    Args:
        dim: Hidden dimension
        n_groups: Number of groups for compete softmax
        delta_init: Initial bias for delta gate
        r_h_mode: Constraint mode for r_h ('spectral_norm', 'clamp', 'none')
        r_h_init: Initial value for r_h (small for stability)
    """

    def __init__(self, dim, n_groups=32, delta_init=-2.0,
                 r_h_mode='spectral_norm', r_h_init=0.1, **kwargs):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.group_size = dim // n_groups
        self.r_h_mode = r_h_mode

        assert dim % n_groups == 0, f"dim {dim} must be divisible by n_groups {n_groups}"

        # Diagonal r_h (stored in log space)
        self.log_r_h = nn.Parameter(torch.full((dim,), float(r_h_init)).abs().log())
        self.sign_r_h = nn.Parameter(torch.ones(dim), requires_grad=False)

        # Diagonal r_delta (stored in log space)
        self.log_r_delta = nn.Parameter(torch.full((dim,), 0.1).abs().log())
        self.sign_r_delta = nn.Parameter(torch.ones(dim), requires_grad=False)

        # W_x for input transformation
        self.W_x = nn.Parameter(torch.empty(dim, dim))

        # Delta gate
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Output projection
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        # RMSNorm scale (in log space, log(1) = 0)
        self.log_gamma = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x, gain=0.5)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: tuple of (log_h0, sign_h0) each [B, dim], or None

        Returns:
            output: [T, B, dim] selective output
            log_h: [T+1, B, dim] log magnitudes
            sign_h: [T+1, B, dim] signs
            h_linear: [T, B, dim] linear hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            log_h0 = torch.full((B, self.dim), LOG_ZERO, device=x.device, dtype=x.dtype)
            sign_h0 = torch.ones(B, self.dim, device=x.device, dtype=x.dtype)
        else:
            log_h0, sign_h0 = h0

        # Use CUDA kernel when available
        if HASTE_LOG_DIAG_TRIPLE_R_AVAILABLE and x.is_cuda:
            return LogSpaceDiagTripleRFunction.apply(
                self.training, self.n_groups, x, log_h0, sign_h0,
                self.W_x, self.log_r_h, self.sign_r_h,
                self.log_r_delta, self.sign_r_delta,
                self.W_delta, self.W_out, self.b, self.b_delta, self.log_gamma
            )
        else:
            return self._forward_pytorch(x, log_h0, sign_h0)

    def _forward_pytorch(self, x, log_h0, sign_h0):
        """Pure PyTorch implementation with diagonal r_h and r_delta."""
        T, B, D = x.shape

        # Get r_h and r_delta from log space
        r_h = torch.exp(self.log_r_h) * self.sign_r_h
        r_delta = torch.exp(self.log_r_delta) * self.sign_r_delta

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

            # v = r_h * h_prev + W_x @ x + b (diagonal r_h)
            v = r_h * h_prev_linear + x_t @ self.W_x.T + self.b
            candidate = torch.tanh(v)

            # Delta gate: sigmoid(W_delta @ x + r_delta * h_prev + b_delta)
            delta_raw = x_t @ self.W_delta.T + r_delta * h_prev_linear + self.b_delta
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


class LogSpaceDiagTripleR(nn.Module):
    """
    Log-Space Diagonal Triple R Elman layer for use in LadderLM.

    Level log_6: Diagonal r_h and r_delta for efficient recurrence.
    More efficient than log_5 (full R matrices) with similar expressivity.

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor
        n_groups: Number of groups for compete softmax
        delta_init: Initial bias for delta gate
        dropout: Dropout rate
        r_h_mode: Constraint mode for r_h
        **kwargs: Ignored (for API compatibility)
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0,
                 dropout=0.0, r_h_mode='spectral_norm', **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.r_h_mode = r_h_mode

        # Adjust n_groups for inner dimension
        while self.d_inner % n_groups != 0 and n_groups > 1:
            n_groups -= 1

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Log-space diagonal triple R cell
        self.cell = LogSpaceDiagTripleRCell(
            self.d_inner, n_groups=n_groups, delta_init=delta_init,
            r_h_mode=r_h_mode
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
    'LogSpaceDiagTripleRCell',
    'LogSpaceDiagTripleR',
    'LOGSPACE_LEVEL_6_AVAILABLE',
    'HASTE_LOG_DIAG_TRIPLE_R_AVAILABLE',
]
