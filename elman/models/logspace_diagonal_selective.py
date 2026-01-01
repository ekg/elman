"""
Log-Space Level 2: Diagonal Selective Log-Space Polynomial

Like log_1 but uses diagonal recurrence (like Mamba2's diagonal A).

Architecture:
    # Diagonal recurrence (element-wise r_h, not full W_h matrix)
    α_t = 1 + softplus(W_α @ x_t + b_α)
    v = r_h * h_prev + W_x @ x + b        # r_h is diagonal vector
    log|h_cand| = α_t * log|v|
    log|h_bounded| = -softplus(-log|h_cand|)
    δ = sigmoid(W_δ @ x_t + b_δ)
    h_new = (1-δ) * h_prev + δ * h_bounded

    # Selective output
    compete = softmax(h.reshape(groups), dim=-1)
    output = compete * silu(W_out @ h)

Key difference from log_1:
- Uses diagonal r_h vector instead of full recurrence
- r_h is stored in log space: (log|r_h|, sign(r_h))
- More parameter efficient, similar to Mamba2's diagonal A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import log-space utilities from log_0
from .logspace_polynomial import (
    LOG_ZERO, LOG_EPS, to_log_space, from_log_space,
    signed_log_add, soft_bound_log
)

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    # Re-enabled: GEMM transpose and gradient scaling now fixed
    HASTE_DIAG_SELECTIVE_AVAILABLE = hasattr(hasty_pytorch_lib, 'logspace_diag_selective_forward')
except ImportError:
    hasty_pytorch_lib = None
    HASTE_DIAG_SELECTIVE_AVAILABLE = False

LOGSPACE_LEVEL_2_AVAILABLE = True  # PyTorch fallback always available


class LogSpaceDiagSelectiveFunction(torch.autograd.Function):
    """Autograd function for Log-Space Diagonal Selective Elman (Haste kernel) with fused RMSNorm."""

    @staticmethod
    def forward(ctx, training, x, log_h0, sign_h0, W_x, log_r_h, sign_r_h,
                W_alpha, b_alpha, W_delta, W_out, b, b_delta, log_gamma, n_groups):
        results = hasty_pytorch_lib.logspace_diag_selective_forward(
            training,
            x.contiguous(),
            log_h0.contiguous(),
            sign_h0.contiguous(),
            W_x.contiguous(),
            log_r_h.contiguous(),
            sign_r_h.contiguous(),
            W_alpha.contiguous(),
            b_alpha.contiguous(),
            W_delta.contiguous(),
            W_out.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            log_gamma.contiguous(),
            n_groups
        )
        log_h, sign_h, output, log_v_cache, sign_v_cache, alpha_cache, \
            log_h_unbounded_cache, delta_cache, weight_rh_cache, alpha_raw_cache, \
            h_linear_cache, compete_cache, log_rms_cache = results

        if training:
            ctx.save_for_backward(
                x, W_x, log_r_h, sign_r_h, W_alpha, W_delta, W_out, log_gamma,
                log_h, sign_h, log_v_cache, sign_v_cache, alpha_cache,
                alpha_raw_cache, log_h_unbounded_cache, delta_cache, weight_rh_cache,
                h_linear_cache, compete_cache, log_rms_cache
            )
            ctx.n_groups = n_groups

        return output, log_h, sign_h, h_linear_cache

    @staticmethod
    def backward(ctx, d_output, d_log_h, d_sign_h, d_h_linear):
        (x, W_x, log_r_h, sign_r_h, W_alpha, W_delta, W_out, log_gamma,
         log_h, sign_h, log_v_cache, sign_v_cache, alpha_cache,
         alpha_raw_cache, log_h_unbounded_cache, delta_cache, weight_rh_cache,
         h_linear_cache, compete_cache, log_rms_cache) = ctx.saved_tensors

        dx, dW_x, d_log_r_h, dW_alpha, db_alpha, dW_delta, dW_out, db, db_delta, d_log_gamma = \
            hasty_pytorch_lib.logspace_diag_selective_backward(
                W_x, log_r_h, sign_r_h, W_alpha, W_delta, W_out, log_gamma,
                x, log_h, sign_h, log_v_cache, sign_v_cache, alpha_cache,
                alpha_raw_cache, log_h_unbounded_cache, delta_cache, weight_rh_cache,
                h_linear_cache, compete_cache, log_rms_cache,
                d_output.contiguous(),
                ctx.n_groups
            )

        return (None, dx, None, None, dW_x, d_log_r_h, None,
                dW_alpha, db_alpha, dW_delta, dW_out, db, db_delta, d_log_gamma, None)


class LogSpaceDiagonalSelectiveCell(nn.Module):
    """
    Log-Space Diagonal Selective Elman cell - Log-Space Level 2.

    Uses diagonal recurrence (element-wise r_h) instead of full W_h matrix.
    This is more parameter efficient and similar to Mamba2's diagonal A.
    Includes fused LogSpaceRMSNorm for bounded gradients.

    Args:
        dim: Hidden dimension
        n_groups: Number of groups for compete softmax
        alpha_init: Initial value for alpha bias
        delta_init: Initial bias for delta gate
    """

    def __init__(self, dim, n_groups=32, alpha_init=0.0, delta_init=-2.0):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.group_size = dim // n_groups

        assert dim % n_groups == 0, f"dim {dim} must be divisible by n_groups {n_groups}"

        # Input projection (no learnable bias - adding in linear space causes gradient explosion)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        # Register zero bias as buffer (not trainable) for CUDA kernel compatibility
        self.register_buffer('b', torch.zeros(dim))

        # Diagonal recurrence (stored as log|r_h|, sign(r_h))
        # This replaces the full W_h matrix with a learnable diagonal
        self.log_r_h = nn.Parameter(torch.full((dim,), -2.0))
        self.sign_r_h = nn.Parameter(torch.ones(dim), requires_grad=False)

        # Input-dependent alpha
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), alpha_init))

        # Delta gate
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Output projection for selective output
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        # Log-space RMSNorm scale (fused into CUDA kernel)
        # Stored in log-space: log(gamma) where gamma is the scale
        self.log_gamma = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_alpha, gain=0.1)
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

        # Use CUDA kernel when available (with fused RMSNorm)
        if HASTE_DIAG_SELECTIVE_AVAILABLE and x.is_cuda:
            return LogSpaceDiagSelectiveFunction.apply(
                self.training, x, log_h0, sign_h0,
                self.W_x, self.log_r_h, self.sign_r_h,
                self.W_alpha, self.b_alpha, self.W_delta,
                self.W_out, self.b, self.b_delta, self.log_gamma, self.n_groups
            )
        else:
            return self._forward_pytorch(x, log_h0, sign_h0)

    def _forward_pytorch(self, x, log_h0, sign_h0):
        """Pure PyTorch implementation."""
        T, B, D = x.shape

        log_h_list = [log_h0]
        sign_h_list = [sign_h0]
        output_list = []
        h_linear_list = []

        for t in range(T):
            log_h_prev = log_h_list[-1]
            sign_h_prev = sign_h_list[-1]
            x_t = x[t]

            # Input-dependent alpha
            # Cap alpha to [1, 2] to prevent gradient explosion
            alpha_raw = x_t @ self.W_alpha.T + self.b_alpha
            alpha = 1.0 + torch.clamp(F.softplus(alpha_raw), max=1.0)

            # DIAGONAL recurrence: r_h[d] * h_prev[d] in log space (element-wise)
            # Constrain log_r_h to be <= -0.1 (so r_h <= 0.9) for stability
            log_r_h_clamped = torch.clamp(self.log_r_h, max=-0.1)
            log_rh_hp = log_r_h_clamped + log_h_prev  # [B, dim] broadcasting
            sign_rh_hp = self.sign_r_h * sign_h_prev

            # W_x @ x (no bias - causes gradient explosion, no W_h since we use diagonal r_h)
            linear_input = x_t @ self.W_x.T
            log_input, sign_input = to_log_space(linear_input)

            # Add: v = r_h * h_prev + input
            log_v, sign_v = signed_log_add(log_rh_hp, sign_rh_hp, log_input, sign_input)

            # Polynomial activation
            log_cand = alpha * log_v
            sign_cand = sign_v

            # Soft bound
            log_cand_bounded = soft_bound_log(log_cand)

            # Delta gate
            delta_raw = x_t @ self.W_delta.T + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Gated update in log space
            log_one_minus_delta = F.logsigmoid(-delta_raw)
            log_delta = F.logsigmoid(delta_raw)

            log_term1 = log_one_minus_delta + log_h_prev
            sign_term1 = sign_h_prev
            log_term2 = log_delta + log_cand_bounded
            sign_term2 = sign_cand

            log_h_new, sign_h_new = signed_log_add(log_term1, sign_term1, log_term2, sign_term2)

            # Apply RMSNorm in log-space (logsumexp for bounded gradients)
            log_h2 = 2 * log_h_new  # log(h^2)
            log_mean_h2 = torch.logsumexp(log_h2, dim=-1, keepdim=True) - torch.log(
                torch.tensor(self.dim, dtype=x.dtype, device=x.device))
            log_rms = log_mean_h2 / 2
            log_h_normed = log_h_new - log_rms + self.log_gamma

            # Convert normalized log to linear for output
            h_linear = from_log_space(log_h_normed, sign_h_new)

            # Selective output: compete × silu
            h_reshaped = h_linear.view(B, self.n_groups, self.group_size)
            compete = F.softmax(h_reshaped, dim=-1)
            compete = compete.view(B, D)

            w_out_h = h_linear @ self.W_out.T
            silu_out = F.silu(w_out_h)

            output = compete * silu_out

            log_h_list.append(log_h_new)
            sign_h_list.append(sign_h_new)
            output_list.append(output)
            h_linear_list.append(h_linear)

        log_h = torch.stack(log_h_list, dim=0)
        sign_h = torch.stack(sign_h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        h_linear = torch.stack(h_linear_list, dim=0)

        return output, log_h, sign_h, h_linear


class LogSpaceDiagonalSelective(nn.Module):
    """
    Log-Space Diagonal Selective Elman layer for use in LadderLM.

    Like LogSpaceSelective but uses diagonal recurrence (more parameter efficient).
    RMSNorm is fused into CUDA kernel for maximum throughput.

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor
        n_groups: Number of groups for compete softmax
        alpha_init: Initial value for alpha bias
        delta_init: Initial bias for delta gate
        dropout: Dropout rate
        **kwargs: Ignored (for API compatibility)
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, alpha_init=0.0,
                 delta_init=-2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Adjust n_groups for inner dimension
        while self.d_inner % n_groups != 0 and n_groups > 1:
            n_groups -= 1

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Log-space diagonal selective cell (includes fused RMSNorm)
        self.cell = LogSpaceDiagonalSelectiveCell(
            self.d_inner, n_groups=n_groups,
            alpha_init=alpha_init, delta_init=delta_init
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
            h_final: [B, d_inner] final hidden state for TBPTT
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)

        # Transpose to [T, B, d_inner] for cell
        x_proj = x_proj.transpose(0, 1).contiguous()

        # Run cell - output is already normalized via fused RMSNorm in CUDA
        output, log_h, sign_h, h_linear = self.cell(x_proj, h0)
        # output: [T, B, d_inner] - compete × silu with normalized h

        # Transpose back to [B, T, d_inner]
        output = output.transpose(0, 1).contiguous()

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        # Final hidden state for TBPTT (as tuple for log-space cell)
        log_h_final = log_h[-1]    # [B, d_inner]
        sign_h_final = sign_h[-1]  # [B, d_inner]
        h_final = (log_h_final, sign_h_final)

        return output, h_final


__all__ = [
    'LogSpaceDiagonalSelectiveCell',
    'LogSpaceDiagonalSelective',
    'LOGSPACE_LEVEL_2_AVAILABLE',
    'HASTE_DIAG_SELECTIVE_AVAILABLE',
]
