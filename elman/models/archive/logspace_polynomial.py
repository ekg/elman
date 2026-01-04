"""
Log-Space Level 0: Polynomial Elman with Input-Dependent Alpha

The first truly log-space level with bounded gradients throughout.

Key innovations:
1. INPUT-DEPENDENT alpha: α_t = 1 + softplus(W_α @ x_t + b_α)  (guaranteed > 1)
2. Polynomial activation: log|h| = α * log|v|  (gradient = α, constant per element)
3. Soft bounding: log_bounded = -softplus(-log_h)  (magnitude ≤ 1, gradient ∈ [0,1])
4. All operations stay in log-space with bounded gradients

Architecture:
    # Pre-activation
    v = r_h * h_prev + W_x @ x + b           # Signed log arithmetic

    # Input-dependent polynomial exponent
    α_t = 1 + softplus(W_α @ x_t + b_α)      # Always > 1

    # Polynomial activation in log space
    log|h_unbounded| = α_t * log|v|          # Gradient = α
    sign(h) = sign(v)

    # Soft bound to magnitude ≤ 1
    log|h_bounded| = -softplus(-log|h_unbounded|)

    # Gated update
    δ = sigmoid(W_δ @ x_t + b_δ)
    h_new = (1-δ) * h_prev + δ * h_bounded

Gradient analysis:
    - Through softplus bound: sigmoid(-x) ∈ [0,1] ✓
    - Through polynomial: α (> 1, input-dependent) ✓
    - Through logsumexp aggregation: softmax weights ∈ [0,1] ✓
    - NO GRADIENT VANISHING from saturation!

Key difference from tanh-based levels:
    - Tanh: gradient → 0 when saturated
    - Polynomial: gradient = α, constant regardless of value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    # Re-enabled: Fused logsumexp RMSNorm backward now implemented
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'logspace_polynomial_forward')
except ImportError:
    hasty_pytorch_lib = None
    HASTE_AVAILABLE = False

LOGSPACE_LEVEL_0_AVAILABLE = True  # PyTorch fallback always available

# Constants matching CUDA kernel
LOG_ZERO = -40.0
LOG_EPS = 1e-10


def to_log_space(x, soft_eps=1e-6):
    """
    Convert linear tensor to (log|x|, sign(x)) representation.

    Uses soft-abs: sqrt(x^2 + eps^2) to bound gradients near zero.
    The gradient d/dx sqrt(x^2 + eps^2) = x / sqrt(x^2 + eps^2) is bounded by 1.
    """
    sign_x = torch.sign(x)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
    # Soft-abs: sqrt(x^2 + eps^2) bounds gradient to 1 near zero
    soft_abs_x = torch.sqrt(x * x + soft_eps * soft_eps)
    log_x = torch.log(soft_abs_x)
    return log_x, sign_x


class _GradScale(torch.autograd.Function):
    """Scale gradients by a fixed factor during backward pass."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def grad_scale(x, scale=0.01):
    """Scale gradients during backward pass."""
    return _GradScale.apply(x, scale)


def from_log_space(log_x, sign_x):
    """Convert (log|x|, sign(x)) back to linear tensor."""
    mask = log_x > LOG_ZERO + 1.0
    # Removed grad_scale - was killing recurrence gradients
    result = sign_x * torch.exp(torch.clamp(log_x, max=20.0))
    return torch.where(mask, result, torch.zeros_like(result))


def signed_log_add(log_a, sign_a, log_b, sign_b):
    """Add two signed log-space values. Returns (log_result, sign_result)."""
    # Handle zeros
    a_is_zero = log_a <= LOG_ZERO + 1.0
    b_is_zero = log_b <= LOG_ZERO + 1.0

    max_log = torch.maximum(log_a, log_b)
    min_log = torch.minimum(log_a, log_b)
    diff = min_log - max_log

    # Determine signs
    a_is_max = log_a >= log_b
    sign_max = torch.where(a_is_max, sign_a, sign_b)
    sign_min = torch.where(a_is_max, sign_b, sign_a)
    same_sign = sign_max * sign_min > 0

    # Same sign: log(|a| + |b|) = max + log(1 + exp(diff))
    log_same = max_log + torch.log1p(torch.exp(diff))

    # Opposite sign: log(||a| - |b||) = max + log(1 - exp(diff))
    exp_diff = torch.exp(diff)
    # Clamp to avoid log(0) in near-cancellation
    log_opp = max_log + torch.log1p(-torch.clamp(exp_diff, max=0.9999))

    log_result = torch.where(same_sign, log_same, log_opp)
    sign_result = sign_max

    # Handle zeros
    log_result = torch.where(a_is_zero, log_b, log_result)
    sign_result = torch.where(a_is_zero, sign_b, sign_result)
    log_result = torch.where(b_is_zero & ~a_is_zero, log_a, log_result)
    sign_result = torch.where(b_is_zero & ~a_is_zero, sign_a, sign_result)

    return log_result, sign_result


def soft_bound_log(log_h):
    """Soft upper bound at magnitude 1: output = -softplus(-log_h)."""
    # When log_h > 0 (|h| > 1): squash toward 0
    # When log_h <= 0 (|h| <= 1): mostly preserve
    return -F.softplus(-log_h)


class LogSpaceRMSNorm(nn.Module):
    """
    RMSNorm computed entirely in log-space using logsumexp for bounded gradients.

    Standard RMSNorm: y = x / rms(x) * gamma
    where rms(x) = sqrt(mean(x^2))

    In log-space:
    - log(x^2) = 2 * log|x|
    - log(mean(x^2)) = logsumexp(2 * log|x|) - log(n)
    - log(rms) = log(mean(x^2)) / 2
    - log|y| = log|x| - log(rms)

    This keeps gradients bounded through logsumexp's softmax-like jacobian.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Learnable scale in log-space
        self.log_gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, log_x, sign_x):
        """
        Args:
            log_x: [*, dim] log magnitudes
            sign_x: [*, dim] signs

        Returns:
            log_y: [*, dim] normalized log magnitudes
            sign_y: [*, dim] signs (unchanged)
        """
        # Compute log(x^2) = 2 * log|x|
        log_x2 = 2 * log_x

        # log(mean(x^2)) = logsumexp(log(x^2)) - log(n)
        # logsumexp gives bounded gradients!
        log_mean_x2 = torch.logsumexp(log_x2, dim=-1, keepdim=True) - torch.log(torch.tensor(self.dim, dtype=log_x.dtype, device=log_x.device))

        # log(rms) = log(mean(x^2)) / 2
        log_rms = log_mean_x2 / 2

        # Normalize: log|y| = log|x| - log(rms) + log(gamma)
        log_y = log_x - log_rms + self.log_gamma

        # Signs unchanged by normalization
        return log_y, sign_x


class LogSpacePolynomialFunction(torch.autograd.Function):
    """Autograd function for Log-Space Polynomial Elman (Haste kernel with fused RMSNorm + selective output)."""

    @staticmethod
    def forward(ctx, training, n_groups, x, log_h0, sign_h0, W_x, log_r_h, sign_r_h,
                W_alpha, b_alpha, W_delta, W_out, b, b_delta, log_gamma):
        results = hasty_pytorch_lib.logspace_polynomial_forward(
            training,
            n_groups,
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
            log_gamma.contiguous()
        )
        log_h, sign_h, output, h_linear_cache, log_v_cache, sign_v_cache, alpha_cache, \
            log_h_unbounded_cache, delta_cache, weight_rh_cache, alpha_raw_cache, \
            log_rms_cache, compete_cache = results

        if training:
            ctx.n_groups = n_groups
            # Save for backward (fused logsumexp gradient + selective output)
            ctx.save_for_backward(
                x, W_x, log_r_h, sign_r_h, W_alpha, W_delta, W_out, log_gamma,
                log_h, sign_h, log_v_cache, sign_v_cache, alpha_cache,
                alpha_raw_cache, log_h_unbounded_cache, delta_cache, weight_rh_cache,
                h_linear_cache, compete_cache, log_rms_cache
            )

        return log_h, sign_h, output

    @staticmethod
    def backward(ctx, d_log_h, d_sign_h, d_output):
        n_groups = ctx.n_groups
        (x, W_x, log_r_h, sign_r_h, W_alpha, W_delta, W_out, log_gamma,
         log_h, sign_h, log_v_cache, sign_v_cache, alpha_cache,
         alpha_raw_cache, log_h_unbounded_cache, delta_cache, weight_rh_cache,
         h_linear_cache, compete_cache, log_rms_cache) = ctx.saved_tensors

        dx, dW_x, d_log_r_h, dW_alpha, db_alpha, dW_delta, dW_out, db, db_delta, d_log_gamma = \
            hasty_pytorch_lib.logspace_polynomial_backward(
                n_groups,
                W_x, log_r_h, sign_r_h, W_alpha, W_delta, W_out, log_gamma,
                x, log_h, sign_h, log_v_cache, sign_v_cache, alpha_cache,
                alpha_raw_cache, log_h_unbounded_cache, delta_cache, weight_rh_cache,
                h_linear_cache, compete_cache, log_rms_cache,
                d_output.contiguous()
            )

        return (None, None, dx, None, None, dW_x, d_log_r_h, None,
                dW_alpha, db_alpha, dW_delta, dW_out, db, db_delta, d_log_gamma)


class LogSpacePolynomialCell(nn.Module):
    """
    Log-Space Polynomial Elman cell - Log-Space Level 0.

    Stores hidden state as (log|h|, sign(h)) pairs.
    Uses polynomial activation with input-dependent exponent.
    Includes fused RMSNorm with logsumexp for bounded gradients.
    NOW includes selective output (compete × silu) for gradient stability.

    Args:
        dim: Hidden dimension
        n_groups: Number of groups for compete softmax (default 8)
        alpha_init: Initial value for alpha bias (default 0.0 -> softplus(0) = 0.69, so α ≈ 1.69)
        delta_init: Initial bias for delta gate (default -2.0 -> sigmoid(-2) ≈ 0.12)
    """

    def __init__(self, dim, n_groups=8, alpha_init=0.0, delta_init=-2.0):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups  # NEW: for selective output

        # Input projection (no learnable bias - adding in linear space causes gradient explosion)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        # Register zero bias as buffer (not trainable) for CUDA kernel compatibility
        self.register_buffer('b', torch.zeros(dim))

        # Diagonal recurrence (stored as log|r_h|, sign(r_h))
        # Initialize r_h to small positive values -> log_r_h ~ -2, sign = +1
        self.log_r_h = nn.Parameter(torch.full((dim,), -2.0))
        self.sign_r_h = nn.Parameter(torch.ones(dim), requires_grad=False)

        # Input-dependent alpha: α = 1 + softplus(W_alpha @ x + b_alpha)
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), alpha_init))

        # Delta gate
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # NEW: Output projection for selective output (compete × silu)
        self.W_out = nn.Parameter(torch.empty(dim, dim))

        # Fused RMSNorm scale in log-space (learned)
        self.log_gamma = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Smaller initialization for gradient stability in log-space
        nn.init.xavier_uniform_(self.W_x, gain=0.5)
        nn.init.xavier_uniform_(self.W_alpha, gain=0.1)  # Small init for alpha
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)
        nn.init.xavier_uniform_(self.W_out, gain=0.5)  # NEW: output projection

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence (linear space)
            h0: tuple of (log_h0, sign_h0) each [B, dim], or None

        Returns:
            log_h: [T+1, B, dim] log magnitudes including initial state
            sign_h: [T+1, B, dim] signs including initial state
            output: [T, B, dim] SELECTIVE output (compete × silu) for gradient stability
        """
        T, B, D = x.shape

        if h0 is None:
            # Initialize to small positive values in log space
            log_h0 = torch.full((B, self.dim), LOG_ZERO, device=x.device, dtype=x.dtype)
            sign_h0 = torch.ones(B, self.dim, device=x.device, dtype=x.dtype)
        else:
            log_h0, sign_h0 = h0

        # Use CUDA kernel when available (with fused RMSNorm + selective output)
        if HASTE_AVAILABLE and x.is_cuda:
            return LogSpacePolynomialFunction.apply(
                self.training, self.n_groups, x, log_h0, sign_h0,
                self.W_x, self.log_r_h, self.sign_r_h,
                self.W_alpha, self.b_alpha, self.W_delta, self.W_out,
                self.b, self.b_delta, self.log_gamma
            )
        else:
            return self._forward_pytorch(x, log_h0, sign_h0)

    def _forward_pytorch(self, x, log_h0, sign_h0):
        """Pure PyTorch implementation for testing (includes selective output)."""
        T, B, D = x.shape
        group_size = D // self.n_groups

        log_h_list = [log_h0]
        sign_h_list = [sign_h0]
        output_list = []

        for t in range(T):
            log_h_prev = log_h_list[-1]
            sign_h_prev = sign_h_list[-1]
            x_t = x[t]

            # Compute input-dependent alpha: α = 1 + softplus(W_alpha @ x + b_alpha)
            # Cap alpha to [1, 2] to prevent gradient explosion
            alpha_raw = x_t @ self.W_alpha.T + self.b_alpha
            alpha = 1.0 + torch.clamp(F.softplus(alpha_raw), max=1.0)

            # r_h * h_prev in log space
            # Constrain log_r_h to be <= -0.1 (so r_h <= 0.9) for stability
            log_r_h_clamped = torch.clamp(self.log_r_h, max=-0.1)
            log_rh_hp = log_r_h_clamped + log_h_prev
            sign_rh_hp = self.sign_r_h * sign_h_prev

            # W_x @ x in linear space, then convert to log (no bias - causes gradient explosion)
            linear_input = x_t @ self.W_x.T
            log_input, sign_input = to_log_space(linear_input)

            # Add: v = r_h * h_prev + input (signed log addition)
            log_v, sign_v = signed_log_add(log_rh_hp, sign_rh_hp, log_input, sign_input)

            # Polynomial activation: log|h_cand| = α * log|v|
            log_cand = alpha * log_v
            sign_cand = sign_v

            # Soft bound to magnitude <= 1
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

            log_h_list.append(log_h_new)
            sign_h_list.append(sign_h_new)

            # Apply RMSNorm in log-space (matching fused CUDA kernel)
            log_h2 = 2 * log_h_new  # log(h^2)
            log_mean_h2 = torch.logsumexp(log_h2, dim=-1, keepdim=True) - torch.log(
                torch.tensor(self.dim, dtype=x.dtype, device=x.device))
            log_rms = log_mean_h2 / 2
            log_h_normed = log_h_new - log_rms + self.log_gamma
            h_linear = from_log_space(log_h_normed, sign_h_new)  # [B, D]

            # NEW: Selective output (compete × silu) for gradient stability
            # Reshape for group softmax: [B, n_groups, group_size]
            h_grouped = h_linear.view(B, self.n_groups, group_size)
            compete = F.softmax(h_grouped, dim=-1)  # [B, n_groups, group_size]
            compete = compete.view(B, D)  # [B, D]

            # Compute w_out_h = W_out @ h_linear
            w_out_h = h_linear @ self.W_out.T  # [B, D]

            # SiLU activation
            silu_val = w_out_h * torch.sigmoid(w_out_h)

            # Final output = compete * silu
            output_t = compete * silu_val
            output_list.append(output_t)

        log_h = torch.stack(log_h_list, dim=0)
        sign_h = torch.stack(sign_h_list, dim=0)
        output = torch.stack(output_list, dim=0)

        return log_h, sign_h, output


class LogSpacePolynomial(nn.Module):
    """
    Log-Space Polynomial Elman layer for use in LadderLM.

    This is the first level that operates entirely in log-space with bounded gradients.
    Uses polynomial activation with input-dependent exponent for nonlinearity.
    Uses FUSED LogSpaceRMSNorm in CUDA kernel for bounded gradient flow.
    NOW includes selective output (compete × silu) for gradient stability.

    Matches interface of StockElman/GatedElman/etc for drop-in use in LadderLM.

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor (d_inner = dim * expansion)
        n_groups: Number of groups for compete softmax (default 8)
        alpha_init: Initial value for alpha bias
        delta_init: Initial bias for delta gate
        dropout: Dropout rate (applied after output projection)
        **kwargs: Ignored (for API compatibility with other levels)
    """

    def __init__(self, dim, expansion=1.0, n_groups=8, alpha_init=0.0, delta_init=-2.0,
                 dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_groups = n_groups

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Log-space polynomial cell (now includes fused RMSNorm + selective output)
        self.cell = LogSpacePolynomialCell(
            self.d_inner, n_groups=n_groups, alpha_init=alpha_init, delta_init=delta_init
        )

        # Output projection (after selective output from cell)
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
            x: [B, T, dim] input sequence (note: B, T order for LadderLM compatibility)
            h0: tuple of (log_h0, sign_h0) or None

        Returns:
            output: [B, T, dim] output for residual connection
            h_final: tuple (log_h, sign_h) final hidden state for TBPTT
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Transpose to [T, B, d_inner] for cell
        x_proj = x_proj.transpose(0, 1).contiguous()

        # Run cell - returns log-space hidden states and SELECTIVE output
        log_h, sign_h, cell_output = self.cell(x_proj, h0)
        # log_h: [T+1, B, d_inner], cell_output: [T, B, d_inner] from compete × silu

        # Transpose back to [B, T, d_inner]
        cell_output_bt = cell_output.transpose(0, 1).contiguous()

        # Output projection
        output = self.out_proj(cell_output_bt)  # [B, T, dim]
        output = self.dropout(output)

        # Final hidden state for TBPTT (last timestep)
        # Return as (log_h, sign_h) tuple so it can be used directly by cell
        log_h_final = log_h[-1]   # [B, d_inner]
        sign_h_final = sign_h[-1]  # [B, d_inner]
        h_final = (log_h_final, sign_h_final)

        return output, h_final


__all__ = [
    'LogSpacePolynomialCell',
    'LogSpacePolynomial',
    'LogSpaceRMSNorm',
    'LOGSPACE_LEVEL_0_AVAILABLE',
    'HASTE_AVAILABLE',
    'to_log_space',
    'from_log_space',
    'signed_log_add',
    'soft_bound_log',
]
