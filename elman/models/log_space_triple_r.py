"""
Level log_5: Log-Space Triple R Elman

Log-space storage with THREE R matrices for complete recurrence control:
- R_h: Hidden state recurrence
- R_x: Input transformation (replaces W_x for symmetry)
- R_delta: Delta gate modulation from hidden state (unused but kept)

Architecture:
    # All projections
    v = R_x @ x + R_h @ h_prev + b
    delta_raw = W_delta @ x + b_delta
    delta = sigmoid(delta_raw)
    h_new = (1-delta) * h_prev + delta * tanh(v)

    # h+x selective output (like Mamba2)
    output = h * silu(h + x + b_gate)

Key features:
- Triple R matrices for symmetric input/hidden/gate recurrence
- h+x selective output: input-dependent gating like Mamba2
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
    """Autograd function for Log-Space Triple R Elman with h+x gating (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, x, log_h0, sign_h0, R_h, R_x, R_delta,
                W_delta, b, b_delta, b_gate):
        results = hasty_pytorch_lib.logspace_triple_r_forward(
            training,
            x.contiguous(),
            log_h0.contiguous(),
            sign_h0.contiguous(),
            R_h.contiguous(),
            R_x.contiguous(),
            R_delta.contiguous(),
            W_delta.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            b_gate.contiguous()
        )
        log_h, sign_h, output, v_cache, delta_cache, gate_cache = results

        if training:
            ctx.save_for_backward(
                x, R_h, R_x, R_delta, W_delta, b_gate,
                log_h, sign_h, v_cache, delta_cache, gate_cache
            )

        # Return h_linear as None since it's not computed in CUDA kernel
        return output, log_h, sign_h, None

    @staticmethod
    def backward(ctx, d_output, d_log_h, d_sign_h, d_h_linear):
        (x, R_h, R_x, R_delta, W_delta, b_gate,
         log_h, sign_h, v_cache, delta_cache, gate_cache) = ctx.saved_tensors

        dx, dR_h, dR_x, dR_delta, dW_delta, db, db_delta, db_gate = \
            hasty_pytorch_lib.logspace_triple_r_backward(
                R_h, R_x, R_delta, W_delta, b_gate,
                x, log_h, sign_h, v_cache, delta_cache, gate_cache,
                d_output.contiguous()
            )

        return (None, dx, None, None, dR_h, dR_x, dR_delta,
                dW_delta, db, db_delta, db_gate)


class LogSpaceTripleRCell(nn.Module):
    """
    Log-Space Triple R Elman cell - Level log_5.

    Log-space storage with three R matrices for complete
    recurrence control: R_h, R_x, R_delta.

    h+x selective output: output = h * silu(h + x + b_gate)

    Args:
        dim: Hidden dimension
        delta_init: Initial bias for delta gate
        r_h_mode: Constraint mode for R_h/R_delta matrices
        r_h_init_gain: Initial gain for R_h initialization
    """

    def __init__(self, dim, delta_init=-2.0,
                 r_h_mode='spectral_norm', r_h_init_gain=0.1,
                 spectral_radius=0.9, diagonal_r_delta=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.r_h_mode = r_h_mode
        self.r_h_init_gain = r_h_init_gain
        self.spectral_radius = spectral_radius  # Target spectral radius (lower = more stable)
        self.diagonal_r_delta = diagonal_r_delta  # Use diagonal r_delta for stability

        # R_h matrix with optional constraints
        if r_h_mode == 'scaled_orthogonal':
            self.register_buffer('R_h_base', torch.empty(dim, dim))
            self.r_h_log_scale = nn.Parameter(torch.tensor(-2.2))  # sigmoid(-2.2) ≈ 0.1
        else:
            self.R_h = nn.Parameter(torch.empty(dim, dim))

        # R_delta: full matrix or diagonal vector
        if diagonal_r_delta:
            # Diagonal r_delta is inherently stable (no eigenvalue blowup)
            self.r_delta = nn.Parameter(torch.full((dim,), 0.1))
        elif r_h_mode == 'scaled_orthogonal':
            self.register_buffer('R_delta_base', torch.empty(dim, dim))
            self.r_delta_log_scale = nn.Parameter(torch.tensor(-2.2))
        else:
            self.R_delta = nn.Parameter(torch.empty(dim, dim))

        # R_x with spectral norm (input matrix, but still constrain for stability)
        self.R_x = nn.Parameter(torch.empty(dim, dim))

        # Bias
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Delta gate also uses linear W_delta @ x
        self.W_delta = nn.Parameter(torch.empty(dim, dim))

        # h+x selective gate bias
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        if self.r_h_mode == 'scaled_orthogonal':
            nn.init.orthogonal_(self.R_h_base)
            if not self.diagonal_r_delta:
                nn.init.orthogonal_(self.R_delta_base)
        else:
            nn.init.orthogonal_(self.R_h, gain=self.r_h_init_gain)
            if not self.diagonal_r_delta:
                nn.init.orthogonal_(self.R_delta, gain=0.1)
        nn.init.orthogonal_(self.R_x, gain=1.0)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)

    def _apply_spectral_norm(self, W, name, target_radius=None):
        """Apply spectral normalization to a weight matrix."""
        if target_radius is None:
            target_radius = self.spectral_radius
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

    def get_R_x(self):
        """Get the effective R_x matrix - no constraint needed for input projection."""
        # R_x is not recurrent, so no spectral norm needed
        return self.R_x

    def get_R_delta(self):
        """Get the effective R_delta matrix/vector with constraints applied."""
        if self.diagonal_r_delta:
            # Diagonal: just return the vector (clamp to prevent blowup)
            return torch.clamp(self.r_delta, -0.5, 0.5)
        elif self.r_h_mode == 'scaled_orthogonal':
            scale = torch.sigmoid(self.r_delta_log_scale)
            return scale * self.R_delta_base
        elif self.r_h_mode == 'spectral_norm':
            # Use very low radius for gate modulation (critical for stability)
            return self._apply_spectral_norm(self.R_delta, 'R_delta', target_radius=0.01)
        else:
            return self.R_delta

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: tuple of (log_h0, sign_h0) each [B, dim], or None

        Returns:
            output: [T, B, dim] h+x selective output
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
        R_x = self.get_R_x()
        R_delta = self.get_R_delta()

        # Use CUDA kernel when available
        if HASTE_LOG_TRIPLE_R_AVAILABLE and x.is_cuda and not self.diagonal_r_delta:
            return LogSpaceTripleRFunction.apply(
                self.training, x, log_h0, sign_h0,
                R_h, R_x, R_delta,
                self.W_delta, self.b, self.b_delta, self.b_gate
            )
        else:
            return self._forward_pytorch(x, log_h0, sign_h0, R_h, R_x, R_delta)

    def _forward_pytorch(self, x, log_h0, sign_h0, R_h, R_x, R_delta):
        """Pure PyTorch implementation with triple R matrices and h+x gating."""
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
            v = x_t @ R_x.T + h_prev_linear @ R_h.T + self.b
            candidate = torch.tanh(v)

            # Delta gate: sigmoid(W_delta @ x + b_delta)
            # NOTE: R_delta @ h_prev removed - creates unstable feedback loop
            delta_raw = x_t @ self.W_delta.T + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Gated update
            h_new = (1 - delta) * h_prev_linear + delta * candidate

            # Convert to log space
            log_h_new, sign_h_new = to_log_space(h_new)

            # h+x selective output: output = h * silu(h + x + b_gate)
            gate = F.silu(h_new + x_t + self.b_gate)
            output = h_new * gate

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
    h+x selective output: output = h * silu(h + x + b_gate)

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor
        delta_init: Initial bias for delta gate
        dropout: Dropout rate
        r_h_mode: Constraint mode for R_h/R_delta matrices
        r_h_init_gain: Initial gain for R_h initialization
        spectral_radius: Target spectral radius for constrained matrices (default 0.9)
        diagonal_r_delta: Use diagonal r_delta for stability (default True)
        **kwargs: Ignored (for API compatibility)
    """

    def __init__(self, dim, expansion=1.0, delta_init=-2.0,
                 dropout=0.0, r_h_mode='spectral_norm', r_h_init_gain=0.1,
                 spectral_radius=0.99, diagonal_r_delta=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.r_h_mode = r_h_mode
        self.diagonal_r_delta = diagonal_r_delta

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Log-space triple R cell
        self.cell = LogSpaceTripleRCell(
            self.d_inner, delta_init=delta_init,
            r_h_mode=r_h_mode, r_h_init_gain=r_h_init_gain,
            spectral_radius=spectral_radius, diagonal_r_delta=diagonal_r_delta
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


if __name__ == "__main__":
    print("Testing LogSpaceTripleR (log_5)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_LOG_TRIPLE_R_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LogSpaceTripleR(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Hidden state: log_h={h[0].shape}, sign_h={h[1].shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Verify h+x gating components
    print(f"b_gate shape: {model.cell.b_gate.shape} (should be [{model.d_inner}])")
    print(f"R_h shape: {model.cell.R_h.shape}")
    print(f"R_x shape: {model.cell.R_x.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Log-Space Triple R (log_5) test passed!")
