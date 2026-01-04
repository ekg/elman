"""
Level 7: Diagonal Triple R Elman (Linear Space) with Mamba2-inspired improvements

Like Level 5 (Linear Triple R) but with DIAGONAL r_h and r_delta vectors
instead of full matrices. More efficient O(d) recurrence.

Enhanced with Mamba2 features:
- Conv1d (d_conv=4) for local context before recurrence
- Internal RMSNorm for stability
- Log-space decay: A_log parameterization (r_h = -exp(A_log)) for stable learning

Architecture:
    x_conv = conv1d(x, d_conv=4)                -- local context
    x_norm = rms_norm(x_conv)                   -- internal normalization
    v = r_h * h_prev + W_x @ x_norm + b         -- DIAGONAL r_h (element-wise)
    delta_raw = W_delta @ x_norm + r_delta * h_prev + b_delta
    delta = sigmoid(delta_raw)
    h_new = (1 - delta) * h_prev + delta * tanh(v)
    output = h_new * silu(W_gate @ x_norm + b_gate)  -- learned gate projection

Key features:
- Diagonal r_h for efficient recurrence (O(d) vs O(d^2))
- Diagonal r_delta for hidden-state modulated gating
- Conv1d for local temporal context (like Mamba2)
- Log-space A parameterization for stable decay learning
- Internal RMSNorm for training stability
- Learned gate projection: output = h * silu(W_gate @ x + b_gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
try:
    import hasty_pytorch_lib
    HASTE_DIAG_TRIPLE_R_AVAILABLE = hasattr(hasty_pytorch_lib, 'diag_triple_r_forward')
except ImportError:
    hasty_pytorch_lib = None
    HASTE_DIAG_TRIPLE_R_AVAILABLE = False

LEVEL_7_AVAILABLE = True  # PyTorch fallback always available


class DiagTripleRFunction(torch.autograd.Function):
    """Autograd function for Diagonal Triple R Elman with learned gate projection (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, n_groups, x, h0,
                W_x, r_h, r_delta, W_delta, W_gate, b, b_delta, b_gate):
        results = hasty_pytorch_lib.diag_triple_r_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            r_h.contiguous(),
            r_delta.contiguous(),
            W_delta.contiguous(),
            W_gate.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            b_gate.contiguous(),
            n_groups
        )

        h, output, v_cache, delta_cache, gate_cache = results

        if training:
            ctx.save_for_backward(
                W_x, r_h, r_delta, W_delta, W_gate,
                x, h, v_cache, delta_cache, gate_cache
            )
            ctx.n_groups = n_groups

        return output, h

    @staticmethod
    def backward(ctx, d_output, d_h):
        (W_x, r_h, r_delta, W_delta, W_gate,
         x, h, v_cache, delta_cache, gate_cache) = ctx.saved_tensors

        grads = hasty_pytorch_lib.diag_triple_r_backward(
            W_x, r_h, r_delta, W_delta, W_gate,
            x, h, v_cache, delta_cache, gate_cache,
            d_output.contiguous(),
            ctx.n_groups
        )

        # Unpack: dx, dW_x, d_r_h, d_r_delta, dW_delta, dW_gate, db, db_delta, db_gate
        dx, dW_x, d_r_h, d_r_delta, dW_delta, dW_gate, db, db_delta, db_gate = grads

        # Return gradients in same order as forward inputs
        return (None, None, dx, None,
                dW_x, d_r_h, d_r_delta, dW_delta, dW_gate, db, db_delta, db_gate)


class DiagTripleRCell(nn.Module):
    """
    Diagonal Triple R Elman cell - Level 7.

    Linear-space diagonal recurrence with r_h and r_delta vectors.
    Efficient O(d) recurrence with learned gate projection output.

    Uses log-space A parameterization for stable decay learning:
        A_log -> r_h = -exp(A_log)

    This ensures r_h is always negative (stable decay) and gradients
    flow smoothly through the exponential.

    Architecture:
        r_h = -exp(A_log)  # stable decay
        v = r_h * h_prev + W_x @ x + b
        delta = sigmoid(W_delta @ x + r_delta * h_prev + b_delta)
        h_new = (1 - delta) * h_prev + delta * tanh(v)
        output = h_new * silu(W_gate @ x + b_gate)  # learned gate projection!

    Args:
        dim: Hidden dimension
        n_groups: Number of groups (unused, kept for API compat)
        delta_init: Initial bias for delta gate
        A_log_init: Initial value for A_log (default -2.0 -> r_h ≈ -0.135)
    """

    def __init__(self, dim, n_groups=32, delta_init=-2.0, A_log_init=-2.0,
                 use_A_log=True, r_h_init=0.9, **kwargs):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups  # Kept for API compat
        self.use_A_log = use_A_log

        if use_A_log:
            # Log-space A for stable decay learning
            # r_h = -exp(A_log), so A_log = log(-r_h)
            # For r_h = -0.135 (like Mamba2 A init), A_log ≈ -2.0
            self.A_log = nn.Parameter(torch.full((dim,), float(A_log_init)))
            self._r_h_direct = None
        else:
            # Direct r_h parameterization (original)
            self.A_log = None
            self._r_h_direct = nn.Parameter(torch.full((dim,), float(r_h_init)))

        # Diagonal r_delta
        self.r_delta = nn.Parameter(torch.full((dim,), 0.1))

        # W_x for input transformation
        self.W_x = nn.Parameter(torch.empty(dim, dim))

        # Delta gate
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # Learned gate projection: output = h * silu(W_gate @ x + b_gate)
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x, gain=0.5)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)
        nn.init.xavier_uniform_(self.W_gate)

    @property
    def r_h(self):
        """Compute r_h - from A_log if enabled, else direct."""
        if self.use_A_log:
            return -torch.exp(self.A_log)
        else:
            return self._r_h_direct

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state, or None

        Returns:
            output: [T, B, dim] output with learned gate projection
            h: [T+1, B, dim] hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Use CUDA kernel when available
        if HASTE_DIAG_TRIPLE_R_AVAILABLE and x.is_cuda:
            return DiagTripleRFunction.apply(
                self.training, self.n_groups, x, h0,
                self.W_x, self.r_h, self.r_delta,
                self.W_delta, self.W_gate, self.b, self.b_delta, self.b_gate
            )
        else:
            return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation with diagonal r_h, r_delta, and learned gate projection."""
        T, B, D = x.shape

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # v = r_h * h_prev + W_x @ x + b (diagonal r_h)
            v = self.r_h * h_prev + x_t @ self.W_x.T + self.b
            candidate = torch.tanh(v)

            # Delta gate: sigmoid(W_delta @ x + r_delta * h_prev + b_delta)
            delta_raw = x_t @ self.W_delta.T + self.r_delta * h_prev + self.b_delta
            delta = torch.sigmoid(delta_raw)

            # Gated update
            h_new = (1 - delta) * h_prev + delta * candidate

            # Learned gate projection: output = h * silu(W_gate @ x + b_gate)
            gate_proj = x_t @ self.W_gate.T + self.b_gate
            gate = F.silu(gate_proj)
            output = h_new * gate

            h_list.append(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)

        return output, h


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (internal)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [*, dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DiagTripleR(nn.Module):
    """
    Diagonal Triple R Elman layer for use in LadderLM.

    Level 7: Linear-space diagonal r_h and r_delta for efficient recurrence.
    Enhanced with Mamba2-inspired features (each can be toggled for ablation):
    - Conv1d (d_conv=4) for local temporal context
    - Internal RMSNorm for training stability
    - Log-space A parameterization in the cell

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor
        n_groups: Number of groups (kept for API compat)
        delta_init: Initial bias for delta gate
        d_conv: Conv1d kernel size for local context (default 4)
        use_conv: Enable conv1d (default True)
        use_internal_norm: Enable internal RMSNorm (default True)
        use_A_log: Enable log-space A parameterization (default True)
        dropout: Dropout rate
        **kwargs: Ignored (for API compatibility)
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0,
                 d_conv=4, use_conv=True, use_internal_norm=True, use_A_log=True,
                 dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.d_conv = d_conv
        self.use_conv = use_conv
        self.use_internal_norm = use_internal_norm
        self.use_A_log = use_A_log

        # Adjust n_groups for inner dimension
        while self.d_inner % n_groups != 0 and n_groups > 1:
            n_groups -= 1

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Conv1d for local context (like Mamba2)
        if use_conv:
            # Groups=d_inner for depthwise convolution (efficient)
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,  # Causal padding
                groups=self.d_inner,  # Depthwise
                bias=True,
            )
        else:
            self.conv1d = None

        # Internal RMSNorm (like Mamba2)
        if use_internal_norm:
            self.norm = RMSNorm(self.d_inner)
        else:
            self.norm = None

        # Diagonal Triple R cell with optional A_log parameterization
        self.cell = DiagTripleRCell(
            self.d_inner, n_groups=n_groups, delta_init=delta_init,
            use_A_log=use_A_log
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # Conv1d is already initialized by PyTorch defaults

    def forward(self, x, h0=None):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner] initial hidden state or None

        Returns:
            output: [B, T, dim] output for residual connection
            h_final: [B, d_inner] final hidden state for TBPTT
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Conv1d for local context (like Mamba2)
        if self.use_conv and self.conv1d is not None:
            # Conv1d expects [B, C, T], so transpose
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)  # [B, d_inner, T + d_conv - 1]
            x_conv = x_conv[:, :, :T]  # Causal: keep only first T (no future info)
            x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]
            # Apply SiLU activation after conv (like Mamba2)
            x_proj = F.silu(x_conv)

        # Internal RMSNorm (this is INSIDE the layer, separate from external norm)
        if self.use_internal_norm and self.norm is not None:
            x_proj = self.norm(x_proj)  # [B, T, d_inner]

        # Transpose to [T, B, d_inner] for cell
        x_proj = x_proj.transpose(0, 1).contiguous()

        # Run cell
        output, h = self.cell(x_proj, h0)

        # Transpose back
        output = output.transpose(0, 1).contiguous()

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        # Final hidden state for TBPTT
        h_final = h[-1]

        return output, h_final


__all__ = [
    'DiagTripleRCell',
    'DiagTripleR',
    'RMSNorm',
    'LEVEL_7_AVAILABLE',
    'HASTE_DIAG_TRIPLE_R_AVAILABLE',
]


if __name__ == "__main__":
    # Quick test to verify model construction and parameter counting
    import sys

    dim = 768
    expansion = 1.0

    print("Testing Level 7 (DiagTripleR) with Mamba2 improvements...")
    print(f"dim={dim}, expansion={expansion}")
    print()

    # Test with all features
    model_all = DiagTripleR(dim, expansion=expansion, use_conv=True, use_internal_norm=True, use_A_log=True)
    params_all = sum(p.numel() for p in model_all.parameters())
    print(f"All features (conv+norm+A_log): {params_all:,} params")

    # Test without conv
    model_no_conv = DiagTripleR(dim, expansion=expansion, use_conv=False, use_internal_norm=True, use_A_log=True)
    params_no_conv = sum(p.numel() for p in model_no_conv.parameters())
    print(f"No conv: {params_no_conv:,} params (+{params_all - params_no_conv:,} from conv)")

    # Test without norm
    model_no_norm = DiagTripleR(dim, expansion=expansion, use_conv=True, use_internal_norm=False, use_A_log=True)
    params_no_norm = sum(p.numel() for p in model_no_norm.parameters())
    print(f"No norm: {params_no_norm:,} params (+{params_all - params_no_norm:,} from norm)")

    # Test without A_log (original)
    model_no_alog = DiagTripleR(dim, expansion=expansion, use_conv=True, use_internal_norm=True, use_A_log=False)
    params_no_alog = sum(p.numel() for p in model_no_alog.parameters())
    print(f"No A_log: {params_no_alog:,} params (A_log vs r_h: same count)")

    # Test baseline (no new features)
    model_base = DiagTripleR(dim, expansion=expansion, use_conv=False, use_internal_norm=False, use_A_log=False)
    params_base = sum(p.numel() for p in model_base.parameters())
    print(f"Baseline (no improvements): {params_base:,} params")
    print()

    # Forward pass test
    x = torch.randn(2, 16, dim)  # [B, T, dim]
    print("Forward pass test...")
    out, h = model_all(x)
    print(f"Input: {x.shape} -> Output: {out.shape}, Hidden: {h.shape}")
    print("OK!")
