"""
Level 7: Diagonal Triple R Elman (Linear Space)

Like Level 5 (Linear Triple R) but with DIAGONAL r_h and r_delta vectors
instead of full matrices. More efficient O(d) recurrence.

Architecture:
    v = r_h * h_prev + W_x @ x + b              -- DIAGONAL r_h (element-wise)
    delta_raw = W_delta @ x + r_delta * h_prev + b_delta  -- DIAGONAL r_delta
    delta = sigmoid(delta_raw)
    h_new = (1 - delta) * h_prev + delta * tanh(v)

    # Selective output
    compete = softmax(h_new.reshape(groups), dim=-1)
    output = compete * silu(W_out @ h_new)

Key features:
- Diagonal r_h for efficient recurrence (O(d) vs O(d^2))
- Diagonal r_delta for hidden-state modulated gating
- Linear-space hidden state (simpler than log-space variants)
- More efficient than full R matrices while retaining expressivity
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
    """Autograd function for Diagonal Triple R Elman (Haste kernel)."""

    @staticmethod
    def forward(ctx, training, n_groups, x, h0,
                W_x, r_h, r_delta, W_delta, W_out, b, b_delta):
        results = hasty_pytorch_lib.diag_triple_r_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            r_h.contiguous(),
            r_delta.contiguous(),
            W_delta.contiguous(),
            W_out.contiguous(),
            b.contiguous(),
            b_delta.contiguous(),
            n_groups
        )

        h, output, v_cache, delta_cache, compete_cache = results

        if training:
            ctx.save_for_backward(
                W_x, r_h, r_delta, W_delta, W_out,
                x, h, v_cache, delta_cache, compete_cache
            )
            ctx.n_groups = n_groups

        return output, h

    @staticmethod
    def backward(ctx, d_output, d_h):
        (W_x, r_h, r_delta, W_delta, W_out,
         x, h, v_cache, delta_cache, compete_cache) = ctx.saved_tensors

        grads = hasty_pytorch_lib.diag_triple_r_backward(
            W_x, r_h, r_delta, W_delta, W_out,
            x, h, v_cache, delta_cache, compete_cache,
            d_output.contiguous(),
            ctx.n_groups
        )

        # Unpack: dx, dW_x, d_r_h, d_r_delta, dW_delta, dW_out, db, db_delta
        dx, dW_x, d_r_h, d_r_delta, dW_delta, dW_out, db, db_delta = grads

        # Return gradients in same order as forward inputs
        return (None, None, dx, None,
                dW_x, d_r_h, d_r_delta, dW_delta, dW_out, db, db_delta)


class DiagTripleRCell(nn.Module):
    """
    Diagonal Triple R Elman cell - Level 7.

    Linear-space diagonal recurrence with r_h and r_delta vectors.
    Efficient O(d) recurrence with h+x selective output gating.

    Architecture:
        v = r_h * h_prev + W_x @ x + b
        delta = sigmoid(W_delta @ x + r_delta * h_prev + b_delta)
        h_new = (1 - delta) * h_prev + delta * tanh(v)
        output = h_new * silu(h_new + x + b_gate)  # h+x gating!

    Args:
        dim: Hidden dimension
        n_groups: Number of groups (unused, kept for API compat)
        delta_init: Initial bias for delta gate
        r_h_init: Initial value for r_h
    """

    def __init__(self, dim, n_groups=32, delta_init=-2.0, r_h_init=0.9, **kwargs):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups  # Kept for API compat, not used with h+x gate

        # Diagonal r_h (decay factor)
        self.r_h = nn.Parameter(torch.full((dim,), float(r_h_init)))

        # Diagonal r_delta
        self.r_delta = nn.Parameter(torch.full((dim,), 0.1))

        # W_x for input transformation
        self.W_x = nn.Parameter(torch.empty(dim, dim))

        # Delta gate
        self.W_delta = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_delta = nn.Parameter(torch.full((dim,), delta_init))

        # h+x selective gate bias (replaces W_out matrix!)
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x, gain=0.5)
        nn.init.xavier_uniform_(self.W_delta, gain=0.1)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state, or None

        Returns:
            output: [T, B, dim] selective output (h * silu(h + x + b_gate))
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
                self.W_delta, self.b_gate, self.b, self.b_delta  # b_gate replaces W_out
            )
        else:
            return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation with diagonal r_h, r_delta, and h+x gate."""
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

            # h+x selective output: output = h * silu(h + x + b_gate)
            gate = F.silu(h_new + x_t + self.b_gate)
            output = h_new * gate

            h_list.append(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)

        return output, h


class DiagTripleR(nn.Module):
    """
    Diagonal Triple R Elman layer for use in LadderLM.

    Level 7: Linear-space diagonal r_h and r_delta for efficient recurrence.
    More efficient than Level 5 (full R matrices) with similar expressivity.

    Args:
        dim: Model dimension
        expansion: Hidden state expansion factor
        n_groups: Number of groups for compete softmax
        delta_init: Initial bias for delta gate
        dropout: Dropout rate
        **kwargs: Ignored (for API compatibility)
    """

    def __init__(self, dim, expansion=1.0, n_groups=32, delta_init=-2.0,
                 dropout=0.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Adjust n_groups for inner dimension
        while self.d_inner % n_groups != 0 and n_groups > 1:
            n_groups -= 1

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Diagonal Triple R cell
        self.cell = DiagTripleRCell(
            self.d_inner, n_groups=n_groups, delta_init=delta_init
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
            h0: [B, d_inner] initial hidden state or None

        Returns:
            output: [B, T, dim] output for residual connection
            h_final: [B, d_inner] final hidden state for TBPTT
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)

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
    'LEVEL_7_AVAILABLE',
    'HASTE_DIAG_TRIPLE_R_AVAILABLE',
]
