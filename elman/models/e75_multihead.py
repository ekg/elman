"""
E75 Multi-Head: Gated Delta Matrix with H Independent Heads

Multi-head version of E75 (Gated Delta Rule) where each head maintains
its own n_state x n_state matrix state.

Architecture per head h:
    k_h = W_k_h @ x          # [n_state]
    v_h = W_v_h @ x          # [n_state]
    q_h = W_q_h @ x          # [n_state]
    beta_h = sigmoid(W_beta_h @ x + b_beta_h)  # [n_state] per-row forget gate

    k_norm = k_h / ||k_h||
    r = S_h @ k_norm          # retrieve
    delta = v_h - r           # delta
    S_h = tanh(beta_h * S_h + outer(delta, k_norm))  # gated update

    Sq_h = S_h @ q_h
    out_h = Sq_h * silu(Sq_h)  # [n_state]

Output: concat(out_0, out_1, ..., out_{H-1})  # [H * n_state]

Benefits:
- H independent memory systems (like multi-head attention)
- Each head can specialize for different types of associations
- Total state: H * n_state^2 (linear in H)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class E75MultiHeadCell(nn.Module):
    """
    E75 Multi-Head Gated Delta Matrix cell.

    H independent heads, each with its own n_state x n_state matrix state.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 4,
        init_beta_bias: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads

        # Fused projections: [H * n_state, dim] for efficiency
        # Each head gets its own slice of the projection
        self.W_k = nn.Parameter(torch.empty(n_heads * n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_heads * n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_heads * n_state, dim))
        self.W_beta = nn.Parameter(torch.empty(n_heads * n_state, dim))

        # Per-head beta biases: [H, n_state]
        self.b_beta = nn.Parameter(torch.full((n_heads, n_state), init_beta_bias))

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        H = self.n_heads

        # Initialize each head's projections with xavier
        for h in range(H):
            start = h * n
            end = (h + 1) * n
            nn.init.xavier_uniform_(self.W_k[start:end])
            nn.init.xavier_uniform_(self.W_v[start:end])
            nn.init.xavier_uniform_(self.W_q[start:end])
            nn.init.xavier_uniform_(self.W_beta[start:end])

    def forward(
        self,
        x: torch.Tensor,
        S_list: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [T, B, dim] input sequence
            S_list: list of H tensors, each [B, n_state, n_state] initial matrix states

        Returns:
            output: [T, B, H * n_state] concatenated outputs from all heads
            S_list: list of H final matrix states [B, n_state, n_state]
        """
        T, B, D = x.shape
        n = self.n_state
        H = self.n_heads

        # Initialize states if not provided
        if S_list is None:
            S_list = [torch.zeros(B, n, n, device=x.device, dtype=x.dtype) for _ in range(H)]

        # Project all inputs at once: [T*B, dim] @ [dim, H*n] -> [T*B, H*n]
        x_flat = x.reshape(T * B, D)

        # Compute all projections: [T, B, H, n]
        k_all = (x_flat @ self.W_k.T).reshape(T, B, H, n)
        v_all = (x_flat @ self.W_v.T).reshape(T, B, H, n)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, H, n)

        # Beta with bias: [T, B, H, n]
        beta_proj = (x_flat @ self.W_beta.T).reshape(T, B, H, n)
        beta_all = torch.sigmoid(beta_proj + self.b_beta)  # Broadcasting [H, n] over [T, B, H, n]

        # Clone S_list for in-place updates
        S_list = [S.clone() for S in S_list]

        outputs = []
        for t in range(T):
            head_outputs = []

            for h in range(H):
                # Get projections for this head at this timestep: [B, n]
                k = k_all[t, :, h]
                v = v_all[t, :, h]
                q = q_all[t, :, h]
                beta = beta_all[t, :, h]  # [B, n]

                # Normalize k
                k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n]

                # Retrieve from memory: S @ k_norm -> [B, n]
                retrieved = torch.einsum('bij,bj->bi', S_list[h], k_norm)

                # Delta update with forget gate
                delta = v - retrieved  # [B, n]
                outer = torch.einsum('bi,bj->bij', delta, k_norm)  # [B, n, n]

                # Gated update: S = tanh(beta * S + outer)
                # beta: [B, n] -> [B, n, 1] for row-wise gating
                S_list[h] = torch.tanh(beta.unsqueeze(-1) * S_list[h] + outer)

                # Self-gating output: Sq * silu(Sq)
                Sq = torch.einsum('bij,bj->bi', S_list[h], q)  # [B, n]
                out_h = Sq * F.silu(Sq)  # [B, n]
                head_outputs.append(out_h)

            # Concatenate all head outputs: [B, H * n]
            out_t = torch.cat(head_outputs, dim=-1)
            outputs.append(out_t)

        # Stack outputs: [T, B, H * n]
        output = torch.stack(outputs, dim=0)
        return output, S_list


class E75MultiHead(nn.Module):
    """
    E75 Multi-Head: Gated Delta Matrix with H Independent Heads - Full layer.

    Each head maintains its own n_state x n_state matrix state.
    Total output dimension: H * n_state
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        init_beta_bias: float = 2.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.n_heads = n_heads
        self.use_conv = use_conv

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

        self.cell = E75MultiHeadCell(
            self.d_inner,
            n_state=n_state,
            n_heads=n_heads,
            init_beta_bias=init_beta_bias,
        )

        # Output projection: H * n_state -> dim
        self.out_proj = nn.Linear(n_heads * n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input sequence
            hidden: Optional list of H matrices, each [B, n_state, n_state]

        Returns:
            output: [B, T, dim] output sequence
            hidden: list of H final matrix states [B, n_state, n_state]
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)

        # Optional conv
        if self.use_conv:
            x_proj = x_proj.transpose(1, 2)
            x_proj = self.conv1d(x_proj)[:, :, :T]
            x_proj = x_proj.transpose(1, 2)

        # Apply SiLU activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.transpose(0, 1).contiguous()

        # Run cell
        cell_out, S_list = self.cell(x_rnn, hidden)

        # Transpose back: [B, T, H * n_state]
        cell_out = cell_out.transpose(0, 1).contiguous()

        # Output projection and dropout
        output = self.out_proj(cell_out)
        output = self.dropout(output)

        return output, S_list

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return (f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, '
                f'n_heads={self.n_heads}, LEVEL=75_MULTIHEAD')


if __name__ == "__main__":
    print("Testing E75 Multi-Head (Gated Delta Matrix with H Heads)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f"Device: {device}")

    # Test dimensions
    B, T, dim = 4, 32, 512
    n_state = 32
    n_heads = 4

    print(f"\nConfig: B={B}, T={T}, dim={dim}, n_state={n_state}, n_heads={n_heads}")
    print(f"Total state size per batch: {n_heads} * {n_state}^2 = {n_heads * n_state * n_state}")

    # Create model
    model = E75MultiHead(
        dim=dim,
        expansion=2.0,
        n_state=n_state,
        n_heads=n_heads,
    ).to(device).to(dtype)

    print(f"\nModel parameters: {model.get_num_params():,}")

    # Test forward pass
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    out, S_list = model(x)
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Number of state matrices: {len(S_list)}")
    for i, S in enumerate(S_list):
        print(f"    S[{i}]: {S.shape}")

    # Test backward pass
    loss = out.sum()
    loss.backward()
    print("\nBackward pass: OK")

    # Test with provided hidden state (TBPTT scenario)
    print("\n--- Testing with provided hidden state ---")
    S_init = [torch.randn(B, n_state, n_state, device=device, dtype=dtype) for _ in range(n_heads)]

    out2, S_list2 = model(x, hidden=S_init)
    print(f"Output with init state: {out2.shape}")

    # Verify states are different from initialization
    for i in range(n_heads):
        diff = (S_list2[i] - S_init[i]).abs().mean().item()
        print(f"  S[{i}] changed by avg {diff:.6f}")

    # Test multiple head configurations
    print("\n--- Testing different head configurations ---")
    configs = [
        (2, 48),   # 2 heads, 48 state -> 4608 state size
        (4, 32),   # 4 heads, 32 state -> 4096 state size
        (8, 24),   # 8 heads, 24 state -> 4608 state size
        (8, 16),   # 8 heads, 16 state -> 2048 state size
    ]

    for H, n in configs:
        model_test = E75MultiHead(
            dim=dim,
            expansion=1.0,
            n_state=n,
            n_heads=H,
        ).to(device).to(dtype)

        x_test = torch.randn(B, T, dim, device=device, dtype=dtype)
        out_test, S_test = model_test(x_test)

        params = model_test.get_num_params()
        state_size = H * n * n
        print(f"  H={H}, n_state={n}: params={params:,}, state_size={state_size}, output_dim={H*n}")

        # Quick backward test
        out_test.sum().backward()

    print("\n" + "=" * 60)
    print("All tests passed!")
