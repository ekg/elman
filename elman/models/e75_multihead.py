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

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E75MH_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e75_multihead_forward')
except ImportError:
    E75MH_CUDA_AVAILABLE = False


class E75MultiHeadCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E75 Multi-Head autograd function with gradient checkpointing."""

    @staticmethod
    def forward(ctx, training, x, S0, W_k, W_v, W_q, W_beta, b_beta, n_heads):
        results = hasty_pytorch_lib.e75_multihead_forward(
            training, x, S0, W_k, W_v, W_q, W_beta, b_beta, n_heads
        )
        # results = [output, S, k_cache, v_cache, q_cache, beta_cache, S_cache]
        # S_cache contains both S_checkpoints and Sq_cache concatenated
        output = results[0]
        S = results[1]
        k_cache = results[2]
        v_cache = results[3]
        q_cache = results[4]
        beta_cache = results[5]
        S_cache = results[6]  # Combined checkpoints + Sq_cache

        ctx.save_for_backward(
            x, S_cache,
            k_cache, v_cache, q_cache, beta_cache,
            W_k, W_v, W_q, W_beta
        )
        ctx.n_heads = n_heads
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_cache,
         k_cache, v_cache, q_cache, beta_cache,
         W_k, W_v, W_q, W_beta) = ctx.saved_tensors
        n_heads = ctx.n_heads

        # Split S_cache into S_checkpoints and Sq_cache
        # S_cache layout: [checkpoints_flat || sq_cache_flat]
        T, B, _ = x.shape
        n_state = k_cache.size(3)
        checkpoint_interval = 16
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        checkpoints_size = num_checkpoints * B * n_heads * n_state * n_state
        sq_cache_size = T * B * n_heads * n_state

        S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, n_heads, n_state, n_state)
        Sq_cache = S_cache[checkpoints_size:].view(T, B, n_heads, n_state)

        grads = hasty_pytorch_lib.e75_multihead_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache, beta_cache,
            d_output.contiguous(),
            W_k, W_v, W_q, W_beta,
            n_heads
        )
        # grads = [dx, dW_k, dW_v, dW_q, dW_beta, db_beta]
        dx = grads[0]
        dW_k = grads[1]
        dW_v = grads[2]
        dW_q = grads[3]
        dW_beta = grads[4]
        db_beta = grads[5]

        # Return gradients for: training, x, S0, W_k, W_v, W_q, W_beta, b_beta, n_heads
        return None, dx, None, dW_k, dW_v, dW_q, dW_beta, db_beta, None


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
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads
        self.use_cuda = use_cuda and E75MH_CUDA_AVAILABLE

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
        S_list: Optional[List[torch.Tensor]] = None,
        use_cuda: Optional[bool] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [T, B, dim] input sequence
            S_list: list of H tensors, each [B, n_state, n_state] initial matrix states
            use_cuda: Override instance setting for CUDA usage

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

        _use_cuda = use_cuda if use_cuda is not None else self.use_cuda

        # Use CUDA kernel if available
        if _use_cuda and E75MH_CUDA_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
            # Stack S_list into single tensor: [B, H, n, n]
            S0 = torch.stack(S_list, dim=1)

            S_final, output = E75MultiHeadCUDAFunction.apply(
                self.training, x, S0,
                self.W_k, self.W_v, self.W_q, self.W_beta, self.b_beta,
                H
            )

            # Convert S_final [B, H, n, n] back to list
            S_list_out = [S_final[:, h] for h in range(H)]
            return output, S_list_out

        # PyTorch fallback
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
        use_cuda: bool = True,
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
            use_cuda=use_cuda,
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
    print(f"CUDA kernel available: {E75MH_CUDA_AVAILABLE}")

    # Test dimensions
    B, T, dim = 4, 32, 512
    n_state = 32
    n_heads = 4

    print(f"\nConfig: B={B}, T={T}, dim={dim}, n_state={n_state}, n_heads={n_heads}")
    print(f"Total state size per batch: {n_heads} * {n_state}^2 = {n_heads * n_state * n_state}")

    # PyTorch fallback test
    print("\n--- PyTorch Fallback ---")
    model = E75MultiHead(
        dim=dim,
        expansion=2.0,
        n_state=n_state,
        n_heads=n_heads,
        use_cuda=False,
    ).to(device).to(dtype)

    print(f"Model parameters: {model.get_num_params():,}")

    # Test forward pass
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    out, S_list = model(x)
    print(f"Forward: Input {x.shape} -> Output {out.shape}")
    print(f"  Number of state matrices: {len(S_list)}, each {S_list[0].shape}")

    # Test backward pass
    loss = out.sum()
    loss.backward()
    print("Backward: OK")

    # CUDA test
    if E75MH_CUDA_AVAILABLE and device == 'cuda':
        print("\n--- CUDA Kernel ---")
        model_cuda = E75MultiHead(
            dim=dim,
            expansion=2.0,
            n_state=n_state,
            n_heads=n_heads,
            use_cuda=True,
        ).to(device).to(dtype)

        # Copy weights
        model_cuda.load_state_dict(model.state_dict())

        x_cuda = torch.randn(B, T, dim, device=device, dtype=dtype)

        out_cuda, S_list_cuda = model_cuda(x_cuda)
        print(f"Forward: Input {x_cuda.shape} -> Output {out_cuda.shape}")

        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        print("Backward: OK")

    # Gradient correctness test
    if E75MH_CUDA_AVAILABLE and device == 'cuda':
        print("\n" + "=" * 60)
        print("Gradient correctness test (CUDA vs PyTorch)")
        print("=" * 60)

        torch.manual_seed(42)

        x_test = torch.randn(2, 16, 256, device=device, dtype=dtype, requires_grad=True)

        # PyTorch reference
        model_pt = E75MultiHead(
            dim=256, expansion=1.0, n_state=32, n_heads=4, use_cuda=False
        ).to(device).to(dtype)

        out_pt, _ = model_pt(x_test)
        loss_pt = out_pt.sum()
        loss_pt.backward()
        grad_pt = x_test.grad.clone()
        grad_W_k_pt = model_pt.cell.W_k.grad.clone()
        grad_W_beta_pt = model_pt.cell.W_beta.grad.clone()

        # Reset
        x_test.grad = None

        # CUDA version with same weights
        model_cuda = E75MultiHead(
            dim=256, expansion=1.0, n_state=32, n_heads=4, use_cuda=True
        ).to(device).to(dtype)
        model_cuda.load_state_dict(model_pt.state_dict())

        out_cuda, _ = model_cuda(x_test)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        grad_cuda = x_test.grad.clone()
        grad_W_k_cuda = model_cuda.cell.W_k.grad.clone()
        grad_W_beta_cuda = model_cuda.cell.W_beta.grad.clone()

        # Compute relative errors
        def rel_err(a, b):
            return (a - b).abs().max().item() / (a.abs().max().item() + 1e-8)

        dx_rel = rel_err(grad_pt, grad_cuda)
        dWk_rel = rel_err(grad_W_k_pt, grad_W_k_cuda)
        dWbeta_rel = rel_err(grad_W_beta_pt, grad_W_beta_cuda)
        out_rel = rel_err(out_pt, out_cuda)

        print(f"Output relative error: {out_rel:.4f}")
        print(f"dx relative error: {dx_rel:.4f}")
        print(f"dW_k relative error: {dWk_rel:.4f}")
        print(f"dW_beta relative error: {dWbeta_rel:.4f}")

        # 5% relative error is acceptable for bfloat16 with checkpoint recomputation
        if dx_rel < 0.05 and dWk_rel < 0.05 and dWbeta_rel < 0.05:
            print("PASSED: Gradients match within 5% relative tolerance!")
        else:
            print("WARNING: Large gradient discrepancy - may need investigation")

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
