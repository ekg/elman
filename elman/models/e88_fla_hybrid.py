"""
E88: FLA-GDN Hybrid with Nonlinear Matrix State

Combines FLA-GatedDeltaNet's proven design elements with E75's nonlinear
matrix state mechanism. Key adaptations from FLA-GDN:

1. **Mamba2-style exponential decay** (replaces sigmoid beta):
   - A_log: learned log eigenvalues (like Mamba2)
   - dt_bias: learned time-step bias
   - a_proj: input-dependent alpha gate
   - Decay: g = -exp(A_log) * softplus(a_proj(x) + dt_bias)

2. **Output gating** (FLA-GDN style):
   - g_proj: output gate projection
   - Gated output: o * sigmoid(g)

3. **Short convolutions** (FLA-GDN style):
   - Depthwise conv on k, v, q after projection
   - bias=False (matching FLA default)
   - SiLU activation fused

4. **L2-normalized Q and K** in retrieval

Architecture per head h:
    # Projections with conv+silu
    k_h = silu(conv(W_k @ x))   # [n_state]
    v_h = silu(conv(W_v @ x))   # [n_state]
    q_h = silu(conv(W_q @ x))   # [n_state]

    # Mamba2-style exponential decay
    g_h = -exp(A_log) * softplus(a_proj(x) + dt_bias)  # scalar per head

    # L2 normalize k and q
    k_norm = k_h / ||k_h||
    q_norm = q_h / ||q_h||

    # Matrix state update (NONLINEAR - key differentiator from FLA-GDN)
    r = S_h @ k_norm            # retrieve
    delta = v_h - r             # delta
    S_h = tanh(exp(g_h) * S_h + outer(delta, k_norm))  # exp decay + nonlinear

    # Output with gating
    Sq_h = S_h @ q_norm
    out_h = Sq_h * sigmoid(g_out_h)  # gated output

Output: out_proj(concat(out_0, ..., out_{H-1}))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    # E88 native CUDA kernel
    E88_NATIVE_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_forward')
    # Legacy E75 kernels for backwards compatibility
    E88_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e75_multihead_forward')
    E88_PRECOMPUTED_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e75_multihead_precomputed_forward')
except ImportError:
    E88_NATIVE_CUDA_AVAILABLE = False
    E88_CUDA_AVAILABLE = False
    E88_PRECOMPUTED_CUDA_AVAILABLE = False

# Backwards compat
E75MH_CUDA_AVAILABLE = E88_CUDA_AVAILABLE
E75MH_PRECOMPUTED_CUDA_AVAILABLE = E88_PRECOMPUTED_CUDA_AVAILABLE


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


class E88FLAHybridCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E88 FLA Hybrid autograd function.

    E88 uses:
    - Rectangular state [n_state x head_v_dim]
    - Mamba2-style scalar exponential decay per head
    - L2-normalized k and q (done externally before calling this)
    - Self-gating output: out = Sq * silu(Sq)
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, S0, n_heads):
        """
        Args:
            training: bool
            k: [T, B, H, n_state] L2 normalized keys
            v: [T, B, H, head_v_dim] values
            q: [T, B, H, n_state] L2 normalized queries
            decay: [T, B, H] exponential decay factors
            S0: [B, H, n_state, head_v_dim] initial state
            n_heads: int
        """
        results = hasty_pytorch_lib.e88_fla_hybrid_forward(
            training, k, v, q, decay, S0, n_heads
        )
        # results = [S_final, output, S_cache]
        S_final = results[0]  # [B, H, n_state, head_v_dim]
        output = results[1]   # [T, B, H, head_v_dim]
        S_cache = results[2]  # Combined checkpoints + Sq_cache

        ctx.save_for_backward(k, v, q, decay, S_cache)
        ctx.n_heads = n_heads
        ctx.n_state = k.size(-1)
        ctx.head_v_dim = v.size(-1)
        return S_final, output

    @staticmethod
    def backward(ctx, dS, d_output):
        k, v, q, decay, S_cache = ctx.saved_tensors
        n_heads = ctx.n_heads
        n_state = ctx.n_state
        head_v_dim = ctx.head_v_dim

        T, B, H, _ = k.shape
        checkpoint_interval = 16
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1

        # Split S_cache into S_checkpoints and Sq_cache
        checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim
        sq_cache_size = T * B * H * head_v_dim

        S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, H, n_state, head_v_dim)
        Sq_cache = S_cache[checkpoints_size:checkpoints_size + sq_cache_size].view(T, B, H, head_v_dim)

        grads = hasty_pytorch_lib.e88_fla_hybrid_backward(
            k, v, q, decay,
            S_checkpoints, Sq_cache,
            d_output.contiguous(),
            n_heads
        )
        # grads = [d_k, d_v, d_q, d_decay]
        d_k = grads[0]
        d_v = grads[1]
        d_q = grads[2]
        d_decay = grads[3]

        # Return gradients for: training, k, v, q, decay, S0, n_heads
        return None, d_k, d_v, d_q, d_decay, None, None


class E75MultiHeadPrecomputedCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E75 Multi-Head with pre-computed k, v, q, beta.

    Used for post-projection convolution mode (FLA-GDN style).
    k, v, q have already had conv+silu applied.
    beta has already had sigmoid applied.
    """

    @staticmethod
    def forward(ctx, training, k, v, q, beta, S0, n_heads):
        """
        Args:
            training: bool
            k: [T, B, H, n_state] pre-computed (with conv+silu)
            v: [T, B, H, n_state] pre-computed (with conv+silu)
            q: [T, B, H, n_state] pre-computed (with conv+silu)
            beta: [T, B, H, n_state] pre-computed (with sigmoid)
            S0: [B, H, n_state, n_state] initial state
            n_heads: int
        """
        results = hasty_pytorch_lib.e75_multihead_precomputed_forward(
            training, k, v, q, beta, S0, n_heads
        )
        # results = [output, S, S_cache]
        output = results[0]  # [T, B, H, n_state]
        S = results[1]       # [B, H, n_state, n_state]
        S_cache = results[2] # Combined checkpoints + Sq_cache

        ctx.save_for_backward(k, v, q, beta, S_cache)
        ctx.n_heads = n_heads
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        k, v, q, beta, S_cache = ctx.saved_tensors
        n_heads = ctx.n_heads

        # Split S_cache into S_checkpoints and Sq_cache
        T, B, H, n_state = k.shape
        checkpoint_interval = 16
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        checkpoints_size = num_checkpoints * B * H * n_state * n_state
        sq_cache_size = T * B * H * n_state

        S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, H, n_state, n_state)
        Sq_cache = S_cache[checkpoints_size:].view(T, B, H, n_state)

        grads = hasty_pytorch_lib.e75_multihead_precomputed_backward(
            k, v, q, beta,
            S_checkpoints, Sq_cache,
            d_output.contiguous(),
            n_heads
        )
        # grads = [d_k, d_v, d_q, d_beta]
        d_k = grads[0]
        d_v = grads[1]
        d_q = grads[2]
        d_beta = grads[3]

        # Return gradients for: training, k, v, q, beta, S0, n_heads
        return None, d_k, d_v, d_q, d_beta, None, None


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


class E88FLAHybrid(nn.Module):
    """
    E88: FLA-GDN Hybrid with Nonlinear Matrix State.

    Combines FLA-GatedDeltaNet's proven design with E75's nonlinear matrix state.

    Key FLA-GDN elements:
    1. Mamba2-style exponential decay (A_log, dt_bias, a_proj)
    2. Output gating with g_proj
    3. Short convolutions on k, v, q (bias=False)
    4. L2-normalized Q and K

    Kept from E75:
    - Nonlinear matrix state: S = tanh(decay * S + outer(delta, k_norm))
    - Multi-head structure with independent states
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,  # FLA-GDN default
        n_state: int = 32,
        n_heads: int = 8,  # More heads like FLA-GDN
        dropout: float = 0.0,
        d_conv: int = 4,
        use_cuda: bool = True,
        **kwargs
    ):
        super().__init__()

        # Validate n_state is multiple of 8
        if n_state % 8 != 0:
            raise ValueError(f"n_state must be multiple of 8, got {n_state}")

        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads
        self.d_conv = d_conv

        # Key and query dimensions (like FLA-GDN: key_dim = num_heads * head_k_dim)
        self.key_dim = n_heads * n_state
        # Value dimension with expansion (like FLA-GDN: value_dim = expand_v * key_dim)
        self.value_dim = int(n_heads * n_state * expansion)
        self.head_v_dim = self.value_dim // n_heads

        # === Projections (FLA-GDN style) ===
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)

        # === Mamba2-style decay parameters ===
        # a_proj: input-dependent alpha (maps to num_heads scalars)
        self.a_proj = nn.Linear(dim, n_heads, bias=False)

        # A_log: learned log eigenvalues (Mamba2 style)
        A = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # dt_bias: learned time-step bias (Mamba2 style initialization)
        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # === Short convolutions (FLA-GDN style, bias=False) ===
        self.q_conv = nn.Conv1d(
            self.key_dim, self.key_dim, d_conv,
            padding=d_conv - 1, groups=self.key_dim, bias=False
        )
        self.k_conv = nn.Conv1d(
            self.key_dim, self.key_dim, d_conv,
            padding=d_conv - 1, groups=self.key_dim, bias=False
        )
        self.v_conv = nn.Conv1d(
            self.value_dim, self.value_dim, d_conv,
            padding=d_conv - 1, groups=self.value_dim, bias=False
        )

        # === Output gating (FLA-GDN style) ===
        self.g_proj = nn.Linear(dim, self.value_dim, bias=False)

        # === Output projection ===
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

        # === Output normalization (RMSNorm like FLA, works with bfloat16) ===
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.norm_eps = 1e-5

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.g_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.xavier_uniform_(self.a_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input
            hidden: Optional list of H matrices, each [B, n_state, n_state]

        Returns:
            output: [B, T, dim]
            hidden: list of H final matrix states
        """
        B, T, D = x.shape
        n = self.n_state
        H = self.n_heads

        # === Projections ===
        q = self.q_proj(x)  # [B, T, key_dim]
        k = self.k_proj(x)  # [B, T, key_dim]
        v = self.v_proj(x)  # [B, T, value_dim]

        # === Short convolutions with SiLU (FLA-GDN style) ===
        # Conv expects [B, C, T]
        q = F.silu(self.q_conv(q.transpose(1, 2))[:, :, :T]).transpose(1, 2)
        k = F.silu(self.k_conv(k.transpose(1, 2))[:, :, :T]).transpose(1, 2)
        v = F.silu(self.v_conv(v.transpose(1, 2))[:, :, :T]).transpose(1, 2)

        # === Compute Mamba2-style decay ===
        # g = -exp(A_log) * softplus(a_proj(x) + dt_bias)
        # Shape: [B, T, H]
        # Compute in float32 for numerical stability, then cast back
        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(x).float() + self.dt_bias
        )
        # Convert to decay factor: decay = exp(g) in [0, 1]
        decay = g.exp().to(x.dtype)  # [B, T, H], cast back to input dtype

        # === Reshape for per-head processing ===
        # q, k: [B, T, H, n_state]
        q = q.view(B, T, H, n)
        k = k.view(B, T, H, n)
        # v: [B, T, H, head_v_dim]
        v = v.view(B, T, H, self.head_v_dim)

        # === Initialize states ===
        # State shape: [B, H, n_state, head_v_dim] stacked (for CUDA kernel)
        if hidden is None:
            S0 = torch.zeros(B, H, n, self.head_v_dim, device=x.device, dtype=x.dtype)
        else:
            # Convert list to stacked tensor
            S0 = torch.stack(hidden, dim=1)  # [B, H, n, head_v_dim]

        # === Use CUDA kernel if available ===
        use_cuda = (E88_NATIVE_CUDA_AVAILABLE and x.is_cuda and
                    x.dtype == torch.bfloat16 and self.training)

        if use_cuda:
            # L2 normalize k and q (CUDA expects normalized inputs)
            # Autocast may promote norm() to float32, so explicitly cast ALL inputs back
            input_dtype = x.dtype  # Use x.dtype as authoritative (should be bf16)
            k_norm = (k / (k.norm(dim=-1, keepdim=True) + 1e-6)).to(input_dtype)
            q_norm = (q / (q.norm(dim=-1, keepdim=True) + 1e-6)).to(input_dtype)

            # Transpose for CUDA: [B, T, H, dim] -> [T, B, H, dim]
            # Cast ALL inputs to input_dtype to handle any autocast promotions
            k_cuda = k_norm.transpose(0, 1).contiguous()
            v_cuda = v.to(input_dtype).transpose(0, 1).contiguous()
            q_cuda = q_norm.transpose(0, 1).contiguous()
            decay_cuda = decay.to(input_dtype).transpose(0, 1).contiguous()
            S0 = S0.to(input_dtype)

            # Call CUDA kernel via autograd.Function
            S_final, output_cuda = E88FLAHybridCUDAFunction.apply(
                self.training, k_cuda, v_cuda, q_cuda, decay_cuda, S0, H
            )

            # Transpose output back: [T, B, H, head_v_dim] -> [B, T, H, head_v_dim]
            output = output_cuda.transpose(0, 1)

            # Convert S_final back to list for hidden state
            S_list = [S_final[:, h] for h in range(H)]
        else:
            # === PyTorch fallback: Recurrence with nonlinear matrix state ===
            S_list = [S0[:, h].clone() for h in range(H)]  # Convert to list

            outputs = []
            for t in range(T):
                head_outputs = []
                for h in range(H):
                    k_t = k[:, t, h]  # [B, n_state]
                    q_t = q[:, t, h]  # [B, n_state]
                    v_t = v[:, t, h]  # [B, head_v_dim]
                    decay_t = decay[:, t, h:h+1]  # [B, 1]

                    # L2 normalize k and q (FLA-GDN style)
                    k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n_state]
                    q_norm = q_t / (q_t.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n_state]

                    # Retrieve from memory: S @ k_norm -> [B, head_v_dim]
                    retrieved = torch.einsum('biv,bi->bv', S_list[h], k_norm)

                    # Delta update
                    delta = v_t - retrieved  # [B, head_v_dim]

                    # Outer product: [B, n_state, head_v_dim]
                    outer = torch.einsum('bv,bi->biv', delta, k_norm)

                    # Gated update with NONLINEAR tanh (key differentiator from FLA-GDN)
                    # S = tanh(decay * S + outer)
                    S_list[h] = torch.tanh(decay_t.unsqueeze(-1) * S_list[h] + outer)

                    # Query the state: S @ q_norm -> [B, head_v_dim]
                    Sq = torch.einsum('biv,bi->bv', S_list[h], q_norm)

                    # Self-gating: out = Sq * silu(Sq) (matches CUDA kernel)
                    out_h = Sq * F.silu(Sq)
                    head_outputs.append(out_h)

                # Stack head outputs: [B, H, head_v_dim]
                out_t = torch.stack(head_outputs, dim=1)
                outputs.append(out_t)

            # Stack time: [B, T, H, head_v_dim]
            output = torch.stack(outputs, dim=1)

        # === Output gating (FLA-GDN style) ===
        g = self.g_proj(x).view(B, T, H, self.head_v_dim)  # [B, T, H, head_v_dim]
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = output.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        output = output * rms * self.o_norm_weight * torch.sigmoid(g)

        # Reshape: [B, T, value_dim]
        output = output.view(B, T, self.value_dim)

        # Output projection
        output = self.o_proj(output)
        output = self.dropout(output)

        return output, S_list

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return (f'dim={self.dim}, key_dim={self.key_dim}, value_dim={self.value_dim}, '
                f'n_state={self.n_state}, n_heads={self.n_heads}, LEVEL=88_FLA_HYBRID')


# Alias for backwards compatibility
E75MultiHead = E88FLAHybrid


if __name__ == "__main__":
    print("Testing E88 FLA Hybrid (Nonlinear Matrix State + FLA-GDN Design)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f"Device: {device}")

    # Test dimensions
    B, T, dim = 4, 32, 512
    n_state = 32
    n_heads = 8

    print(f"\nConfig: B={B}, T={T}, dim={dim}, n_state={n_state}, n_heads={n_heads}")

    # Test E88FLAHybrid
    print("\n--- E88 FLA Hybrid ---")
    model = E88FLAHybrid(
        dim=dim,
        expansion=2.0,
        n_state=n_state,
        n_heads=n_heads,
    ).to(device).to(dtype)

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Key dim: {model.key_dim}, Value dim: {model.value_dim}")

    # Test forward pass
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    out, S_list = model(x)
    print(f"Forward: Input {x.shape} -> Output {out.shape}")
    print(f"  Number of state matrices: {len(S_list)}, each {S_list[0].shape}")

    # Test backward pass
    loss = out.sum()
    loss.backward()
    print("Backward: OK")

    # Check gradients exist
    grad_k = model.k_proj.weight.grad
    grad_a = model.a_proj.weight.grad
    print(f"  k_proj grad norm: {grad_k.norm().item():.4f}")
    print(f"  a_proj grad norm: {grad_a.norm().item():.4f}")

    # Test multiple head configurations
    print("\n--- Testing different head configurations ---")
    configs = [
        (4, 32, 2.0),   # 4 heads, 32 state, 2x expansion
        (8, 24, 2.0),   # 8 heads, 24 state, 2x expansion
        (8, 32, 1.5),   # 8 heads, 32 state, 1.5x expansion
        (16, 16, 2.0),  # 16 heads, 16 state, 2x expansion
    ]

    for H, n, exp in configs:
        model_test = E88FLAHybrid(
            dim=dim,
            expansion=exp,
            n_state=n,
            n_heads=H,
        ).to(device).to(dtype)

        x_test = torch.randn(B, T, dim, device=device, dtype=dtype)
        out_test, S_test = model_test(x_test)

        params = model_test.get_num_params()
        state_size = H * n * model_test.head_v_dim  # Rectangular state
        print(f"  H={H}, n={n}, exp={exp}: params={params:,}, state_size={state_size}")

        # Quick backward test
        out_test.sum().backward()

    print("\n" + "=" * 60)
    print("All tests passed!")
