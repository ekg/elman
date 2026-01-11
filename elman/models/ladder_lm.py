"""
Language Model wrapper for E-Series Elman models.

E-Series:
    e0: Stock Elman - tanh + h*silu(W_gate@x) gating
    e1: Mamba-Gated Elman - Mamba2-style split projection gating
    e2: Slot Elman - Multi-slot memory (64x like Mamba2)

Architecture matches Mamba exactly:
    - Fused add+norm using mamba_ssm.ops.triton.layer_norm
    - Block pattern: residual = x + residual; x = norm(residual); x = mixer(x)
    - RMSNorm (not LayerNorm) for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import fused ops from mamba_ssm for exact architecture match
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_fn
    FUSED_NORM_AVAILABLE = True
except ImportError:
    FUSED_NORM_AVAILABLE = False
    RMSNorm = nn.RMSNorm  # Fallback to PyTorch

from .stock_elman import StockElman
from .mamba_gated_elman import MambaGatedElman
from .softsign_elman import SoftsignElman
from .diagonal_state_elman import DiagonalStateElman
from .slot_elman import SlotElman
from .lowrank_slot_elman import LowRankSlotElman
from .lowrank_elman import LowRankElman
from .pure_lowrank_elman import PureLowRankElman
from .diagonal_elman import DiagonalElman
from .scaled_lowrank_elman import ScaledLowRankElman
from .hybrid_elman import HybridElman
from .multiscale_elman import MultiScaleElman
from .selective_elman import SelectiveElman
from .selective_gated_elman import SelectiveGatedElman
from .matrix_state_elman import MatrixStateElman
from .selective_wh_elman import SelectiveWhElman
from .haware_gate_elman import HAwareGateElman
from .simplified_gate_elman import SimplifiedGateElman
from .mamba2_informed_elman import Mamba2InformedElman
from .structured_elman import StructuredElman
from .structured_elman_attention import StructuredElmanAttention
from .dual_memory_elman import DualMemoryElman
from .e24_single_gemm import E24Layer
from .e25_entmax import E25DualMemoryElman
from .e26_parallel import E26DualMemoryElman
from .e28_conv_elman import E28ConvElman


def get_ladder_level(level):
    """Get the module class for a specific ladder level.

    Args:
        level: Integer level (0-6, 8-10) or 'mamba2'

    Returns:
        Layer class
    """
    levels = {
        0: StockElman,
        1: MambaGatedElman,
        2: SlotElman,
        3: LowRankSlotElman,
        4: LowRankElman,
        5: PureLowRankElman,
        6: DiagonalElman,
        8: ScaledLowRankElman,
        9: HybridElman,
        10: MultiScaleElman,
        11: SelectiveElman,
        12: SelectiveGatedElman,  # E12: Hidden-state-dependent gating
        14: MatrixStateElman,  # E14: Matrix state with outer product update
        15: SoftsignElman,  # E15: E1 with softsign instead of tanh
        16: DiagonalStateElman,  # E16: Mamba2 efficiency + E1 nonlinearity
        17: SelectiveWhElman,  # E17: Input-dependent gating on W_h @ h
        '18a': lambda **kw: HAwareGateElman(gate_mode=0, **kw),  # E18-A: gate = z + h
        '18b': lambda **kw: HAwareGateElman(gate_mode=1, **kw),  # E18-B: gate = z + Rh
        '18e': lambda **kw: HAwareGateElman(gate_mode=2, **kw),  # E18-E: no gate
        '19a': lambda **kw: SimplifiedGateElman(gate_mode=0, **kw),  # E19-A: gate = Wx + h
        '19b': lambda **kw: SimplifiedGateElman(gate_mode=1, **kw),  # E19-B: gate = h-only
        '19d': lambda **kw: SimplifiedGateElman(gate_mode=2, **kw),  # E19-D: residual + z
        '19e': lambda **kw: SimplifiedGateElman(gate_mode=3, **kw),  # E19-E: residual + Wx + h
        20: Mamba2InformedElman,  # E20: Mamba2-style matrix state
        21: StructuredElman,  # E21: MIMO with nonlinear state
        '21s': lambda **kw: StructuredElman(mimo_rank=4, **kw),  # E21-S: smaller rank
        '21t': lambda **kw: StructuredElman(nonlinearity='tanh', **kw),  # E21-T: tanh
        '21l': lambda **kw: StructuredElman(nonlinearity='linear', **kw),  # E21-L: linear (ablation)
        22: StructuredElmanAttention,  # E22: E21 + state attention (UTM class)
        '22n': lambda **kw: StructuredElmanAttention(attn_type='over_N', **kw),  # E22-N: attention over N
        '22h': lambda **kw: StructuredElmanAttention(attn_type='over_heads', **kw),  # E22-H: attention over heads
        '22k4': lambda **kw: StructuredElmanAttention(attn_period=4, **kw),  # E22-K4: attend every 4 steps
        '22k16': lambda **kw: StructuredElmanAttention(attn_period=16, **kw),  # E22-K16: attend every 16 steps
        23: DualMemoryElman,  # E23: Dual-memory (tape + working memory)
        '23n32': lambda **kw: DualMemoryElman(n_slots=32, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        '23n128': lambda **kw: DualMemoryElman(n_slots=128, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        24: E24Layer,  # E24: True single-GEMM dual memory
        '24n32': lambda **kw: E24Layer(n_slots=32, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        '24n128': lambda **kw: E24Layer(n_slots=128, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        25: E25DualMemoryElman,  # E25: Dual memory with 1.5-entmax attention
        '25n32': lambda **kw: E25DualMemoryElman(n_slots=32, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        '25n128': lambda **kw: E25DualMemoryElman(n_slots=128, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        26: E26DualMemoryElman,  # E26: Parallel dual memory
        '26n32': lambda **kw: E26DualMemoryElman(n_slots=32, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        '26n128': lambda **kw: E26DualMemoryElman(n_slots=128, **{k: v for k, v in kw.items() if k != 'n_slots'}),
        28: E28ConvElman,  # E28: E1 + Mamba2 causal conv
        'mamba2': 'mamba2',  # Special case - handled separately
    }
    if level in levels:
        return levels[level]
    raise ValueError(f"Invalid level {level}. Available: 0-6, 8-17, 18a/b/e, 19a/b/d/e, 20-26, 28, mamba2")


class LadderLM(nn.Module):
    """
    Language Model using Elman Ablation Ladder levels.

    Uses Mamba-style architecture with pre-norm + residual connections.

    Args:
        vocab_size: Size of vocabulary (256 for byte-level)
        dim: Model dimension
        depth: Number of layers
        level: Ablation ladder level (0-3)
        expansion: Hidden state expansion factor
        n_groups: Number of groups for compete softmax (levels 2+)
        delta_init: Initial delta gate bias
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size=256,
        dim=512,
        depth=12,
        level=0,
        expansion=1.0,
        n_groups=32,
        n_slots=8,
        n_banks=4,  # For E10 multi-scale: number of EMA memory banks
        rank=None,
        delta_init=-2.0,
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=0.1,
        core_ratio=0.125,  # For E9 hybrid: fraction of d_inner for dense core
        mamba2_init=False,  # Use Mamba2-style initialization
        state_expansion=2,  # For E16: d_state = d_inner * state_expansion
        use_conv=False,  # Conv1d hurts E-series (nonlinear RNN doesn't need it)
        d_conv=4,  # Conv kernel size (if enabled)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.level = level
        self.n_slots = n_slots
        self.n_banks = n_banks
        self.rank = rank
        self.r_h_mode = r_h_mode
        self.use_conv = use_conv
        self.d_conv = d_conv

        # Get the layer class for this level
        LayerClass = get_ladder_level(level)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Use fused RMSNorm from mamba_ssm (matches Mamba architecture exactly)
        self.fused_add_norm = FUSED_NORM_AVAILABLE
        self.residual_in_fp32 = True  # Keep residual in fp32 for stability

        # Pre-normalization layers (one per recurrent layer) - RMSNorm like Mamba
        self.layer_norms = nn.ModuleList([
            RMSNorm(dim) for _ in range(depth)
        ])

        # Stack of recurrent layers
        self.layers = nn.ModuleList([
            LayerClass(
                dim=dim,
                expansion=expansion,
                n_groups=n_groups,
                n_slots=n_slots,
                n_banks=n_banks,  # For E10 multi-scale
                rank=rank,
                delta_init=delta_init,
                dropout=dropout,
                r_h_mode=r_h_mode,
                r_h_init_gain=r_h_init_gain,
                core_ratio=core_ratio,  # For E9 hybrid
                mamba2_init=mamba2_init,  # Mamba2-style initialization
                state_expansion=state_expansion,  # For E16
                use_conv=use_conv,  # Conv1d before recurrence (like Mamba2)
                d_conv=d_conv,  # Conv kernel size
            )
            for _ in range(depth)
        ])

        # Final layer norm before output - RMSNorm like Mamba
        self.norm = RMSNorm(dim)

        # Output projection to vocabulary (tied with embedding)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        x,
        return_loss=False,
        return_prev_hiddens=False,
        prev_hiddens=None,
        prev_conv_buffers=None,
        actual_length=None,
        doc_boundaries=None,
    ):
        """
        Forward pass compatible with train.py interface.

        Args:
            x: [B, T] input token indices
            return_loss: If True, compute loss (x is [B, T+1] with targets)
            return_prev_hiddens: If True, return hidden states for TBPTT
            prev_hiddens: List of [B, d_inner] per layer, or None
            prev_conv_buffers: Unused, for API compatibility
            actual_length: For masking padded chunks
            doc_boundaries: [B, T] boolean tensor for hidden state reset

        Returns:
            If return_loss: (loss, new_hiddens) or loss
            Else: (logits, new_hiddens) or logits
        """
        if return_loss:
            # x is [B, T+1], split into input and target
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        B, T = inp.shape

        # Embed tokens
        x = self.embedding(inp)  # [B, T, dim]

        # Initialize hidden states if not provided
        if prev_hiddens is None:
            prev_hiddens = [None] * self.depth

        new_hidden_states = []

        # Run through layers with fused add+norm (exactly like Mamba's Block)
        # Pattern: residual = x + residual; x = norm(residual); x = mixer(x)
        residual = None
        for i, (ln, layer) in enumerate(zip(self.layer_norms, self.layers)):
            if self.fused_add_norm:
                # Fused add + RMSNorm (like Mamba)
                x, residual = rms_norm_fn(
                    x,
                    ln.weight,
                    None,  # bias
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=ln.eps,
                )
            else:
                # Non-fused fallback
                residual = (x + residual) if residual is not None else x
                x = ln(residual.to(dtype=ln.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

            # Elman layer forward
            x, h_final = layer(x, prev_hiddens[i])

            new_hidden_states.append(h_final)

        # Final fused add + norm
        if self.fused_add_norm:
            # prenorm=False returns just the normalized output (not a tuple)
            x = rms_norm_fn(
                x,
                self.norm.weight,
                None,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        else:
            x = self.norm((x + residual).to(dtype=self.norm.weight.dtype))
        logits = self.lm_head(x)

        if return_loss:
            # Mask out padded positions if actual_length is provided
            if actual_length is not None:
                device = logits.device
                # Create mask: valid positions are 0 to actual_length-2 (shifted by 1 for targets)
                positions = torch.arange(target.size(1), device=device).unsqueeze(0)
                valid_mask = positions < (actual_length.unsqueeze(1) - 1)
                target = target.clone()
                target[~valid_mask] = -100

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target.reshape(-1),
                ignore_index=-100,
            )
            if return_prev_hiddens:
                # Return (loss, (hidden_states, conv_buffers)) format expected by train.py
                return loss, (new_hidden_states, None)
            return loss

        if return_prev_hiddens:
            return logits, (new_hidden_states, None)
        return logits

    def get_num_params(self):
        """Count parameters."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        level_names = {
            0: "Stock Elman (e0)",
            1: "Mamba-Gated Elman (e1)",
            2: "Slot Elman (e2)",
            3: "Low-Rank Slot Elman (e3)",
        }
        return f'level={self.level} ({level_names.get(self.level, "Unknown")}), dim={self.dim}, depth={self.depth}'


def create_ladder_model(
    target_params: str = "100m",
    level: int = 0,
    vocab_size: int = 256,
    expansion: float = 1.0,
    n_groups: int = 32,
    n_slots: int = 8,
    r_h_mode: str = 'spectral_norm',
    r_h_init_gain: float = 0.1,
    state_expansion: int = 2,
    mamba2_init: bool = False,
    use_conv: bool = False,  # Conv1d hurts E-series (nonlinear RNN doesn't need it)
    d_conv: int = 4,  # Conv kernel size (if enabled)
):
    """
    Create a LadderLM with approximately target_params parameters.

    Uses dynamic parameter counting: creates 1-layer and 2-layer models to
    compute exact params_per_layer, then determines depth to reach target.

    Args:
        target_params: Target parameter count (e.g., "100m", "500m", "1b")
        level: Ablation ladder level (0-3) or 'mamba2'
        vocab_size: Vocabulary size
        expansion: Hidden state expansion
        n_groups: Number of groups for compete softmax
        n_slots: Number of slots for E2/E3 (default: 8)
        r_h_mode: Constraint mode for R_h matrix (for log-space levels)
        r_h_init_gain: Initial gain for R_h orthogonal initialization
        state_expansion: For E16: d_state = d_inner * state_expansion

    Returns:
        LadderLM or Mamba2LM model
    """
    # Handle mamba2 specially
    if level == 'mamba2':
        from .mamba2_baseline import create_mamba2_model
        return create_mamba2_model(target_params=target_params, vocab_size=vocab_size)

    # Parse target
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    # Dimension configs based on target size (expansion can vary per level)
    # Format: target_params -> (dim, default_expansion)
    dim_configs = {
        50_000_000: (512, 1.5),
        100_000_000: (768, 1.5),
        200_000_000: (1024, 1.5),
        350_000_000: (1024, 2.0),
        500_000_000: (1024, 2.5),
        700_000_000: (1280, 2.0),
        1_000_000_000: (1536, 2.0),
        1_300_000_000: (1792, 2.0),
    }

    # Find closest dim config
    closest = min(dim_configs.keys(), key=lambda x: abs(x - target_count))
    dim, default_expansion = dim_configs[closest]

    # Use provided expansion or default from config
    if expansion == 1.0:
        expansion = default_expansion

    # Create a 1-layer model to count base params (embeddings, output, etc)
    model_1layer = LadderLM(
        vocab_size=vocab_size, dim=dim, depth=1, level=level,
        expansion=expansion, n_groups=n_groups, n_slots=n_slots,
        r_h_mode=r_h_mode, r_h_init_gain=r_h_init_gain,
        state_expansion=state_expansion, mamba2_init=mamba2_init,
        use_conv=use_conv, d_conv=d_conv,
    )
    params_1layer = model_1layer.get_num_params()

    # Create a 2-layer model to compute params per layer
    model_2layer = LadderLM(
        vocab_size=vocab_size, dim=dim, depth=2, level=level,
        expansion=expansion, n_groups=n_groups, n_slots=n_slots,
        r_h_mode=r_h_mode, r_h_init_gain=r_h_init_gain,
        state_expansion=state_expansion, mamba2_init=mamba2_init,
        use_conv=use_conv, d_conv=d_conv,
    )
    params_2layer = model_2layer.get_num_params()

    # Compute params per layer
    params_per_layer = params_2layer - params_1layer
    base_params = params_1layer - params_per_layer  # embedding + output

    # Clean up probe models
    del model_1layer, model_2layer

    # Calculate depth needed to reach target
    if params_per_layer > 0:
        depth = max(1, round((target_count - base_params) / params_per_layer))
    else:
        depth = 12  # fallback

    # Ensure reasonable depth bounds
    depth = max(4, min(depth, 48))

    # Create the actual model
    model = LadderLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        level=level,
        expansion=expansion,
        n_groups=n_groups,
        n_slots=n_slots,
        r_h_mode=r_h_mode,
        r_h_init_gain=r_h_init_gain,
        state_expansion=state_expansion,
        mamba2_init=mamba2_init,
        use_conv=use_conv,
        d_conv=d_conv,
    )

    actual_params = model.get_num_params()
    r_h_info = f", r_h_mode={r_h_mode}" if str(level).startswith('log') else ""
    print(f"Created Level {level} model: dim={dim}, depth={depth}, params={actual_params:,}{r_h_info}")

    return model


if __name__ == "__main__":
    print("Testing LadderLM...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for level in range(4):
        print(f"\nLevel {level}:")
        model = LadderLM(
            vocab_size=256,
            dim=256,
            depth=4,
            level=level,
            expansion=1.0,
        ).to(device).bfloat16()

        x = torch.randint(0, 256, (2, 32), device=device)
        logits, hidden = model(x, return_prev_hiddens=True)
        loss = F.cross_entropy(logits.view(-1, 256), x.view(-1))
        loss.backward()

        print(f"  Params: {model.get_num_params():,}")
        print(f"  Logits: {logits.shape}, Loss: {loss.item():.4f}")

    print("\nLadderLM tests passed!")
