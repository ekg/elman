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
from .e30_diagonal_gated import E30DiagonalGated
from .e31_sparse_gated import E31SparseGated
from .e32_no_presilu import E32NoPresilu
from .e33_self_gate import E33SelfGate
from .e34_diagonal_wh import E34DiagonalWh
from .e35_cubic_gate import E35CubicGate
from .e36_linear_recurrence import E36LinearRecurrence
from .e37_tied_weights import E37TiedWeights
from .e37_tied_weights_v2 import E37TiedWeightsV2
from .e38_no_wx import E38NoWx
from .e39_no_bias import E39NoBias
from .e40_no_presilu import E40NoPresilu
from .e41_diagonal_wx import E41DiagonalWx
from .e42_linear_tied import E42LinearTied
from .e43_scalar_decay import E43ScalarDecay
from .e44_diagonal_w import E44DiagonalW
from .e45_pure_accumulation import E45PureAccumulation, E45bWithDecay
from .e46_no_in_proj import E46NoInProj
from .e48_no_projections import E48NoProjections
from .e51_no_self_gate import E51NoSelfGate
from .e52_quadratic_gate import E52QuadraticGate, E52bSignedQuadratic
from .e53_sigmoid_gate import E53SigmoidGate
from .e54_diagonal_no_proj import E54DiagonalNoProj
from .e55_scalar_no_proj import E55ScalarNoProj
from .e56_concat_elman import E56ConcatElman
from .e57_learned_radius import E57LearnedRadius
from .e58_learned_radii import E58LearnedRadii
from .e59_highway import E59Highway, E59bGatedHighway, E59cMixedHighway
from .e60_residual_nonlinear import E60ResidualNonlinear, E60bGatedResidual, E60cForgetGate
from .e61_decay_gated import E61DecayGated, E61bAdditiveDecay, E61cTiedDecay
from .e62_selective_write import E62SelectiveWrite, E62bDecaySelective, E62cTiedSelective
from .e63_nonlinear_delta import E63NonlinearDelta, E63aComplementary, E63bIndependent, E63cHDependent, E63dResidual
from .e63m_matrix_nonlinear import E63mMatrixNonlinear, E63mFull, E63mLite, E63mRNN
from .e64_additive_h import E64AdditiveH
from .e65_diagonal_h import E65DiagonalH
from .e66_lowrank_h import E66LowRankH
from .e67_h_gated import E67HGated, E67HGatedDiagonal, E67HGatedLowRank
from .e68_self_gating import E68SelfGating, E68SelfGatingStandard, E68SelfGatingInverse
from .gated_delta_net import GatedDeltaNet, GatedDeltaNetVector
from .fla_gated_delta import FLAGatedDeltaNetLayer
from .llama_baseline import LlamaLayer
from .e70_matrix_linear import E70MatrixLinear
from .e71_matrix_gated import E71MatrixGated
from .e72_matrix_selfgate import E72MatrixSelfGate, E72MatrixSelfGateStandard, E72MatrixSelfGateInverse
from .e73_matrix_nonlinear import E73MatrixNonlinear, E73MatrixColumn, E73MatrixRow, E73MatrixFull
from .e74_v2 import E74v2
from .e75_gated_delta import E75GatedDelta
from .e75_multihead import E75MultiHead
from .e76_logspace_delta import E76LogSpaceDelta
from .e77_linear_matrix import E77LinearMatrix
from .e78_projected_matrix import E78ProjectedMatrix
from .e79_coupled_matrix import E79CoupledMatrix
from .e83_circular_tower import E83CircularTower
from .e85_input_as_matrix import E85InputAsMatrixLayer
from .e86_input_matrix_delta import E86InputMatrixDeltaLayer
from .e87_sparse_block import E87SparseBlockLayer


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
        30: E30DiagonalGated,  # E30: E1 + diagonal gating (SSM-style selectivity)
        31: E31SparseGated,  # E31: E1 + sparse gating via softplus (default α=1.5)
        '31a': lambda **kw: E31SparseGated(alpha=2.0, **kw),  # E31a: relu gating (strictly sparse)
        '31b': lambda **kw: E31SparseGated(alpha=1.5, **kw),  # E31b: softplus gating (smooth sparse)
        32: E32NoPresilu,  # E32: E1 without pre-silu activation (simplification test)
        33: E33SelfGate,  # E33: E1 with self-gating: output = h * silu(h) instead of h * silu(z)
        34: E34DiagonalWh,  # E34: E33 with diagonal W_h (d vector instead of matrix)
        35: E35CubicGate,  # E35: E1 with cubic gating: output = h^3 instead of h * silu(z)
        36: E36LinearRecurrence,  # E36: Linear recurrence (no tanh!) + self-gate
        37: E37TiedWeights,  # E37: Tied weights: W_x = W_h = W, single GEMM per step
        '37v2': E37TiedWeightsV2,  # E37v2: Tied weights with batched GEMM optimization
        38: E38NoWx,  # E38: No W_x: h = tanh(x + W_h @ h_prev + b), removes input transform
        39: E39NoBias,  # E39: No bias: h = tanh(x + W_h @ h_prev), simplest recurrence
        40: E40NoPresilu,  # E40: No pre-silu: x_proj = in_proj(x), NOT silu(in_proj(x))
        41: E41DiagonalWx,  # E41: Diagonal W_x (d_x vector instead of matrix)
        42: E42LinearTied,  # E42: Linear recurrence + tied weights (E36 + E37)
        43: E43ScalarDecay,  # E43: Scalar decay (λ replaces d×d matrix)
        44: E44DiagonalW,  # E44: Diagonal W (per-dimension decay, Mamba2-style)
        45: E45PureAccumulation,  # E45: Pure accumulation (W=I, no params in recurrence)
        '45b': E45bWithDecay,  # E45b: Pure accumulation + learned scalar decay
        46: E46NoInProj,  # E46: No in_proj (recurrence on raw embeddings)
        48: E48NoProjections,  # E48: No projections at all (minimal recurrent layer)
        51: E51NoSelfGate,  # E51: No self-gate (linear output)
        52: E52QuadraticGate,  # E52: Pure quadratic gate (h²)
        '52b': E52bSignedQuadratic,  # E52b: Signed quadratic (h * |h|)
        53: E53SigmoidGate,  # E53: Sigmoid gate only (silu, not h * silu)
        54: E54DiagonalNoProj,  # E54: Diagonal W + no projections (Mamba2-style minimal)
        55: E55ScalarNoProj,  # E55: Scalar + no projections (ultimate minimal)
        56: E56ConcatElman,  # E56: Concat Elman - W @ [x;h] instead of W_x @ x + W_h @ h
        57: E57LearnedRadius,  # E57: E1 with learned spectral radius (scalar)
        58: E58LearnedRadii,  # E58: E1 with per-dimension learned radii
        59: E59Highway,  # E59: Highway Elman (residual recurrence, gradient=I)
        '59b': E59bGatedHighway,  # E59b: Gated residual highway
        '59c': E59cMixedHighway,  # E59c: Mixed residual + small recurrent
        60: E60ResidualNonlinear,  # E60: Residual nonlinear (h + tanh(Wh + Ux))
        '60b': E60bGatedResidual,  # E60b: Gated residual nonlinear
        '60c': E60cForgetGate,  # E60c: Forget-gate style (GRU-like)
        61: E61DecayGated,  # E61: Decay-gated (α·h + (1-α)·v, Mamba2-style)
        '61b': E61bAdditiveDecay,  # E61b: Additive decay (α·h + v)
        '61c': E61cTiedDecay,  # E61c: Tied decay (single-gate GRU)
        62: E62SelectiveWrite,  # E62: Selective write ((1-k)·h + k·v, DeltaNet-style)
        '62b': E62bDecaySelective,  # E62b: Decay + selective (α·(1-k)·h + k·v)
        '62c': E62cTiedSelective,  # E62c: Tied selective (GRU-style)
        63: E63NonlinearDelta,  # E63: Nonlinear delta (UTM-class! v=tanh(Wh+Ux))
        '63a': E63aComplementary,  # E63a: Complementary gates (GRU-style)
        '63b': E63bIndependent,  # E63b: Independent gates (LSTM-style)
        '63c': E63cHDependent,  # E63c: H-dependent gates (maximum expressivity)
        '63d': E63dResidual,  # E63d: Residual nonlinear (h + α*tanh(Wh+Ux))
        '63m': E63mMatrixNonlinear,  # E63m: Matrix state + nonlinear retrieval (O(d²) state)
        '63m-full': E63mFull,  # E63m-full: Full d×d matrix state
        '63m-lite': E63mLite,  # E63m-lite: Reduced-rank N×d matrix
        '63m-rnn': E63mRNN,  # E63m-rnn: + output recurrence (Delta RNN)
        64: E64AdditiveH,  # E64: Additive h-dependence v=tanh(h+Wx) - O(d) UTM
        65: E65DiagonalH,  # E65: Diagonal h-dependence v=tanh(d*h+Wx) - O(d) UTM
        66: E66LowRankH,  # E66: Low-rank h-dependence v=tanh(UVh+Wx) - O(d*r) UTM
        '66r16': lambda **kw: E66LowRankH(rank=16, **kw),  # E66 rank=16
        '66r64': lambda **kw: E66LowRankH(rank=64, **kw),  # E66 rank=64
        '66r128': lambda **kw: E66LowRankH(rank=128, **kw),  # E66 rank=128
        67: E67HGated,  # E67: H-dependent gate α=σ(Wx+d*h) - O(d) UTM
        '67d': E67HGatedDiagonal,  # E67d: Diagonal h in gate
        '67lr': E67HGatedLowRank,  # E67lr: Low-rank h in gate
        68: E68SelfGating,  # E68: Self-gating v=tanh(Wx)*σ(h) - O(d) UTM
        '68s': E68SelfGatingStandard,  # E68s: Standard self-gating
        '68i': E68SelfGatingInverse,  # E68i: Inverse (resist overwrite)
        'gdn': GatedDeltaNet,  # GatedDeltaNet: ICLR 2025 baseline (matrix state)
        'gdn-vec': GatedDeltaNetVector,  # GatedDeltaNet Vector: Simplified (vector state)
        'fla-gdn': FLAGatedDeltaNetLayer,  # FLA GatedDeltaNet: Optimized Triton kernels (ICLR 2025)
        'llama': LlamaLayer,  # Llama Transformer: attention baseline
        70: E70MatrixLinear,  # E70: Matrix Linear (E42-style) - linear accum + self-gate
        '70n32': lambda **kw: E70MatrixLinear(n_state=32, **kw),
        '70n128': lambda **kw: E70MatrixLinear(n_state=128, **kw),
        71: E71MatrixGated,  # E71: Matrix Gated (E67-style) - S affects gate
        '71n32': lambda **kw: E71MatrixGated(n_state=32, **kw),
        '71n128': lambda **kw: E71MatrixGated(n_state=128, **kw),
        72: E72MatrixSelfGate,  # E72: Matrix Self-Gate (E68-style) - S gates value
        '72s': E72MatrixSelfGateStandard,  # Standard: content enables writing
        '72i': E72MatrixSelfGateInverse,  # Inverse: content resists writing
        73: E73MatrixNonlinear,  # E73: Matrix Nonlinear (E1-style) - S inside tanh
        '73c': E73MatrixColumn,  # Column modulation
        '73r': E73MatrixRow,  # Row modulation
        '73f': E73MatrixFull,  # Full element-wise modulation
        75: E75GatedDelta,  # E75: Gated Delta (E74 delta rule + E61 forget gate)
        '75n32': lambda **kw: E75GatedDelta(**{**kw, 'n_state': 32}),
        '75n48': lambda **kw: E75GatedDelta(**{**kw, 'n_state': 48}),
        '75n64': lambda **kw: E75GatedDelta(**{**kw, 'n_state': 64}),
        '75n96': lambda **kw: E75GatedDelta(**{**kw, 'n_state': 96}),
        # E75 Multi-Head variants (H independent matrix states)
        'E75h2': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 2}),
        'E75h4': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4}),
        'E75h8': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8}),
        'E75h2n48': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 2, 'n_state': 48}),
        'E75h4n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 24}),
        'E75h8n16': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 16}),
        'E75h4n32': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 4, 'n_state': 32}),
        'E75h8n24': lambda **kw: E75MultiHead(**{**kw, 'n_heads': 8, 'n_state': 24}),
        # E76: Log-Space Gated Delta (E75 + Mamba2/FLA-GDN stability techniques)
        # Default: tanh + log_gate (nonlinear recurrence with stable params)
        76: E76LogSpaceDelta,
        '76n32': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 32}),
        '76n48': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 48}),
        '76n64': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 64}),
        '76n96': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 96}),
        # E76 configuration variants:
        # -t = tanh (nonlinear), -l = linear, -log = log_gate, -sig = sigmoid_gate
        '76-t-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'use_tanh': True, 'log_space_gate': True}),
        '76-t-sig': lambda **kw: E76LogSpaceDelta(**{**kw, 'use_tanh': True, 'log_space_gate': False}),
        '76-l-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'use_tanh': False, 'log_space_gate': True}),
        '76-l-sig': lambda **kw: E76LogSpaceDelta(**{**kw, 'use_tanh': False, 'log_space_gate': False}),
        # With n_state variants
        '76n32-t-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 32, 'use_tanh': True, 'log_space_gate': True}),
        '76n48-t-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 48, 'use_tanh': True, 'log_space_gate': True}),
        '76n64-t-log': lambda **kw: E76LogSpaceDelta(**{**kw, 'n_state': 64, 'use_tanh': True, 'log_space_gate': True}),
        # E74v2: Extended delta rule variants (CUDA kernel support)
        '74v2': E74v2,  # E74v2 base: delta update, output gate
        '74v2-delta': lambda **kw: E74v2(update_type='delta', **kw),
        '74v2-residual': lambda **kw: E74v2(update_type='residual', **kw),
        '74v2-ntm': lambda **kw: E74v2(update_type='ntm', **kw),
        '74v2-retrieved_gate': lambda **kw: E74v2(update_type='retrieved_gate', **kw),
        '74v2-ema': lambda **kw: E74v2(update_type='ema', **kw),
        '74v2-delta-input': lambda **kw: E74v2(update_type='delta', gate_type='input', **kw),
        '74v2-ema-input': lambda **kw: E74v2(update_type='ema', gate_type='input', **kw),
        # E77: Linear Matrix State (E42's linear recurrence + matrix state + fused projections)
        77: E77LinearMatrix,
        '77n32': lambda **kw: E77LinearMatrix(**{**kw, 'n_state': 32}),
        '77n48': lambda **kw: E77LinearMatrix(**{**kw, 'n_state': 48}),
        '77n64': lambda **kw: E77LinearMatrix(**{**kw, 'n_state': 64}),
        '77n96': lambda **kw: E77LinearMatrix(**{**kw, 'n_state': 96}),
        # E78: Projected Matrix State (E77 + random projection for sparse efficient state)
        78: E78ProjectedMatrix,
        '78n64s32': lambda **kw: E78ProjectedMatrix(**{**kw, 'n_effective': 64, 'n_small': 32}),
        '78n128s32': lambda **kw: E78ProjectedMatrix(**{**kw, 'n_effective': 128, 'n_small': 32}),
        '78n256s64': lambda **kw: E78ProjectedMatrix(**{**kw, 'n_effective': 256, 'n_small': 64}),
        # E79: Coupled Memory-Modulation Matrix System
        # Two coupled matrices (S content + M modulation) with mutual gating control
        79: E79CoupledMatrix,
        '79n32': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 32}),
        '79n48': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 48}),
        '79n64': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 64}),
        '79n96': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 96}),
        # E79 bias ablations
        '79nb': lambda **kw: E79CoupledMatrix(**{**kw, 'use_bias': False}),  # No bias
        '79ib': lambda **kw: E79CoupledMatrix(**{**kw, 'input_bias': True}),  # Input-dependent bias
        '79n32nb': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 32, 'use_bias': False}),
        '79n32ib': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 32, 'input_bias': True}),
        '79n48nb': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 48, 'use_bias': False}),
        '79n48ib': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 48, 'input_bias': True}),
        '79n64nb': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 64, 'use_bias': False}),
        '79n64ib': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 64, 'input_bias': True}),
        '79n96nb': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 96, 'use_bias': False}),
        '79n96ib': lambda **kw: E79CoupledMatrix(**{**kw, 'n_state': 96, 'input_bias': True}),

        # E83: Circular K-Tower (K matrices in mutual gating circle)
        # Default: K=3, n_state=32, fixed bias
        83: E83CircularTower,
        # K=2 (like E79 but circular)
        '83k2': lambda **kw: E83CircularTower(**{**kw, 'K': 2, 'n_state': 32}),
        '83k2nb': lambda **kw: E83CircularTower(**{**kw, 'K': 2, 'n_state': 32, 'use_bias': False}),
        '83k2ib': lambda **kw: E83CircularTower(**{**kw, 'K': 2, 'n_state': 32, 'input_bias': True}),
        # K=4, n_state=32 (4 heads, 4K state)
        '83k4n32': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 32}),
        '83k4n32nb': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 32, 'use_bias': False}),
        '83k4n32ib': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 32, 'input_bias': True}),
        # K=4, n_state=24 (4 heads, 2.3K state)
        '83k4n24': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 24}),
        '83k4n24nb': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 24, 'use_bias': False}),
        '83k4n24ib': lambda **kw: E83CircularTower(**{**kw, 'K': 4, 'n_state': 24, 'input_bias': True}),
        # K=8, n_state=24 (8 heads, 4.6K state)
        '83k8n24': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 24}),
        '83k8n24nb': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 24, 'use_bias': False}),
        '83k8n24ib': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 24, 'input_bias': True}),
        # K=8, n_state=16 (8 heads, 2K state)
        '83k8n16': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 16}),
        '83k8n16nb': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 16, 'use_bias': False}),
        '83k8n16ib': lambda **kw: E83CircularTower(**{**kw, 'K': 8, 'n_state': 16, 'input_bias': True}),

        # E85: Input-As-Matrix (dim = n_state^2, input IS the transformation matrix)
        # Default: n_state=32 -> dim=1024
        85: E85InputAsMatrixLayer,
        # n_state variants (dim = n_state^2)
        '85n16': lambda **kw: E85InputAsMatrixLayer(**{**kw, 'n_state': 16}),  # dim=256
        '85n24': lambda **kw: E85InputAsMatrixLayer(**{**kw, 'n_state': 24}),  # dim=576
        '85n32': lambda **kw: E85InputAsMatrixLayer(**{**kw, 'n_state': 32}),  # dim=1024
        '85n48': lambda **kw: E85InputAsMatrixLayer(**{**kw, 'n_state': 48}),  # dim=2304

        # E86: Input-as-Matrix Delta Rule (E85's input-as-matrix + E75's delta rule)
        # Default: n_state=32, n_heads=1 -> cell_dim=1024, output=32
        86: E86InputMatrixDeltaLayer,
        # n_state variants (single head)
        '86n16': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 16}),  # cell_dim=256, out=16
        '86n24': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 24}),  # cell_dim=576, out=24
        '86n32': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 32}),  # cell_dim=1024, out=32
        '86n48': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 48}),  # cell_dim=2304, out=48
        # Multi-head variants for capacity scaling
        '86h2': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_heads': 2}),  # 2 heads, cell_dim=2048, out=64
        '86h4': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_heads': 4}),  # 4 heads, cell_dim=4096, out=128
        '86h2n24': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 24, 'n_heads': 2}),  # cell_dim=1152, out=48
        '86h4n24': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 24, 'n_heads': 4}),  # cell_dim=2304, out=96
        '86h4n16': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 16, 'n_heads': 4}),  # cell_dim=1024, out=64
        '86h8n16': lambda **kw: E86InputMatrixDeltaLayer(**{**kw, 'n_state': 16, 'n_heads': 8}),  # cell_dim=2048, out=128

        # E87: Content-Gated Sparse Block Memory
        # n_blocks blocks of n_state×n_state, top_k updated per step
        '87': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 4, 'top_k': 2}),
        '87b4k2': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 4, 'top_k': 2}),
        '87b4k1': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 4, 'top_k': 1}),
        '87b8k2': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 24, 'n_blocks': 8, 'top_k': 2}),
        '87b8k4': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 24, 'n_blocks': 8, 'top_k': 4}),
        '87b4k2n48': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 48, 'n_blocks': 4, 'top_k': 2}),
        '87b6k2': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 6, 'top_k': 2}),
        '87b6k3': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 32, 'n_blocks': 6, 'top_k': 3}),
        '87b8k3': lambda **kw: E87SparseBlockLayer(**{**kw, 'n_state': 24, 'n_blocks': 8, 'top_k': 3}),

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
    raise ValueError(f"Invalid level {level}. Available: 0-6, 8-17, 18a/b/e, 19a/b/d/e, 20-26, 28, 30-68, gdn, gdn-vec, fla-gdn, llama, mamba2")


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
        n_state=64,  # For E70-E73: matrix state size (S is n_state x n_state)
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
        self.n_state = n_state
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
                n_state=n_state,  # For E70-E73 matrix state
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
