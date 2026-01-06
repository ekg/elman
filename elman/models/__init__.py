"""
Elman Models - E-Series for Mamba2 comparison

E-Series:
    e0: Stock Elman - tanh recurrence + h*silu(W_gate@x) gating
    e1: Mamba-Gated Elman - Mamba2-style split projection gating
    e2: Slot Elman - Multi-slot memory (64x more capacity like Mamba2)

All support:
    - Optional conv1d for local context
    - Mamba2-style silu gating

Archived experimental levels are in elman/models/archive/.

Usage:
    from elman.models import StockElman, MambaGatedElman, SlotElman
    from elman.models import LadderLM, create_ladder_model

    # Create an e0 model
    model = create_ladder_model("500m", level=0)

    # Create an e1 model
    model = create_ladder_model("500m", level=1)

    # Create an e2 model (64 memory slots)
    model = create_ladder_model("500m", level=2)
"""

# E0: Stock Elman (base of e-series)
from .stock_elman import StockElman, StockElmanCell, LEVEL_0_AVAILABLE

# E1: Mamba-Gated Elman (Mamba2-style split projection)
from .mamba_gated_elman import MambaGatedElman, MambaGatedElmanCell
LEVEL_1_AVAILABLE = True

# E2: Slot Elman (Multi-slot memory like Mamba2)
from .slot_elman import SlotElman, SlotElmanCell
LEVEL_2_AVAILABLE = True

# E3: Low-Rank Slot Elman (independent low-rank W_h per slot)
from .lowrank_slot_elman import LowRankSlotElman, LowRankSlotElmanCell
LEVEL_3_AVAILABLE = True

# E4: Low-Rank Elman (SVD-style for fat hidden state)
from .lowrank_elman import LowRankElman, LowRankElmanCell
LEVEL_4_AVAILABLE = True

# E5: Pure Low-Rank Elman (no projections, all low-rank on full dim)
from .pure_lowrank_elman import PureLowRankElman, PureLowRankElmanCell
LEVEL_5_AVAILABLE = True

# E6: Diagonal Elman (per-channel scalar recurrence + low-rank mix)
from .diagonal_elman import DiagonalElman, DiagonalElmanCell
LEVEL_6_AVAILABLE = True

# Language model wrapper
from .ladder_lm import LadderLM, create_ladder_model

# Mamba2 baseline for comparison
try:
    from .mamba2_baseline import Mamba2LM, create_mamba2_model, MAMBA2_AVAILABLE
except ImportError:
    Mamba2LM = None
    create_mamba2_model = None
    MAMBA2_AVAILABLE = False


def get_available_levels():
    """Return dict of available ladder levels."""
    levels = {
        0: ("Stock Elman (e0)", LEVEL_0_AVAILABLE, StockElman),
        1: ("Mamba-Gated Elman (e1)", LEVEL_1_AVAILABLE, MambaGatedElman),
        2: ("Slot Elman (e2)", LEVEL_2_AVAILABLE, SlotElman),
        3: ("Low-Rank Slot Elman (e3)", LEVEL_3_AVAILABLE, LowRankSlotElman),
        4: ("Low-Rank Elman (e4)", LEVEL_4_AVAILABLE, LowRankElman),
        5: ("Pure Low-Rank Elman (e5)", LEVEL_5_AVAILABLE, PureLowRankElman),
        6: ("Diagonal Elman (e6)", LEVEL_6_AVAILABLE, DiagonalElman),
    }
    return levels


def get_ladder_level(level):
    """Get the module class for a specific ladder level."""
    levels = get_available_levels()
    if level not in levels:
        raise ValueError(f"Invalid level {level}. Available: {list(levels.keys())}")
    name, available, cls = levels[level]
    if not available:
        raise ImportError(f"Level {level} ({name}) is not available.")
    return cls


__all__ = [
    # E0: Stock Elman
    'StockElman', 'StockElmanCell', 'LEVEL_0_AVAILABLE',
    # E1: Mamba-Gated Elman
    'MambaGatedElman', 'MambaGatedElmanCell', 'LEVEL_1_AVAILABLE',
    # E2: Slot Elman
    'SlotElman', 'SlotElmanCell', 'LEVEL_2_AVAILABLE',
    # E3: Low-Rank Slot Elman
    'LowRankSlotElman', 'LowRankSlotElmanCell', 'LEVEL_3_AVAILABLE',
    # E4: Low-Rank Elman
    'LowRankElman', 'LowRankElmanCell', 'LEVEL_4_AVAILABLE',
    # E5: Pure Low-Rank Elman
    'PureLowRankElman', 'PureLowRankElmanCell', 'LEVEL_5_AVAILABLE',
    # E6: Diagonal Elman
    'DiagonalElman', 'DiagonalElmanCell', 'LEVEL_6_AVAILABLE',
    # Language model wrapper
    'LadderLM', 'create_ladder_model',
    # Mamba2 baseline
    'Mamba2LM', 'create_mamba2_model', 'MAMBA2_AVAILABLE',
    # Helpers
    'get_available_levels', 'get_ladder_level',
]
