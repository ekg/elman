"""
Elman Models - E-Series for Mamba2 comparison

E-Series:
    e0: Stock Elman - tanh recurrence + h*silu(W_gate@x) gating
    e1: Mamba-Gated Elman - Mamba2-style split projection gating

Both support:
    - Spectral normalization on W_h (radius < 0.99)
    - Optional conv1d for local context

Archived experimental levels are in elman/models/archive/.

Usage:
    from elman.models import StockElman, MambaGatedElman
    from elman.models import LadderLM, create_ladder_model

    # Create an e0 model
    model = create_ladder_model("500m", level=0)

    # Create an e1 model
    model = create_ladder_model("500m", level=1)
"""

# E0: Stock Elman (base of e-series)
from .stock_elman import StockElman, StockElmanCell, LEVEL_0_AVAILABLE

# E1: Mamba-Gated Elman (Mamba2-style split projection)
from .mamba_gated_elman import MambaGatedElman, MambaGatedElmanCell
LEVEL_1_AVAILABLE = True

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
    # Language model wrapper
    'LadderLM', 'create_ladder_model',
    # Mamba2 baseline
    'Mamba2LM', 'create_mamba2_model', 'MAMBA2_AVAILABLE',
    # Helpers
    'get_available_levels', 'get_ladder_level',
]
