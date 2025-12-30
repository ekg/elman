"""
Elman Ablation Ladder - Linear-Space Levels (0-3) and Log-Space Levels

This module implements the systematic ablation from stock Elman to more
complex architectures, including true log-space implementations.

Linear-Space Ablation Ladder:
    Level 0: Stock Elman - Basic tanh recurrence
    Level 1: Gated Elman - + Input-dependent delta gate
    Level 2: Selective Elman - + compete x silu output
    Level 3: Diagonal Selective - Diagonal r_h (like Mamba2's diagonal A)

Log-Space Ablation Ladder:
    Log Level 0: Polynomial Elman - Input-dependent α, polynomial activation
                 Bounded gradients throughout (no vanishing from saturation!)

Usage:
    from elman.models import StockElman, GatedElman, SelectiveElman, DiagonalSelective
    from elman.models import LogSpacePolynomial
    from elman.models import LadderLM, create_ladder_model

    # Create a model at a specific level
    model = create_ladder_model("100m", level=3)
"""

# Import ladder levels
try:
    from .stock_elman import StockElman, StockElmanCell, LEVEL_0_AVAILABLE
except ImportError:
    StockElman = None
    StockElmanCell = None
    LEVEL_0_AVAILABLE = False

try:
    from .gated_elman import GatedElman, GatedElmanCell, LEVEL_1_AVAILABLE
except ImportError:
    GatedElman = None
    GatedElmanCell = None
    LEVEL_1_AVAILABLE = False

try:
    from .selective_elman import SelectiveElman, SelectiveElmanCell, LEVEL_2_AVAILABLE
except ImportError:
    SelectiveElman = None
    SelectiveElmanCell = None
    LEVEL_2_AVAILABLE = False

try:
    from .diagonal_selective import DiagonalSelective, DiagonalSelectiveCell, LEVEL_3_AVAILABLE
except ImportError:
    DiagonalSelective = None
    DiagonalSelectiveCell = None
    LEVEL_3_AVAILABLE = False

# Log-space levels
try:
    from .logspace_polynomial import (
        LogSpacePolynomial, LogSpacePolynomialCell, LOGSPACE_LEVEL_0_AVAILABLE
    )
except ImportError:
    LogSpacePolynomial = None
    LogSpacePolynomialCell = None
    LOGSPACE_LEVEL_0_AVAILABLE = False

try:
    from .logspace_selective import (
        LogSpaceSelective, LogSpaceSelectiveCell, LOGSPACE_LEVEL_1_AVAILABLE
    )
except ImportError:
    LogSpaceSelective = None
    LogSpaceSelectiveCell = None
    LOGSPACE_LEVEL_1_AVAILABLE = False

try:
    from .logspace_diagonal_selective import (
        LogSpaceDiagonalSelective, LogSpaceDiagonalSelectiveCell, LOGSPACE_LEVEL_2_AVAILABLE
    )
except ImportError:
    LogSpaceDiagonalSelective = None
    LogSpaceDiagonalSelectiveCell = None
    LOGSPACE_LEVEL_2_AVAILABLE = False

# Language model wrapper
try:
    from .ladder_lm import LadderLM, create_ladder_model
except ImportError:
    LadderLM = None
    create_ladder_model = None


def get_available_levels():
    """Return dict of available ladder levels."""
    levels = {
        # Linear-space levels
        0: ("Stock Elman", LEVEL_0_AVAILABLE, StockElman),
        1: ("Gated Elman", LEVEL_1_AVAILABLE, GatedElman),
        2: ("Selective Elman", LEVEL_2_AVAILABLE, SelectiveElman),
        3: ("Diagonal Selective", LEVEL_3_AVAILABLE, DiagonalSelective),
        # Log-space levels (prefixed with 'log_')
        'log_0': ("Log-Space Polynomial", LOGSPACE_LEVEL_0_AVAILABLE, LogSpacePolynomial),
        'log_1': ("Log-Space Selective", LOGSPACE_LEVEL_1_AVAILABLE, LogSpaceSelective),
        'log_2': ("Log-Space Diagonal Selective", LOGSPACE_LEVEL_2_AVAILABLE, LogSpaceDiagonalSelective),
    }
    return levels


def get_ladder_level(level):
    """Get the module class for a specific ladder level."""
    levels = get_available_levels()
    if level not in levels:
        raise ValueError(f"Invalid level {level}. Must be 0-3.")
    name, available, cls = levels[level]
    if not available:
        raise ImportError(f"Level {level} ({name}) is not available.")
    return cls


__all__ = [
    # Linear-space level classes
    'StockElman', 'StockElmanCell',
    'GatedElman', 'GatedElmanCell',
    'SelectiveElman', 'SelectiveElmanCell',
    'DiagonalSelective', 'DiagonalSelectiveCell',
    # Log-space level classes
    'LogSpacePolynomial', 'LogSpacePolynomialCell',
    'LogSpaceSelective', 'LogSpaceSelectiveCell',
    'LogSpaceDiagonalSelective', 'LogSpaceDiagonalSelectiveCell',
    # Availability flags
    'LEVEL_0_AVAILABLE', 'LEVEL_1_AVAILABLE', 'LEVEL_2_AVAILABLE', 'LEVEL_3_AVAILABLE',
    'LOGSPACE_LEVEL_0_AVAILABLE', 'LOGSPACE_LEVEL_1_AVAILABLE', 'LOGSPACE_LEVEL_2_AVAILABLE',
    # Language model
    'LadderLM', 'create_ladder_model',
    # Helpers
    'get_available_levels', 'get_ladder_level',
]
