"""
Elman Ablation Ladder - Linear-Space Levels (0-6) and Log-Space Levels (log_0 to log_6)

This module implements the systematic ablation from stock Elman to more
complex architectures, including true log-space implementations.

Linear-Space Ablation Ladder:
    Level 0: Stock Elman - Basic tanh recurrence
    Level 1: Gated Elman - + Input-dependent delta gate
    Level 2: Selective Elman - + compete x silu output
    Level 3: Diagonal Selective - Diagonal r_h (like Mamba2's diagonal A)
    Level 4: Full Recurrence - Full R_h matrix (more expressive)
    Level 5: Linear Triple R - R_h, R_x, R_delta matrices
    Level 6: Linear Polynomial - Input-dependent polynomial activation

Log-Space Ablation Ladder:
    log_0: Log-Space Polynomial - Input-dependent α, polynomial activation
    log_1: Log-Space Selective - + compete x silu output
    log_2: Log-Space Diagonal Selective - Diagonal r_h in log space
    log_3: Log-Space Diagonal (Full) - Full log-space with diagonal r_h
    log_4: Log-Compute Full - Full R_h matrix with logsumexp matmul
    log_5: Log-Space Triple R - R_h, R_x, R_delta (full matrices)
    log_6: Log-Space Diagonal Triple R - Diagonal r_h + r_delta (efficient)

Usage:
    from elman.models import StockElman, GatedElman, SelectiveElman, DiagonalSelective
    from elman.models import FullRecurrence, LinearTripleR, LinearPolynomial
    from elman.models import LogSpacePolynomial, LogSpaceTripleR
    from elman.models import LadderLM, create_ladder_model

    # Create a model at a specific level
    model = create_ladder_model("100m", level='log_2')
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

try:
    from .full_recurrence import FullRecurrence, FullRecurrenceCell, LEVEL_4_AVAILABLE
except ImportError:
    FullRecurrence = None
    FullRecurrenceCell = None
    LEVEL_4_AVAILABLE = False

try:
    from .linear_triple_r import LinearTripleR, LinearTripleRCell, LEVEL_5_AVAILABLE
except ImportError:
    LinearTripleR = None
    LinearTripleRCell = None
    LEVEL_5_AVAILABLE = False

try:
    from .linear_polynomial import LinearPolynomial, LinearPolynomialCell, LEVEL_6_AVAILABLE
except ImportError:
    LinearPolynomial = None
    LinearPolynomialCell = None
    LEVEL_6_AVAILABLE = False

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

try:
    from .log_storage_diagonal import (
        LogStorageDiagonal, LogStorageDiagonalCell, LOG_STORAGE_DIAGONAL_AVAILABLE
    )
except ImportError:
    LogStorageDiagonal = None
    LogStorageDiagonalCell = None
    LOG_STORAGE_DIAGONAL_AVAILABLE = False

try:
    from .log_compute_full import (
        LogComputeFull, LogComputeFullCell, LOG_COMPUTE_FULL_AVAILABLE
    )
except ImportError:
    LogComputeFull = None
    LogComputeFullCell = None
    LOG_COMPUTE_FULL_AVAILABLE = False

try:
    from .log_space_triple_r import (
        LogSpaceTripleR, LogSpaceTripleRCell, LOG_SPACE_TRIPLE_R_AVAILABLE
    )
except ImportError:
    LogSpaceTripleR = None
    LogSpaceTripleRCell = None
    LOG_SPACE_TRIPLE_R_AVAILABLE = False

try:
    from .logspace_diagonal_triple_r import (
        LogSpaceDiagTripleR, LogSpaceDiagTripleRCell, LOGSPACE_LEVEL_6_AVAILABLE
    )
except ImportError:
    LogSpaceDiagTripleR = None
    LogSpaceDiagTripleRCell = None
    LOGSPACE_LEVEL_6_AVAILABLE = False

# Level 7: Linear-space Diagonal Triple R
try:
    from .diagonal_triple_r import (
        DiagTripleR, DiagTripleRCell, LEVEL_7_AVAILABLE
    )
except ImportError:
    DiagTripleR = None
    DiagTripleRCell = None
    LEVEL_7_AVAILABLE = False

# Language model wrapper
try:
    from .ladder_lm import LadderLM, create_ladder_model
except ImportError:
    LadderLM = None
    create_ladder_model = None

# Baseline models for comparison
try:
    from .gru_baseline import GRULM, create_gru_model
    GRU_BASELINE_AVAILABLE = True
except ImportError:
    GRULM = None
    create_gru_model = None
    GRU_BASELINE_AVAILABLE = False

try:
    from .lstm_baseline import LSTMLM, create_lstm_model
    LSTM_BASELINE_AVAILABLE = True
except ImportError:
    LSTMLM = None
    create_lstm_model = None
    LSTM_BASELINE_AVAILABLE = False

try:
    from .mamba2_baseline import Mamba2LM, create_mamba2_model, MAMBA2_AVAILABLE
except ImportError:
    Mamba2LM = None
    create_mamba2_model = None
    MAMBA2_AVAILABLE = False


def get_available_levels():
    """Return dict of available ladder levels."""
    levels = {
        # Linear-space levels (0-7)
        0: ("Stock Elman", LEVEL_0_AVAILABLE, StockElman),
        1: ("Gated Elman", LEVEL_1_AVAILABLE, GatedElman),
        2: ("Selective Elman", LEVEL_2_AVAILABLE, SelectiveElman),
        3: ("Diagonal Selective", LEVEL_3_AVAILABLE, DiagonalSelective),
        4: ("Full Recurrence", LEVEL_4_AVAILABLE, FullRecurrence),
        5: ("Linear Triple R", LEVEL_5_AVAILABLE, LinearTripleR),
        6: ("Linear Polynomial", LEVEL_6_AVAILABLE, LinearPolynomial),
        7: ("Diagonal Triple R", LEVEL_7_AVAILABLE, DiagTripleR),
        # Log-space levels (log_0 to log_6)
        'log_0': ("Log-Space Polynomial", LOGSPACE_LEVEL_0_AVAILABLE, LogSpacePolynomial),
        'log_1': ("Log-Space Selective", LOGSPACE_LEVEL_1_AVAILABLE, LogSpaceSelective),
        'log_2': ("Log-Space Diagonal Selective", LOGSPACE_LEVEL_2_AVAILABLE, LogSpaceDiagonalSelective),
        'log_3': ("Log-Space Diagonal (Full)", LOG_STORAGE_DIAGONAL_AVAILABLE, LogStorageDiagonal),
        'log_4': ("Log-Compute Full", LOG_COMPUTE_FULL_AVAILABLE, LogComputeFull),
        'log_5': ("Log-Space Triple R", LOG_SPACE_TRIPLE_R_AVAILABLE, LogSpaceTripleR),
        'log_6': ("Log-Space Diagonal Triple R", LOGSPACE_LEVEL_6_AVAILABLE, LogSpaceDiagTripleR),
    }
    return levels


def get_ladder_level(level):
    """Get the module class for a specific ladder level."""
    levels = get_available_levels()
    if level not in levels:
        raise ValueError(f"Invalid level {level}. Must be 0-6 or log_0 to log_6.")
    name, available, cls = levels[level]
    if not available:
        raise ImportError(f"Level {level} ({name}) is not available.")
    return cls


__all__ = [
    # Linear-space level classes (0-7)
    'StockElman', 'StockElmanCell',
    'GatedElman', 'GatedElmanCell',
    'SelectiveElman', 'SelectiveElmanCell',
    'DiagonalSelective', 'DiagonalSelectiveCell',
    'FullRecurrence', 'FullRecurrenceCell',
    'LinearTripleR', 'LinearTripleRCell',
    'LinearPolynomial', 'LinearPolynomialCell',
    'DiagTripleR', 'DiagTripleRCell',
    # Log-space level classes (log_0 to log_6)
    'LogSpacePolynomial', 'LogSpacePolynomialCell',
    'LogSpaceSelective', 'LogSpaceSelectiveCell',
    'LogSpaceDiagonalSelective', 'LogSpaceDiagonalSelectiveCell',
    'LogStorageDiagonal', 'LogStorageDiagonalCell',
    'LogComputeFull', 'LogComputeFullCell',
    'LogSpaceTripleR', 'LogSpaceTripleRCell',
    'LogSpaceDiagTripleR', 'LogSpaceDiagTripleRCell',
    # Baseline models
    'GRULM', 'create_gru_model', 'GRU_BASELINE_AVAILABLE',
    'LSTMLM', 'create_lstm_model', 'LSTM_BASELINE_AVAILABLE',
    'Mamba2LM', 'create_mamba2_model', 'MAMBA2_AVAILABLE',
    # Availability flags
    'LEVEL_0_AVAILABLE', 'LEVEL_1_AVAILABLE', 'LEVEL_2_AVAILABLE', 'LEVEL_3_AVAILABLE',
    'LEVEL_4_AVAILABLE', 'LEVEL_5_AVAILABLE', 'LEVEL_6_AVAILABLE', 'LEVEL_7_AVAILABLE',
    'LOGSPACE_LEVEL_0_AVAILABLE', 'LOGSPACE_LEVEL_1_AVAILABLE', 'LOGSPACE_LEVEL_2_AVAILABLE',
    'LOG_STORAGE_DIAGONAL_AVAILABLE', 'LOG_COMPUTE_FULL_AVAILABLE', 'LOG_SPACE_TRIPLE_R_AVAILABLE',
    'LOGSPACE_LEVEL_6_AVAILABLE',
    # Language model
    'LadderLM', 'create_ladder_model',
    # Helpers
    'get_available_levels', 'get_ladder_level',
]
