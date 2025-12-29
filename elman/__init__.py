"""
Elman Ladder - Research framework for non-linear RNN architectures.

See README.md for full documentation.
"""

from .models import (
    StockElman, StockElmanCell,
    GatedElman, GatedElmanCell,
    SelectiveElman, SelectiveElmanCell,
    DiagonalSelective, DiagonalSelectiveCell,
    LadderLM, create_ladder_model,
    get_available_levels, get_ladder_level,
)

__version__ = "0.1.0"
