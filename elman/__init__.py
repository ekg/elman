"""
Elman E-Series - Research framework for Elman RNN vs Mamba2 comparison.

E-Series:
  e0: Stock Elman - tanh recurrence + h*silu(W_gate@x) gating
  e1: Mamba2-style gating - split projection, rnn(x) * silu(z)
"""

from .models import (
    StockElman, StockElmanCell,
    LadderLM, create_ladder_model,
    get_available_levels, get_ladder_level,
)

__version__ = "0.2.0"
