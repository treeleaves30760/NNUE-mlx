"""Shogi (Japanese chess) engine implementing the GameState interface.

Public API is re-exported from submodules for backward compatibility with
code that previously imported from ``src.games.shogi``.
"""

from .state import ShogiState, initial_state
from .constants import (
    PAWN,
    LANCE,
    KNIGHT,
    SILVER,
    GOLD,
    BISHOP,
    ROOK,
    PROMOTED_PAWN,
    PROMOTED_LANCE,
    PROMOTED_KNIGHT,
    PROMOTED_SILVER,
    PROMOTED_BISHOP,
    PROMOTED_ROOK,
)

__all__ = [
    # State
    "ShogiState",
    "initial_state",
    # Public piece-type constants (API surface, king excluded)
    "PAWN",
    "LANCE",
    "KNIGHT",
    "SILVER",
    "GOLD",
    "BISHOP",
    "ROOK",
    "PROMOTED_PAWN",
    "PROMOTED_LANCE",
    "PROMOTED_KNIGHT",
    "PROMOTED_SILVER",
    "PROMOTED_BISHOP",
    "PROMOTED_ROOK",
]
