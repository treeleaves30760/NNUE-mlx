"""Mini Shogi (5x5 Shogi) game engine package.

Implements the GameState interface for the 5x5 variant of Shogi known as
Gogo Shogi (5五将棋). Each side has 6 pieces: King, Rook, Bishop, Gold,
Silver, and Pawn. Captured pieces can be dropped back onto the board.
"""

from .state import MiniShogiState, initial_state
from .constants import (
    # Piece-type interface indices
    PAWN,
    SILVER,
    GOLD,
    BISHOP,
    ROOK,
    PROMOTED_PAWN,
    PROMOTED_SILVER,
    PROMOTED_BISHOP,
    PROMOTED_ROOK,
    # Board/piece numeric values
    EMPTY,
    PAWN_VAL,
    SILVER_VAL,
    GOLD_VAL,
    BISHOP_VAL,
    ROOK_VAL,
    KING_VAL,
    TOKIN_VAL,
    PRO_SILVER_VAL,
    HORSE_VAL,
    DRAGON_VAL,
    MAX_PIECE_VAL,
    # Board geometry
    BOARD_SIZE,
    NUM_SQUARES,
    NUM_PIECE_TYPES,
    HAND_PIECE_TYPES,
)

__all__ = [
    # State class and factory
    "MiniShogiState",
    "initial_state",
    # Piece-type interface indices
    "PAWN",
    "SILVER",
    "GOLD",
    "BISHOP",
    "ROOK",
    "PROMOTED_PAWN",
    "PROMOTED_SILVER",
    "PROMOTED_BISHOP",
    "PROMOTED_ROOK",
    # Board/piece numeric values
    "EMPTY",
    "PAWN_VAL",
    "SILVER_VAL",
    "GOLD_VAL",
    "BISHOP_VAL",
    "ROOK_VAL",
    "KING_VAL",
    "TOKIN_VAL",
    "PRO_SILVER_VAL",
    "HORSE_VAL",
    "DRAGON_VAL",
    "MAX_PIECE_VAL",
    # Board geometry
    "BOARD_SIZE",
    "NUM_SQUARES",
    "NUM_PIECE_TYPES",
    "HAND_PIECE_TYPES",
]
