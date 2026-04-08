"""Los Alamos mini chess (6x6) subpackage.

Re-exports the public API so that callers can continue to use:
    from src.games.minichess import MiniChessState, initial_state
    from src.games.minichess import PAWN, KNIGHT, ROOK, QUEEN, KING, ...
"""

from .state import MiniChessState, initial_state
from .constants import (
    BOARD_SIZE,
    EMPTY,
    FIFTY_MOVE_LIMIT,
    KING,
    KNIGHT,
    KNIGHT_IDX,
    NUM_PIECE_TYPES,
    NUM_SQUARES,
    PAWN,
    PAWN_IDX,
    QUEEN,
    QUEEN_IDX,
    ROOK,
    ROOK_IDX,
)

__all__ = [
    # Main class and factory
    "MiniChessState",
    "initial_state",
    # Board / game constants
    "BOARD_SIZE",
    "NUM_SQUARES",
    "NUM_PIECE_TYPES",
    "FIFTY_MOVE_LIMIT",
    # Piece type values
    "EMPTY",
    "PAWN",
    "KNIGHT",
    "ROOK",
    "QUEEN",
    "KING",
    # Piece index constants
    "PAWN_IDX",
    "KNIGHT_IDX",
    "ROOK_IDX",
    "QUEEN_IDX",
]
