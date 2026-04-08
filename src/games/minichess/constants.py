"""Los Alamos mini chess (6x6) — constants, Zobrist tables, and board helpers.

Square index layout (row-major, rank-major):
  row 5 (rank 6): squares 30-35   (a6 .. f6)
  row 4 (rank 5): squares 24-29
  row 3 (rank 4): squares 18-23
  row 2 (rank 3): squares 12-17
  row 1 (rank 2): squares  6-11
  row 0 (rank 1): squares  0- 5   (a1 .. f1)

Column (file) index: sq % 6  (0=a .. 5=f)
Row  (rank) index:   sq // 6 (0=rank1 .. 5=rank6)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..base import BLACK, WHITE, GameConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 6
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE  # 36

# Piece type values stored in the board array (unsigned; sign encodes colour)
EMPTY = 0
PAWN = 1
KNIGHT = 2
ROOK = 3
QUEEN = 4
KING = 5

# Piece-type indices used in pieces_on_board() and Zobrist tables.
# King is excluded here (it is returned separately via king_square()).
#   PAWN_IDX=0, KNIGHT_IDX=1, ROOK_IDX=2, QUEEN_IDX=3
PAWN_IDX = 0
KNIGHT_IDX = 1
ROOK_IDX = 2
QUEEN_IDX = 3

# Map from board-array piece value to pieces_on_board index
_PIECE_TO_IDX: dict[int, int] = {
    PAWN: PAWN_IDX,
    KNIGHT: KNIGHT_IDX,
    ROOK: ROOK_IDX,
    QUEEN: QUEEN_IDX,
}

# Number of non-king piece types per side
NUM_PIECE_TYPES = 4

# 50-move rule: draw after 50 moves without pawn advance or capture
FIFTY_MOVE_LIMIT = 100  # half-moves (plies)

# ---------------------------------------------------------------------------
# Game configuration (singleton)
# ---------------------------------------------------------------------------

_CONFIG = GameConfig(
    name="minichess",
    board_height=BOARD_SIZE,
    board_width=BOARD_SIZE,
    num_piece_types=NUM_PIECE_TYPES,
    has_drops=False,
    has_promotion=True,
)

# ---------------------------------------------------------------------------
# Zobrist hashing tables (fixed seed for reproducibility)
# ---------------------------------------------------------------------------

def _build_zobrist_tables() -> Tuple[np.ndarray, np.ndarray]:
    """Build Zobrist random number tables with a fixed seed.

    Returns:
        piece_table: shape (2, 5, 36) -- [colour, piece_type_idx 0-4, square]
                     Index 4 is used for the King (piece value 5).
        side_table:  scalar uint64 XORed in when it is Black's turn.
    """
    rng = np.random.default_rng(seed=0xDEADBEEF_CAFEBABE)
    # We store 5 piece-type slots (indices 0-4), where index i corresponds to
    # board piece value (i+1): Pawn=slot0, Knight=slot1, Rook=slot2,
    # Queen=slot3, King=slot4.
    piece_table = rng.integers(
        low=0, high=np.iinfo(np.uint64).max, size=(2, 5, NUM_SQUARES),
        dtype=np.uint64,
    )
    side_key = rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
    return piece_table, side_key


_ZOBRIST_PIECES, _ZOBRIST_SIDE = _build_zobrist_tables()


def _piece_slot(piece_value: int) -> int:
    """Convert unsigned board piece value (1-5) to Zobrist slot index (0-4)."""
    return piece_value - 1


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _sq(row: int, col: int) -> int:
    """Convert (row, col) to flat square index."""
    return row * BOARD_SIZE + col


def _row(sq: int) -> int:
    return sq // BOARD_SIZE


def _col(sq: int) -> int:
    return sq % BOARD_SIZE


def _on_board(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


# ---------------------------------------------------------------------------
# Direction constants
# ---------------------------------------------------------------------------

# Knight move offsets (delta_row, delta_col)
_KNIGHT_DELTAS: List[Tuple[int, int]] = [
    (-2, -1), (-2, +1), (-1, -2), (-1, +2),
    (+1, -2), (+1, +2), (+2, -1), (+2, +1),
]

# Rook / Queen sliding directions
_ROOK_DIRS: List[Tuple[int, int]] = [
    (-1, 0), (+1, 0), (0, -1), (0, +1),
]

# Queen also slides diagonally (but note: no Bishop in this variant)
_QUEEN_DIRS: List[Tuple[int, int]] = _ROOK_DIRS + [
    (-1, -1), (-1, +1), (+1, -1), (+1, +1),
]

# King moves one step in any direction
_KING_DELTAS: List[Tuple[int, int]] = _QUEEN_DIRS  # same offsets, 1 step only


# ---------------------------------------------------------------------------
# Initial position
# ---------------------------------------------------------------------------

def _build_initial_board() -> np.ndarray:
    """Return the initial 36-element board array for Los Alamos chess.

    White rank 1 (row 0): R N Q K N R
    White rank 2 (row 1): 6 pawns
    Black rank 6 (row 5): r n q k n r   (stored as negative values)
    Black rank 5 (row 4): 6 pawns       (stored as negative values)
    """
    board = np.zeros(NUM_SQUARES, dtype=np.int8)

    # White back rank (row 0, squares 0-5): R N Q K N R
    back_rank = [ROOK, KNIGHT, QUEEN, KING, KNIGHT, ROOK]
    for col, piece in enumerate(back_rank):
        board[_sq(0, col)] = piece

    # White pawns (row 1, squares 6-11)
    for col in range(BOARD_SIZE):
        board[_sq(1, col)] = PAWN

    # Black back rank (row 5, squares 30-35): r n q k n r (mirror columns)
    for col, piece in enumerate(back_rank):
        board[_sq(5, col)] = -piece

    # Black pawns (row 4, squares 24-29)
    for col in range(BOARD_SIZE):
        board[_sq(4, col)] = -PAWN

    return board
