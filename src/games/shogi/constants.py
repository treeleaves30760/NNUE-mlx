"""Constants, Zobrist tables, board geometry helpers, direction sets, and
low-level movement primitives for Shogi.

Board layout (9x9):
  - Index 0 = square 1a (top-left from sente's view = gote's back rank, file 9)
  - Index 80 = square 9i (bottom-right from sente's view = sente's back rank, file 1)
  - sq = rank * 9 + file_index   (rank 0..8, file_index 0..8)
  - From sente's perspective: rank 0 is gote's side, rank 8 is sente's side

Piece encoding (sign encodes colour):
  positive = sente (WHITE=0), negative = gote (BLACK=1)
  0=empty
  1=Pawn  2=Lance  3=Knight  4=Silver  5=Gold  6=Bishop  7=Rook  8=King
  9=+Pawn(Tokin)  10=+Lance  11=+Knight  12=+Silver  13=+Bishop(Horse)  14=+Rook(Dragon)

Piece-type constants for the public API (0-indexed, king excluded):
  PAWN=0 LANCE=1 KNIGHT=2 SILVER=3 GOLD=4 BISHOP=5 ROOK=6
  PROMOTED_PAWN=7 PROMOTED_LANCE=8 PROMOTED_KNIGHT=9 PROMOTED_SILVER=10
  PROMOTED_BISHOP=11 PROMOTED_ROOK=12
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.games.base import BLACK, WHITE, GameConfig

# ---------------------------------------------------------------------------
# Public piece-type constants (API surface, king excluded)
# ---------------------------------------------------------------------------
PAWN = 0
LANCE = 1
KNIGHT = 2
SILVER = 3
GOLD = 4
BISHOP = 5
ROOK = 6
PROMOTED_PAWN = 7
PROMOTED_LANCE = 8
PROMOTED_KNIGHT = 9
PROMOTED_SILVER = 10
PROMOTED_BISHOP = 11
PROMOTED_ROOK = 12

# Mapping from API piece-type constant to the internal board encoding value
# (always positive; multiply by -1 for gote)
_API_TO_BOARD: List[int] = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
# Inverse: board value (1..14) -> API piece-type constant
_BOARD_TO_API: List[int] = [
    -1,  # 0 = empty (unused)
    PAWN,           # 1
    LANCE,          # 2
    KNIGHT,         # 3
    SILVER,         # 4
    GOLD,           # 5
    BISHOP,         # 6
    ROOK,           # 7
    -1,             # 8 = King (not in piece-type list)
    PROMOTED_PAWN,  # 9
    PROMOTED_LANCE, # 10
    PROMOTED_KNIGHT,# 11
    PROMOTED_SILVER,# 12
    PROMOTED_BISHOP,# 13
    PROMOTED_ROOK,  # 14
]

# Board values for hand pieces (unpromoted, no king)
_HAND_BOARD_VALUES = [1, 2, 3, 4, 5, 6, 7]  # pawn..rook

# ---------------------------------------------------------------------------
# Zobrist table (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(0xDEADBEEF_CAFEBABE)

# Piece placements: board_value in 1..14, side 0..1, square 0..80
_Z_PIECE = _RNG.integers(low=0, high=2**63, size=(15, 2, 81), dtype=np.uint64)
# Hand pieces: piece_type 0..6 (pawn..rook), side 0..1, count 0..38
_Z_HAND = _RNG.integers(low=0, high=2**63, size=(7, 2, 39), dtype=np.uint64)
# Side to move
_Z_SIDE = _RNG.integers(low=0, high=2**63, size=2, dtype=np.uint64)

# ---------------------------------------------------------------------------
# Static game config
# ---------------------------------------------------------------------------
_CONFIG = GameConfig(
    name="shogi",
    board_height=9,
    board_width=9,
    num_piece_types=13,
    has_drops=True,
    has_promotion=True,
)

# ---------------------------------------------------------------------------
# Board geometry helpers
# ---------------------------------------------------------------------------

def _rank(sq: int) -> int:
    """Return rank (0=top/gote side, 8=bottom/sente side)."""
    return sq // 9

def _file(sq: int) -> int:
    """Return file index (0=left, 8=right from sente's view)."""
    return sq % 9

def _sq(rank: int, file_idx: int) -> int:
    return rank * 9 + file_idx

def _in_bounds(rank: int, file_idx: int) -> bool:
    return 0 <= rank < 9 and 0 <= file_idx < 9

# Promotion zone: the 3 ranks closest to the opponent
# For sente (WHITE=0): ranks 0, 1, 2
# For gote  (BLACK=1): ranks 6, 7, 8
def _in_promo_zone(sq: int, side: int) -> bool:
    r = _rank(sq)
    return r <= 2 if side == WHITE else r >= 6

# ---------------------------------------------------------------------------
# Direction constants
# One-step deltas for each piece (from the moving side's perspective).
# Directions are expressed as (delta_rank, delta_file) where "forward" for
# sente is rank-decreasing (toward rank 0).
# We store them for sente; for gote we negate delta_rank.
# ---------------------------------------------------------------------------
_KING_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_GOLD_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,0)]  # no diagonal-back
_SILVER_DIRS = [(-1,-1),(-1,0),(-1,1),(1,-1),(1,1)]

# Sliding directions
_ROOK_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
_BISHOP_DIRS = [(-1,-1),(-1,1),(1,-1),(1,1)]
_DRAGON_DIRS = _ROOK_DIRS + [(-1,-1),(-1,1),(1,-1),(1,1)]  # rook + one-step bishop
_HORSE_DIRS = _BISHOP_DIRS + [(-1,0),(1,0),(0,-1),(0,1)]   # bishop + one-step rook

# ---------------------------------------------------------------------------
# Low-level movement primitives (used by movegen.py and check detection)
# ---------------------------------------------------------------------------

def _apply_perspective(dr: int, df: int, side: int) -> Tuple[int, int]:
    """Flip rank direction for gote."""
    return (-dr if side == BLACK else dr, df)

def _one_step_moves(
    board: np.ndarray, sq: int, side: int, deltas: List[Tuple[int, int]],
) -> List[int]:
    """Return list of target squares reachable by one-step moves."""
    r, f = _rank(sq), _file(sq)
    result: List[int] = []
    sign = 1 if side == WHITE else -1
    for dr, df in deltas:
        pr, pf = _apply_perspective(dr, df, side)
        nr, nf = r + pr, f + pf
        if not _in_bounds(nr, nf):
            continue
        target = _sq(nr, nf)
        occupant = board[target]
        # Cannot capture own piece
        if occupant != 0 and (occupant > 0) == (sign > 0):
            continue
        result.append(target)
    return result

def _sliding_moves(
    board: np.ndarray, sq: int, side: int, dirs: List[Tuple[int, int]],
) -> List[int]:
    """Return list of target squares reachable by sliding moves."""
    r, f = _rank(sq), _file(sq)
    result: List[int] = []
    sign = 1 if side == WHITE else -1
    for dr, df in dirs:
        cr, cf = r + dr, f + df
        while _in_bounds(cr, cf):
            target = _sq(cr, cf)
            occupant = board[target]
            if occupant != 0:
                if (occupant > 0) != (sign > 0):
                    result.append(target)  # capture enemy
                break
            result.append(target)
            cr += dr
            cf += df
    return result

def _lance_moves(board: np.ndarray, sq: int, side: int) -> List[int]:
    """Lance slides straight forward (rank-decreasing for sente)."""
    r, f = _rank(sq), _file(sq)
    result: List[int] = []
    sign = 1 if side == WHITE else -1
    dr = -1 if side == WHITE else 1
    cr = r + dr
    while 0 <= cr < 9:
        target = _sq(cr, f)
        occupant = board[target]
        if occupant != 0:
            if (occupant > 0) != (sign > 0):
                result.append(target)
            break
        result.append(target)
        cr += dr
    return result

def _knight_moves(board: np.ndarray, sq: int, side: int) -> List[int]:
    """Knight jumps: 2 forward + 1 sideways."""
    r, f = _rank(sq), _file(sq)
    result: List[int] = []
    sign = 1 if side == WHITE else -1
    dr = -2 if side == WHITE else 2
    for df in (-1, 1):
        nr, nf = r + dr, f + df
        if not _in_bounds(nr, nf):
            continue
        target = _sq(nr, nf)
        occupant = board[target]
        if occupant != 0 and (occupant > 0) == (sign > 0):
            continue
        result.append(target)
    return result

def _raw_targets(board: np.ndarray, sq: int, side: int, piece_val: int) -> List[int]:
    """Return all squares the piece on sq can move to (ignoring check).

    piece_val is the absolute board encoding (1..14).
    """
    match piece_val:
        case 1:   # Pawn - one step forward
            dr = -1 if side == WHITE else 1
            r, f = _rank(sq), _file(sq)
            nr = r + dr
            if not _in_bounds(nr, f):
                return []
            target = _sq(nr, f)
            sign = 1 if side == WHITE else -1
            occupant = board[target]
            if occupant != 0 and (occupant > 0) == (sign > 0):
                return []
            return [target]
        case 2:   # Lance
            return _lance_moves(board, sq, side)
        case 3:   # Knight
            return _knight_moves(board, sq, side)
        case 4:   # Silver
            return _one_step_moves(board, sq, side, _SILVER_DIRS)
        case 5:   # Gold
            return _one_step_moves(board, sq, side, _GOLD_DIRS)
        case 6:   # Bishop
            return _sliding_moves(board, sq, side, _BISHOP_DIRS)
        case 7:   # Rook
            return _sliding_moves(board, sq, side, _ROOK_DIRS)
        case 8:   # King
            return _one_step_moves(board, sq, side, _KING_DIRS)
        case 9 | 10 | 11 | 12:  # +Pawn, +Lance, +Knight, +Silver -> move like gold
            return _one_step_moves(board, sq, side, _GOLD_DIRS)
        case 13:  # +Bishop (Horse) -> bishop slides + one-step orthogonal
            return (
                _sliding_moves(board, sq, side, _BISHOP_DIRS)
                + _one_step_moves(board, sq, side, _ROOK_DIRS)
            )
        case 14:  # +Rook (Dragon) -> rook slides + one-step diagonal
            return (
                _sliding_moves(board, sq, side, _ROOK_DIRS)
                + _one_step_moves(board, sq, side, _BISHOP_DIRS)
            )
        case _:
            return []
