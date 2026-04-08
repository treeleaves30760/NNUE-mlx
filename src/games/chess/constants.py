"""Constants, Zobrist tables, pre-computed attack tables, and board geometry
helpers for the chess engine.

Board layout: flat numpy array of length 64.
  - Index 0 = a1 (rank 1, file a), index 63 = h8 (rank 8, file h).
  - square(rank, file) = rank * 8 + file  where rank/file are 0-based.

Piece encoding:
  0 = empty
  1 = Pawn, 2 = Knight, 3 = Bishop, 4 = Rook, 5 = Queen, 6 = King
  Positive = White, Negative = Black.

Piece-type constants matching the base interface:
  PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4
  (King excluded from pieces_on_board; handled by king_square.)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.games.base import GameConfig

# ---------------------------------------------------------------------------
# Piece-type constants (base-interface mapping)
# ---------------------------------------------------------------------------
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4

# Internal board encoding (always positive; sign encodes color)
_EMPTY = 0
_PAWN = 1
_KNIGHT = 2
_BISHOP = 3
_ROOK = 4
_QUEEN = 5
_KING = 6

# Map from base-interface piece_type to internal encoding
_TYPE_TO_INTERNAL = {PAWN: _PAWN, KNIGHT: _KNIGHT, BISHOP: _BISHOP, ROOK: _ROOK, QUEEN: _QUEEN}
# Map from internal encoding to base-interface piece_type (king excluded)
_INTERNAL_TO_TYPE = {_PAWN: PAWN, _KNIGHT: KNIGHT, _BISHOP: BISHOP, _ROOK: ROOK, _QUEEN: QUEEN}

# ---------------------------------------------------------------------------
# Zobrist tables  (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(seed=20240101)

# piece_zobrist[piece_internal 1..6][color 0..1][square 0..63]
_piece_zobrist: np.ndarray = _rng.integers(
    low=0, high=2**63, size=(7, 2, 64), dtype=np.uint64
)
# side-to-move: XOR in when it is Black's turn
_side_zobrist: np.uint64 = np.uint64(_rng.integers(0, 2**63, dtype=np.uint64))
# castling rights: 4 bits (WK, WQ, BK, BQ) -> 16 independent hashes
_castling_zobrist: np.ndarray = _rng.integers(0, 2**63, size=16, dtype=np.uint64)
# en-passant file (0-7); 0 = no en-passant
_ep_zobrist: np.ndarray = _rng.integers(0, 2**63, size=8, dtype=np.uint64)

# ---------------------------------------------------------------------------
# Static game config
# ---------------------------------------------------------------------------
_CHESS_CONFIG = GameConfig(
    name="chess",
    board_height=8,
    board_width=8,
    num_piece_types=5,   # Pawn, Knight, Bishop, Rook, Queen
    has_drops=False,
    has_promotion=True,
)

# ---------------------------------------------------------------------------
# Square helpers
# ---------------------------------------------------------------------------

def _sq(rank: int, file: int) -> int:
    """Convert (rank, file) both 0-based to a square index."""
    return rank * 8 + file


def _rank(sq: int) -> int:
    return sq >> 3


def _file(sq: int) -> int:
    return sq & 7


# ---------------------------------------------------------------------------
# Pre-computed attack / move tables
# ---------------------------------------------------------------------------

def _build_knight_attacks() -> List[int]:
    """Return bitmask (as Python int) of knight attack squares for each square."""
    masks = []
    for sq in range(64):
        r, f = _rank(sq), _file(sq)
        bb = 0
        for dr, df in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                bb |= 1 << _sq(nr, nf)
        masks.append(bb)
    return masks


def _build_king_attacks() -> List[int]:
    masks = []
    for sq in range(64):
        r, f = _rank(sq), _file(sq)
        bb = 0
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    bb |= 1 << _sq(nr, nf)
        masks.append(bb)
    return masks


_KNIGHT_ATTACKS: List[int] = _build_knight_attacks()
_KING_ATTACKS: List[int] = _build_king_attacks()

# Ray directions for sliding pieces: (delta_rank, delta_file)
_ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_QUEEN_DIRS = _ROOK_DIRS + _BISHOP_DIRS

# Pre-compute ray squares (list of lists) for each square and each direction
def _build_rays() -> Dict[Tuple[int, int], List[List[int]]]:
    """Return {(dr, df): [ray_squares_from_sq_0, ..., ray_squares_from_sq_63]}."""
    rays: Dict[Tuple[int, int], List[List[int]]] = {}
    for dr, df in _QUEEN_DIRS:
        ray_list: List[List[int]] = []
        for sq in range(64):
            r, f = _rank(sq), _file(sq)
            ray: List[int] = []
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                ray.append(_sq(nr, nf))
                nr += dr
                nf += df
            ray_list.append(ray)
        rays[(dr, df)] = ray_list
    return rays


_RAYS = _build_rays()

# ---------------------------------------------------------------------------
# Pawn attack tables
# ---------------------------------------------------------------------------

def _build_pawn_attacks() -> Tuple[List[int], List[int]]:
    """Return (white_attacks[64], black_attacks[64]) as bitmasks."""
    white, black = [], []
    for sq in range(64):
        r, f = _rank(sq), _file(sq)
        wb, bb = 0, 0
        if r + 1 < 8:
            if f - 1 >= 0:
                wb |= 1 << _sq(r + 1, f - 1)
            if f + 1 < 8:
                wb |= 1 << _sq(r + 1, f + 1)
        if r - 1 >= 0:
            if f - 1 >= 0:
                bb |= 1 << _sq(r - 1, f - 1)
            if f + 1 < 8:
                bb |= 1 << _sq(r - 1, f + 1)
        white.append(wb)
        black.append(bb)
    return white, black


_PAWN_ATTACKS_WHITE, _PAWN_ATTACKS_BLACK = _build_pawn_attacks()

# Promotion piece types (internal encodings) offered for a pawn promotion
_PROMO_INTERNALS = [_QUEEN, _ROOK, _BISHOP, _KNIGHT]

# Map internal encoding -> base interface piece type for promotions
_PROMO_INTERNAL_TO_BASE = {
    _QUEEN: QUEEN, _ROOK: ROOK, _BISHOP: BISHOP, _KNIGHT: KNIGHT
}
_PROMO_BASE_TO_INTERNAL = {v: k for k, v in _PROMO_INTERNAL_TO_BASE.items()}
