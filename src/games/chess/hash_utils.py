"""Zobrist hash computation and board evaluation utilities for chess."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.games.base import WHITE, BLACK

from .constants import (
    _EMPTY, _PAWN, _KNIGHT, _BISHOP, _ROOK, _QUEEN, _KING,
    _PROMO_BASE_TO_INTERNAL,
    _piece_zobrist, _side_zobrist, _castling_zobrist, _ep_zobrist,
    _sq, _rank, _file,
)


def _compute_hash(
    board: np.ndarray,
    side: int,
    castling: int,
    ep_square: int,
) -> int:
    """Compute the full Zobrist hash from scratch."""
    h = np.uint64(0)
    for sq in range(64):
        v = int(board[sq])
        if v == 0:
            continue
        color = WHITE if v > 0 else BLACK
        internal = abs(v)
        h ^= _piece_zobrist[internal, color, sq]
    if side == BLACK:
        h ^= _side_zobrist
    h ^= _castling_zobrist[castling & 0xF]
    if 0 <= ep_square < 64:
        h ^= _ep_zobrist[_file(ep_square)]
    return int(h)


# Rook-square -> castling-right mask to clear when that square is touched
_CASTLING_ROOK_UPDATE: Dict[int, int] = {
    _sq(0, 0): ~0b0010,   # a1 WQ
    _sq(0, 7): ~0b0001,   # h1 WK
    _sq(7, 0): ~0b1000,   # a8 BQ
    _sq(7, 7): ~0b0100,   # h8 BK
}


def _update_castling(
    castling: int,
    side: int,
    mover_abs: int,
    from_sq: int,
    to_sq: int,
) -> int:
    """Return the new castling-rights bitmask after a move."""
    new_castling = castling
    if mover_abs == _KING:
        if side == WHITE:
            new_castling &= ~0b0011
        else:
            new_castling &= ~0b1100
    if from_sq in _CASTLING_ROOK_UPDATE:
        new_castling &= _CASTLING_ROOK_UPDATE[from_sq]
    if to_sq in _CASTLING_ROOK_UPDATE:
        new_castling &= _CASTLING_ROOK_UPDATE[to_sq]
    return new_castling


def _update_hash(
    h_in: "np.uint64",
    old_castling: int,
    new_castling: int,
    old_ep: int,
    new_ep: int,
    mover: int,
    mover_abs: int,
    captured: int,
    from_sq: int,
    to_sq: int,
    promotion,
) -> int:
    """Compute the updated Zobrist hash for a move, returned as a plain int."""
    h = h_in
    mover_color = WHITE if mover > 0 else BLACK
    opp_color = 1 - mover_color

    h ^= _side_zobrist
    h ^= _castling_zobrist[old_castling & 0xF]
    h ^= _castling_zobrist[new_castling & 0xF]

    if 0 <= old_ep < 64:
        h ^= _ep_zobrist[_file(old_ep)]
    if 0 <= new_ep < 64:
        h ^= _ep_zobrist[_file(new_ep)]

    h ^= _piece_zobrist[mover_abs, mover_color, from_sq]

    if captured != 0:
        h ^= _piece_zobrist[abs(captured), opp_color, to_sq]

    if mover_abs == _PAWN and to_sq == old_ep and old_ep >= 0:
        color_sign = 1 if mover > 0 else -1
        ep_victim_rank = _rank(to_sq) - color_sign
        ep_victim_sq = _sq(ep_victim_rank, _file(to_sq))
        h ^= _piece_zobrist[_PAWN, opp_color, ep_victim_sq]

    if promotion is not None:
        placed_abs = _PROMO_BASE_TO_INTERNAL[promotion]
    else:
        placed_abs = mover_abs
    h ^= _piece_zobrist[placed_abs, mover_color, to_sq]

    if mover_abs == _KING and abs(_file(to_sq) - _file(from_sq)) == 2:
        rank = _rank(from_sq)
        if _file(to_sq) > _file(from_sq):
            rook_from = _sq(rank, 7)
            rook_to = _sq(rank, 5)
        else:
            rook_from = _sq(rank, 0)
            rook_to = _sq(rank, 3)
        h ^= _piece_zobrist[_ROOK, mover_color, rook_from]
        h ^= _piece_zobrist[_ROOK, mover_color, rook_to]

    return int(h)


def _insufficient_material(board: np.ndarray) -> bool:
    """Return True if neither side can force checkmate."""
    white_pieces: Dict[int, int] = {}
    black_pieces: Dict[int, int] = {}
    for sq in range(64):
        v = int(board[sq])
        if v == 0 or abs(v) == _KING:
            continue
        if v > 0:
            white_pieces[v] = white_pieces.get(v, 0) + 1
        else:
            black_pieces[-v] = black_pieces.get(-v, 0) + 1

    for pieces in (white_pieces, black_pieces):
        if _PAWN in pieces or _ROOK in pieces or _QUEEN in pieces:
            return False

    def minor_count(pieces: Dict[int, int]) -> int:
        return pieces.get(_KNIGHT, 0) + pieces.get(_BISHOP, 0)

    wm = minor_count(white_pieces)
    bm = minor_count(black_pieces)

    if wm <= 1 and bm == 0:
        return True
    if bm <= 1 and wm == 0:
        return True

    if (wm == 1 and bm == 1
            and _BISHOP in white_pieces and _BISHOP in black_pieces):
        w_bishop_sq = next(sq for sq in range(64) if board[sq] == _BISHOP)
        b_bishop_sq = next(sq for sq in range(64) if board[sq] == -_BISHOP)
        if (_rank(w_bishop_sq) + _file(w_bishop_sq)) % 2 == (
                _rank(b_bishop_sq) + _file(b_bishop_sq)) % 2:
            return True

    return False
