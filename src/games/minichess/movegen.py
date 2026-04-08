"""Los Alamos mini chess (6x6) — move generation and position helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..base import BLACK, WHITE, Move
from .constants import (
    BOARD_SIZE,
    EMPTY,
    KING,
    KNIGHT,
    NUM_SQUARES,
    PAWN,
    QUEEN,
    ROOK,
    _KING_DELTAS,
    _KNIGHT_DELTAS,
    _QUEEN_DIRS,
    _ROOK_DIRS,
    _ZOBRIST_PIECES,
    _ZOBRIST_SIDE,
    _col,
    _on_board,
    _piece_slot,
    _row,
    _sq,
)

# ---------------------------------------------------------------------------
# Move generation (pseudo-legal — king-in-check filtering is done separately)
# ---------------------------------------------------------------------------


def _pseudo_legal_moves(board: np.ndarray, side: int) -> List[Move]:
    """Generate all pseudo-legal moves (ignores check) for the given side."""
    moves: List[Move] = []
    sign = +1 if side == WHITE else -1  # sign of own pieces

    for sq in range(NUM_SQUARES):
        piece = board[sq]
        if piece == EMPTY or (piece > 0) != (side == WHITE):
            # Square is empty or occupied by the opponent
            continue

        abs_piece = abs(piece)
        row = _row(sq)
        col = _col(sq)

        if abs_piece == PAWN:
            _gen_pawn_moves(board, sq, row, col, sign, moves)
        elif abs_piece == KNIGHT:
            _gen_knight_moves(board, sq, row, col, sign, moves)
        elif abs_piece == ROOK:
            _gen_sliding_moves(board, sq, row, col, sign, _ROOK_DIRS, moves)
        elif abs_piece == QUEEN:
            _gen_sliding_moves(board, sq, row, col, sign, _QUEEN_DIRS, moves)
        elif abs_piece == KING:
            _gen_king_moves(board, sq, row, col, sign, moves)

    return moves


def _gen_pawn_moves(
    board: np.ndarray,
    sq: int,
    row: int,
    col: int,
    sign: int,    # +1 for White, -1 for Black
    moves: List[Move],
) -> None:
    """Generate pseudo-legal pawn moves from square sq."""
    forward = +1 if sign == +1 else -1  # direction of advance
    promo_rank = BOARD_SIZE - 1 if sign == +1 else 0

    # One square forward (no double-step in Los Alamos chess)
    fwd_row = row + forward
    if _on_board(fwd_row, col):
        fwd_sq = _sq(fwd_row, col)
        if board[fwd_sq] == EMPTY:
            if fwd_row == promo_rank:
                # Mandatory promotion to Queen
                moves.append(Move(from_sq=sq, to_sq=fwd_sq, promotion=QUEEN))
            else:
                moves.append(Move(from_sq=sq, to_sq=fwd_sq))

    # Diagonal captures (left and right)
    for dcol in (-1, +1):
        cap_col = col + dcol
        cap_row = row + forward
        if _on_board(cap_row, cap_col):
            cap_sq = _sq(cap_row, cap_col)
            target = board[cap_sq]
            # Must capture an opponent's piece (sign mismatch)
            if target != EMPTY and (target > 0) != (sign > 0):
                if cap_row == promo_rank:
                    moves.append(Move(from_sq=sq, to_sq=cap_sq, promotion=QUEEN))
                else:
                    moves.append(Move(from_sq=sq, to_sq=cap_sq))


def _gen_knight_moves(
    board: np.ndarray,
    sq: int,
    row: int,
    col: int,
    sign: int,
    moves: List[Move],
) -> None:
    for dr, dc in _KNIGHT_DELTAS:
        nr, nc = row + dr, col + dc
        if _on_board(nr, nc):
            dest = _sq(nr, nc)
            target = board[dest]
            # Can move to empty or opponent-occupied square
            if target == EMPTY or (target > 0) != (sign > 0):
                moves.append(Move(from_sq=sq, to_sq=dest))


def _gen_sliding_moves(
    board: np.ndarray,
    sq: int,
    row: int,
    col: int,
    sign: int,
    directions: List[Tuple[int, int]],
    moves: List[Move],
) -> None:
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        while _on_board(nr, nc):
            dest = _sq(nr, nc)
            target = board[dest]
            if target == EMPTY:
                moves.append(Move(from_sq=sq, to_sq=dest))
            else:
                # Occupied: capture if opponent's piece, then stop sliding
                if (target > 0) != (sign > 0):
                    moves.append(Move(from_sq=sq, to_sq=dest))
                break
            nr += dr
            nc += dc


def _gen_king_moves(
    board: np.ndarray,
    sq: int,
    row: int,
    col: int,
    sign: int,
    moves: List[Move],
) -> None:
    for dr, dc in _KING_DELTAS:
        nr, nc = row + dr, col + dc
        if _on_board(nr, nc):
            dest = _sq(nr, nc)
            target = board[dest]
            if target == EMPTY or (target > 0) != (sign > 0):
                moves.append(Move(from_sq=sq, to_sq=dest))


# ---------------------------------------------------------------------------
# Apply a move to a board (returns new board array)
# ---------------------------------------------------------------------------

def _apply_move(board: np.ndarray, move: Move) -> np.ndarray:
    """Return a new board with the given move applied."""
    new_board = board.copy()
    piece = new_board[move.from_sq]
    new_board[move.to_sq] = piece
    new_board[move.from_sq] = EMPTY

    # Promotion: replace pawn with queen
    if move.promotion is not None:
        sign = +1 if piece > 0 else -1
        new_board[move.to_sq] = sign * move.promotion

    return new_board


# ---------------------------------------------------------------------------
# Attack / check detection
# ---------------------------------------------------------------------------

def _find_king(board: np.ndarray, side: int) -> int:
    """Return the square of the given side's king, or -1 if not found."""
    king_val = KING if side == WHITE else -KING
    for sq in range(NUM_SQUARES):
        if board[sq] == king_val:
            return sq
    return -1  # should never happen in a legal position


def _is_square_attacked(board: np.ndarray, sq: int, by_side: int) -> bool:
    """Return True if square sq is attacked by any piece belonging to by_side."""
    sign = +1 if by_side == WHITE else -1

    row = _row(sq)
    col = _col(sq)

    # Check knight attacks
    for dr, dc in _KNIGHT_DELTAS:
        nr, nc = row + dr, col + dc
        if _on_board(nr, nc):
            piece = board[_sq(nr, nc)]
            if piece == sign * KNIGHT:
                return True

    # Check rook / queen attacks along ranks and files
    for dr, dc in _ROOK_DIRS:
        nr, nc = row + dr, col + dc
        while _on_board(nr, nc):
            piece = board[_sq(nr, nc)]
            if piece != EMPTY:
                if piece == sign * ROOK or piece == sign * QUEEN:
                    return True
                break
            nr += dr
            nc += dc

    # Check queen / diagonal attacks
    diagonal_dirs = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    for dr, dc in diagonal_dirs:
        nr, nc = row + dr, col + dc
        while _on_board(nr, nc):
            piece = board[_sq(nr, nc)]
            if piece != EMPTY:
                if piece == sign * QUEEN:
                    return True
                break
            nr += dr
            nc += dc

    # Check pawn attacks
    # A White pawn at (row-1, col±1) attacks sq; a Black pawn at (row+1, col±1) attacks sq.
    pawn_attack_row = row - 1 if by_side == WHITE else row + 1
    if _on_board(pawn_attack_row, col):
        for dc in (-1, +1):
            pc = col + dc
            if _on_board(pawn_attack_row, pc):
                piece = board[_sq(pawn_attack_row, pc)]
                if piece == sign * PAWN:
                    return True

    # Check king proximity (to prevent kings from moving adjacent to each other)
    for dr, dc in _KING_DELTAS:
        nr, nc = row + dr, col + dc
        if _on_board(nr, nc):
            piece = board[_sq(nr, nc)]
            if piece == sign * KING:
                return True

    return False


def _is_in_check(board: np.ndarray, side: int) -> bool:
    """Return True if the given side's king is currently in check."""
    king_sq = _find_king(board, side)
    if king_sq == -1:
        return True  # king captured — treated as check for safety
    opponent = BLACK if side == WHITE else WHITE
    return _is_square_attacked(board, king_sq, by_side=opponent)


# ---------------------------------------------------------------------------
# Zobrist hash computation
# ---------------------------------------------------------------------------

def _compute_hash(board: np.ndarray, side: int) -> int:
    """Compute Zobrist hash for a position from scratch."""
    h = np.uint64(0)
    for sq in range(NUM_SQUARES):
        piece = int(board[sq])
        if piece == EMPTY:
            continue
        colour = WHITE if piece > 0 else BLACK
        slot = _piece_slot(abs(piece))
        h ^= _ZOBRIST_PIECES[colour, slot, sq]
    if side == BLACK:
        h ^= _ZOBRIST_SIDE
    return int(h)

