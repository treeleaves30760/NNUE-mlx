"""Move generation and attack detection for the chess engine."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.games.base import WHITE, BLACK, Move

from .constants import (
    _EMPTY, _PAWN, _KNIGHT, _BISHOP, _ROOK, _QUEEN, _KING,
    _PROMO_INTERNALS, _PROMO_INTERNAL_TO_BASE, _PROMO_BASE_TO_INTERNAL,
    _KNIGHT_ATTACKS, _KING_ATTACKS,
    _ROOK_DIRS, _BISHOP_DIRS, _QUEEN_DIRS, _RAYS,
    _PAWN_ATTACKS_WHITE, _PAWN_ATTACKS_BLACK,
    _sq, _rank, _file,
)


def _make_initial_board() -> np.ndarray:
    board = np.zeros(64, dtype=np.int8)
    back_row = [_ROOK, _KNIGHT, _BISHOP, _QUEEN, _KING, _BISHOP, _KNIGHT, _ROOK]
    for f, p in enumerate(back_row):
        board[_sq(0, f)] = p
        board[_sq(7, f)] = -p
    for f in range(8):
        board[_sq(1, f)] = _PAWN
        board[_sq(6, f)] = -_PAWN
    return board


def _apply_move_to_board(board: np.ndarray, move: Move, ep_square: int) -> None:
    """Mutate board in-place to reflect move. Handles en-passant, castling, promotions."""
    from_sq = move.from_sq
    to_sq = move.to_sq
    mover = int(board[from_sq])
    mover_abs = abs(mover)
    color_sign = 1 if mover > 0 else -1

    board[from_sq] = _EMPTY

    if mover_abs == _PAWN and to_sq == ep_square and ep_square >= 0:
        captured_rank = _rank(to_sq) - color_sign
        board[_sq(captured_rank, _file(to_sq))] = _EMPTY

    if mover_abs == _KING:
        file_diff = _file(to_sq) - _file(from_sq)
        if abs(file_diff) == 2:
            rank = _rank(from_sq)
            if file_diff > 0:
                board[_sq(rank, 7)] = _EMPTY
                board[_sq(rank, 5)] = _ROOK * color_sign
            else:
                board[_sq(rank, 0)] = _EMPTY
                board[_sq(rank, 3)] = _ROOK * color_sign

    if move.promotion is not None:
        internal = _PROMO_BASE_TO_INTERNAL[move.promotion]
        board[to_sq] = internal * color_sign
    else:
        board[to_sq] = mover


def _generate_pseudo_moves(
    board: np.ndarray, side: int, castling: int, ep_square: int,
) -> List[Move]:
    """Generate all pseudo-legal moves (may leave own king in check)."""
    moves: List[Move] = []
    sign = 1 if side == WHITE else -1

    for sq in range(64):
        piece = int(board[sq])
        if piece * sign <= 0:
            continue
        piece_abs = abs(piece)
        if piece_abs == _PAWN:
            _gen_pawn_moves(board, sq, side, ep_square, moves)
        elif piece_abs == _KNIGHT:
            _gen_leaper_moves(board, sq, _KNIGHT_ATTACKS[sq], sign, moves)
        elif piece_abs == _BISHOP:
            _gen_slider_moves(board, sq, _BISHOP_DIRS, sign, moves)
        elif piece_abs == _ROOK:
            _gen_slider_moves(board, sq, _ROOK_DIRS, sign, moves)
        elif piece_abs == _QUEEN:
            _gen_slider_moves(board, sq, _QUEEN_DIRS, sign, moves)
        elif piece_abs == _KING:
            _gen_leaper_moves(board, sq, _KING_ATTACKS[sq], sign, moves)
            _gen_castling_moves(board, sq, side, castling, moves)
    return moves


def _gen_pawn_moves(
    board: np.ndarray, sq: int, side: int, ep_square: int, moves: List[Move],
) -> None:
    rank = _rank(sq)
    file = _file(sq)
    forward = 1 if side == WHITE else -1
    start_rank = 1 if side == WHITE else 6
    promo_rank = 6 if side == WHITE else 1

    new_rank = rank + forward
    if 0 <= new_rank < 8:
        to_sq = _sq(new_rank, file)
        if board[to_sq] == _EMPTY:
            if rank == promo_rank:
                for pt in _PROMO_INTERNALS:
                    moves.append(Move(sq, to_sq, promotion=_PROMO_INTERNAL_TO_BASE[pt]))
            else:
                moves.append(Move(sq, to_sq))
                if rank == start_rank:
                    to_sq2 = _sq(new_rank + forward, file)
                    if board[to_sq2] == _EMPTY:
                        moves.append(Move(sq, to_sq2))

    opp_sign = -1 if side == WHITE else 1
    for df in (-1, 1):
        nf = file + df
        if 0 <= nf < 8 and 0 <= new_rank < 8:
            to_sq = _sq(new_rank, nf)
            is_capture = (int(board[to_sq]) * opp_sign > 0)
            is_ep = (to_sq == ep_square)
            if is_capture or is_ep:
                if rank == promo_rank:
                    for pt in _PROMO_INTERNALS:
                        moves.append(Move(sq, to_sq, promotion=_PROMO_INTERNAL_TO_BASE[pt]))
                else:
                    moves.append(Move(sq, to_sq))


def _gen_leaper_moves(
    board: np.ndarray, sq: int, attack_bb: int, sign: int, moves: List[Move],
) -> None:
    bb = attack_bb
    while bb:
        lsb = bb & (-bb)
        to_sq = lsb.bit_length() - 1
        if int(board[to_sq]) * sign <= 0:
            moves.append(Move(sq, to_sq))
        bb ^= lsb


def _gen_slider_moves(
    board: np.ndarray, sq: int, directions: List[Tuple[int, int]],
    sign: int, moves: List[Move],
) -> None:
    for dr, df in directions:
        for to_sq in _RAYS[(dr, df)][sq]:
            target = int(board[to_sq])
            if target * sign > 0:
                break
            moves.append(Move(sq, to_sq))
            if target != _EMPTY:
                break


def _gen_castling_moves(
    board: np.ndarray, king_sq: int, side: int, castling: int, moves: List[Move],
) -> None:
    rank = 0 if side == WHITE else 7
    opponent = 1 - side

    ks_bit = 0b0001 if side == WHITE else 0b0100
    if castling & ks_bit:
        if board[_sq(rank, 5)] == _EMPTY and board[_sq(rank, 6)] == _EMPTY:
            if not (_square_attacked(board, _sq(rank, 4), opponent)
                    or _square_attacked(board, _sq(rank, 5), opponent)
                    or _square_attacked(board, _sq(rank, 6), opponent)):
                moves.append(Move(_sq(rank, 4), _sq(rank, 6)))

    qs_bit = 0b0010 if side == WHITE else 0b1000
    if castling & qs_bit:
        if (board[_sq(rank, 1)] == _EMPTY
                and board[_sq(rank, 2)] == _EMPTY
                and board[_sq(rank, 3)] == _EMPTY):
            if not (_square_attacked(board, _sq(rank, 4), opponent)
                    or _square_attacked(board, _sq(rank, 3), opponent)
                    or _square_attacked(board, _sq(rank, 2), opponent)):
                moves.append(Move(_sq(rank, 4), _sq(rank, 2)))


def _square_attacked(board: np.ndarray, sq: int, by_side: int) -> bool:
    """Return True if sq is attacked by any piece belonging to by_side."""
    sign = 1 if by_side == WHITE else -1

    # Pawn attacks (reverse lookup)
    attacker_pawn_bb = _PAWN_ATTACKS_BLACK[sq] if by_side == WHITE else _PAWN_ATTACKS_WHITE[sq]
    bb = attacker_pawn_bb
    while bb:
        lsb = bb & (-bb)
        y = lsb.bit_length() - 1
        if int(board[y]) == _PAWN * sign:
            return True
        bb ^= lsb

    bb = _KNIGHT_ATTACKS[sq]
    while bb:
        lsb = bb & (-bb)
        y = lsb.bit_length() - 1
        if int(board[y]) == _KNIGHT * sign:
            return True
        bb ^= lsb

    bb = _KING_ATTACKS[sq]
    while bb:
        lsb = bb & (-bb)
        y = lsb.bit_length() - 1
        if int(board[y]) == _KING * sign:
            return True
        bb ^= lsb

    for dr, df in _ROOK_DIRS:
        for y in _RAYS[(dr, df)][sq]:
            v = int(board[y])
            if v != _EMPTY:
                if v == _ROOK * sign or v == _QUEEN * sign:
                    return True
                break

    for dr, df in _BISHOP_DIRS:
        for y in _RAYS[(dr, df)][sq]:
            v = int(board[y])
            if v != _EMPTY:
                if v == _BISHOP * sign or v == _QUEEN * sign:
                    return True
                break

    return False
