"""FEN parsing and serialization for the chess engine."""

from __future__ import annotations

import numpy as np

from src.games.base import WHITE, BLACK

from .constants import (
    _PAWN, _KNIGHT, _BISHOP, _ROOK, _QUEEN, _KING,
    _sq, _rank, _file,
)
from .hash_utils import _compute_hash


def from_fen(fen: str) -> "ChessState":
    """Parse a FEN string and return a ChessState."""
    from .state import ChessState

    parts = fen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Invalid FEN: {fen!r}")

    piece_map = {
        "P": _PAWN, "N": _KNIGHT, "B": _BISHOP, "R": _ROOK, "Q": _QUEEN, "K": _KING,
        "p": -_PAWN, "n": -_KNIGHT, "b": -_BISHOP, "r": -_ROOK, "q": -_QUEEN, "k": -_KING,
    }
    board = np.zeros(64, dtype=np.int8)
    rows = parts[0].split("/")
    if len(rows) != 8:
        raise ValueError(f"FEN board must have 8 rows: {fen!r}")
    for rank_idx, row in enumerate(reversed(rows)):
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                board[_sq(rank_idx, file_idx)] = piece_map[ch]
                file_idx += 1

    side = WHITE if parts[1] == "w" else BLACK

    castling = 0
    if "K" in parts[2]:
        castling |= 0b0001
    if "Q" in parts[2]:
        castling |= 0b0010
    if "k" in parts[2]:
        castling |= 0b0100
    if "q" in parts[2]:
        castling |= 0b1000

    ep_square = -1
    if parts[3] != "-":
        ep_file = ord(parts[3][0]) - ord("a")
        ep_rank = int(parts[3][1]) - 1
        ep_square = _sq(ep_rank, ep_file)

    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    hash_val = _compute_hash(board, side, castling, ep_square)
    king_sqs = (
        int(np.argmax(board == _KING)),
        int(np.argmax(board == -_KING)),
    )
    return ChessState(
        board=board,
        side=side,
        castling=castling,
        ep_square=ep_square,
        halfmove=halfmove,
        fullmove=fullmove,
        zobrist=hash_val,
        history=(hash_val,),
        king_sqs=king_sqs,
    )


def to_fen(state) -> str:
    """Return the FEN string for a ChessState."""
    piece_chars = {
        1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
        -1: "p", -2: "n", -3: "b", -4: "r", -5: "q", -6: "k",
    }
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            v = int(state.board[_sq(rank, file)])
            if v == 0:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece_chars[v]
        if empty:
            row += str(empty)
        rows.append(row)
    fen_board = "/".join(rows)
    side = "w" if state._side == WHITE else "b"
    castling_parts = ""
    if state._castling & 0b0001:
        castling_parts += "K"
    if state._castling & 0b0010:
        castling_parts += "Q"
    if state._castling & 0b0100:
        castling_parts += "k"
    if state._castling & 0b1000:
        castling_parts += "q"
    castling_str = castling_parts or "-"
    if state._ep_square >= 0:
        ep_file = "abcdefgh"[_file(state._ep_square)]
        ep_rank = str(_rank(state._ep_square) + 1)
        ep_str = ep_file + ep_rank
    else:
        ep_str = "-"
    return f"{fen_board} {side} {castling_str} {ep_str} {state._halfmove} {state._fullmove}"
