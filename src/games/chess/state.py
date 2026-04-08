"""ChessState class and initial_state factory for the chess engine."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.games.base import BLACK, WHITE, GameConfig, GameState, Move

from .constants import (
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN,
    _EMPTY, _PAWN, _KNIGHT, _KING,
    _INTERNAL_TO_TYPE,
    _side_zobrist, _ep_zobrist,
    _CHESS_CONFIG,
    _sq, _rank, _file,
)
from .movegen import (
    _make_initial_board,
    _apply_move_to_board,
    _generate_pseudo_moves,
    _square_attacked,
)
from .hash_utils import _compute_hash, _update_castling, _update_hash, _insufficient_material


class ChessState(GameState):
    """Immutable chess position."""

    __slots__ = (
        "board", "_side", "_castling", "_ep_square",
        "_halfmove", "_fullmove", "_hash", "_history", "_king_sqs",
    )

    def __init__(
        self, board: np.ndarray, side: int, castling: int, ep_square: int,
        halfmove: int, fullmove: int, zobrist: int, history: tuple,
        king_sqs: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.board = board
        self._side = side
        self._castling = castling
        self._ep_square = ep_square
        self._halfmove = halfmove
        self._fullmove = fullmove
        self._hash = zobrist
        self._history = history
        if king_sqs is not None:
            self._king_sqs = king_sqs
        else:
            self._king_sqs = (
                int(np.argmax(board == _KING)),
                int(np.argmax(board == -_KING)),
            )

    def config(self) -> GameConfig:
        return _CHESS_CONFIG

    def side_to_move(self) -> int:
        return self._side

    def king_square(self, side: int) -> int:
        return self._king_sqs[side]

    def pieces_on_board(self) -> List[Tuple[int, int, int]]:
        result: List[Tuple[int, int, int]] = []
        for sq in range(64):
            v = int(self.board[sq])
            if v == 0:
                continue
            color = WHITE if v > 0 else BLACK
            internal = abs(v)
            if internal == _KING:
                continue
            piece_type = _INTERNAL_TO_TYPE[internal]
            result.append((piece_type, color, sq))
        return result

    def hand_pieces(self, side: int) -> Dict[int, int]:
        return {}

    def zobrist_hash(self) -> int:
        return int(self._hash)

    def board_array(self):
        return self.board

    def copy(self) -> "ChessState":
        return ChessState(
            board=self.board.copy(), side=self._side, castling=self._castling,
            ep_square=self._ep_square, halfmove=self._halfmove,
            fullmove=self._fullmove, zobrist=self._hash,
            history=self._history, king_sqs=self._king_sqs,
        )

    def is_terminal(self) -> bool:
        return self.result() is not None

    def result(self) -> Optional[float]:
        if self._halfmove >= 100:
            return 0.5
        if self._history.count(self._hash) >= 3:
            return 0.5
        if _insufficient_material(self.board):
            return 0.5
        moves = self.legal_moves()
        if not moves:
            if self.is_check():
                return 0.0
            return 0.5
        return None

    def is_check(self) -> bool:
        king_sq = self.king_square(self._side)
        return _square_attacked(self.board, king_sq, 1 - self._side)

    def legal_moves(self) -> List[Move]:
        pseudo = _generate_pseudo_moves(self.board, self._side, self._castling, self._ep_square)
        legal: List[Move] = []
        for mv in pseudo:
            scratch = self.board.copy()
            _apply_move_to_board(scratch, mv, self._ep_square)
            king_sq = int(np.argmax(scratch == (_KING if self._side == WHITE else -_KING)))
            if not _square_attacked(scratch, king_sq, 1 - self._side):
                legal.append(mv)
        return legal

    def make_move(self, move: Move) -> "ChessState":
        new_board = self.board.copy()
        from_sq = move.from_sq
        to_sq = move.to_sq
        mover = int(new_board[from_sq])
        mover_abs = abs(mover)
        captured = int(new_board[to_sq])
        old_ep = self._ep_square

        _apply_move_to_board(new_board, move, old_ep)
        new_castling = _update_castling(self._castling, self._side, mover_abs, from_sq, to_sq)
        new_ep = (from_sq + to_sq) >> 1 if (mover_abs == _PAWN and abs(to_sq - from_sq) == 16) else -1
        is_capture = (captured != 0) or (mover_abs == _PAWN and to_sq == old_ep)
        new_half = 0 if (mover_abs == _PAWN or is_capture) else self._halfmove + 1
        new_full = self._fullmove + (1 if self._side == BLACK else 0)

        new_hash = _update_hash(
            np.uint64(self._hash), self._castling, new_castling,
            old_ep, new_ep, mover, mover_abs, captured,
            from_sq, to_sq, move.promotion,
        )
        new_history = self._history + (new_hash,)

        w_king, b_king = self._king_sqs
        if mover_abs == _KING:
            if self._side == WHITE:
                w_king = to_sq
            else:
                b_king = to_sq

        return ChessState(
            board=new_board, side=1 - self._side, castling=new_castling,
            ep_square=new_ep, halfmove=new_half, fullmove=new_full,
            zobrist=new_hash, history=new_history, king_sqs=(w_king, b_king),
        )

    def make_null_move(self) -> "ChessState":
        h = np.uint64(self._hash)
        h ^= _side_zobrist
        if 0 <= self._ep_square < 64:
            h ^= _ep_zobrist[_file(self._ep_square)]
        return ChessState(
            board=self.board, side=1 - self._side, castling=self._castling,
            ep_square=-1, halfmove=self._halfmove, fullmove=self._fullmove,
            zobrist=int(h), history=self._history, king_sqs=self._king_sqs,
        )

    def __repr__(self) -> str:
        piece_chars = {
            0: ".", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
            -1: "p", -2: "n", -3: "b", -4: "r", -5: "q", -6: "k",
        }
        lines = []
        for rank in range(7, -1, -1):
            row = ""
            for file in range(8):
                row += piece_chars[int(self.board[_sq(rank, file)])] + " "
            lines.append(f"{rank + 1}  {row}")
        lines.append("   a b c d e f g h")
        side_str = "White" if self._side == WHITE else "Black"
        lines.append(f"Side to move: {side_str}")
        return "\n".join(lines)

    def to_fen(self) -> str:
        from .fen import to_fen
        return to_fen(self)


def initial_state() -> ChessState:
    board = _make_initial_board()
    castling = 0b1111
    ep_square = -1
    hash_val = _compute_hash(board, WHITE, castling, ep_square)
    return ChessState(
        board=board, side=WHITE, castling=castling, ep_square=ep_square,
        halfmove=0, fullmove=1, zobrist=hash_val,
        history=(hash_val,), king_sqs=(_sq(0, 4), _sq(7, 4)),
    )
