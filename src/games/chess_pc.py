"""Fast chess engine using python-chess bitboard library.

Drop-in replacement for chess.py that wraps chess.Board to implement
the GameState interface. Legal move generation is 50-100x faster than
the pure-Python implementation thanks to bitboard representation.

Square mapping: Both python-chess and our interface use a1=0, h8=63.
Piece encoding: We map to our convention (positive=white, negative=black,
1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess
import chess.polyglot
import numpy as np

from .base import BLACK, WHITE, GameConfig, GameState, Move

# Piece-type constants (base-interface mapping, king excluded)
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4

# python-chess piece type → our internal board value (1-6)
_PC_TO_INTERNAL = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}

# python-chess piece type → our feature piece type (0-4, king excluded)
_PC_TO_FEATURE_TYPE = {
    chess.PAWN: PAWN, chess.KNIGHT: KNIGHT, chess.BISHOP: BISHOP,
    chess.ROOK: ROOK, chess.QUEEN: QUEEN,
}

# Our feature piece type → python-chess promotion piece type
_PROMO_TO_PC = {
    QUEEN: chess.QUEEN, ROOK: chess.ROOK,
    BISHOP: chess.BISHOP, KNIGHT: chess.KNIGHT,
}
_PC_TO_PROMO = {v: k for k, v in _PROMO_TO_PC.items()}

_CHESS_CONFIG = GameConfig(
    name="chess",
    board_height=8,
    board_width=8,
    num_piece_types=5,
    has_drops=False,
    has_promotion=True,
)


class PythonChessState(GameState):
    """Chess position backed by python-chess bitboard engine."""

    __slots__ = ("_board", "_board_arr_cache")

    def __init__(self, board: chess.Board) -> None:
        self._board = board
        self._board_arr_cache: Optional[np.ndarray] = None

    def config(self) -> GameConfig:
        return _CHESS_CONFIG

    def side_to_move(self) -> int:
        return WHITE if self._board.turn == chess.WHITE else BLACK

    def king_square(self, side: int) -> int:
        color = chess.WHITE if side == WHITE else chess.BLACK
        return self._board.king(color)

    def pieces_on_board(self) -> List[Tuple[int, int, int]]:
        result: List[Tuple[int, int, int]] = []
        for sq, piece in self._board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue
            ft = _PC_TO_FEATURE_TYPE[piece.piece_type]
            color = WHITE if piece.color == chess.WHITE else BLACK
            result.append((ft, color, sq))
        return result

    def hand_pieces(self, side: int) -> Dict[int, int]:
        return {}

    def zobrist_hash(self) -> int:
        return chess.polyglot.zobrist_hash(self._board)

    def board_array(self) -> np.ndarray:
        if self._board_arr_cache is not None:
            return self._board_arr_cache
        arr = np.zeros(64, dtype=np.int8)
        for sq, piece in self._board.piece_map().items():
            val = _PC_TO_INTERNAL[piece.piece_type]
            if piece.color == chess.BLACK:
                val = -val
            arr[sq] = val
        self._board_arr_cache = arr
        return arr

    def copy(self) -> "PythonChessState":
        return PythonChessState(self._board.copy())

    def is_terminal(self) -> bool:
        # Don't use claim_draw=True here because python-chess's
        # can_claim_threefold_repetition() internally pops/replays the
        # entire move stack, which breaks when make_move_inplace has
        # pushed search-tree moves onto the stack.
        return self._board.is_game_over(claim_draw=False)

    def result(self) -> Optional[float]:
        if not self._board.is_game_over(claim_draw=False):
            return None
        outcome = self._board.outcome(claim_draw=False)
        if outcome is None:
            return 0.5
        if outcome.winner is None:
            return 0.5
        # outcome.winner is True for WHITE, False for BLACK
        winner = WHITE if outcome.winner else BLACK
        stm = self.side_to_move()
        return 1.0 if winner == stm else 0.0

    def is_check(self) -> bool:
        return self._board.is_check()

    def legal_moves(self) -> List[Move]:
        moves = []
        for m in self._board.legal_moves:
            promo = None
            if m.promotion is not None:
                promo = _PC_TO_PROMO.get(m.promotion)
            moves.append(Move(from_sq=m.from_square, to_sq=m.to_square,
                              promotion=promo))
        return moves

    def make_move(self, move: Move) -> "PythonChessState":
        pc_promo = None
        if move.promotion is not None:
            pc_promo = _PROMO_TO_PC.get(move.promotion, chess.QUEEN)
        pc_move = chess.Move(move.from_sq, move.to_sq, promotion=pc_promo)
        new_board = self._board.copy()
        new_board.push(pc_move)
        return PythonChessState(new_board)

    def make_move_inplace(self, move: Move):
        """Apply move in-place, return undo token (the chess.Move)."""
        pc_promo = None
        if move.promotion is not None:
            pc_promo = _PROMO_TO_PC.get(move.promotion, chess.QUEEN)
        pc_move = chess.Move(move.from_sq, move.to_sq, promotion=pc_promo)
        self._board.push(pc_move)
        self._board_arr_cache = None
        return pc_move

    def unmake_move(self, undo) -> None:
        """Undo a move applied with make_move_inplace."""
        self._board.pop()
        self._board_arr_cache = None

    def __repr__(self) -> str:
        return f"PythonChessState(\n{self._board}\n)"


def initial_state() -> PythonChessState:
    """Return the standard starting chess position."""
    return PythonChessState(chess.Board())


def from_fen(fen: str) -> PythonChessState:
    """Parse a FEN string and return a PythonChessState."""
    return PythonChessState(chess.Board(fen))
