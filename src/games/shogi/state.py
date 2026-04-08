"""ShogiState class and initial_state() factory for Shogi."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.games.base import BLACK, WHITE, GameConfig, GameState, Move
from .constants import (
    _BOARD_TO_API,
    _CONFIG,
    _Z_HAND,
    _Z_PIECE,
    _Z_SIDE,
    _sq,
)
from .movegen import (
    PAWN,
    _apply_move,
    _demote,
    _expand_board_moves,
    _expand_drop_moves,
    _is_in_check,
    _is_uchifuzume,
    _king_sq,
)

class ShogiState(GameState):
    """Immutable Shogi game state."""

    __slots__ = (
        "_board",        # np.ndarray shape (81,) dtype int8
        "_sente_hand",   # Dict[api_piece_type, count]
        "_gote_hand",
        "_side",         # WHITE or BLACK
        "_hash",         # Zobrist hash (int)
        "_history",      # Tuple of Zobrist hashes (for repetition detection)
        "_terminal",     # None | "checkmate" | "draw"
        "_result_val",   # None | float
        "_king_sqs",     # cached (sente_king_sq, gote_king_sq)
    )

    def __init__(
        self,
        board: np.ndarray,
        sente_hand: Dict[int, int],
        gote_hand: Dict[int, int],
        side: int,
        history: Tuple[int, ...] = (),
        king_sqs: Optional[Tuple[int, int]] = None,
        hash_val: Optional[int] = None,
    ) -> None:
        self._board = board
        self._sente_hand = sente_hand
        self._gote_hand = gote_hand
        self._side = side
        self._hash = hash_val if hash_val is not None else self._compute_hash()
        self._history = history
        self._terminal = None
        self._result_val = None
        if king_sqs is not None:
            self._king_sqs = king_sqs
        else:
            self._king_sqs = (_king_sq(board, WHITE), _king_sq(board, BLACK))

    def config(self) -> GameConfig:
        return _CONFIG

    def side_to_move(self) -> int:
        return self._side

    def king_square(self, side: int) -> int:
        return self._king_sqs[side]

    def pieces_on_board(self) -> List[Tuple[int, int, int]]:
        result: List[Tuple[int, int, int]] = []
        for sq in range(81):
            p = self._board[sq]
            if p == 0:
                continue
            pv = abs(p)
            if pv == 8:
                continue  # king not included
            api_pt = _BOARD_TO_API[pv]
            color = WHITE if p > 0 else BLACK
            result.append((api_pt, color, sq))
        return result

    def hand_pieces(self, side: int) -> Dict[int, int]:
        h = self._sente_hand if side == WHITE else self._gote_hand
        return dict(h)

    def zobrist_hash(self) -> int:
        return self._hash

    def board_array(self):
        return self._board

    def copy(self) -> "ShogiState":
        return ShogiState(
            self._board.copy(),
            dict(self._sente_hand),
            dict(self._gote_hand),
            self._side,
            self._history,
            king_sqs=self._king_sqs,
        )

    def is_check(self) -> bool:
        return _is_in_check(self._board, self._side)

    def is_terminal(self) -> bool:
        self._ensure_terminal()
        return self._terminal is not None

    def result(self) -> Optional[float]:
        self._ensure_terminal()
        return self._result_val

    def legal_moves(self) -> List[Move]:
        return self._compute_legal_moves()

    def make_move(self, move: Move) -> "ShogiState":
        new_board, sh, gh = _apply_move(
            self._board, self._sente_hand, self._gote_hand, move, self._side
        )
        new_side = BLACK if self._side == WHITE else WHITE
        new_history = self._history + (self._hash,)
        # Incremental Zobrist hash; flip side first
        h = self._hash
        side = self._side
        sign = 1 if side == WHITE else -1
        mover_color = WHITE if side == WHITE else BLACK
        opp_color = 1 - mover_color
        h ^= int(_Z_SIDE[side])
        h ^= int(_Z_SIDE[new_side])
        if move.from_sq is None:
            # Drop move: update board hash and hand hash
            pt = move.drop_piece  # API piece type 0..6
            board_val = pt + 1
            h ^= int(_Z_PIECE[board_val][mover_color][move.to_sq])
            old_hand = self._sente_hand if side == WHITE else self._gote_hand
            old_cnt = old_hand.get(pt, 0)
            new_cnt = old_cnt - 1
            if old_cnt > 0:
                h ^= int(_Z_HAND[pt][mover_color][old_cnt])
            if new_cnt > 0:
                h ^= int(_Z_HAND[pt][mover_color][new_cnt])
        else:
            from_sq = move.from_sq
            to_sq = move.to_sq
            pv = abs(int(self._board[from_sq]))
            captured = int(self._board[to_sq])
            # Remove mover from source
            h ^= int(_Z_PIECE[pv][mover_color][from_sq])
            # Remove captured piece from destination and add (demoted) to hand
            if captured != 0:
                cv = abs(captured)
                h ^= int(_Z_PIECE[cv][opp_color][to_sq])
                demoted = _demote(cv)
                if demoted != 0:
                    api_pt = _BOARD_TO_API[demoted]
                    old_hand = self._sente_hand if side == WHITE else self._gote_hand
                    old_cnt = old_hand.get(api_pt, 0)
                    new_cnt = old_cnt + 1
                    if old_cnt > 0:
                        h ^= int(_Z_HAND[api_pt][mover_color][old_cnt])
                    h ^= int(_Z_HAND[api_pt][mover_color][new_cnt])
            # Place piece at destination (possibly promoted)
            new_pv = move.promotion if move.promotion is not None else pv
            h ^= int(_Z_PIECE[new_pv][mover_color][to_sq])
        w_king, b_king = self._king_sqs
        if move.from_sq is not None and abs(int(self._board[move.from_sq])) == 8:
            if side == WHITE:
                w_king = move.to_sq
            else:
                b_king = move.to_sq
        return ShogiState(new_board, sh, gh, new_side, new_history,
                          king_sqs=(w_king, b_king), hash_val=h)

    def make_null_move(self) -> "ShogiState":
        """Pass the turn without moving any piece (for null move pruning)."""
        new_side = BLACK if self._side == WHITE else WHITE
        h = self._hash
        h ^= int(_Z_SIDE[self._side])
        h ^= int(_Z_SIDE[new_side])
        return ShogiState(self._board, self._sente_hand, self._gote_hand,
                          new_side, self._history,
                          king_sqs=self._king_sqs, hash_val=h)

    def _compute_hash(self) -> int:
        h = int(_Z_SIDE[self._side])
        for sq in range(81):
            p = self._board[sq]
            if p == 0:
                continue
            pv = abs(p)
            color = WHITE if p > 0 else BLACK
            h ^= int(_Z_PIECE[pv][color][sq])
        for pt in range(7):
            cnt_s = self._sente_hand.get(pt, 0)
            cnt_g = self._gote_hand.get(pt, 0)
            if cnt_s:
                h ^= int(_Z_HAND[pt][WHITE][cnt_s])
            if cnt_g:
                h ^= int(_Z_HAND[pt][BLACK][cnt_g])
        return h

    def _compute_legal_moves(self) -> List[Move]:
        board = self._board
        side = self._side
        hand = self._sente_hand if side == WHITE else self._gote_hand
        sh = self._sente_hand
        gh = self._gote_hand
        pseudo_board = _expand_board_moves(board, side)
        pseudo_drops = _expand_drop_moves(board, hand, side)
        legal: List[Move] = []
        # Filter board moves
        for m in pseudo_board:
            nb, nsh, ngh = _apply_move(board, sh, gh, m, side)
            if not _is_in_check(nb, side):
                legal.append(m)
        # Filter drop moves
        for m in pseudo_drops:
            # Uchifuzume check for pawn drops
            if m.drop_piece == PAWN:
                if _is_uchifuzume(board, sh, gh, m.to_sq, side):
                    continue
            nb, nsh, ngh = _apply_move(board, sh, gh, m, side)
            if not _is_in_check(nb, side):
                legal.append(m)
        return legal

    def _ensure_terminal(self) -> None:
        """Lazily compute terminal status."""
        if self._terminal is not None:
            return
        # King captured -> the side whose king is missing loses.
        for side in [WHITE, BLACK]:
            king_val = 8 if side == WHITE else -8
            if not np.any(self._board == king_val):
                self._terminal = "king_captured"
                self._result_val = 0.0 if side == self._side else 1.0
                return
        # Repetition check: fourfold repetition (3 previous + now) -> draw
        if self._history.count(self._hash) >= 3:
            self._terminal = "draw"
            self._result_val = 0.5
            return
        moves = self._compute_legal_moves()
        if moves:
            return  # game continues
        # No legal moves: the side to move loses
        # (In standard shogi stalemate is treated as a loss)
        self._terminal = "checkmate"
        self._result_val = 0.0  # side to move loses


def initial_state() -> ShogiState:
    """Return the standard Shogi starting position.

    Gote (BLACK=1) pieces are at ranks 0-2 (negative values).
    Sente (WHITE=0) pieces are at ranks 6-8 (positive values).
    Row 0 (gote back rank): L N S G K G S N L
    Row 1: gote rook (file 1), gote bishop (file 7)
    Row 2: gote pawns; Row 6: sente pawns
    Row 7: sente bishop (file 1), sente rook (file 7)
    Row 8 (sente back rank): L N S G K G S N L
    """
    board = np.zeros(81, dtype=np.int8)
    # Piece values (positive = sente encoding)
    L, N, S, G, K, B, R = 2, 3, 4, 5, 8, 6, 7  # noqa: E741
    # Gote back rank (row 0) - negative for gote
    back_rank = [L, N, S, G, K, G, S, N, L]
    for f, piece in enumerate(back_rank):
        board[_sq(0, f)] = -piece
    # Gote rook & bishop (row 1)
    board[_sq(1, 7)] = -B  # bishop at file 7
    board[_sq(1, 1)] = -R  # rook at file 1
    # Gote pawns (row 2)
    for f in range(9):
        board[_sq(2, f)] = -1  # -Pawn
    # Sente pawns (row 6)
    for f in range(9):
        board[_sq(6, f)] = 1  # +Pawn
    # Sente rook & bishop (row 7)
    board[_sq(7, 1)] = B   # bishop at file 1
    board[_sq(7, 7)] = R   # rook at file 7
    # Sente back rank (row 8)
    for f, piece in enumerate(back_rank):
        board[_sq(8, f)] = piece
    return ShogiState(
        board=board,
        sente_hand={},
        gote_hand={},
        side=WHITE,  # Sente moves first
        history=(),
        king_sqs=(_sq(8, 4), _sq(0, 4)),
    )
