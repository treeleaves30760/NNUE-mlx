"""Mini Shogi main state class and initial-position factory."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.games.base import BLACK, WHITE, GameConfig, GameState, Move
from .constants import (
    NUM_SQUARES, BOARD_SIZE,
    KING_VAL, PAWN,
    HAND_PIECE_TYPES, _BOARD_VAL_TO_PIECE_TYPE, _PIECE_TYPE_TO_BOARD_VAL,
    _DEMOTE, _CONFIG,
    _INITIAL_BOARD,
    _ZOBRIST_BLACK_TO_MOVE,
    _sq, _sign, _piece_color,
    _compute_zobrist, _update_zobrist_remove, _update_zobrist_place,
    _update_zobrist_hand,
    PAWN_VAL, SILVER_VAL, GOLD_VAL, BISHOP_VAL, ROOK_VAL,
    TOKIN_VAL, PRO_SILVER_VAL, HORSE_VAL, DRAGON_VAL,
)
from .movegen import _generate_pseudolegal_moves, _is_in_check

class MiniShogiState(GameState):
    """Immutable Mini Shogi position.

    The board is a flat numpy int8 array of length 25.  Positive values are
    sente (WHITE) pieces, negative values are gote (BLACK) pieces.
    """

    __slots__ = (
        "_board",
        "_hands",
        "_side",
        "_hash",
        "_history",   # tuple of past hashes for repetition detection
        "_terminal",  # cached terminal flag
        "_result_val",
        "_king_sqs",  # cached (sente_king_sq, gote_king_sq)
    )

    def __init__(
        self,
        board: np.ndarray,
        hands: Tuple[Dict[int, int], Dict[int, int]],
        side: int,
        hash_val: int,
        history: Tuple[int, ...],
        terminal: Optional[bool] = None,
        result_val: Optional[float] = None,
        king_sqs: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._board = board
        self._hands = hands
        self._side = side
        self._hash = hash_val
        self._history = history
        self._terminal = terminal
        self._result_val = result_val
        if king_sqs is not None:
            self._king_sqs = king_sqs
        else:
            w_king = b_king = -1
            w_val, b_val = KING_VAL, -KING_VAL
            for sq in range(NUM_SQUARES):
                v = board[sq]
                if v == w_val:
                    w_king = sq
                elif v == b_val:
                    b_king = sq
            self._king_sqs = (w_king, b_king)
    def config(self) -> GameConfig:
        return _CONFIG
    def side_to_move(self) -> int:
        return self._side
    def king_square(self, side: int) -> int:
        return self._king_sqs[side]
    def pieces_on_board(self) -> List[Tuple[int, int, int]]:
        """Return (piece_type, color, square) for every non-king piece."""
        result = []
        for sq in range(NUM_SQUARES):
            val = int(self._board[sq])
            if val == 0:
                continue
            abs_v = abs(val)
            if abs_v == KING_VAL:
                continue
            pt = _BOARD_VAL_TO_PIECE_TYPE[abs_v]
            color = WHITE if val > 0 else BLACK
            result.append((pt, color, sq))
        return result
    def hand_pieces(self, side: int) -> Dict[int, int]:
        return dict(self._hands[side])
    def zobrist_hash(self) -> int:
        return self._hash
    def board_array(self):
        return self._board
    def copy(self) -> "MiniShogiState":
        return MiniShogiState(
            board=self._board.copy(),
            hands=(dict(self._hands[WHITE]), dict(self._hands[BLACK])),
            side=self._side,
            hash_val=self._hash,
            history=self._history,
            terminal=self._terminal,
            result_val=self._result_val,
            king_sqs=self._king_sqs,
        )
    def is_check(self) -> bool:
        return _is_in_check(self._board, self._side)
    def is_terminal(self) -> bool:
        if self._terminal is not None:
            return self._terminal
        self._compute_terminal()
        return self._terminal  # type: ignore[return-value]
    def result(self) -> Optional[float]:
        if self._terminal is None:
            self._compute_terminal()
        return self._result_val
    def _compute_terminal(self) -> None:
        """Determine whether the current position is terminal."""
        # King captured -> the side whose king is missing loses.
        for side in [0, 1]:
            king_val = _sign(side) * KING_VAL
            if not any(self._board[sq] == king_val for sq in range(NUM_SQUARES)):
                self._terminal = True
                # side's king is gone -> side lost.
                # If it's side's turn, result = 0.0 (current side loses).
                # If it's opponent's turn, result = 1.0 (current side wins).
                self._result_val = 0.0 if side == self._side else 1.0
                return
        # Fourfold repetition -> draw.
        if self._history.count(self._hash) >= 4:
            self._terminal = True
            self._result_val = 0.5
            return
        # No legal moves -> checkmate (side to move loses).
        if not self.legal_moves():
            self._terminal = True
            self._result_val = 0.0  # current side loses
            return
        self._terminal = False
        self._result_val = None
    def legal_moves(self) -> List[Move]:
        """Return all fully legal moves (pseudo-legal filtered for check)."""
        pseudo = _generate_pseudolegal_moves(self._board, self._side, self._hands)
        legal: List[Move] = []
        for move in pseudo:
            new_state = self._apply_move_unchecked(move)
            # The move is legal if it does not leave our king in check.
            if not _is_in_check(new_state._board, self._side):
                # Additional rule: no drop-pawn checkmate (uchifuzume).
                if (move.from_sq is None and move.drop_piece == PAWN
                        and new_state._is_checkmate_for_opponent()):
                    continue
                legal.append(move)
        return legal
    def _is_checkmate_for_opponent(self) -> bool:
        """Return True if the opponent (side that just moved against) is in checkmate."""
        opp = self._side  # after make_move, the side flipped; but here we haven't flipped
        # Actually this method is called on the new_state after applying the move,
        # where _side has already been flipped to the opponent who is now to move.
        return not bool(
            [m for m in _generate_pseudolegal_moves(self._board, self._side, self._hands)
             if not _is_in_check(self._apply_move_unchecked(m)._board, self._side)]
        )

    def make_move(self, move: Move) -> "MiniShogiState":
        """Apply a move and return a new, fully validated MiniShogiState."""
        new_state = self._apply_move_unchecked(move)
        # Append current hash to history before returning.
        new_history = self._history + (new_state._hash,)
        return MiniShogiState(
            board=new_state._board,
            hands=new_state._hands,
            side=new_state._side,
            hash_val=new_state._hash,
            history=new_history,
            king_sqs=new_state._king_sqs,
        )

    def _apply_move_unchecked(self, move: Move) -> "MiniShogiState":
        """Apply a move without legality checking; return the resulting state."""
        board = self._board.copy()
        hands = (dict(self._hands[WHITE]), dict(self._hands[BLACK]))
        side = self._side
        opp = 1 - side
        sign = _sign(side)
        h = self._hash
        if move.from_sq is None:
            # --- Drop ---
            pt = move.drop_piece
            assert pt is not None
            board_val = _PIECE_TYPE_TO_BOARD_VAL[pt] * sign
            # Remove from hand.
            old_cnt = hands[side][pt]
            hands[side][pt] -= 1
            h = _update_zobrist_hand(h, side, pt, old_cnt, old_cnt - 1)
            # Place on board.
            board[move.to_sq] = board_val
            h = _update_zobrist_place(h, move.to_sq, board_val)
        else:
            from_sq = move.from_sq
            to_sq = move.to_sq
            piece_val = int(board[from_sq])
            abs_v = abs(piece_val)
            # Remove piece from source.
            h = _update_zobrist_remove(h, from_sq, piece_val)
            board[from_sq] = 0
            # Capture if any.
            captured = int(board[to_sq])
            if captured != 0:
                h = _update_zobrist_remove(h, to_sq, captured)
                # Demote captured piece and add to hand (kings are never captured
                # in legal play and cannot go to hand; skip them defensively).
                cap_abs = abs(captured)
                if cap_abs != KING_VAL:
                    demoted_abs = _DEMOTE.get(cap_abs, cap_abs)
                    # Map demoted board value to hand piece type.
                    cap_pt = _BOARD_VAL_TO_PIECE_TYPE[demoted_abs]
                    old_cnt = hands[side].get(cap_pt, 0)
                    hands[side][cap_pt] = old_cnt + 1
                    h = _update_zobrist_hand(h, side, cap_pt, old_cnt, old_cnt + 1)
            # Promotion.
            if move.promotion is not None:
                new_board_val = _PIECE_TYPE_TO_BOARD_VAL[move.promotion] * sign
            else:
                new_board_val = piece_val
            board[to_sq] = new_board_val
            h = _update_zobrist_place(h, to_sq, new_board_val)
        # Flip side to move.
        if side == BLACK:
            h ^= _ZOBRIST_BLACK_TO_MOVE  # remove BLACK flag
        else:
            h ^= _ZOBRIST_BLACK_TO_MOVE  # add BLACK flag
        # Propagate cached king squares
        w_king, b_king = self._king_sqs
        if move.from_sq is not None and abs(int(self._board[move.from_sq])) == KING_VAL:
            if side == WHITE:
                w_king = move.to_sq
            else:
                b_king = move.to_sq
        return MiniShogiState(
            board=board,
            hands=(hands[WHITE], hands[BLACK]),
            side=opp,
            hash_val=h,
            history=self._history,
            king_sqs=(w_king, b_king),
        )

    def make_null_move(self) -> "MiniShogiState":
        """Pass the turn without moving any piece (for null move pruning)."""
        new_side = 1 - self._side
        h = self._hash ^ _ZOBRIST_BLACK_TO_MOVE
        return MiniShogiState(
            board=self._board,
            hands=self._hands,
            side=new_side,
            hash_val=h,
            history=self._history,
            king_sqs=self._king_sqs,
        )

    def __repr__(self) -> str:
        lines = []
        piece_chars = {
            PAWN_VAL: "P", SILVER_VAL: "S", GOLD_VAL: "G",
            BISHOP_VAL: "B", ROOK_VAL: "R", KING_VAL: "K",
            TOKIN_VAL: "T", PRO_SILVER_VAL: "+S", HORSE_VAL: "+B", DRAGON_VAL: "+R",
        }
        lines.append(f"Side to move: {'Sente(W)' if self._side == WHITE else 'Gote(B)'}")
        lines.append("  01234")
        for rank in range(BOARD_SIZE):
            row = f"{rank} "
            for file in range(BOARD_SIZE):
                val = int(self._board[_sq(rank, file)])
                if val == 0:
                    row += "."
                else:
                    c = piece_chars.get(abs(val), "?")
                    row += c[0].upper() if val > 0 else c[0].lower()
            lines.append(row)
        lines.append(f"Sente hand: {self._hands[WHITE]}")
        lines.append(f"Gote  hand: {self._hands[BLACK]}")
        return "\n".join(lines)

def initial_state() -> MiniShogiState:
    """Return the standard Mini Shogi starting position."""
    board = _INITIAL_BOARD.copy()
    hands: Tuple[Dict[int, int], Dict[int, int]] = (
        {pt: 0 for pt in HAND_PIECE_TYPES},
        {pt: 0 for pt in HAND_PIECE_TYPES},
    )
    h = _compute_zobrist(board, hands, WHITE)
    return MiniShogiState(
        board=board,
        hands=hands,
        side=WHITE,
        hash_val=h,
        history=(h,),
        king_sqs=(_sq(4, 4), _sq(0, 0)),
    )
