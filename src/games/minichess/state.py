"""Los Alamos mini chess (6x6) — MiniChessState class and factory function."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..base import BLACK, WHITE, GameConfig, GameState, Move
from .constants import (
    BOARD_SIZE,
    EMPTY,
    FIFTY_MOVE_LIMIT,
    KING,
    KNIGHT,
    NUM_SQUARES,
    PAWN,
    QUEEN,
    ROOK,
    _CONFIG,
    _PIECE_TO_IDX,
    _ZOBRIST_PIECES,
    _ZOBRIST_SIDE,
    _build_initial_board,
    _col,
    _piece_slot,
    _row,
    _sq,
)
from .movegen import (
    _apply_move,
    _compute_hash,
    _find_king,
    _is_in_check,
    _pseudo_legal_moves,
)

# ---------------------------------------------------------------------------
# MiniChessState
# ---------------------------------------------------------------------------

class MiniChessState(GameState):
    """Immutable Los Alamos mini chess position."""

    __slots__ = (
        "_board",       # np.ndarray[int8], length 36
        "_side",        # WHITE (0) or BLACK (1)
        "_halfmoves",   # half-move clock for 50-move rule
        "_hash",        # cached Zobrist hash
        "_terminal",    # cached terminal status: None = not computed
        "_result",      # cached result: None = not computed / not terminal
        "_legal",       # cached list of legal moves
        "_king_sqs",    # cached (white_king_sq, black_king_sq)
    )

    def __init__(
        self,
        board: np.ndarray,
        side: int,
        halfmoves: int = 0,
        *,
        hash_val: Optional[int] = None,
        king_sqs: Optional[Tuple[int, int]] = None,
    ) -> None:
        object.__setattr__(self, "_board", board)
        object.__setattr__(self, "_side", side)
        object.__setattr__(self, "_halfmoves", halfmoves)
        h = hash_val if hash_val is not None else _compute_hash(board, side)
        object.__setattr__(self, "_hash", h)
        object.__setattr__(self, "_terminal", None)
        object.__setattr__(self, "_result", None)
        object.__setattr__(self, "_legal", None)
        if king_sqs is not None:
            object.__setattr__(self, "_king_sqs", king_sqs)
        else:
            object.__setattr__(self, "_king_sqs", (
                _find_king(board, WHITE),
                _find_king(board, BLACK),
            ))

    # ------------------------------------------------------------------
    # GameState interface implementation
    # ------------------------------------------------------------------

    def config(self) -> GameConfig:
        return _CONFIG

    def side_to_move(self) -> int:
        return self._side

    def king_square(self, side: int) -> int:
        return self._king_sqs[side]

    def pieces_on_board(self) -> List[Tuple[int, int, int]]:
        """Return (piece_type_idx, colour, square) for all non-king pieces."""
        result: List[Tuple[int, int, int]] = []
        for sq in range(NUM_SQUARES):
            piece = int(self._board[sq])
            if piece == EMPTY:
                continue
            abs_p = abs(piece)
            if abs_p == KING:
                continue  # king excluded per interface contract
            colour = WHITE if piece > 0 else BLACK
            result.append((_PIECE_TO_IDX[abs_p], colour, sq))
        return result

    def hand_pieces(self, side: int) -> Dict[int, int]:
        # No drops in chess variants
        return {}

    def zobrist_hash(self) -> int:
        return self._hash

    def legal_moves(self) -> List[Move]:
        if self._legal is not None:
            return self._legal

        pseudo = _pseudo_legal_moves(self._board, self._side)
        legal: List[Move] = []
        for move in pseudo:
            new_board = _apply_move(self._board, move)
            # A move is legal only if it does not leave own king in check
            if not _is_in_check(new_board, self._side):
                legal.append(move)

        object.__setattr__(self, "_legal", legal)
        return legal

    def make_move(self, move: Move) -> "MiniChessState":
        """Apply move and return a new immutable state."""
        from_sq = move.from_sq
        to_sq = move.to_sq
        mover = int(self._board[from_sq])
        mover_abs = abs(mover)
        captured = int(self._board[to_sq])

        new_board = _apply_move(self._board, move)
        new_side = BLACK if self._side == WHITE else WHITE

        # Update halfmove clock: reset on pawn move or capture, else increment
        if mover_abs == PAWN or captured != EMPTY:
            new_halfmoves = 0
        else:
            new_halfmoves = self._halfmoves + 1

        # Incremental Zobrist hash
        h = np.uint64(self._hash)
        mover_color = WHITE if mover > 0 else BLACK

        # Flip side
        h ^= _ZOBRIST_SIDE

        # Remove mover from source
        h ^= _ZOBRIST_PIECES[mover_color, _piece_slot(mover_abs), from_sq]

        # Remove captured piece at destination
        if captured != EMPTY:
            cap_color = WHITE if captured > 0 else BLACK
            h ^= _ZOBRIST_PIECES[cap_color, _piece_slot(abs(captured)), to_sq]

        # Place piece at destination (possibly promoted)
        if move.promotion is not None:
            placed_abs = abs(move.promotion)
        else:
            placed_abs = mover_abs
        h ^= _ZOBRIST_PIECES[mover_color, _piece_slot(placed_abs), to_sq]

        # Propagate cached king squares
        w_king, b_king = self._king_sqs
        if mover_abs == KING:
            if self._side == WHITE:
                w_king = to_sq
            else:
                b_king = to_sq

        return MiniChessState(new_board, new_side, new_halfmoves,
                              hash_val=int(h), king_sqs=(w_king, b_king))

    def make_null_move(self) -> "MiniChessState":
        """Pass the turn without moving any piece (for null move pruning)."""
        h = np.uint64(self._hash)
        h ^= _ZOBRIST_SIDE
        new_side = BLACK if self._side == WHITE else WHITE
        return MiniChessState(self._board, new_side, self._halfmoves,
                              hash_val=int(h), king_sqs=self._king_sqs)

    def is_terminal(self) -> bool:
        if self._terminal is not None:
            return self._terminal

        terminal = self._compute_terminal()
        object.__setattr__(self, "_terminal", terminal)
        return terminal

    def result(self) -> Optional[float]:
        if not self.is_terminal():
            return None
        if self._result is not None:
            return self._result

        res = self._compute_result()
        object.__setattr__(self, "_result", res)
        return res

    def is_check(self) -> bool:
        """Return True if the side to move is currently in check."""
        return _is_in_check(self._board, self._side)

    def board_array(self):
        return self._board

    def copy(self) -> "MiniChessState":
        return MiniChessState(
            self._board.copy(),
            self._side,
            self._halfmoves,
            hash_val=self._hash,
            king_sqs=self._king_sqs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_terminal(self) -> bool:
        """Determine whether this position is a terminal state."""
        # 50-move rule (in half-moves / plies)
        if self._halfmoves >= FIFTY_MOVE_LIMIT:
            return True
        # King captured (defensive — should not occur in legal play)
        if _find_king(self._board, WHITE) == -1:
            return True
        if _find_king(self._board, BLACK) == -1:
            return True
        # Checkmate or stalemate: no legal moves available
        if len(self.legal_moves()) == 0:
            return True
        return False

    def _compute_result(self) -> float:
        """Compute the result from the side-to-move's perspective."""
        # 50-move draw
        if self._halfmoves >= FIFTY_MOVE_LIMIT:
            return 0.5
        # King captured — opponent wins, so side to move loses
        stm = self._side
        opp = BLACK if stm == WHITE else WHITE
        if _find_king(self._board, stm) == -1:
            return 0.0  # own king is gone
        if _find_king(self._board, opp) == -1:
            return 1.0  # opponent's king is gone
        # No legal moves
        if len(self.legal_moves()) == 0:
            if _is_in_check(self._board, stm):
                return 0.0  # checkmate — side to move loses
            else:
                return 0.5  # stalemate
        return 0.5  # fallback (should not reach here if is_terminal is correct)

    # -- Debug helpers --------------------------------------------------

    def __repr__(self) -> str:
        return f"MiniChessState(side={self._side}, halfmoves={self._halfmoves})"

    def render(self) -> str:
        """Return a human-readable ASCII board representation."""
        piece_chars = {
            PAWN:   ("P", "p"),
            KNIGHT: ("N", "n"),
            ROOK:   ("R", "r"),
            QUEEN:  ("Q", "q"),
            KING:   ("K", "k"),
        }
        lines = []
        for row in range(BOARD_SIZE - 1, -1, -1):  # rank 6 at top
            line = f"{row + 1} |"
            for col in range(BOARD_SIZE):
                piece = int(self._board[_sq(row, col)])
                if piece == EMPTY:
                    line += " ."
                else:
                    chars = piece_chars[abs(piece)]
                    line += " " + (chars[0] if piece > 0 else chars[1])
            lines.append(line)
        lines.append("   " + " ".join(f" {chr(ord('a') + c)}" for c in range(BOARD_SIZE)))
        side_str = "White" if self._side == WHITE else "Black"
        lines.append(f"Side to move: {side_str}  |  Halfmoves: {self._halfmoves}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def initial_state() -> MiniChessState:
    """Return the starting position for Los Alamos mini chess."""
    board = _build_initial_board()
    return MiniChessState(board, side=WHITE, halfmoves=0,
                          king_sqs=(_sq(0, 3), _sq(5, 3)))
