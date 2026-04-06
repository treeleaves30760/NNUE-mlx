"""Los Alamos mini chess (6x6) engine.

Rules summary:
- 6x6 board, 36 squares (index 0 = a1 bottom-left, index 35 = f6 top-right)
- No bishops, no castling, no en passant
- Pawns move 1 square forward only (no double-step)
- Pawns promote to Queen only upon reaching the last rank
- Win by checkmate or capturing the opponent's king
- Draw by stalemate or 50-move rule

Piece encoding (stored in board array, white positive / black negative):
  0 = empty
  1 = Pawn
  2 = Knight
  3 = Rook
  4 = Queen
  5 = King

Square index layout (row-major, rank-major):
  row 5 (rank 6): squares 30-35   (a6 .. f6)
  row 4 (rank 5): squares 24-29
  row 3 (rank 4): squares 18-23
  row 2 (rank 3): squares 12-17
  row 1 (rank 2): squares  6-11
  row 0 (rank 1): squares  0- 5   (a1 .. f1)

Column (file) index: sq % 6  (0=a .. 5=f)
Row  (rank) index:   sq // 6 (0=rank1 .. 5=rank6)
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BLACK, WHITE, GameConfig, GameState, Move

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 6
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE  # 36

# Piece type values stored in the board array (unsigned; sign encodes colour)
EMPTY = 0
PAWN = 1
KNIGHT = 2
ROOK = 3
QUEEN = 4
KING = 5

# Piece-type indices used in pieces_on_board() and Zobrist tables.
# King is excluded here (it is returned separately via king_square()).
#   PAWN_IDX=0, KNIGHT_IDX=1, ROOK_IDX=2, QUEEN_IDX=3
PAWN_IDX = 0
KNIGHT_IDX = 1
ROOK_IDX = 2
QUEEN_IDX = 3

# Map from board-array piece value to pieces_on_board index
_PIECE_TO_IDX: dict[int, int] = {
    PAWN: PAWN_IDX,
    KNIGHT: KNIGHT_IDX,
    ROOK: ROOK_IDX,
    QUEEN: QUEEN_IDX,
}

# Number of non-king piece types per side
NUM_PIECE_TYPES = 4

# 50-move rule: draw after 50 moves without pawn advance or capture
FIFTY_MOVE_LIMIT = 100  # half-moves (plies)

# ---------------------------------------------------------------------------
# Game configuration (singleton)
# ---------------------------------------------------------------------------

_CONFIG = GameConfig(
    name="minichess",
    board_height=BOARD_SIZE,
    board_width=BOARD_SIZE,
    num_piece_types=NUM_PIECE_TYPES,
    has_drops=False,
    has_promotion=True,
)

# ---------------------------------------------------------------------------
# Zobrist hashing tables (fixed seed for reproducibility)
# ---------------------------------------------------------------------------

def _build_zobrist_tables() -> Tuple[np.ndarray, np.ndarray]:
    """Build Zobrist random number tables with a fixed seed.

    Returns:
        piece_table: shape (2, 5, 36) -- [colour, piece_type_idx 0-4, square]
                     Index 4 is used for the King (piece value 5).
        side_table:  scalar uint64 XORed in when it is Black's turn.
    """
    rng = np.random.default_rng(seed=0xDEADBEEF_CAFEBABE)
    # We store 5 piece-type slots (indices 0-4), where index i corresponds to
    # board piece value (i+1): Pawn=slot0, Knight=slot1, Rook=slot2,
    # Queen=slot3, King=slot4.
    piece_table = rng.integers(
        low=0, high=np.iinfo(np.uint64).max, size=(2, 5, NUM_SQUARES),
        dtype=np.uint64,
    )
    side_key = rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
    return piece_table, side_key


_ZOBRIST_PIECES, _ZOBRIST_SIDE = _build_zobrist_tables()


def _piece_slot(piece_value: int) -> int:
    """Convert unsigned board piece value (1-5) to Zobrist slot index (0-4)."""
    return piece_value - 1


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _sq(row: int, col: int) -> int:
    """Convert (row, col) to flat square index."""
    return row * BOARD_SIZE + col


def _row(sq: int) -> int:
    return sq // BOARD_SIZE


def _col(sq: int) -> int:
    return sq % BOARD_SIZE


def _on_board(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


# ---------------------------------------------------------------------------
# Initial position
# ---------------------------------------------------------------------------

def _build_initial_board() -> np.ndarray:
    """Return the initial 36-element board array for Los Alamos chess.

    White rank 1 (row 0): R N Q K N R
    White rank 2 (row 1): 6 pawns
    Black rank 6 (row 5): r n q k n r   (stored as negative values)
    Black rank 5 (row 4): 6 pawns       (stored as negative values)
    """
    board = np.zeros(NUM_SQUARES, dtype=np.int8)

    # White back rank (row 0, squares 0-5): R N Q K N R
    back_rank = [ROOK, KNIGHT, QUEEN, KING, KNIGHT, ROOK]
    for col, piece in enumerate(back_rank):
        board[_sq(0, col)] = piece

    # White pawns (row 1, squares 6-11)
    for col in range(BOARD_SIZE):
        board[_sq(1, col)] = PAWN

    # Black back rank (row 5, squares 30-35): r n q k n r (mirror columns)
    for col, piece in enumerate(back_rank):
        board[_sq(5, col)] = -piece

    # Black pawns (row 4, squares 24-29)
    for col in range(BOARD_SIZE):
        board[_sq(4, col)] = -PAWN

    return board


# ---------------------------------------------------------------------------
# Move generation (pseudo-legal — king-in-check filtering is done separately)
# ---------------------------------------------------------------------------

# Knight move offsets (delta_row, delta_col)
_KNIGHT_DELTAS: List[Tuple[int, int]] = [
    (-2, -1), (-2, +1), (-1, -2), (-1, +2),
    (+1, -2), (+1, +2), (+2, -1), (+2, +1),
]

# Rook / Queen sliding directions
_ROOK_DIRS: List[Tuple[int, int]] = [
    (-1, 0), (+1, 0), (0, -1), (0, +1),
]

# Queen also slides diagonally (but note: no Bishop in this variant)
_QUEEN_DIRS: List[Tuple[int, int]] = _ROOK_DIRS + [
    (-1, -1), (-1, +1), (+1, -1), (+1, +1),
]

# King moves one step in any direction
_KING_DELTAS: List[Tuple[int, int]] = _QUEEN_DIRS  # same offsets, 1 step only


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

        # King captured (should only arise if the engine allows an illegal
        # last move, but we handle it defensively)
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

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

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
