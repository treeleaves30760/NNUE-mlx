"""Complete chess engine implementing the GameState interface.

Board layout: flat numpy array of length 64.
  - Index 0 = a1 (rank 1, file a), index 63 = h8 (rank 8, file h).
  - square(rank, file) = rank * 8 + file  where rank/file are 0-based.

Piece encoding:
  0 = empty
  1 = Pawn, 2 = Knight, 3 = Bishop, 4 = Rook, 5 = Queen, 6 = King
  Positive = White, Negative = Black.

Piece-type constants matching the base interface:
  PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4
  (King excluded from pieces_on_board; handled by king_square.)
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BLACK, WHITE, GameConfig, GameState, Move

# ---------------------------------------------------------------------------
# Piece-type constants (base-interface mapping)
# ---------------------------------------------------------------------------
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4

# Internal board encoding (always positive; sign encodes color)
_EMPTY = 0
_PAWN = 1
_KNIGHT = 2
_BISHOP = 3
_ROOK = 4
_QUEEN = 5
_KING = 6

# Map from base-interface piece_type to internal encoding
_TYPE_TO_INTERNAL = {PAWN: _PAWN, KNIGHT: _KNIGHT, BISHOP: _BISHOP, ROOK: _ROOK, QUEEN: _QUEEN}
# Map from internal encoding to base-interface piece_type (king excluded)
_INTERNAL_TO_TYPE = {_PAWN: PAWN, _KNIGHT: KNIGHT, _BISHOP: BISHOP, _ROOK: ROOK, _QUEEN: QUEEN}

# ---------------------------------------------------------------------------
# Zobrist tables  (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(seed=20240101)

# piece_zobrist[piece_internal 1..6][color 0..1][square 0..63]
_piece_zobrist: np.ndarray = _rng.integers(
    low=0, high=2**63, size=(7, 2, 64), dtype=np.uint64
)
# side-to-move: XOR in when it is Black's turn
_side_zobrist: np.uint64 = np.uint64(_rng.integers(0, 2**63, dtype=np.uint64))
# castling rights: 4 bits (WK, WQ, BK, BQ) -> 16 independent hashes
_castling_zobrist: np.ndarray = _rng.integers(0, 2**63, size=16, dtype=np.uint64)
# en-passant file (0-7); 0 = no en-passant
_ep_zobrist: np.ndarray = _rng.integers(0, 2**63, size=8, dtype=np.uint64)

# ---------------------------------------------------------------------------
# Static game config
# ---------------------------------------------------------------------------
_CHESS_CONFIG = GameConfig(
    name="chess",
    board_height=8,
    board_width=8,
    num_piece_types=5,   # Pawn, Knight, Bishop, Rook, Queen
    has_drops=False,
    has_promotion=True,
)

# ---------------------------------------------------------------------------
# Square helpers
# ---------------------------------------------------------------------------

def _sq(rank: int, file: int) -> int:
    """Convert (rank, file) both 0-based to a square index."""
    return rank * 8 + file


def _rank(sq: int) -> int:
    return sq >> 3


def _file(sq: int) -> int:
    return sq & 7


# ---------------------------------------------------------------------------
# Pre-computed attack / move tables
# ---------------------------------------------------------------------------

def _build_knight_attacks() -> List[int]:
    """Return bitmask (as Python int) of knight attack squares for each square."""
    masks = []
    for sq in range(64):
        r, f = _rank(sq), _file(sq)
        bb = 0
        for dr, df in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                bb |= 1 << _sq(nr, nf)
        masks.append(bb)
    return masks


def _build_king_attacks() -> List[int]:
    masks = []
    for sq in range(64):
        r, f = _rank(sq), _file(sq)
        bb = 0
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    bb |= 1 << _sq(nr, nf)
        masks.append(bb)
    return masks


_KNIGHT_ATTACKS: List[int] = _build_knight_attacks()
_KING_ATTACKS: List[int] = _build_king_attacks()

# Ray directions for sliding pieces: (delta_rank, delta_file)
_ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_QUEEN_DIRS = _ROOK_DIRS + _BISHOP_DIRS

# Pre-compute ray squares (list of lists) for each square and each direction
def _build_rays() -> Dict[Tuple[int, int], List[List[int]]]:
    """Return {(dr, df): [ray_squares_from_sq_0, ..., ray_squares_from_sq_63]}."""
    rays: Dict[Tuple[int, int], List[List[int]]] = {}
    for dr, df in _QUEEN_DIRS:
        ray_list: List[List[int]] = []
        for sq in range(64):
            r, f = _rank(sq), _file(sq)
            ray: List[int] = []
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                ray.append(_sq(nr, nf))
                nr += dr
                nf += df
            ray_list.append(ray)
        rays[(dr, df)] = ray_list
    return rays


_RAYS = _build_rays()

# ---------------------------------------------------------------------------
# Pawn attack tables
# ---------------------------------------------------------------------------

def _build_pawn_attacks() -> Tuple[List[int], List[int]]:
    """Return (white_attacks[64], black_attacks[64]) as bitmasks."""
    white, black = [], []
    for sq in range(64):
        r, f = _rank(sq), _file(sq)
        wb, bb = 0, 0
        if r + 1 < 8:
            if f - 1 >= 0:
                wb |= 1 << _sq(r + 1, f - 1)
            if f + 1 < 8:
                wb |= 1 << _sq(r + 1, f + 1)
        if r - 1 >= 0:
            if f - 1 >= 0:
                bb |= 1 << _sq(r - 1, f - 1)
            if f + 1 < 8:
                bb |= 1 << _sq(r - 1, f + 1)
        white.append(wb)
        black.append(bb)
    return white, black


_PAWN_ATTACKS_WHITE, _PAWN_ATTACKS_BLACK = _build_pawn_attacks()

# Promotion piece types (internal encodings) offered for a pawn promotion
_PROMO_INTERNALS = [_QUEEN, _ROOK, _BISHOP, _KNIGHT]

# Map internal encoding -> base interface piece type for promotions
_PROMO_INTERNAL_TO_BASE = {
    _QUEEN: QUEEN, _ROOK: ROOK, _BISHOP: BISHOP, _KNIGHT: KNIGHT
}
_PROMO_BASE_TO_INTERNAL = {v: k for k, v in _PROMO_INTERNAL_TO_BASE.items()}

# ---------------------------------------------------------------------------
# Initial board setup
# ---------------------------------------------------------------------------

def _make_initial_board() -> np.ndarray:
    board = np.zeros(64, dtype=np.int8)
    # White pieces on ranks 0 and 1 (a1..h1 and a2..h2)
    back_row = [_ROOK, _KNIGHT, _BISHOP, _QUEEN, _KING, _BISHOP, _KNIGHT, _ROOK]
    for f, p in enumerate(back_row):
        board[_sq(0, f)] = p          # white
        board[_sq(7, f)] = -p         # black
    for f in range(8):
        board[_sq(1, f)] = _PAWN      # white pawns
        board[_sq(6, f)] = -_PAWN     # black pawns
    return board


# ---------------------------------------------------------------------------
# ChessState
# ---------------------------------------------------------------------------

class ChessState(GameState):
    """Immutable chess position.

    Attributes:
        board         -- np.ndarray[int8] shape (64,)
        _side         -- WHITE or BLACK
        _castling     -- int bitmask: bit0=WK, bit1=WQ, bit2=BK, bit3=BQ
        _ep_square    -- square index for en-passant target, or -1
        _halfmove     -- half-move clock (for fifty-move rule)
        _fullmove     -- full-move counter
        _hash         -- Zobrist hash (uint64)
        _history      -- tuple of previous Zobrist hashes (for threefold repetition)
    """

    __slots__ = (
        "board",
        "_side",
        "_castling",
        "_ep_square",
        "_halfmove",
        "_fullmove",
        "_hash",
        "_history",
        "_king_sqs",
    )

    def __init__(
        self,
        board: np.ndarray,
        side: int,
        castling: int,
        ep_square: int,
        halfmove: int,
        fullmove: int,
        zobrist: int,
        history: tuple,
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

    # ------------------------------------------------------------------
    # GameState interface
    # ------------------------------------------------------------------

    def config(self) -> GameConfig:
        return _CHESS_CONFIG

    def side_to_move(self) -> int:
        return self._side

    def king_square(self, side: int) -> int:
        """Return the square of the king for the given side."""
        return self._king_sqs[side]

    def pieces_on_board(self) -> List[Tuple[int, int, int]]:
        """Return [(piece_type, color, square)] for all non-king pieces."""
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
        """Chess has no drops; always returns empty dict."""
        return {}

    def zobrist_hash(self) -> int:
        return int(self._hash)

    def board_array(self):
        return self.board

    def copy(self) -> "ChessState":
        return ChessState(
            board=self.board.copy(),
            side=self._side,
            castling=self._castling,
            ep_square=self._ep_square,
            halfmove=self._halfmove,
            fullmove=self._fullmove,
            zobrist=self._hash,
            history=self._history,
            king_sqs=self._king_sqs,
        )

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.result() is not None

    def result(self) -> Optional[float]:
        """Game result from the side-to-move's perspective."""
        # Fifty-move rule
        if self._halfmove >= 100:
            return 0.5
        # Threefold repetition
        if self._history.count(self._hash) >= 3:
            return 0.5
        # Insufficient material
        if _insufficient_material(self.board):
            return 0.5
        # No legal moves -> checkmate or stalemate
        moves = self.legal_moves()
        if not moves:
            if self.is_check():
                return 0.0   # side to move is checkmated
            return 0.5       # stalemate
        return None

    def is_check(self) -> bool:
        """Return True if the side to move is currently in check."""
        king_sq = self.king_square(self._side)
        return _square_attacked(self.board, king_sq, 1 - self._side)

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def legal_moves(self) -> List[Move]:
        """Generate all legal moves for the side to move."""
        pseudo = _generate_pseudo_moves(self.board, self._side, self._castling, self._ep_square)
        legal: List[Move] = []
        for mv in pseudo:
            # Apply the move on a scratch board and check for check
            scratch = self.board.copy()
            _apply_move_to_board(scratch, mv, self._ep_square)
            king_sq = int(np.argmax(scratch == (_KING if self._side == WHITE else -_KING)))
            if not _square_attacked(scratch, king_sq, 1 - self._side):
                legal.append(mv)
        return legal

    def make_move(self, move: Move) -> "ChessState":
        """Apply a move and return a completely new ChessState."""
        new_board = self.board.copy()
        from_sq = move.from_sq
        to_sq = move.to_sq

        mover = int(new_board[from_sq])
        mover_abs = abs(mover)
        captured = int(new_board[to_sq])
        old_ep = self._ep_square

        # ---------- apply move to board ----------
        _apply_move_to_board(new_board, move, self._ep_square)

        # ---------- new castling rights ----------
        new_castling = self._castling
        # Moving the king
        if mover_abs == _KING:
            if self._side == WHITE:
                new_castling &= ~0b0011   # clear WK and WQ
            else:
                new_castling &= ~0b1100   # clear BK and BQ
        # Moving or capturing a rook
        _castling_rook_update = {
            _sq(0, 0): ~0b0010,   # a1 WQ
            _sq(0, 7): ~0b0001,   # h1 WK
            _sq(7, 0): ~0b1000,   # a8 BQ
            _sq(7, 7): ~0b0100,   # h8 BK
        }
        if from_sq in _castling_rook_update:
            new_castling &= _castling_rook_update[from_sq]
        if to_sq in _castling_rook_update:
            new_castling &= _castling_rook_update[to_sq]

        # ---------- new en-passant square ----------
        new_ep = -1
        if mover_abs == _PAWN and abs(to_sq - from_sq) == 16:
            # Double pawn push
            new_ep = (from_sq + to_sq) >> 1   # midpoint square

        # ---------- half-move clock ----------
        is_capture = (captured != 0) or (mover_abs == _PAWN and to_sq == old_ep)
        new_half = 0 if (mover_abs == _PAWN or is_capture) else self._halfmove + 1

        # ---------- full-move counter ----------
        new_full = self._fullmove + (1 if self._side == BLACK else 0)

        # ---------- Zobrist hash (incremental) ----------
        h = np.uint64(self._hash)
        mover_color = WHITE if mover > 0 else BLACK
        opp_color = 1 - mover_color

        # Flip side to move
        h ^= _side_zobrist

        # Castling rights change
        h ^= _castling_zobrist[self._castling & 0xF]
        h ^= _castling_zobrist[new_castling & 0xF]

        # En-passant change
        if 0 <= old_ep < 64:
            h ^= _ep_zobrist[_file(old_ep)]
        if 0 <= new_ep < 64:
            h ^= _ep_zobrist[_file(new_ep)]

        # Remove mover from source square
        h ^= _piece_zobrist[mover_abs, mover_color, from_sq]

        # Remove captured piece at destination (if normal capture)
        if captured != 0:
            h ^= _piece_zobrist[abs(captured), opp_color, to_sq]

        # En-passant capture: remove the victim pawn from its actual square
        if mover_abs == _PAWN and to_sq == old_ep and old_ep >= 0:
            color_sign = 1 if mover > 0 else -1
            ep_victim_rank = _rank(to_sq) - color_sign
            ep_victim_sq = _sq(ep_victim_rank, _file(to_sq))
            h ^= _piece_zobrist[_PAWN, opp_color, ep_victim_sq]

        # Place piece at destination (possibly promoted)
        if move.promotion is not None:
            placed_abs = _PROMO_BASE_TO_INTERNAL[move.promotion]
        else:
            placed_abs = mover_abs
        h ^= _piece_zobrist[placed_abs, mover_color, to_sq]

        # Castling: rook also moves
        if mover_abs == _KING and abs(_file(to_sq) - _file(from_sq)) == 2:
            rank = _rank(from_sq)
            if _file(to_sq) > _file(from_sq):
                # Kingside: rook h1/h8 -> f1/f8
                rook_from = _sq(rank, 7)
                rook_to = _sq(rank, 5)
            else:
                # Queenside: rook a1/a8 -> d1/d8
                rook_from = _sq(rank, 0)
                rook_to = _sq(rank, 3)
            h ^= _piece_zobrist[_ROOK, mover_color, rook_from]
            h ^= _piece_zobrist[_ROOK, mover_color, rook_to]

        new_hash = int(h)

        # ---------- history for repetition ----------
        new_history = self._history + (new_hash,)

        # ---------- king squares ----------
        w_king, b_king = self._king_sqs
        if mover_abs == _KING:
            if self._side == WHITE:
                w_king = to_sq
            else:
                b_king = to_sq

        return ChessState(
            board=new_board,
            side=1 - self._side,
            castling=new_castling,
            ep_square=new_ep,
            halfmove=new_half,
            fullmove=new_full,
            zobrist=new_hash,
            history=new_history,
            king_sqs=(w_king, b_king),
        )

    # ------------------------------------------------------------------
    # Utility / debug
    # ------------------------------------------------------------------

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
        """Return the FEN string for this position."""
        piece_chars = {
            1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
            -1: "p", -2: "n", -3: "b", -4: "r", -5: "q", -6: "k",
        }
        rows = []
        for rank in range(7, -1, -1):
            row = ""
            empty = 0
            for file in range(8):
                v = int(self.board[_sq(rank, file)])
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
        side = "w" if self._side == WHITE else "b"
        castling_parts = ""
        if self._castling & 0b0001:
            castling_parts += "K"
        if self._castling & 0b0010:
            castling_parts += "Q"
        if self._castling & 0b0100:
            castling_parts += "k"
        if self._castling & 0b1000:
            castling_parts += "q"
        castling_str = castling_parts or "-"
        if self._ep_square >= 0:
            ep_file = "abcdefgh"[_file(self._ep_square)]
            ep_rank = str(_rank(self._ep_square) + 1)
            ep_str = ep_file + ep_rank
        else:
            ep_str = "-"
        return f"{fen_board} {side} {castling_str} {ep_str} {self._halfmove} {self._fullmove}"


# ---------------------------------------------------------------------------
# Board mutation helper
# ---------------------------------------------------------------------------

def _apply_move_to_board(board: np.ndarray, move: Move, ep_square: int) -> None:
    """Mutate `board` in-place to reflect `move`.

    Handles: normal moves, captures, en-passant, castling, and promotions.
    `ep_square` is the en-passant target square *before* the move is applied.
    """
    from_sq = move.from_sq
    to_sq = move.to_sq
    mover = int(board[from_sq])
    mover_abs = abs(mover)
    color_sign = 1 if mover > 0 else -1  # +1 for white, -1 for black

    board[from_sq] = _EMPTY

    # En-passant capture
    if mover_abs == _PAWN and to_sq == ep_square and ep_square >= 0:
        # Remove the captured pawn (it sits one rank behind the to_sq)
        captured_rank = _rank(to_sq) - color_sign   # white moves up (+1), black down (-1)
        board[_sq(captured_rank, _file(to_sq))] = _EMPTY

    # Castling: also move the rook
    if mover_abs == _KING:
        file_diff = _file(to_sq) - _file(from_sq)
        if abs(file_diff) == 2:
            rank = _rank(from_sq)
            if file_diff > 0:
                # Kingside
                board[_sq(rank, 7)] = _EMPTY
                board[_sq(rank, 5)] = _ROOK * color_sign
            else:
                # Queenside
                board[_sq(rank, 0)] = _EMPTY
                board[_sq(rank, 3)] = _ROOK * color_sign

    # Place piece at destination (handle promotions)
    if move.promotion is not None:
        internal = _PROMO_BASE_TO_INTERNAL[move.promotion]
        board[to_sq] = internal * color_sign
    else:
        board[to_sq] = mover


# ---------------------------------------------------------------------------
# Pseudo-move generation
# ---------------------------------------------------------------------------

def _generate_pseudo_moves(
    board: np.ndarray,
    side: int,
    castling: int,
    ep_square: int,
) -> List[Move]:
    """Generate all pseudo-legal moves (may leave own king in check)."""
    moves: List[Move] = []
    sign = 1 if side == WHITE else -1  # positive for own pieces

    for sq in range(64):
        piece = int(board[sq])
        if piece * sign <= 0:
            continue  # empty or opponent piece

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
    board: np.ndarray,
    sq: int,
    side: int,
    ep_square: int,
    moves: List[Move],
) -> None:
    rank = _rank(sq)
    file = _file(sq)
    forward = 1 if side == WHITE else -1
    start_rank = 1 if side == WHITE else 6
    promo_rank = 6 if side == WHITE else 1   # rank from which next step is promotion

    new_rank = rank + forward

    # Single push
    if 0 <= new_rank < 8:
        to_sq = _sq(new_rank, file)
        if board[to_sq] == _EMPTY:
            if rank == promo_rank:
                for pt in _PROMO_INTERNALS:
                    moves.append(Move(sq, to_sq, promotion=_PROMO_INTERNAL_TO_BASE[pt]))
            else:
                moves.append(Move(sq, to_sq))
                # Double push from starting rank
                if rank == start_rank:
                    to_sq2 = _sq(new_rank + forward, file)
                    if board[to_sq2] == _EMPTY:
                        moves.append(Move(sq, to_sq2))

    # Captures
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
    board: np.ndarray,
    sq: int,
    attack_bb: int,
    sign: int,
    moves: List[Move],
) -> None:
    bb = attack_bb
    while bb:
        lsb = bb & (-bb)
        to_sq = lsb.bit_length() - 1
        target = int(board[to_sq])
        if target * sign <= 0:  # empty or opponent
            moves.append(Move(sq, to_sq))
        bb ^= lsb


def _gen_slider_moves(
    board: np.ndarray,
    sq: int,
    directions: List[Tuple[int, int]],
    sign: int,
    moves: List[Move],
) -> None:
    for dr, df in directions:
        for to_sq in _RAYS[(dr, df)][sq]:
            target = int(board[to_sq])
            if target * sign > 0:
                break  # own piece blocks
            moves.append(Move(sq, to_sq))
            if target != _EMPTY:
                break  # captured opponent piece; can't go further


def _gen_castling_moves(
    board: np.ndarray,
    king_sq: int,
    side: int,
    castling: int,
    moves: List[Move],
) -> None:
    """Add kingside and queenside castling moves if rights and path are clear."""
    rank = 0 if side == WHITE else 7
    opponent = 1 - side

    # Kingside: bits 0 (WK) and 2 (BK)
    ks_bit = 0b0001 if side == WHITE else 0b0100
    if castling & ks_bit:
        # f and g files must be empty
        if board[_sq(rank, 5)] == _EMPTY and board[_sq(rank, 6)] == _EMPTY:
            # King must not pass through check: e, f, g squares
            if not (_square_attacked(board, _sq(rank, 4), opponent)
                    or _square_attacked(board, _sq(rank, 5), opponent)
                    or _square_attacked(board, _sq(rank, 6), opponent)):
                moves.append(Move(_sq(rank, 4), _sq(rank, 6)))

    # Queenside: bits 1 (WQ) and 3 (BQ)
    qs_bit = 0b0010 if side == WHITE else 0b1000
    if castling & qs_bit:
        # b, c, d files must be empty
        if (board[_sq(rank, 1)] == _EMPTY
                and board[_sq(rank, 2)] == _EMPTY
                and board[_sq(rank, 3)] == _EMPTY):
            # King must not pass through check: e, d, c squares
            if not (_square_attacked(board, _sq(rank, 4), opponent)
                    or _square_attacked(board, _sq(rank, 3), opponent)
                    or _square_attacked(board, _sq(rank, 2), opponent)):
                moves.append(Move(_sq(rank, 4), _sq(rank, 2)))


# ---------------------------------------------------------------------------
# Attack detection
# ---------------------------------------------------------------------------

def _square_attacked(board: np.ndarray, sq: int, by_side: int) -> bool:
    """Return True if `sq` is attacked by any piece belonging to `by_side`."""
    sign = 1 if by_side == WHITE else -1  # sign of attacking pieces on board

    # Pawn attacks
    pawn_attacks = _PAWN_ATTACKS_WHITE[sq] if by_side == BLACK else _PAWN_ATTACKS_BLACK[sq]
    # "Pawn attacks from sq" when reversed: we look at squares that a pawn *on sq*
    # would capture from the perspective of the *attacker*.
    # Equivalently: if a white pawn on sq would attack a square X, and the attacker
    # is black, then a black pawn on X attacks sq.
    # Correct approach: use the opponent's pawn attack table indexed by sq.
    # White pawns attack upward; black attack downward.
    # A white pawn on X attacks sq means sq is in _PAWN_ATTACKS_WHITE[X].
    # So we check _PAWN_ATTACKS_BLACK[sq] for white attackers (those squares
    # that have white pawns that can reach sq) -- wait, let me be precise:
    #
    # _PAWN_ATTACKS_WHITE[X] = set of squares attacked by a WHITE pawn at X.
    # We want: does any white pawn Y attack sq?
    # That is: sq in _PAWN_ATTACKS_WHITE[Y] for some Y.
    # Equivalently: Y in _PAWN_ATTACKS_BLACK[sq]   (reverse attack bitmap).
    # So for WHITE attackers, check _PAWN_ATTACKS_BLACK[sq].
    # For BLACK attackers, check _PAWN_ATTACKS_WHITE[sq].
    attacker_pawn_bb = _PAWN_ATTACKS_BLACK[sq] if by_side == WHITE else _PAWN_ATTACKS_WHITE[sq]
    bb = attacker_pawn_bb
    while bb:
        lsb = bb & (-bb)
        y = lsb.bit_length() - 1
        if int(board[y]) == _PAWN * sign:
            return True
        bb ^= lsb

    # Knight attacks
    bb = _KNIGHT_ATTACKS[sq]
    while bb:
        lsb = bb & (-bb)
        y = lsb.bit_length() - 1
        if int(board[y]) == _KNIGHT * sign:
            return True
        bb ^= lsb

    # King attacks
    bb = _KING_ATTACKS[sq]
    while bb:
        lsb = bb & (-bb)
        y = lsb.bit_length() - 1
        if int(board[y]) == _KING * sign:
            return True
        bb ^= lsb

    # Sliding pieces
    for dr, df in _ROOK_DIRS:
        for y in _RAYS[(dr, df)][sq]:
            v = int(board[y])
            if v != _EMPTY:
                if v == _ROOK * sign or v == _QUEEN * sign:
                    return True
                break  # blocked

    for dr, df in _BISHOP_DIRS:
        for y in _RAYS[(dr, df)][sq]:
            v = int(board[y])
            if v != _EMPTY:
                if v == _BISHOP * sign or v == _QUEEN * sign:
                    return True
                break  # blocked

    return False


# ---------------------------------------------------------------------------
# Zobrist hash computation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Insufficient material detection
# ---------------------------------------------------------------------------

def _insufficient_material(board: np.ndarray) -> bool:
    """Return True if neither side can force checkmate (insufficient material)."""
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

    # Any pawn, rook, or queen present -> sufficient material
    for pieces in (white_pieces, black_pieces):
        if _PAWN in pieces or _ROOK in pieces or _QUEEN in pieces:
            return False

    # Count minor pieces
    def minor_count(pieces: Dict[int, int]) -> int:
        return pieces.get(_KNIGHT, 0) + pieces.get(_BISHOP, 0)

    wm = minor_count(white_pieces)
    bm = minor_count(black_pieces)

    # KvK, KNvK, KBvK, KvKN, KvKB
    if wm <= 1 and bm == 0:
        return True
    if bm <= 1 and wm == 0:
        return True

    # KBvKB with bishops on same color
    if (wm == 1 and bm == 1
            and _BISHOP in white_pieces and _BISHOP in black_pieces):
        w_bishop_sq = next(sq for sq in range(64) if board[sq] == _BISHOP)
        b_bishop_sq = next(sq for sq in range(64) if board[sq] == -_BISHOP)
        if (_rank(w_bishop_sq) + _file(w_bishop_sq)) % 2 == (
                _rank(b_bishop_sq) + _file(b_bishop_sq)) % 2:
            return True

    return False


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def initial_state() -> ChessState:
    """Return the standard starting chess position."""
    board = _make_initial_board()
    # All castling rights granted: WK | WQ | BK | BQ
    castling = 0b1111
    ep_square = -1
    hash_val = _compute_hash(board, WHITE, castling, ep_square)
    return ChessState(
        board=board,
        side=WHITE,
        castling=castling,
        ep_square=ep_square,
        halfmove=0,
        fullmove=1,
        zobrist=hash_val,
        history=(hash_val,),
        king_sqs=(_sq(0, 4), _sq(7, 4)),
    )


# ---------------------------------------------------------------------------
# FEN parser (convenience helper for testing / debugging)
# ---------------------------------------------------------------------------

def from_fen(fen: str) -> ChessState:
    """Parse a FEN string and return a ChessState.

    Raises:
        ValueError: If the FEN string is malformed.
    """
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
