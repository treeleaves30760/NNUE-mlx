"""Mini Shogi (5x5 Shogi) game engine.

Implements the GameState interface for the 5x5 variant of Shogi known as
Gogo Shogi (5五将棋). Each side has 6 pieces: King, Rook, Bishop, Gold,
Silver, and Pawn. Captured pieces can be dropped back onto the board.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BLACK, WHITE, GameConfig, GameState, Move

# ---------------------------------------------------------------------------
# Board / piece constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 5
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE

# Piece type values stored in the board array (positive = sente/WHITE,
# negative = gote/BLACK, zero = empty square).
EMPTY = 0
PAWN_VAL = 1
SILVER_VAL = 2
GOLD_VAL = 3
BISHOP_VAL = 4
ROOK_VAL = 5
KING_VAL = 6
TOKIN_VAL = 7       # promoted pawn
PRO_SILVER_VAL = 8  # promoted silver
HORSE_VAL = 9       # promoted bishop
DRAGON_VAL = 10     # promoted rook

# Maximum board value used for array sizing.
MAX_PIECE_VAL = DRAGON_VAL

# Piece-type indices used in the GameState interface (pieces_on_board /
# hand_pieces). King is implicit; only non-king piece types appear here.
PAWN = 0
SILVER = 1
GOLD = 2
BISHOP = 3
ROOK = 4
PROMOTED_PAWN = 5
PROMOTED_SILVER = 6
PROMOTED_BISHOP = 7
PROMOTED_ROOK = 8

# Number of non-king piece types (used for NNUE feature sizing).
NUM_PIECE_TYPES = 9  # PAWN..PROMOTED_ROOK

# Pieces that can live in a player's hand (always unpromoted).
HAND_PIECE_TYPES = (PAWN, SILVER, GOLD, BISHOP, ROOK)

# Map from board array value -> interface piece-type index.
_BOARD_VAL_TO_PIECE_TYPE: Dict[int, int] = {
    PAWN_VAL: PAWN,
    SILVER_VAL: SILVER,
    GOLD_VAL: GOLD,
    BISHOP_VAL: BISHOP,
    ROOK_VAL: ROOK,
    TOKIN_VAL: PROMOTED_PAWN,
    PRO_SILVER_VAL: PROMOTED_SILVER,
    HORSE_VAL: PROMOTED_BISHOP,
    DRAGON_VAL: PROMOTED_ROOK,
}

# Map from interface piece-type index -> board array value.
_PIECE_TYPE_TO_BOARD_VAL: Dict[int, int] = {v: k for k, v in _BOARD_VAL_TO_PIECE_TYPE.items()}

# Map from board array value -> demoted (unpromoted) board array value.
_DEMOTE: Dict[int, int] = {
    TOKIN_VAL: PAWN_VAL,
    PRO_SILVER_VAL: SILVER_VAL,
    HORSE_VAL: BISHOP_VAL,
    DRAGON_VAL: ROOK_VAL,
}

# Board array values that can promote and their promoted counterparts.
_PROMOTE: Dict[int, int] = {v: k for k, v in _DEMOTE.items()}
# Gold and King cannot promote.

# ---------------------------------------------------------------------------
# Game configuration
# ---------------------------------------------------------------------------

_CONFIG = GameConfig(
    name="minishogi",
    board_height=BOARD_SIZE,
    board_width=BOARD_SIZE,
    num_piece_types=NUM_PIECE_TYPES,
    has_drops=True,
    has_promotion=True,
)

# ---------------------------------------------------------------------------
# Promotion zones
# Row indices where pieces enter the promotion zone.
# Sente (WHITE, moves toward rank 0) promotes on rank 0.
# Gote  (BLACK, moves toward rank 4) promotes on rank 4.
# ---------------------------------------------------------------------------

_PROMO_RANK = {WHITE: 0, BLACK: BOARD_SIZE - 1}

# ---------------------------------------------------------------------------
# Piece movement tables
# ---------------------------------------------------------------------------

# Direction offsets as (drank, dfile).  Positive drank = toward larger rank
# index (toward gote side for a sente piece, i.e. backward for sente).
# "Forward" for sente is drank = -1 (toward rank 0).
# "Forward" for gote  is drank = +1 (toward rank 4).

_ALL_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Gold moves: forward, forward-diag x2, sideways x2, backward straight.
# For sente: forward = (-1,0); diagonals = (-1,-1), (-1,1); sides = (0,-1),(0,1); back = (1,0).
_GOLD_DIRS_SENTE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]
_GOLD_DIRS_GOTE  = [( 1, -1), ( 1, 0), ( 1, 1), (0, -1), (0, 1), (-1, 0)]

# Silver moves: all 4 diagonals + 1 forward straight.
_SILVER_DIRS_SENTE = [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 1)]
_SILVER_DIRS_GOTE  = [( 1, -1), ( 1, 0), ( 1, 1), (-1, -1), (-1, 1)]

# Pawn moves: one square forward only.
_PAWN_DIR_SENTE = [(-1, 0)]
_PAWN_DIR_GOTE  = [( 1, 0)]

# Slider directions.
_BISHOP_SLIDERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ROOK_SLIDERS   = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Extra one-step additions for promoted sliders.
_HORSE_EXTRA  = _ROOK_SLIDERS   # promoted bishop gets orthogonal 1-step
_DRAGON_EXTRA = _BISHOP_SLIDERS  # promoted rook gets diagonal 1-step

# ---------------------------------------------------------------------------
# Zobrist hash tables (fixed seed for reproducibility)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(seed=20240101)

# board_zobrist[square][piece_encoding]  where piece_encoding = piece_value + 10
# (shift so that negative gote pieces become non-negative indices 0..9, and
# positive sente pieces become 11..20; index 10 = empty, not used in XOR).
_PIECE_OFFSET = MAX_PIECE_VAL  # shift: board_val + _PIECE_OFFSET -> table index
_ZOBRIST_BOARD = _rng.integers(
    low=0, high=2**63, size=(NUM_SQUARES, MAX_PIECE_VAL * 2 + 1), dtype=np.int64
)

# hand_zobrist[side][piece_type_idx][count]  counts 0..3 per piece type
_MAX_HAND_COUNT = 4
_ZOBRIST_HAND = _rng.integers(
    low=0, high=2**63,
    size=(2, len(HAND_PIECE_TYPES), _MAX_HAND_COUNT),
    dtype=np.int64,
)

# Side-to-move hash (XOR in when gote/BLACK is to move).
_ZOBRIST_BLACK_TO_MOVE = int(_rng.integers(low=0, high=2**63, dtype=np.int64))

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sq(rank: int, file: int) -> int:
    """Convert (rank, file) coordinates to a flat square index."""
    return rank * BOARD_SIZE + file


def _rank(sq: int) -> int:
    return sq // BOARD_SIZE


def _file(sq: int) -> int:
    return sq % BOARD_SIZE


def _on_board(rank: int, file: int) -> bool:
    return 0 <= rank < BOARD_SIZE and 0 <= file < BOARD_SIZE


def _sign(side: int) -> int:
    """Return +1 for sente (WHITE), -1 for gote (BLACK)."""
    return 1 if side == WHITE else -1


def _piece_color(val: int) -> Optional[int]:
    """Return WHITE/BLACK/None for a board array value."""
    if val > 0:
        return WHITE
    if val < 0:
        return BLACK
    return None


def _abs_val(val: int) -> int:
    """Absolute piece type value (strip color)."""
    return abs(val)


# ---------------------------------------------------------------------------
# Initial board setup
# ---------------------------------------------------------------------------
#
# Gote (row 0):  K  G  S  B  R    files 0..4
# Gote (row 1):  .  .  .  .  P
# Empty row 2:   .  .  .  .  .
# Sente (row 3): P  .  .  .  .
# Sente (row 4): R  B  S  G  K    files 0..4
#
# Gote pieces are stored as negative values.

def _make_initial_board() -> np.ndarray:
    board = np.zeros(NUM_SQUARES, dtype=np.int8)
    # Gote (BLACK, negative) back rank at row 0: K G S B R
    board[_sq(0, 0)] = -KING_VAL
    board[_sq(0, 1)] = -GOLD_VAL
    board[_sq(0, 2)] = -SILVER_VAL
    board[_sq(0, 3)] = -BISHOP_VAL
    board[_sq(0, 4)] = -ROOK_VAL
    # Gote pawn at row 1 file 4
    board[_sq(1, 4)] = -PAWN_VAL
    # Sente (WHITE, positive) pawn at row 3 file 0
    board[_sq(3, 0)] = PAWN_VAL
    # Sente back rank at row 4: R B S G K
    board[_sq(4, 0)] = ROOK_VAL
    board[_sq(4, 1)] = BISHOP_VAL
    board[_sq(4, 2)] = SILVER_VAL
    board[_sq(4, 3)] = GOLD_VAL
    board[_sq(4, 4)] = KING_VAL
    return board


_INITIAL_BOARD = _make_initial_board()

# Empty hand: {piece_type: 0} for PAWN..ROOK
_EMPTY_HAND: Dict[int, int] = {pt: 0 for pt in HAND_PIECE_TYPES}


# ---------------------------------------------------------------------------
# Move generation helpers
# ---------------------------------------------------------------------------

def _generate_pseudolegal_moves(
    board: np.ndarray,
    side: int,
    hands: Tuple[Dict[int, int], Dict[int, int]],
) -> List[Move]:
    """Generate all pseudo-legal moves for *side* (before legality filtering).

    A pseudo-legal move is one that obeys the piece movement rules but may
    leave the moving side's king in check.
    """
    sign = _sign(side)
    moves: List[Move] = []

    for sq in range(NUM_SQUARES):
        val = int(board[sq])
        if val == 0 or _piece_color(val) != side:
            continue
        abs_v = abs(val)
        rank = _rank(sq)
        file = _file(sq)
        promo_rank = _PROMO_RANK[side]

        # ---- Step movers ------------------------------------------------
        if abs_v == KING_VAL:
            dirs = _ALL_8
            for dr, df in dirs:
                nr, nf = rank + dr, file + df
                if not _on_board(nr, nf):
                    continue
                target = int(board[_sq(nr, nf)])
                if _piece_color(target) == side:
                    continue  # own piece blocks
                moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf)))

        elif abs_v == GOLD_VAL or abs_v in (TOKIN_VAL, PRO_SILVER_VAL):
            # Gold and promoted pieces that move like gold.
            dirs = _GOLD_DIRS_SENTE if side == WHITE else _GOLD_DIRS_GOTE
            for dr, df in dirs:
                nr, nf = rank + dr, file + df
                if not _on_board(nr, nf):
                    continue
                target = int(board[_sq(nr, nf)])
                if _piece_color(target) == side:
                    continue
                moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf)))

        elif abs_v == SILVER_VAL:
            dirs = _SILVER_DIRS_SENTE if side == WHITE else _SILVER_DIRS_GOTE
            for dr, df in dirs:
                nr, nf = rank + dr, file + df
                if not _on_board(nr, nf):
                    continue
                target = int(board[_sq(nr, nf)])
                if _piece_color(target) == side:
                    continue
                # Optional promotion: piece moves into or out of promo zone.
                dest_rank = nr
                in_promo = (rank == promo_rank or dest_rank == promo_rank)
                if in_promo and abs_v in _PROMOTE:
                    # Offer both promoting and non-promoting versions.
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf),
                                      promotion=_BOARD_VAL_TO_PIECE_TYPE[_PROMOTE[abs_v]]))
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf)))
                else:
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf)))

        elif abs_v == PAWN_VAL:
            dirs = _PAWN_DIR_SENTE if side == WHITE else _PAWN_DIR_GOTE
            for dr, df in dirs:
                nr, nf = rank + dr, file + df
                if not _on_board(nr, nf):
                    continue
                target = int(board[_sq(nr, nf)])
                if _piece_color(target) == side:
                    continue
                dest_rank = nr
                # Must promote if landing on last rank; may promote if crossing
                # into promo zone (same as last rank in mini shogi).
                if dest_rank == promo_rank:
                    # Forced promotion.
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf),
                                      promotion=PROMOTED_PAWN))
                elif rank == promo_rank:
                    # Moving out of promo zone; still offer optional promo.
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf),
                                      promotion=PROMOTED_PAWN))
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf)))
                else:
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf)))

        # ---- Slider movers -----------------------------------------------
        elif abs_v == BISHOP_VAL:
            _add_slider_moves(board, side, sq, rank, file, _BISHOP_SLIDERS,
                              abs_v, promo_rank, moves)

        elif abs_v == ROOK_VAL:
            _add_slider_moves(board, side, sq, rank, file, _ROOK_SLIDERS,
                              abs_v, promo_rank, moves)

        elif abs_v == HORSE_VAL:
            # Promoted bishop: diagonal slides + orthogonal 1-step.
            _add_slider_moves(board, side, sq, rank, file, _BISHOP_SLIDERS,
                              abs_v, promo_rank, moves, no_promote=True)
            for dr, df in _HORSE_EXTRA:
                nr, nf = rank + dr, file + df
                if not _on_board(nr, nf):
                    continue
                target = int(board[_sq(nr, nf)])
                if _piece_color(target) != side:
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf)))

        elif abs_v == DRAGON_VAL:
            # Promoted rook: orthogonal slides + diagonal 1-step.
            _add_slider_moves(board, side, sq, rank, file, _ROOK_SLIDERS,
                              abs_v, promo_rank, moves, no_promote=True)
            for dr, df in _DRAGON_EXTRA:
                nr, nf = rank + dr, file + df
                if not _on_board(nr, nf):
                    continue
                target = int(board[_sq(nr, nf)])
                if _piece_color(target) != side:
                    moves.append(Move(from_sq=sq, to_sq=_sq(nr, nf)))

    # ---- Drop moves -------------------------------------------------------
    my_hand = hands[side]
    for pt, cnt in my_hand.items():
        if cnt == 0:
            continue
        board_val = _PIECE_TYPE_TO_BOARD_VAL[pt]
        for sq in range(NUM_SQUARES):
            if board[sq] != 0:
                continue  # must drop on empty square
            r = _rank(sq)
            f = _file(sq)
            # Pawn cannot be dropped on last rank (no legal moves from there).
            if board_val == PAWN_VAL and r == promo_rank:
                continue
            # Nifu: cannot drop a pawn on a file that already has an unpromoted
            # own pawn.
            if board_val == PAWN_VAL and _nifu(board, side, f):
                continue
            moves.append(Move(from_sq=None, to_sq=sq, drop_piece=pt))

    return moves


def _add_slider_moves(
    board: np.ndarray,
    side: int,
    sq: int,
    rank: int,
    file: int,
    directions: List[Tuple[int, int]],
    abs_v: int,
    promo_rank: int,
    moves: List[Move],
    no_promote: bool = False,
) -> None:
    """Append all slider moves in given directions to the moves list."""
    for dr, df in directions:
        nr, nf = rank + dr, file + df
        while _on_board(nr, nf):
            target_sq = _sq(nr, nf)
            target = int(board[target_sq])
            if _piece_color(target) == side:
                break  # own piece blocks
            can_promo = (not no_promote) and (abs_v in _PROMOTE)
            in_promo = can_promo and (rank == promo_rank or nr == promo_rank)
            if in_promo:
                moves.append(Move(from_sq=sq, to_sq=target_sq,
                                  promotion=_BOARD_VAL_TO_PIECE_TYPE[_PROMOTE[abs_v]]))
                moves.append(Move(from_sq=sq, to_sq=target_sq))
            else:
                moves.append(Move(from_sq=sq, to_sq=target_sq))
            if target != 0:
                break  # capture stops sliding
            nr += dr
            nf += df


def _nifu(board: np.ndarray, side: int, file: int) -> bool:
    """Return True if side already has an unpromoted pawn on the given file."""
    sign = _sign(side)
    pawn_val = sign * PAWN_VAL
    for rank in range(BOARD_SIZE):
        if board[_sq(rank, file)] == pawn_val:
            return True
    return False


# ---------------------------------------------------------------------------
# Check detection
# ---------------------------------------------------------------------------

def _is_in_check(board: np.ndarray, side: int) -> bool:
    """Return True if *side*'s king is under attack."""
    # Find king square.
    king_val = _sign(side) * KING_VAL
    king_sq = -1
    for sq in range(NUM_SQUARES):
        if board[sq] == king_val:
            king_sq = sq
            break
    if king_sq == -1:
        return False  # no king found (shouldn't happen)
    return _is_square_attacked(board, king_sq, side)


def _is_square_attacked(board: np.ndarray, sq: int, defending_side: int) -> bool:
    """Return True if *sq* is attacked by the opponent of *defending_side*."""
    attacker_side = 1 - defending_side
    attacker_sign = _sign(attacker_side)
    rank = _rank(sq)
    file = _file(sq)

    # --- Check for pawn attacks ---
    # A sente (WHITE) pawn attacks one square forward for sente, i.e. at lower
    # rank. So a gote (BLACK) pawn at (rank+1, file) attacks sq if attacker is BLACK.
    pawn_dr = 1 if attacker_side == WHITE else -1  # from attacker's perspective
    # Actually: an attacker's pawn stands behind sq and attacks forward.
    # Sente pawn attacks the square one rank lower (rank - 1 from pawn's position).
    # So sq is attacked by a sente pawn at (rank+1, file).
    # Gote pawn attacks one rank higher; sq attacked by gote pawn at (rank-1, file).
    pawn_r = rank + (1 if attacker_side == WHITE else -1)
    if _on_board(pawn_r, file):
        v = int(board[_sq(pawn_r, file)])
        if v == attacker_sign * PAWN_VAL:
            return True

    # --- Check for silver attacks ---
    silver_dirs = _SILVER_DIRS_SENTE if attacker_side == WHITE else _SILVER_DIRS_GOTE
    for dr, df in silver_dirs:
        nr, nf = rank + dr, file + df
        if _on_board(nr, nf):
            v = int(board[_sq(nr, nf)])
            if v == attacker_sign * SILVER_VAL:
                return True

    # --- Check for gold / tokin / promoted-silver attacks ---
    gold_dirs = _GOLD_DIRS_SENTE if attacker_side == WHITE else _GOLD_DIRS_GOTE
    for dr, df in gold_dirs:
        nr, nf = rank + dr, file + df
        if _on_board(nr, nf):
            v = int(board[_sq(nr, nf)])
            abs_v = abs(v)
            if _piece_color(v) == attacker_side and abs_v in (GOLD_VAL, TOKIN_VAL, PRO_SILVER_VAL):
                return True

    # --- Check for king attacks ---
    for dr, df in _ALL_8:
        nr, nf = rank + dr, file + df
        if _on_board(nr, nf):
            v = int(board[_sq(nr, nf)])
            if v == attacker_sign * KING_VAL:
                return True

    # --- Check for bishop / horse attacks (diagonal) ---
    for dr, df in _BISHOP_SLIDERS:
        nr, nf = rank + dr, file + df
        while _on_board(nr, nf):
            v = int(board[_sq(nr, nf)])
            if v != 0:
                abs_v = abs(v)
                if _piece_color(v) == attacker_side and abs_v in (BISHOP_VAL, HORSE_VAL):
                    return True
                break  # blocked by any piece
            nr += dr
            nf += df

    # --- Check for rook / dragon attacks (orthogonal) ---
    for dr, df in _ROOK_SLIDERS:
        nr, nf = rank + dr, file + df
        while _on_board(nr, nf):
            v = int(board[_sq(nr, nf)])
            if v != 0:
                abs_v = abs(v)
                if _piece_color(v) == attacker_side and abs_v in (ROOK_VAL, DRAGON_VAL):
                    return True
                break
            nr += dr
            nf += df

    # --- Horse 1-step orthogonal attacks ---
    for dr, df in _HORSE_EXTRA:
        nr, nf = rank + dr, file + df
        if _on_board(nr, nf):
            v = int(board[_sq(nr, nf)])
            if _piece_color(v) == attacker_side and abs(v) == HORSE_VAL:
                return True

    # --- Dragon 1-step diagonal attacks ---
    for dr, df in _DRAGON_EXTRA:
        nr, nf = rank + dr, file + df
        if _on_board(nr, nf):
            v = int(board[_sq(nr, nf)])
            if _piece_color(v) == attacker_side and abs(v) == DRAGON_VAL:
                return True

    return False


# ---------------------------------------------------------------------------
# Zobrist hash computation
# ---------------------------------------------------------------------------

def _compute_zobrist(
    board: np.ndarray,
    hands: Tuple[Dict[int, int], Dict[int, int]],
    side_to_move: int,
) -> int:
    h = 0
    for sq in range(NUM_SQUARES):
        val = int(board[sq])
        if val != 0:
            idx = val + _PIECE_OFFSET
            h ^= int(_ZOBRIST_BOARD[sq, idx])
    for side in (WHITE, BLACK):
        for i, pt in enumerate(HAND_PIECE_TYPES):
            cnt = hands[side].get(pt, 0)
            if cnt > 0:
                h ^= int(_ZOBRIST_HAND[side, i, cnt])
    if side_to_move == BLACK:
        h ^= _ZOBRIST_BLACK_TO_MOVE
    return h


def _update_zobrist_remove(h: int, sq: int, val: int) -> int:
    """XOR out a piece from the hash."""
    idx = val + _PIECE_OFFSET
    return h ^ int(_ZOBRIST_BOARD[sq, idx])


def _update_zobrist_place(h: int, sq: int, val: int) -> int:
    """XOR in a piece into the hash."""
    idx = val + _PIECE_OFFSET
    return h ^ int(_ZOBRIST_BOARD[sq, idx])


def _update_zobrist_hand(h: int, side: int, pt: int, old_cnt: int, new_cnt: int) -> int:
    """Update hash for hand piece count change."""
    i = HAND_PIECE_TYPES.index(pt)
    if old_cnt > 0:
        h ^= int(_ZOBRIST_HAND[side, i, old_cnt])
    if new_cnt > 0:
        h ^= int(_ZOBRIST_HAND[side, i, new_cnt])
    return h


# ---------------------------------------------------------------------------
# MiniShogiState
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # GameState interface
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Utility / display
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

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
