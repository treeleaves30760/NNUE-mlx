"""Mini Shogi constants, piece tables, board geometry, and Zobrist helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.games.base import BLACK, WHITE, GameConfig

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


def _piece_color(val: int):
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
# Zobrist update helpers
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
