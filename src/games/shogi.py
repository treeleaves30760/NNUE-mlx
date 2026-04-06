"""Shogi (Japanese chess) engine implementing the GameState interface.

Board layout (9x9):
  - Index 0 = square 1a (top-left from sente's view = gote's back rank, file 9)
  - Index 80 = square 9i (bottom-right from sente's view = sente's back rank, file 1)
  - sq = rank * 9 + file_index   (rank 0..8, file_index 0..8)
  - From sente's perspective: rank 0 is gote's side, rank 8 is sente's side

Piece encoding (sign encodes colour):
  positive = sente (WHITE=0), negative = gote (BLACK=1)
  0=empty
  1=Pawn  2=Lance  3=Knight  4=Silver  5=Gold  6=Bishop  7=Rook  8=King
  9=+Pawn(Tokin)  10=+Lance  11=+Knight  12=+Silver  13=+Bishop(Horse)  14=+Rook(Dragon)

Piece-type constants for the public API (0-indexed, king excluded):
  PAWN=0 LANCE=1 KNIGHT=2 SILVER=3 GOLD=4 BISHOP=5 ROOK=6
  PROMOTED_PAWN=7 PROMOTED_LANCE=8 PROMOTED_KNIGHT=9 PROMOTED_SILVER=10
  PROMOTED_BISHOP=11 PROMOTED_ROOK=12
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.games.base import BLACK, WHITE, GameConfig, GameState, Move

# ---------------------------------------------------------------------------
# Public piece-type constants (API surface, king excluded)
# ---------------------------------------------------------------------------
PAWN = 0
LANCE = 1
KNIGHT = 2
SILVER = 3
GOLD = 4
BISHOP = 5
ROOK = 6
PROMOTED_PAWN = 7
PROMOTED_LANCE = 8
PROMOTED_KNIGHT = 9
PROMOTED_SILVER = 10
PROMOTED_BISHOP = 11
PROMOTED_ROOK = 12

# Mapping from API piece-type constant to the internal board encoding value
# (always positive; multiply by -1 for gote)
_API_TO_BOARD: List[int] = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
# Inverse: board value (1..14) -> API piece-type constant
_BOARD_TO_API: List[int] = [
    -1,  # 0 = empty (unused)
    PAWN,           # 1
    LANCE,          # 2
    KNIGHT,         # 3
    SILVER,         # 4
    GOLD,           # 5
    BISHOP,         # 6
    ROOK,           # 7
    -1,             # 8 = King (not in piece-type list)
    PROMOTED_PAWN,  # 9
    PROMOTED_LANCE, # 10
    PROMOTED_KNIGHT,# 11
    PROMOTED_SILVER,# 12
    PROMOTED_BISHOP,# 13
    PROMOTED_ROOK,  # 14
]

# Board values for hand pieces (unpromoted, no king)
_HAND_BOARD_VALUES = [1, 2, 3, 4, 5, 6, 7]  # pawn..rook

# ---------------------------------------------------------------------------
# Zobrist table (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(0xDEADBEEF_CAFEBABE)

# Piece placements: board_value in 1..14, side 0..1, square 0..80
_Z_PIECE = _RNG.integers(
    low=0, high=2**63, size=(15, 2, 81), dtype=np.uint64
)
# Hand pieces: piece_type 0..6 (pawn..rook), side 0..1, count 0..38
_Z_HAND = _RNG.integers(
    low=0, high=2**63, size=(7, 2, 39), dtype=np.uint64
)
# Side to move
_Z_SIDE = _RNG.integers(low=0, high=2**63, size=2, dtype=np.uint64)


# ---------------------------------------------------------------------------
# Static game config
# ---------------------------------------------------------------------------
_CONFIG = GameConfig(
    name="shogi",
    board_height=9,
    board_width=9,
    num_piece_types=13,
    has_drops=True,
    has_promotion=True,
)

# ---------------------------------------------------------------------------
# Board geometry helpers
# ---------------------------------------------------------------------------

def _rank(sq: int) -> int:
    """Return rank (0=top/gote side, 8=bottom/sente side)."""
    return sq // 9


def _file(sq: int) -> int:
    """Return file index (0=left, 8=right from sente's view)."""
    return sq % 9


def _sq(rank: int, file_idx: int) -> int:
    return rank * 9 + file_idx


def _in_bounds(rank: int, file_idx: int) -> bool:
    return 0 <= rank < 9 and 0 <= file_idx < 9


# Promotion zone: the 3 ranks closest to the opponent
# For sente (WHITE=0): ranks 0, 1, 2
# For gote  (BLACK=1): ranks 6, 7, 8
def _in_promo_zone(sq: int, side: int) -> bool:
    r = _rank(sq)
    return r <= 2 if side == WHITE else r >= 6


# ---------------------------------------------------------------------------
# Piece move-generation helpers (return list of (rank, file) deltas or
# sliding directions for each raw board value)
# ---------------------------------------------------------------------------

# One-step deltas for each piece (from the moving side's perspective).
# Directions are expressed as (delta_rank, delta_file) where "forward" for
# sente is rank-decreasing (toward rank 0).
# We store them for sente; for gote we negate delta_rank.

_KING_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_GOLD_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,0)]  # no diagonal-back
_SILVER_DIRS = [(-1,-1),(-1,0),(-1,1),(1,-1),(1,1)]

# Sliding directions
_ROOK_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
_BISHOP_DIRS = [(-1,-1),(-1,1),(1,-1),(1,1)]
_DRAGON_DIRS = _ROOK_DIRS + [(-1,-1),(-1,1),(1,-1),(1,1)]  # rook + one-step bishop
_HORSE_DIRS = _BISHOP_DIRS + [(-1,0),(1,0),(0,-1),(0,1)]   # bishop + one-step rook


def _apply_perspective(dr: int, df: int, side: int) -> Tuple[int, int]:
    """Flip rank direction for gote."""
    return (-dr if side == BLACK else dr, df)


def _one_step_moves(
    board: np.ndarray,
    sq: int,
    side: int,
    deltas: List[Tuple[int, int]],
) -> List[int]:
    """Return list of target squares reachable by one-step moves."""
    r, f = _rank(sq), _file(sq)
    result: List[int] = []
    sign = 1 if side == WHITE else -1
    for dr, df in deltas:
        pr, pf = _apply_perspective(dr, df, side)
        nr, nf = r + pr, f + pf
        if not _in_bounds(nr, nf):
            continue
        target = _sq(nr, nf)
        occupant = board[target]
        # Cannot capture own piece
        if occupant != 0 and (occupant > 0) == (sign > 0):
            continue
        result.append(target)
    return result


def _sliding_moves(
    board: np.ndarray,
    sq: int,
    side: int,
    dirs: List[Tuple[int, int]],
) -> List[int]:
    """Return list of target squares reachable by sliding moves."""
    r, f = _rank(sq), _file(sq)
    result: List[int] = []
    sign = 1 if side == WHITE else -1
    for dr, df in dirs:
        cr, cf = r + dr, f + df
        while _in_bounds(cr, cf):
            target = _sq(cr, cf)
            occupant = board[target]
            if occupant != 0:
                if (occupant > 0) != (sign > 0):
                    result.append(target)  # capture enemy
                break
            result.append(target)
            cr += dr
            cf += df
    return result


def _lance_moves(board: np.ndarray, sq: int, side: int) -> List[int]:
    """Lance slides straight forward (rank-decreasing for sente)."""
    r, f = _rank(sq), _file(sq)
    result: List[int] = []
    sign = 1 if side == WHITE else -1
    dr = -1 if side == WHITE else 1
    cr = r + dr
    while 0 <= cr < 9:
        target = _sq(cr, f)
        occupant = board[target]
        if occupant != 0:
            if (occupant > 0) != (sign > 0):
                result.append(target)
            break
        result.append(target)
        cr += dr
    return result


def _knight_moves(board: np.ndarray, sq: int, side: int) -> List[int]:
    """Knight jumps: 2 forward + 1 sideways."""
    r, f = _rank(sq), _file(sq)
    result: List[int] = []
    sign = 1 if side == WHITE else -1
    dr = -2 if side == WHITE else 2
    for df in (-1, 1):
        nr, nf = r + dr, f + df
        if not _in_bounds(nr, nf):
            continue
        target = _sq(nr, nf)
        occupant = board[target]
        if occupant != 0 and (occupant > 0) == (sign > 0):
            continue
        result.append(target)
    return result


def _raw_targets(board: np.ndarray, sq: int, side: int, piece_val: int) -> List[int]:
    """Return all squares the piece on sq can move to (ignoring check).

    piece_val is the absolute board encoding (1..14).
    """
    match piece_val:
        case 1:   # Pawn - one step forward
            dr = -1 if side == WHITE else 1
            r, f = _rank(sq), _file(sq)
            nr = r + dr
            if not _in_bounds(nr, f):
                return []
            target = _sq(nr, f)
            sign = 1 if side == WHITE else -1
            occupant = board[target]
            if occupant != 0 and (occupant > 0) == (sign > 0):
                return []
            return [target]
        case 2:   # Lance
            return _lance_moves(board, sq, side)
        case 3:   # Knight
            return _knight_moves(board, sq, side)
        case 4:   # Silver
            return _one_step_moves(board, sq, side, _SILVER_DIRS)
        case 5:   # Gold
            return _one_step_moves(board, sq, side, _GOLD_DIRS)
        case 6:   # Bishop
            return _sliding_moves(board, sq, side, _BISHOP_DIRS)
        case 7:   # Rook
            return _sliding_moves(board, sq, side, _ROOK_DIRS)
        case 8:   # King
            return _one_step_moves(board, sq, side, _KING_DIRS)
        case 9 | 10 | 11 | 12:  # +Pawn, +Lance, +Knight, +Silver -> move like gold
            return _one_step_moves(board, sq, side, _GOLD_DIRS)
        case 13:  # +Bishop (Horse) -> bishop slides + one-step orthogonal
            return (
                _sliding_moves(board, sq, side, _BISHOP_DIRS)
                + _one_step_moves(board, sq, side, _ROOK_DIRS)
            )
        case 14:  # +Rook (Dragon) -> rook slides + one-step diagonal
            return (
                _sliding_moves(board, sq, side, _ROOK_DIRS)
                + _one_step_moves(board, sq, side, _BISHOP_DIRS)
            )
        case _:
            return []


# ---------------------------------------------------------------------------
# Promotion helpers
# ---------------------------------------------------------------------------

# Pieces that can promote (board values 1..7)
_CAN_PROMOTE = {1, 2, 3, 4, 6, 7}  # pawn, lance, knight, silver, bishop, rook
# Promoted version mapping (board value -> promoted board value)
_PROMOTE_TO = {1: 9, 2: 10, 3: 11, 4: 12, 6: 13, 7: 14}

# Pieces that MUST promote to avoid being stranded
# pawn and lance cannot exist on rank 0 (sente) / rank 8 (gote)
# knight cannot exist on rank 0..1 (sente) / rank 7..8 (gote)
def _must_promote(piece_val: int, to_sq: int, side: int) -> bool:
    """Return True if the piece must promote when landing on to_sq."""
    r = _rank(to_sq)
    if side == WHITE:
        if piece_val == 1 or piece_val == 2:  # pawn, lance
            return r == 0
        if piece_val == 3:  # knight
            return r <= 1
    else:  # BLACK / gote
        if piece_val == 1 or piece_val == 2:
            return r == 8
        if piece_val == 3:
            return r >= 7
    return False


def _promotion_possible(piece_val: int, from_sq: int, to_sq: int, side: int) -> bool:
    """Return True if promotion is possible for this move (piece can promote and
    either from or to square is in the promotion zone)."""
    if piece_val not in _CAN_PROMOTE:
        return False
    return _in_promo_zone(from_sq, side) or _in_promo_zone(to_sq, side)


# ---------------------------------------------------------------------------
# Check detection
# ---------------------------------------------------------------------------

def _king_sq(board: np.ndarray, side: int) -> int:
    """Find the king square for the given side."""
    king_val = 8 if side == WHITE else -8
    idx = int(np.argmax(board == king_val))
    return idx


try:
    from src.accel import is_square_attacked_shogi as _c_is_sq_attacked
except ImportError:
    _c_is_sq_attacked = None


def _is_square_attacked(board: np.ndarray, sq: int, by_side: int) -> bool:
    """Return True if sq is attacked by any piece of by_side."""
    if _c_is_sq_attacked is not None:
        return _c_is_sq_attacked(bytes(board), sq, by_side)

    # Python fallback
    sign = 1 if by_side == WHITE else -1
    for s in range(81):
        p = board[s]
        if p == 0:
            continue
        if (p > 0) != (sign > 0):
            continue
        pv = abs(p)
        if pv == 8:
            continue  # handled separately to avoid recursion
        targets = _raw_targets(board, s, by_side, pv)
        if sq in targets:
            return True
    # Also check king attacks
    ksq = _king_sq(board, by_side)
    r, f = _rank(sq), _file(sq)
    kr, kf = _rank(ksq), _file(ksq)
    if max(abs(r - kr), abs(f - kf)) == 1:
        return True
    return False


def _is_in_check(board: np.ndarray, side: int) -> bool:
    """Return True if 'side' is in check."""
    ksq = _king_sq(board, side)
    opponent = BLACK if side == WHITE else WHITE
    return _is_square_attacked(board, ksq, opponent)


# ---------------------------------------------------------------------------
# Pseudo-move expansion (before legality filtering)
# ---------------------------------------------------------------------------

def _expand_board_moves(
    board: np.ndarray, side: int
) -> List[Move]:
    """Generate all pseudo-legal board moves (no captures of own pieces,
    but does not check for leaving king in check).
    Includes both promoted and non-promoted versions where applicable."""
    moves: List[Move] = []
    sign = 1 if side == WHITE else -1
    for sq in range(81):
        p = board[sq]
        if p == 0 or (p > 0) != (sign > 0):
            continue
        pv = abs(p)
        if pv == 8:
            # King moves
            for tsq in _one_step_moves(board, sq, side, _KING_DIRS):
                moves.append(Move(from_sq=sq, to_sq=tsq))
            continue
        targets = _raw_targets(board, sq, side, pv)
        for tsq in targets:
            must_promo = _must_promote(pv, tsq, side)
            can_promo = _promotion_possible(pv, sq, tsq, side)
            if must_promo:
                # Only promoted version
                moves.append(Move(from_sq=sq, to_sq=tsq, promotion=_PROMOTE_TO[pv]))
            elif can_promo:
                # Both versions
                moves.append(Move(from_sq=sq, to_sq=tsq, promotion=_PROMOTE_TO[pv]))
                moves.append(Move(from_sq=sq, to_sq=tsq))
            else:
                moves.append(Move(from_sq=sq, to_sq=tsq))
    return moves


def _expand_drop_moves(
    board: np.ndarray,
    hand: Dict[int, int],
    side: int,
) -> List[Move]:
    """Generate all pseudo-legal drop moves for side.

    Restrictions enforced here:
    - Only drop on empty squares
    - Cannot drop where piece has no legal moves (pawn/lance on last rank,
      knight on last 2 ranks)
    - Nifu: cannot drop a pawn on a file that already has an unpromoted pawn
      of the same side
    Note: uchifuzume (pawn drop checkmate) is checked separately during
    legal-move filtering.
    """
    moves: List[Move] = []
    sign = 1 if side == WHITE else -1

    # Pre-compute files that already have a sente/gote pawn (for nifu)
    pawn_files: set = set()
    for sq in range(81):
        p = board[sq]
        if (p > 0) == (sign > 0) and abs(p) == 1:
            pawn_files.add(_file(sq))

    for pt in range(7):  # PAWN..ROOK
        if hand.get(pt, 0) == 0:
            continue
        board_val = pt + 1  # API piece type 0..6 maps to board value 1..7
        for sq in range(81):
            if board[sq] != 0:
                continue
            r = _rank(sq)
            f = _file(sq)
            # Pawn / Lance cannot land on last rank
            if board_val in (1, 2):
                if side == WHITE and r == 0:
                    continue
                if side == BLACK and r == 8:
                    continue
            # Knight cannot land on last 2 ranks
            if board_val == 3:
                if side == WHITE and r <= 1:
                    continue
                if side == BLACK and r >= 7:
                    continue
            # Nifu
            if board_val == 1 and f in pawn_files:
                continue
            moves.append(Move(from_sq=None, to_sq=sq, drop_piece=pt))
    return moves


# ---------------------------------------------------------------------------
# Apply a move to produce a new board + hand state
# ---------------------------------------------------------------------------

def _apply_move(
    board: np.ndarray,
    sente_hand: Dict[int, int],
    gote_hand: Dict[int, int],
    move: Move,
    side: int,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """Return (new_board, new_sente_hand, new_gote_hand) after applying move.

    Does not mutate the inputs.
    """
    new_board = board.copy()
    sh = dict(sente_hand)
    gh = dict(gote_hand)
    sign = 1 if side == WHITE else -1
    hand = sh if side == WHITE else gh

    if move.from_sq is None:
        # Drop move
        pt = move.drop_piece  # API piece type 0..6
        board_val = pt + 1
        new_board[move.to_sq] = sign * board_val
        hand[pt] = hand.get(pt, 0) - 1
        if hand[pt] == 0:
            del hand[pt]
    else:
        # Board move
        captured = new_board[move.to_sq]
        piece = new_board[move.from_sq]
        pv = abs(piece)

        # Determine what the piece becomes
        if move.promotion is not None:
            new_piece_val = move.promotion
        else:
            new_piece_val = pv

        new_board[move.from_sq] = 0
        new_board[move.to_sq] = sign * new_piece_val

        # If a piece was captured, add its unpromoted version to hand
        if captured != 0:
            cv = abs(captured)
            # Demote if promoted; kings (cv=8) return 0 and are not added to hand
            demoted = _demote(cv)
            if demoted != 0:
                # Captured piece goes to the capturing side's hand
                own_hand = sh if side == WHITE else gh
                api_pt = _BOARD_TO_API[demoted]
                own_hand[api_pt] = own_hand.get(api_pt, 0) + 1

    return new_board, sh, gh


def _demote(board_val: int) -> int:
    """Return the unpromoted board value (1..7) for a given board value.

    Returns 0 for the king (board_val=8); kings cannot go to hand.
    """
    if board_val == 8:
        return 0  # king never goes to hand
    if board_val <= 7:
        return board_val
    # 9->1, 10->2, 11->3, 12->4, 13->6, 14->7
    _DEMOTE_MAP = {9: 1, 10: 2, 11: 3, 12: 4, 13: 6, 14: 7}
    return _DEMOTE_MAP[board_val]


# ---------------------------------------------------------------------------
# Uchifuzume check (pawn drop giving immediate checkmate is illegal)
# ---------------------------------------------------------------------------

def _is_uchifuzume(
    board: np.ndarray,
    sente_hand: Dict[int, int],
    gote_hand: Dict[int, int],
    drop_sq: int,
    side: int,
) -> bool:
    """Return True if dropping a pawn on drop_sq by side constitutes uchifuzume."""
    # Apply the pawn drop tentatively
    new_board, sh, gh = _apply_move(
        board, sente_hand, gote_hand,
        Move(from_sq=None, to_sq=drop_sq, drop_piece=PAWN),
        side,
    )
    opponent = BLACK if side == WHITE else WHITE
    # The opponent must be in check after the drop
    if not _is_in_check(new_board, opponent):
        return False
    # Check whether the opponent has any legal escape
    opp_hand = sh if opponent == WHITE else gh
    pseudo = _expand_board_moves(new_board, opponent) + _expand_drop_moves(
        new_board, opp_hand, opponent
    )
    for m in pseudo:
        nb, nsh, ngh = _apply_move(new_board, sh, gh, m, opponent)
        if not _is_in_check(nb, opponent):
            return False  # opponent can escape -> not uchifuzume
    return True  # no escape -> uchifuzume, drop is illegal


# ---------------------------------------------------------------------------
# ShogiState
# ---------------------------------------------------------------------------

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
            self._king_sqs = (
                _king_sq(board, WHITE),
                _king_sq(board, BLACK),
            )

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

        # Incremental Zobrist hash
        h = self._hash
        side = self._side
        sign = 1 if side == WHITE else -1
        mover_color = WHITE if side == WHITE else BLACK
        opp_color = 1 - mover_color

        # Flip side
        h ^= int(_Z_SIDE[side])
        h ^= int(_Z_SIDE[new_side])

        if move.from_sq is None:
            # Drop move
            pt = move.drop_piece  # API piece type 0..6
            board_val = pt + 1
            placed_val = sign * board_val

            # Place piece on board
            h ^= int(_Z_PIECE[board_val][mover_color][move.to_sq])

            # Update hand: remove old count, add new count
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
            piece = int(self._board[from_sq])
            pv = abs(piece)
            captured = int(self._board[to_sq])

            # Remove mover from source
            h ^= int(_Z_PIECE[pv][mover_color][from_sq])

            # Remove captured piece from destination
            if captured != 0:
                cv = abs(captured)
                h ^= int(_Z_PIECE[cv][opp_color][to_sq])

                # Captured piece (demoted) goes to hand
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
            if move.promotion is not None:
                new_pv = move.promotion
            else:
                new_pv = pv
            h ^= int(_Z_PIECE[new_pv][mover_color][to_sq])

        # Propagate cached king squares
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

        # Repetition check (fourfold repetition -> draw)
        if self._history.count(self._hash) >= 3:
            # The same position has occurred 4 times total (3 previous + now)
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


# ---------------------------------------------------------------------------
# Initial position factory
# ---------------------------------------------------------------------------

def initial_state() -> ShogiState:
    """Return the standard Shogi starting position.

    Gote (BLACK=1) pieces are at ranks 0-2 (negative values).
    Sente (WHITE=0) pieces are at ranks 6-8 (positive values).

    Row 0 (gote back rank, left to right from sente's view):
      file 0..8: L  N  S  G  K  G  S  N  L
    Row 1:
      file 1: R (gote rook), file 7: B (gote bishop)
    Row 2: gote pawns (all files)
    Row 6: sente pawns (all files)
    Row 7:
      file 1: B (sente bishop), file 7: R (sente rook)
    Row 8 (sente back rank):
      file 0..8: L  N  S  G  K  G  S  N  L
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
