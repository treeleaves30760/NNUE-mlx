"""Promotion helpers, attack detection, move expansion, board mutation,
and uchifuzume for Shogi."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.games.base import BLACK, WHITE, Move
from .constants import (
    PAWN,
    _BOARD_TO_API,
    _KING_DIRS,
    _GOLD_DIRS,
    _SILVER_DIRS,
    _ROOK_DIRS,
    _BISHOP_DIRS,
    _rank,
    _file,
    _sq,
    _in_bounds,
    _in_promo_zone,
    _one_step_moves,
    _sliding_moves,
    _lance_moves,
    _knight_moves,
    _raw_targets,
)

# Pieces that can promote (board values 1..7)
_CAN_PROMOTE = {1, 2, 3, 4, 6, 7}  # pawn, lance, knight, silver, bishop, rook
# Promoted version mapping (board value -> promoted board value)
_PROMOTE_TO = {1: 9, 2: 10, 3: 11, 4: 12, 6: 13, 7: 14}


def _must_promote(piece_val: int, to_sq: int, side: int) -> bool:
    """Return True if the piece must promote when landing on to_sq.

    Pieces that MUST promote to avoid being stranded:
    pawn and lance cannot exist on rank 0 (sente) / rank 8 (gote)
    knight cannot exist on rank 0..1 (sente) / rank 7..8 (gote)
    """
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


def _expand_board_moves(board: np.ndarray, side: int) -> List[Move]:
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
    board: np.ndarray, hand: Dict[int, int], side: int,
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
