"""Mini Shogi move generation and attack detection."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from src.games.base import BLACK, WHITE, Move
from .constants import (
    NUM_SQUARES, BOARD_SIZE,
    KING_VAL, GOLD_VAL, SILVER_VAL, PAWN_VAL,
    BISHOP_VAL, ROOK_VAL, HORSE_VAL, DRAGON_VAL,
    TOKIN_VAL, PRO_SILVER_VAL,
    _ALL_8,
    _GOLD_DIRS_SENTE, _GOLD_DIRS_GOTE,
    _SILVER_DIRS_SENTE, _SILVER_DIRS_GOTE,
    _PAWN_DIR_SENTE, _PAWN_DIR_GOTE,
    _BISHOP_SLIDERS, _ROOK_SLIDERS,
    _HORSE_EXTRA, _DRAGON_EXTRA,
    _PROMO_RANK, _PROMOTE, _BOARD_VAL_TO_PIECE_TYPE, _PIECE_TYPE_TO_BOARD_VAL,
    PROMOTED_PAWN,
    _sq, _rank, _file, _on_board, _sign, _piece_color,
)

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
        # ---- Step movers ----
        if abs_v == KING_VAL:
            for dr, df in _ALL_8:
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
        # ---- Slider movers ----
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
    # ---- Drop moves ----
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
            # Nifu: cannot drop a pawn on a file that already has an unpromoted own pawn.
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


try:
    from src.accel import is_square_attacked_minishogi as _c_is_sq_attacked_mini
except ImportError:
    _c_is_sq_attacked_mini = None


def _is_square_attacked(board: np.ndarray, sq: int, defending_side: int) -> bool:
    """Return True if *sq* is attacked by the opponent of *defending_side*."""
    attacker_side = 1 - defending_side
    if _c_is_sq_attacked_mini is not None:
        return _c_is_sq_attacked_mini(bytes(board), sq, attacker_side)
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
