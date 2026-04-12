"""Rule-based and NNUE evaluators for all chess variants.

This module has two top-level layers:

  1. :class:`NNUEEvaluator` — wraps an incremental accumulator so the
     alpha-beta search can get a network evaluation of a position with
     incremental updates on pushes/pops.

  2. :class:`RuleBasedEvaluator` and :class:`MaterialEvaluator` —
     per-variant hand-tuned evaluators used for rule-based play and as
     bootstrap labels for NNUE training data. Each variant has its own
     evaluation function with variant-correct piece values, board
     geometry (mirror arithmetic), and positional features.

The variants dispatched:
  * ``chess``     — 8x8, tapered eval with pawn structure, king safety,
                    mobility, and classical piece bonuses
  * ``minichess`` — 6x6 Los Alamos, full PSTs + basic positional terms
  * ``shogi``     — 9x9, material + hand premium + king safety + advance
  * ``minishogi`` — 5x5, material + hand premium + advance + basic king
                    safety

All rule-based evaluators return scores in centipawn-like units from
the side-to-move's perspective. Start positions are (approximately)
symmetric and should score ~0. Evaluators are pure functions of the
state: no hidden state, no randomness — which is what makes them safe
to use as training labels.
"""

import numpy as np
from typing import List, Optional

from src.features.base import FeatureSet
from src.games.base import GameState, Move
from src.model.accumulator import IncrementalAccumulator

try:
    from src.accel import AcceleratedAccumulator as _AccelAccum
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False

# C-backed chess evaluator — identical semantics to _chess_rule_based but
# ~80x faster per call. Imported opportunistically; the Python fallback
# remains the source of truth and is still used for minichess.
try:
    from src.accel._nnue_accel import chess_c_evaluate as _chess_c_evaluate
    _HAS_CHESS_C_EVAL = True
except ImportError:
    _HAS_CHESS_C_EVAL = False


class NNUEEvaluator:
    """Wraps the incremental accumulator for use in alpha-beta search."""

    def __init__(self, accumulator: IncrementalAccumulator,
                 feature_set: FeatureSet):
        self.accumulator = accumulator
        self.feature_set = feature_set

    def set_position(self, state: GameState):
        """Initialize accumulator for a root position (full recompute)."""
        wf = self.feature_set.active_features(state, 0)
        bf = self.feature_set.active_features(state, 1)
        self.accumulator.refresh(wf, bf)

    # Scale raw NNUE output to centipawn-like range for search.
    # Trained with OUTPUT_SCALE=32 in loss, so model output of 1.0 ~ 32 cp.
    EVAL_OUTPUT_SCALE = 128.0

    def evaluate(self, state: GameState) -> float:
        """Evaluate current position using the accumulator.

        Returns score from side-to-move's perspective in centipawn-like units.
        """
        return self.accumulator.evaluate(state.side_to_move()) * self.EVAL_OUTPUT_SCALE

    def push_move(self, state_before: GameState, move: Move,
                  state_after: GameState):
        """Update accumulator for a new move (incremental or full refresh)."""
        self.accumulator.push()

        for perspective in [0, 1]:
            delta = self.feature_set.feature_delta(
                state_before, move, state_after, perspective
            )
            if delta is None:
                # King moved, need full refresh for this perspective
                features = self.feature_set.active_features(state_after, perspective)
                self.accumulator.refresh_perspective(perspective, features)
            else:
                added, removed = delta
                self.accumulator.update(perspective, added, removed)

    def push_move_refresh(self, state_after: GameState):
        """Full accumulator refresh for use with make_move_inplace."""
        self.accumulator.push()
        wf = self.feature_set.active_features(state_after, 0)
        bf = self.feature_set.active_features(state_after, 1)
        self.accumulator.refresh(wf, bf)

    def pop_move(self):
        """Restore accumulator state after unmaking a move."""
        self.accumulator.pop()

    @classmethod
    def from_numpy(cls, npz_path: str, feature_set: FeatureSet) -> "NNUEEvaluator":
        """Create evaluator from exported numpy weights.

        Uses the accelerated C extension (NEON SIMD + Accelerate) when
        available, falling back to the pure-numpy IncrementalAccumulator.
        Supports int16 quantized weights (auto-detected from dtype).
        """
        data = np.load(npz_path)
        return cls._build_from_data(data, feature_set)

    @classmethod
    def from_weights_dict(cls, weights: dict, feature_set: FeatureSet) -> "NNUEEvaluator":
        """Create evaluator from a pre-loaded dict of numpy arrays.

        Keys must match: feature_table.weight, ft_bias, l1.weight, l1.bias,
        l2.weight, l2.bias, output.weight, output.bias.
        Supports int16 quantized weights (auto-detected from dtype).
        """
        return cls._build_from_data(weights, feature_set)

    @classmethod
    def _build_from_data(cls, data, feature_set: FeatureSet) -> "NNUEEvaluator":
        """Build evaluator from weight data (dict or NpzFile)."""
        ft_weight = data["feature_table.weight"]
        is_quantized = ft_weight.dtype == np.int16
        quant_scale = float(data["quant_scale"]) if "quant_scale" in data else 512.0

        if is_quantized and _HAS_ACCEL:
            # Pass int16 weights directly to C extension (auto-detects dtype)
            accumulator = _AccelAccum(
                ft_weight=ft_weight,
                ft_bias=data["ft_bias"],
                l1_weight=data["l1.weight"],
                l1_bias=data["l1.bias"],
                l2_weight=data["l2.weight"],
                l2_bias=data["l2.bias"],
                out_weight=data["output.weight"],
                out_bias=data["output.bias"],
                quant_scale=quant_scale,
            )
        elif is_quantized:
            # No C extension: dequantize to float32 for numpy fallback
            accumulator = IncrementalAccumulator(
                ft_weight=ft_weight.astype(np.float32) / quant_scale,
                ft_bias=data["ft_bias"].astype(np.float32) / quant_scale,
                l1_weight=data["l1.weight"],
                l1_bias=data["l1.bias"],
                l2_weight=data["l2.weight"],
                l2_bias=data["l2.bias"],
                out_weight=data["output.weight"],
                out_bias=data["output.bias"],
            )
        else:
            AccumClass = _AccelAccum if _HAS_ACCEL else IncrementalAccumulator
            accumulator = AccumClass(
                ft_weight=ft_weight,
                ft_bias=data["ft_bias"],
                l1_weight=data["l1.weight"],
                l1_bias=data["l1.bias"],
                l2_weight=data["l2.weight"],
                l2_bias=data["l2.bias"],
                out_weight=data["output.weight"],
                out_bias=data["output.bias"],
            )
        return cls(accumulator, feature_set)


def _game_name(state: GameState) -> str:
    cfg = state.config() if hasattr(state, "config") else None
    return getattr(cfg, "name", "chess") if cfg is not None else "chess"


# ===========================================================================
# Shogi rule-based evaluation (9x9)
# ===========================================================================
#
# Shogi board encoding (see src/games/shogi/constants.py):
#   1=Pawn 2=Lance 3=Knight 4=Silver 5=Gold 6=Bishop 7=Rook 8=King
#   9=+Pawn(Tokin) 10=+Lance 11=+Knight 12=+Silver 13=+Horse 14=+Dragon
#   sq = rank * 9 + file   (rank 0 = gote's side, rank 8 = sente's side)
#
# Piece values are loosely in line with Apery/YaneuraOu but scaled to match
# the existing chess centipawn range (pawn ~100). Hand pieces are worth a
# small premium over board pieces because they can be dropped anywhere.

_SHOGI_BOARD_VALUES = [
    0,      # 0 empty
    100,    # 1 Pawn
    430,    # 2 Lance
    450,    # 3 Knight
    640,    # 4 Silver
    690,    # 5 Gold
    890,    # 6 Bishop
    1040,   # 7 Rook
    0,      # 8 King (game-ending; handled separately)
    520,    # 9 +Pawn (Tokin)
    530,    # 10 +Lance
    540,    # 11 +Knight
    570,    # 12 +Silver
    1150,   # 13 +Bishop (Horse)
    1300,   # 14 +Rook (Dragon)
]

# Hand values indexed by API piece type (PAWN=0..ROOK=6).
_SHOGI_HAND_VALUES = [115, 480, 510, 720, 780, 950, 1100]

_SHOGI_PIECE_ADVANCE = [
    0, 6, 4, 7, 3, 1, 0, 3, 0, 2, 2, 2, 2, 0, 2,
]


# MVV-LVA values exported for shogi move-ordering. Keys are (board_code - 1)
# because MoveOrdering does ``piece_values.get(abs(board[sq]) - 1, 100)``.
SHOGI_MVV_LVA_VALUES = {
    0:  100,   # 1 Pawn
    1:  430,   # 2 Lance
    2:  450,   # 3 Knight
    3:  640,   # 4 Silver
    4:  690,   # 5 Gold
    5:  890,   # 6 Bishop
    6:  1040,  # 7 Rook
    7:  20000, # 8 King
    8:  520,   # 9 Tokin
    9:  530,   # 10 +Lance
    10: 540,   # 11 +Knight
    11: 570,   # 12 +Silver
    12: 1150,  # 13 Horse
    13: 1300,  # 14 Dragon
}


def _shogi_material(state: GameState) -> float:
    """Material + hand + piece advancement from sente's view."""
    board = state.board_array()
    score = 0.0
    n = len(_SHOGI_BOARD_VALUES)
    for sq in range(len(board)):
        piece = int(board[sq])
        if piece == 0:
            continue
        pv = piece if piece > 0 else -piece
        if pv >= n:
            continue
        value = _SHOGI_BOARD_VALUES[pv]
        adv_w = _SHOGI_PIECE_ADVANCE[pv]
        rank = sq // 9
        if piece > 0:
            score += value + adv_w * (8 - rank)
        else:
            score -= value + adv_w * rank
    h_len = len(_SHOGI_HAND_VALUES)
    for pt, cnt in state.hand_pieces(0).items():
        if 0 <= pt < h_len:
            score += _SHOGI_HAND_VALUES[pt] * cnt
    for pt, cnt in state.hand_pieces(1).items():
        if 0 <= pt < h_len:
            score -= _SHOGI_HAND_VALUES[pt] * cnt
    return score


def _shogi_king_safety(state: GameState) -> float:
    """Sente-POV king safety: king position, defenders, attacker proximity.

    Combines four cheap signals: advancement penalty, central-file penalty,
    defender count in the 3x3 ring (weighted for Gold/Silver), attacker
    proximity inside the 5x5 ring, and a hand-threat modifier when the
    king is already exposed.
    """
    board = state.board_array()
    score = 0.0

    for side, sign, home_rank in ((0, 1, 8), (1, -1, 0)):
        ksq = state.king_square(side)
        if ksq is None or ksq < 0:
            continue
        kr, kf = ksq // 9, ksq % 9

        forward = (home_rank - kr) if side == 0 else (kr - home_rank)
        if forward > 0:
            score += sign * (-35 * forward)
        else:
            score += sign * 15

        if 3 <= kf <= 5:
            score += sign * (-40)
        elif kf in (0, 8):
            score += sign * 20
        elif kf in (1, 7):
            score += sign * 10

        gold_val = 5 if side == 0 else -5
        silver_val = 4 if side == 0 else -4
        friend_pos = (side == 0)

        defender_bonus = 0
        attacker_penalty = 0
        for dr in (-2, -1, 0, 1, 2):
            for df in (-2, -1, 0, 1, 2):
                nr, nf = kr + dr, kf + df
                if not (0 <= nr < 9 and 0 <= nf < 9):
                    continue
                v = int(board[nr * 9 + nf])
                if v == 0:
                    continue
                is_friend = (v > 0) == friend_pos
                cheb = max(abs(dr), abs(df))
                if is_friend:
                    if cheb <= 1:
                        av = abs(v)
                        if av == abs(gold_val) or av == abs(silver_val):
                            defender_bonus += 35
                        elif 9 <= av <= 12:
                            defender_bonus += 30
                        else:
                            defender_bonus += 12
                else:
                    av = abs(v)
                    if av in (7, 14):
                        weight = 90
                    elif av in (6, 13):
                        weight = 70
                    elif av == 5 or 9 <= av <= 12:
                        weight = 45
                    elif av == 4:
                        weight = 40
                    else:
                        weight = 25
                    attacker_penalty += weight * (3 - cheb)

        opp_side = 1 - side
        hand_threat = 0
        for pt, cnt in state.hand_pieces(opp_side).items():
            if 0 <= pt < len(_SHOGI_HAND_VALUES):
                hand_threat += _SHOGI_HAND_VALUES[pt] * cnt
        exposure = max(0, forward) + (1 if 3 <= kf <= 5 else 0)
        if exposure > 0:
            attacker_penalty += (hand_threat * exposure) // 20

        score += sign * defender_bonus
        score -= sign * attacker_penalty

    return score


def _shogi_castle_bonus(state: GameState) -> float:
    """Sente-POV bonus for recognised castle shapes.

    Mino / Yagura / Anaguma are all characterised by two things:

      1. The king has moved OFF the central three files and is still
         near its home rank (this is "tucked away on a wing").
      2. Two or three gold/silver pieces sit adjacent to the king
         forming a defensive wall.

    Rather than pattern-match exact squares (fragile and variant-
    heavy), we score any king that satisfies the *shape* by counting
    gold/silver-class defenders within Chebyshev distance 2. A 3+
    wall gets a flat bonus of 40, plus 15 per extra defender beyond
    3, capped at 85. Exactly the kind of bonus the Python and C
    ports can share line-for-line.
    """
    board = state.board_array()
    score = 0
    for side, sign in ((0, 1), (1, -1)):
        ksq = state.king_square(side)
        if ksq is None or ksq < 0:
            continue
        kr, kf = ksq // 9, ksq % 9

        # Gate: king must be on a wing file (not central) AND on its
        # home-side half of the board. Sente home = ranks 7-8, gote
        # home = ranks 0-1. A central-file king doesn't count as
        # castled regardless of defender wall.
        if 3 <= kf <= 5:
            continue
        if side == 0:
            if kr < 7:
                continue
        else:
            if kr > 1:
                continue

        wall = 0
        for dr in (-2, -1, 0, 1, 2):
            for df in (-2, -1, 0, 1, 2):
                if dr == 0 and df == 0:
                    continue
                nr, nf = kr + dr, kf + df
                if not (0 <= nr < 9 and 0 <= nf < 9):
                    continue
                v = int(board[nr * 9 + nf])
                if v == 0:
                    continue
                is_friend = (v > 0) == (side == 0)
                if not is_friend:
                    continue
                av = abs(v)
                # Gold / silver / promoted gold-likes count as wall.
                if av == 4 or av == 5 or (9 <= av <= 12):
                    wall += 1

        if wall >= 3:
            bonus = 40 + (wall - 3) * 15
            if bonus > 85:
                bonus = 85
            score += sign * bonus

    return score


def _shogi_attack_cluster(state: GameState) -> float:
    """Sente-POV bonus for friendly pieces gathered near the ENEMY king.

    The existing ``_shogi_king_safety`` penalises enemy pieces near
    OUR king. This complementary term rewards the same picture from
    the attacker's side: every friendly piece within Chebyshev
    distance 2 of the enemy king adds to an attack count, weighted
    by piece strength (rook/bishop/dragon/horse >> gold/silver >> pawn).

    Together with hand pieces we also count: if our hand has many
    drop-able pieces and the enemy king is exposed (central file or
    advanced), we credit that as drop pressure. This mirrors the
    existing defender logic and closes a loop in which building an
    attack formation is recognised independently of whether the
    defender was simply absent.
    """
    board = state.board_array()
    score = 0.0

    for side, sign in ((0, 1), (1, -1)):
        opp = 1 - side
        ksq = state.king_square(opp)
        if ksq is None or ksq < 0:
            continue
        kr, kf = ksq // 9, ksq % 9

        attack_weight = 0
        for dr in (-2, -1, 0, 1, 2):
            for df in (-2, -1, 0, 1, 2):
                nr, nf = kr + dr, kf + df
                if not (0 <= nr < 9 and 0 <= nf < 9):
                    continue
                v = int(board[nr * 9 + nf])
                if v == 0:
                    continue
                is_friend = (v > 0) == (side == 0)
                if not is_friend:
                    continue
                av = abs(v)
                cheb = max(abs(dr), abs(df))
                if av in (7, 14):      # Rook / Dragon
                    w = 70
                elif av in (6, 13):    # Bishop / Horse
                    w = 55
                elif av == 5 or 9 <= av <= 12:  # Gold / promoted gold-likes
                    w = 30
                elif av == 4:          # Silver
                    w = 25
                elif av == 2:          # Lance
                    w = 18
                elif av == 3:          # Knight
                    w = 18
                elif av == 1:          # Pawn
                    w = 10
                else:
                    continue
                attack_weight += w * (3 - cheb)

        # Drop pressure from our hand when the enemy king is exposed.
        # Re-use the same exposure test as _shogi_king_safety: advanced
        # king (away from home) or centrally placed king. Note that
        # sente's home rank is 8 and gote's is 0 — when the opponent
        # is sente (opp == 0), their home is rank 8.
        home_rank_opp = 8 if opp == 0 else 0
        forward_opp = (home_rank_opp - kr) if opp == 0 else (kr - home_rank_opp)
        exposure = max(0, forward_opp) + (1 if 3 <= kf <= 5 else 0)
        if exposure > 0:
            hand = state.hand_pieces(side)
            hand_total = 0
            for pt, cnt in hand.items():
                if 0 <= pt < len(_SHOGI_HAND_VALUES):
                    hand_total += _SHOGI_HAND_VALUES[pt] * cnt
            attack_weight += (hand_total * exposure) // 25

        score += sign * attack_weight

    return score


def _shogi_rook_positional(state: GameState) -> float:
    """Sente-POV rook positional bonuses beyond the advance table.

    Two features:
      * **Invasion**: a rook or dragon that has crossed into the enemy's
        three home ranks is worth far more than the advance table alone
        captures — it attacks back-rank pieces and prevents the enemy
        king from retreating comfortably.
      * **File split**: rooks / dragons on files 1 (static rook home)
        or 7 (ranging rook post) get a small bonus for being on a
        classical strong file rather than marooned on a central file.
    """
    board = state.board_array()
    score = 0.0
    for sq in range(81):
        v = int(board[sq])
        if v == 0:
            continue
        av = v if v > 0 else -v
        if av not in (7, 14):  # Rook / Dragon
            continue
        rank = sq // 9
        f = sq % 9
        is_dragon = (av == 14)
        if v > 0:
            if rank <= 2:
                score += 50 if is_dragon else 35
            if f in (1, 7):
                score += 8
        else:
            if rank >= 6:
                score -= 50 if is_dragon else 35
            if f in (1, 7):
                score -= 8
    return score


def _shogi_evaluate(state: GameState) -> float:
    score = (_shogi_material(state)
             + _shogi_king_safety(state)
             + _shogi_rook_positional(state)
             + _shogi_attack_cluster(state)
             + _shogi_castle_bonus(state))
    if state.side_to_move() == 1:
        score = -score
    return score


def _shogi_material_only(state: GameState) -> float:
    score = _shogi_material(state)
    if state.side_to_move() == 1:
        score = -score
    return score


# ===========================================================================
# Chess rule-based evaluation (8x8) — tapered with pawn structure,
# king safety, mobility, and piece bonuses
# ===========================================================================
#
# Scoring is from white's perspective, negated at the end if black is to
# move. All positional sub-scores are in centipawn-like units and add
# linearly to material — mostly classical values from the chess
# programming wiki with some adjustments for balance against the material.

_CHESS_PIECE_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}

# --- Tapered piece-square tables -----------------------------------------
#
# Two tables per piece type: one tuned for the middlegame (emphasising
# central presence, king safety, developed pieces) and one for the
# endgame (emphasising king activity, passed pawn support, long-range
# rook files). Phase-weighted interpolation between the two produces a
# smooth evaluation across the game.
#
# Tables are in "white's view": sq 0 = a1 (white queenside corner), sq 63
# = h8 (black kingside corner). For black pieces we mirror vertically via
# ``_chess_mirror_sq``.

_CHESS_PAWN_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_CHESS_PAWN_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_CHESS_KNIGHT_MG = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
_CHESS_KNIGHT_EG = _CHESS_KNIGHT_MG  # knight value changes little by phase

_CHESS_BISHOP_MG = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10, 10,  5, 10, 10,  5, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
_CHESS_BISHOP_EG = _CHESS_BISHOP_MG

_CHESS_ROOK_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
_CHESS_ROOK_EG = _CHESS_ROOK_MG

_CHESS_QUEEN_MG = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
_CHESS_QUEEN_EG = _CHESS_QUEEN_MG

# Middlegame king PST: corners safer, centre dangerous.
_CHESS_KING_MG = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]
# Endgame king PST: opposite — activity matters, centre dominates.
_CHESS_KING_EG = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]

_CHESS_PST_MG = {
    1: _CHESS_PAWN_MG, 2: _CHESS_KNIGHT_MG, 3: _CHESS_BISHOP_MG,
    4: _CHESS_ROOK_MG, 5: _CHESS_QUEEN_MG, 6: _CHESS_KING_MG,
}
_CHESS_PST_EG = {
    1: _CHESS_PAWN_EG, 2: _CHESS_KNIGHT_EG, 3: _CHESS_BISHOP_EG,
    4: _CHESS_ROOK_EG, 5: _CHESS_QUEEN_EG, 6: _CHESS_KING_EG,
}

# Game-phase weights: how much each piece counts toward "middlegame-ness".
# Queens and rooks dominate phase (their trade pushes toward endgame).
_CHESS_PHASE_WEIGHTS = {1: 0, 2: 1, 3: 1, 4: 2, 5: 4, 6: 0}
_CHESS_PHASE_TOTAL = 24  # 4 knights + 4 bishops + 4 rooks + 2 queens


def _chess_mirror_sq(sq: int) -> int:
    """Vertical mirror for a white-POV PST applied to a black piece."""
    return (7 - sq // 8) * 8 + (sq % 8)


def _chess_material(state: GameState) -> float:
    """Pure material, white-POV, negated at end if black to move."""
    board = state.board_array()
    score = 0.0
    for sq in range(len(board)):
        piece = int(board[sq])
        if piece == 0:
            continue
        abs_piece = abs(piece)
        # Don't count kings in pure material (value 20000 dwarfs real
        # material swings and they're always on the board by game rules).
        if abs_piece == 6:
            continue
        value = _CHESS_PIECE_VALUES.get(abs_piece, 100)
        score += value if piece > 0 else -value
    if state.side_to_move() == 1:
        score = -score
    return score


def _chess_pawn_structure(pawn_files_white: List[int],
                           pawn_files_black: List[int]) -> float:
    """Returns white-POV pawn structure score.

    ``pawn_files_white[f]`` is the count of white pawns on file ``f``;
    same for black. We look at three classical features:
      * doubled pawns  — penalise 2+ pawns on the same file
      * isolated pawns — penalise pawns with no pawns on adjacent files
      * passed pawns require rank info so they're handled separately
    """
    score = 0.0
    for f in range(8):
        wc = pawn_files_white[f]
        bc = pawn_files_black[f]

        # Doubled pawns: 2+ pawns on same file
        if wc >= 2:
            score -= 15 * (wc - 1)
        if bc >= 2:
            score += 15 * (bc - 1)

        # Isolated pawns: no friendly pawns on adjacent files
        left = pawn_files_white[f - 1] if f > 0 else 0
        right = pawn_files_white[f + 1] if f < 7 else 0
        if wc > 0 and left == 0 and right == 0:
            score -= 20 * wc

        left = pawn_files_black[f - 1] if f > 0 else 0
        right = pawn_files_black[f + 1] if f < 7 else 0
        if bc > 0 and left == 0 and right == 0:
            score += 20 * bc

    return score


def _chess_king_shelter(king_sq: Optional[int],
                         friendly_pawn_squares: List[int],
                         side: int) -> int:
    """Return shelter bonus for a castled-or-tucked-in king.

    Counts up to three friendly pawns on the files (kf-1, kf, kf+1)
    directly in front of the king. A pawn one rank ahead is worth 20,
    two ranks ahead 10, further ahead or missing 0. Mid-board or
    advanced kings get no shelter — only kings on their home two ranks
    qualify, so an endgame king in the centre doesn't accidentally
    score positive shelter.
    """
    if king_sq is None or king_sq < 0:
        return 0
    kf = king_sq % 8
    kr = king_sq // 8
    if side == 0:
        if kr > 2:
            return 0
        want_ranks = (kr + 1, kr + 2)
    else:
        if kr < 5:
            return 0
        want_ranks = (kr - 1, kr - 2)

    shelter = 0
    for pf in (kf - 1, kf, kf + 1):
        if not (0 <= pf < 8):
            continue
        best_bonus = 0
        for psq in friendly_pawn_squares:
            if psq % 8 != pf:
                continue
            pr = psq // 8
            if pr == want_ranks[0]:
                best_bonus = max(best_bonus, 20)
            elif pr == want_ranks[1]:
                best_bonus = max(best_bonus, 10)
        shelter += best_bonus
    return shelter


def _chess_connected_pawns(white_pawn_squares: List[int],
                             black_pawn_squares: List[int]) -> int:
    """Phalanx + chain bonus for connected pawns.

    A pawn is "connected" when it has a friendly neighbour on an
    adjacent file — either on the same rank (phalanx) or one rank
    behind (chain / supported). Both configurations are strong: the
    phalanx threatens a safe push, the chain is un-attackable by
    enemy pawns. We award +10 per phalanx pair (counted once by only
    looking rightward) and +8 per chain support.
    """
    score = 0
    w_set = set(white_pawn_squares)
    b_set = set(black_pawn_squares)

    for sq in white_pawn_squares:
        f, r = sq % 8, sq // 8
        # Phalanx: neighbour to the right on same rank (counted once).
        if f < 7 and (sq + 1) in w_set:
            score += 10
        # Chain: supported by friendly pawn diagonally behind.
        if r > 0:
            if f > 0 and ((r - 1) * 8 + (f - 1)) in w_set:
                score += 8
            if f < 7 and ((r - 1) * 8 + (f + 1)) in w_set:
                score += 8

    for sq in black_pawn_squares:
        f, r = sq % 8, sq // 8
        if f < 7 and (sq + 1) in b_set:
            score -= 10
        if r < 7:
            if f > 0 and ((r + 1) * 8 + (f - 1)) in b_set:
                score -= 8
            if f < 7 and ((r + 1) * 8 + (f + 1)) in b_set:
                score -= 8

    return score


def _chess_knight_outposts(white_knight_squares: List[int],
                            black_knight_squares: List[int],
                            white_pawn_squares: List[int],
                            black_pawn_squares: List[int]) -> int:
    """Bonus for knights parked on a supported, unattackable outpost.

    An outpost is a square in the opponent's half of the board where
    the knight is defended by a friendly pawn AND cannot be chased
    away by an enemy pawn (no enemy pawn on the adjacent files far
    enough forward to attack the square in one or two pushes). Outposts
    are among the strongest positional assets in the middlegame — the
    knight can't be exchanged by a minor piece on equal terms and
    controls 8 squares from a secure base.
    """
    score = 0
    w_pawn_set = set(white_pawn_squares)
    b_pawn_set = set(black_pawn_squares)

    for sq in white_knight_squares:
        f, r = sq % 8, sq // 8
        if r < 4:
            continue  # not in enemy half
        supported = False
        if r > 0:
            if f > 0 and ((r - 1) * 8 + (f - 1)) in w_pawn_set:
                supported = True
            elif f < 7 and ((r - 1) * 8 + (f + 1)) in w_pawn_set:
                supported = True
        if not supported:
            continue
        safe = True
        for df in (-1, 1):
            nf = f + df
            if not (0 <= nf < 8):
                continue
            # Any black pawn on this file above (larger rank) could chase.
            for nr in range(r + 1, 8):
                if (nr * 8 + nf) in b_pawn_set:
                    safe = False
                    break
            if not safe:
                break
        if safe:
            # Extra bonus for advanced outpost.
            score += 20 + (r - 4) * 5

    for sq in black_knight_squares:
        f, r = sq % 8, sq // 8
        if r > 3:
            continue
        supported = False
        if r < 7:
            if f > 0 and ((r + 1) * 8 + (f - 1)) in b_pawn_set:
                supported = True
            elif f < 7 and ((r + 1) * 8 + (f + 1)) in b_pawn_set:
                supported = True
        if not supported:
            continue
        safe = True
        for df in (-1, 1):
            nf = f + df
            if not (0 <= nf < 8):
                continue
            for nr in range(r - 1, -1, -1):
                if (nr * 8 + nf) in w_pawn_set:
                    safe = False
                    break
            if not safe:
                break
        if safe:
            score -= 20 + (3 - r) * 5

    return score


# Ray directions for sliding-piece mobility. (drow, dfile) tuples.
_CHESS_ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_CHESS_BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_CHESS_QUEEN_DIRS = _CHESS_ROOK_DIRS + _CHESS_BISHOP_DIRS
_CHESS_KNIGHT_OFFSETS = (
    (1, 2), (1, -2), (-1, 2), (-1, -2),
    (2, 1), (2, -1), (-2, 1), (-2, -1),
)


def _chess_mobility(board, knight_w: List[int], knight_b: List[int],
                     bishop_sq_w: List[int], bishop_sq_b: List[int],
                     rook_sq_w: List[int], rook_sq_b: List[int],
                     queen_sq_w: List[int], queen_sq_b: List[int]) -> int:
    """White-POV mobility bonus for every non-king, non-pawn piece.

    "Mobility" is the count of squares the piece can move to without
    overstepping a friendly piece. Captures count, own blockers stop
    the ray. Each square is worth a small weight (2-4 cp) which adds
    up to a substantial positional signal in open positions and is
    near-zero in closed ones — tapered interpolation is forgiving.

    This function is called at every leaf of alpha-beta so every
    microsecond matters. We flatten all loops, bind hot locals, and
    walk rays with pure integer arithmetic instead of delegating to a
    nested helper. The result is a ~3x speedup over the naive closure
    version.
    """
    score = 0
    b = board  # local binding for fast access inside the hot loops
    knight_offsets = _CHESS_KNIGHT_OFFSETS

    # --- Knights --------------------------------------------------------
    # 3 cp per target square (0 captures own, 1 captures enemy or empty).
    kw_count = 0
    for sq in knight_w:
        r0 = sq >> 3
        f0 = sq & 7
        for dr, df in knight_offsets:
            r = r0 + dr
            f = f0 + df
            if 0 <= r < 8 and 0 <= f < 8:
                t = b[(r << 3) + f]
                if t <= 0:  # empty or enemy
                    kw_count += 1
    kb_count = 0
    for sq in knight_b:
        r0 = sq >> 3
        f0 = sq & 7
        for dr, df in knight_offsets:
            r = r0 + dr
            f = f0 + df
            if 0 <= r < 8 and 0 <= f < 8:
                t = b[(r << 3) + f]
                if t >= 0:
                    kb_count += 1
    score += (kw_count - kb_count) * 3

    # --- Bishops (4 diagonal rays) --------------------------------------
    bw = 0
    for sq in bishop_sq_w:
        r0 = sq >> 3
        f0 = sq & 7
        # +1,+1
        r = r0 + 1
        f = f0 + 1
        while r < 8 and f < 8:
            t = b[(r << 3) + f]
            if t == 0:
                bw += 1
            else:
                if t < 0:
                    bw += 1
                break
            r += 1
            f += 1
        # +1,-1
        r = r0 + 1
        f = f0 - 1
        while r < 8 and f >= 0:
            t = b[(r << 3) + f]
            if t == 0:
                bw += 1
            else:
                if t < 0:
                    bw += 1
                break
            r += 1
            f -= 1
        # -1,+1
        r = r0 - 1
        f = f0 + 1
        while r >= 0 and f < 8:
            t = b[(r << 3) + f]
            if t == 0:
                bw += 1
            else:
                if t < 0:
                    bw += 1
                break
            r -= 1
            f += 1
        # -1,-1
        r = r0 - 1
        f = f0 - 1
        while r >= 0 and f >= 0:
            t = b[(r << 3) + f]
            if t == 0:
                bw += 1
            else:
                if t < 0:
                    bw += 1
                break
            r -= 1
            f -= 1

    bb = 0
    for sq in bishop_sq_b:
        r0 = sq >> 3
        f0 = sq & 7
        r = r0 + 1
        f = f0 + 1
        while r < 8 and f < 8:
            t = b[(r << 3) + f]
            if t == 0:
                bb += 1
            else:
                if t > 0:
                    bb += 1
                break
            r += 1
            f += 1
        r = r0 + 1
        f = f0 - 1
        while r < 8 and f >= 0:
            t = b[(r << 3) + f]
            if t == 0:
                bb += 1
            else:
                if t > 0:
                    bb += 1
                break
            r += 1
            f -= 1
        r = r0 - 1
        f = f0 + 1
        while r >= 0 and f < 8:
            t = b[(r << 3) + f]
            if t == 0:
                bb += 1
            else:
                if t > 0:
                    bb += 1
                break
            r -= 1
            f += 1
        r = r0 - 1
        f = f0 - 1
        while r >= 0 and f >= 0:
            t = b[(r << 3) + f]
            if t == 0:
                bb += 1
            else:
                if t > 0:
                    bb += 1
                break
            r -= 1
            f -= 1
    score += (bw - bb) * 4

    # --- Rooks (4 orthogonal rays) --------------------------------------
    rw = 0
    for sq in rook_sq_w:
        r0 = sq >> 3
        f0 = sq & 7
        # +r
        r = r0 + 1
        while r < 8:
            t = b[(r << 3) + f0]
            if t == 0:
                rw += 1
            else:
                if t < 0:
                    rw += 1
                break
            r += 1
        # -r
        r = r0 - 1
        while r >= 0:
            t = b[(r << 3) + f0]
            if t == 0:
                rw += 1
            else:
                if t < 0:
                    rw += 1
                break
            r -= 1
        # +f
        f = f0 + 1
        while f < 8:
            t = b[(r0 << 3) + f]
            if t == 0:
                rw += 1
            else:
                if t < 0:
                    rw += 1
                break
            f += 1
        # -f
        f = f0 - 1
        while f >= 0:
            t = b[(r0 << 3) + f]
            if t == 0:
                rw += 1
            else:
                if t < 0:
                    rw += 1
                break
            f -= 1

    rb = 0
    for sq in rook_sq_b:
        r0 = sq >> 3
        f0 = sq & 7
        r = r0 + 1
        while r < 8:
            t = b[(r << 3) + f0]
            if t == 0:
                rb += 1
            else:
                if t > 0:
                    rb += 1
                break
            r += 1
        r = r0 - 1
        while r >= 0:
            t = b[(r << 3) + f0]
            if t == 0:
                rb += 1
            else:
                if t > 0:
                    rb += 1
                break
            r -= 1
        f = f0 + 1
        while f < 8:
            t = b[(r0 << 3) + f]
            if t == 0:
                rb += 1
            else:
                if t > 0:
                    rb += 1
                break
            f += 1
        f = f0 - 1
        while f >= 0:
            t = b[(r0 << 3) + f]
            if t == 0:
                rb += 1
            else:
                if t > 0:
                    rb += 1
                break
            f -= 1
    score += (rw - rb) * 3

    # --- Queens (8 directions: orthogonal + diagonal) -------------------
    # Reuse rook + bishop idea but inline for queens directly. Queen
    # mobility is weighted only 2 cp — queens are already worth 900.
    qw = 0
    for sq in queen_sq_w:
        r0 = sq >> 3
        f0 = sq & 7
        for dr, df in _CHESS_QUEEN_DIRS:
            r = r0 + dr
            f = f0 + df
            while 0 <= r < 8 and 0 <= f < 8:
                t = b[(r << 3) + f]
                if t == 0:
                    qw += 1
                else:
                    if t < 0:
                        qw += 1
                    break
                r += dr
                f += df
    qb = 0
    for sq in queen_sq_b:
        r0 = sq >> 3
        f0 = sq & 7
        for dr, df in _CHESS_QUEEN_DIRS:
            r = r0 + dr
            f = f0 + df
            while 0 <= r < 8 and 0 <= f < 8:
                t = b[(r << 3) + f]
                if t == 0:
                    qb += 1
                else:
                    if t > 0:
                        qb += 1
                    break
                r += dr
                f += df
    score += (qw - qb) * 2

    return score


def _chess_passed_pawn_bonus(white_pawn_squares: List[int],
                              black_pawn_squares: List[int]) -> float:
    """Passed pawn bonuses — pawn has no enemy pawn ahead on the same
    file or the two adjacent files. Bonus scales with rank advanced."""
    score = 0.0

    # White passed pawns: check no black pawn on files f-1, f, f+1 with
    # rank strictly greater than this pawn's rank.
    for sq in white_pawn_squares:
        f, r = sq % 8, sq // 8
        blocked = False
        for bsq in black_pawn_squares:
            bf, br = bsq % 8, bsq // 8
            if abs(bf - f) <= 1 and br > r:
                blocked = True
                break
        if not blocked:
            # Rank bonus: pawn's advancement from its starting rank (1).
            advance = r - 1  # 0..6, since promotion at rank 7
            score += 10 + advance * 12

    for sq in black_pawn_squares:
        f, r = sq % 8, sq // 8
        blocked = False
        for wsq in white_pawn_squares:
            wf, wr = wsq % 8, wsq // 8
            if abs(wf - f) <= 1 and wr < r:
                blocked = True
                break
        if not blocked:
            advance = 6 - r  # 0..6
            score -= 10 + advance * 12

    return score


def _chess_rule_based(state: GameState) -> float:
    """Full chess evaluator: material + tapered PST + pawn structure +
    passed pawns + piece bonuses (bishop pair, rook on open file,
    knight outposts, king shelter, mobility).

    Returns score in centipawns from the side-to-move perspective.

    All internal accumulators use *integer* arithmetic so the C port in
    ``src/accel/_chess_eval_c.c`` can produce bit-identical results.
    Scales that would naturally be floats (e.g. ``passed * 1.5``) are
    implemented as rational integer multipliers (``passed * 3 // 2``).
    The final float cast is the API contract — every score caller sees
    the same integer value, just boxed as a Python float.
    """
    board = state.board_array()

    # Accumulators computed in a single board pass — integers throughout.
    mg_score = 0
    eg_score = 0
    phase = 0

    white_bishops = 0
    black_bishops = 0
    white_bishop_squares: List[int] = []
    black_bishop_squares: List[int] = []
    white_rooks: List[int] = []
    black_rooks: List[int] = []
    white_knights: List[int] = []
    black_knights: List[int] = []
    white_queens: List[int] = []
    black_queens: List[int] = []
    white_pawn_squares: List[int] = []
    black_pawn_squares: List[int] = []
    pawn_files_white = [0] * 8
    pawn_files_black = [0] * 8

    for sq in range(64):
        piece = int(board[sq])
        if piece == 0:
            continue
        abs_piece = abs(piece)

        # Track phase (queens/rooks/bishops/knights).
        phase += _CHESS_PHASE_WEIGHTS.get(abs_piece, 0)

        value = _CHESS_PIECE_VALUES.get(abs_piece, 100)
        if abs_piece == 6:
            value = 0  # king not counted in material swing
        pst_mg = _CHESS_PST_MG.get(abs_piece)
        pst_eg = _CHESS_PST_EG.get(abs_piece)

        if piece > 0:
            mg_score += value
            eg_score += value
            if pst_mg:
                mg_score += pst_mg[sq]
                eg_score += pst_eg[sq]
            if abs_piece == 1:
                pawn_files_white[sq % 8] += 1
                white_pawn_squares.append(sq)
            elif abs_piece == 2:
                white_knights.append(sq)
            elif abs_piece == 3:
                white_bishops += 1
                white_bishop_squares.append(sq)
            elif abs_piece == 4:
                white_rooks.append(sq)
            elif abs_piece == 5:
                white_queens.append(sq)
        else:
            mg_score -= value
            eg_score -= value
            if pst_mg:
                m = _chess_mirror_sq(sq)
                mg_score -= pst_mg[m]
                eg_score -= pst_eg[m]
            if abs_piece == 1:
                pawn_files_black[sq % 8] += 1
                black_pawn_squares.append(sq)
            elif abs_piece == 2:
                black_knights.append(sq)
            elif abs_piece == 3:
                black_bishops += 1
                black_bishop_squares.append(sq)
            elif abs_piece == 4:
                black_rooks.append(sq)
            elif abs_piece == 5:
                black_queens.append(sq)

    # --- Pawn structure -----------------------------------------------
    pawn_struct = _chess_pawn_structure(pawn_files_white, pawn_files_black)
    passed = _chess_passed_pawn_bonus(white_pawn_squares, black_pawn_squares)
    connected = _chess_connected_pawns(white_pawn_squares, black_pawn_squares)
    mg_score += pawn_struct + passed + connected
    # Integer scales: passed * 3/2 in EG (worth more in endgames);
    #                 connected * 7/10 in EG (pawn chains matter less).
    eg_score += pawn_struct + (passed * 3) // 2 + (connected * 7) // 10

    # --- Knight outposts ---------------------------------------------
    outposts = _chess_knight_outposts(
        white_knights, black_knights,
        white_pawn_squares, black_pawn_squares,
    )
    mg_score += outposts
    eg_score += outposts // 2  # outposts worth half as much in endgame

    # --- Bishop pair --------------------------------------------------
    if white_bishops >= 2:
        mg_score += 30
        eg_score += 50
    if black_bishops >= 2:
        mg_score -= 30
        eg_score -= 50

    # --- Rooks on open / semi-open files + 7th rank -------------------
    for sq in white_rooks:
        f = sq % 8
        r = sq // 8
        w_pawns_on_file = pawn_files_white[f]
        b_pawns_on_file = pawn_files_black[f]
        if w_pawns_on_file == 0 and b_pawns_on_file == 0:
            mg_score += 20
            eg_score += 10
        elif w_pawns_on_file == 0:
            mg_score += 10
            eg_score += 5
        if r == 6:  # 7th rank (0-indexed)
            mg_score += 25
            eg_score += 15
    for sq in black_rooks:
        f = sq % 8
        r = sq // 8
        w_pawns_on_file = pawn_files_white[f]
        b_pawns_on_file = pawn_files_black[f]
        if w_pawns_on_file == 0 and b_pawns_on_file == 0:
            mg_score -= 20
            eg_score -= 10
        elif b_pawns_on_file == 0:
            mg_score -= 10
            eg_score -= 5
        if r == 1:  # 7th from black's POV
            mg_score -= 25
            eg_score -= 15

    # --- King shelter (MG only) --------------------------------------
    wk_sq = state.king_square(0) if hasattr(state, "king_square") else None
    bk_sq = state.king_square(1) if hasattr(state, "king_square") else None
    mg_score += _chess_king_shelter(wk_sq, white_pawn_squares, 0)
    mg_score -= _chess_king_shelter(bk_sq, black_pawn_squares, 1)

    # --- Mobility (MG heavier, EG lighter) ---------------------------
    mobility = _chess_mobility(
        board,
        white_knights, black_knights,
        white_bishop_squares, black_bishop_squares,
        white_rooks, black_rooks,
        white_queens, black_queens,
    )
    mg_score += mobility
    eg_score += (mobility * 6) // 10  # integer 0.6 scale

    # --- Tapered interpolation ----------------------------------------
    # phase ∈ [0, 24]; 24 = full opening, 0 = pure endgame.
    # Integer formulation (C port produces bit-identical results):
    #   score = (mg * phase + eg * (24 - phase)) // 24
    if phase > _CHESS_PHASE_TOTAL:
        phase = _CHESS_PHASE_TOTAL
    score = (mg_score * phase + eg_score * (_CHESS_PHASE_TOTAL - phase)) \
            // _CHESS_PHASE_TOTAL

    if state.side_to_move() == 1:
        score = -score
    return float(score)


# ===========================================================================
# Minichess rule-based evaluation (6x6 Los Alamos)
# ===========================================================================
#
# Minichess piece codes (src/games/minichess/constants.py):
#   1=Pawn 2=Knight 3=Rook 4=Queen 5=King
#   There is NO bishop. Board is 6x6 with row 0 = white back rank,
#   row 5 = black back rank. Squares indexed sq = row * 6 + col.
#
# Queens dominate at this scale, so piece values are tuned slightly lower
# than standard chess values (the 6x6 board is ~56% the area of chess)
# but kept in the same centipawn range for consistency.

_MINICHESS_PIECE_VALUES = {1: 100, 2: 300, 3: 500, 4: 900, 5: 20000}

_MINICHESS_PAWN_PST = [
    # row 0 (white back rank — pawns never live here)
      0,  0,  0,  0,  0,  0,
    # row 1 (starting rank)
      0,  0, -5, -5,  0,  0,
    # row 2
     10, 10, 20, 20, 10, 10,
    # row 3
     20, 20, 30, 30, 20, 20,
    # row 4 (one step from promotion)
     40, 40, 50, 50, 40, 40,
    # row 5 (promotion target — pawns promote, rarely here)
      0,  0,  0,  0,  0,  0,
]

_MINICHESS_KNIGHT_PST = [
    -30, -15,   0,   0, -15, -30,
    -15,   5,  15,  15,   5, -15,
      0,  15,  25,  25,  15,   0,
      0,  15,  25,  25,  15,   0,
    -15,   5,  15,  15,   5, -15,
    -30, -15,   0,   0, -15, -30,
]

_MINICHESS_ROOK_PST = [
      0,   0,   5,   5,   0,   0,
      0,   0,   5,   5,   0,   0,
      0,   0,   5,   5,   0,   0,
      0,   0,   5,   5,   0,   0,
     10,  10,  10,  10,  10,  10,  # rank 5 = 7th-rank equivalent
      5,   5,   5,   5,   5,   5,
]

_MINICHESS_QUEEN_PST = [
    -20, -10,  -5,  -5, -10, -20,
    -10,   0,   5,   5,   0, -10,
     -5,   5,  10,  10,   5,  -5,
     -5,   5,  10,  10,   5,  -5,
    -10,   0,   5,   5,   0, -10,
    -20, -10,  -5,  -5, -10, -20,
]

_MINICHESS_KING_MG = [
     20,  30,  10,   0,  30,  20,   # row 0 — safe corners for white king
      0,   0, -10, -10,   0,   0,
    -10, -20, -25, -25, -20, -10,
    -20, -25, -30, -30, -25, -20,
    -25, -30, -35, -35, -30, -25,
    -30, -35, -40, -40, -35, -30,
]

_MINICHESS_KING_EG = [
    -30, -20, -10, -10, -20, -30,
    -20,   0,  10,  10,   0, -20,
    -10,  10,  20,  20,  10, -10,
    -10,  10,  20,  20,  10, -10,
    -20,   0,  10,  10,   0, -20,
    -30, -20, -10, -10, -20, -30,
]

_MINICHESS_PST_MG = {
    1: _MINICHESS_PAWN_PST,
    2: _MINICHESS_KNIGHT_PST,
    3: _MINICHESS_ROOK_PST,
    4: _MINICHESS_QUEEN_PST,
    5: _MINICHESS_KING_MG,
}
_MINICHESS_PST_EG = {
    1: _MINICHESS_PAWN_PST,
    2: _MINICHESS_KNIGHT_PST,
    3: _MINICHESS_ROOK_PST,
    4: _MINICHESS_QUEEN_PST,
    5: _MINICHESS_KING_EG,
}

_MINICHESS_PHASE_WEIGHTS = {1: 0, 2: 1, 3: 2, 4: 4, 5: 0}
_MINICHESS_PHASE_TOTAL = 2 * (2 * 1 + 2 * 2 + 1 * 4)  # both sides, knights+rooks+queen


def _minichess_mirror_sq(sq: int) -> int:
    """Vertical mirror on the 6x6 board."""
    return (5 - sq // 6) * 6 + (sq % 6)


def _minichess_material(state: GameState) -> float:
    board = state.board_array()
    score = 0.0
    for sq in range(len(board)):
        piece = int(board[sq])
        if piece == 0:
            continue
        abs_piece = abs(piece)
        if abs_piece == 5:
            continue  # king
        value = _MINICHESS_PIECE_VALUES.get(abs_piece, 100)
        score += value if piece > 0 else -value
    if state.side_to_move() == 1:
        score = -score
    return score


def _minichess_rule_based(state: GameState) -> float:
    """Material + tapered PST + pawn structure + rook open-file bonus.

    6x6 board, no bishops, queens and rooks dominate. Pawn structure
    uses file count only (isolated / doubled), passed pawns are common
    on such a small board so we give a flat rank-based bonus.
    """
    board = state.board_array()

    mg = 0.0
    eg = 0.0
    phase = 0

    white_pawn_files = [0] * 6
    black_pawn_files = [0] * 6
    white_pawn_squares: List[int] = []
    black_pawn_squares: List[int] = []
    white_rooks: List[int] = []
    black_rooks: List[int] = []

    for sq in range(36):
        piece = int(board[sq])
        if piece == 0:
            continue
        abs_piece = abs(piece)
        phase += _MINICHESS_PHASE_WEIGHTS.get(abs_piece, 0)

        value = _MINICHESS_PIECE_VALUES.get(abs_piece, 100)
        if abs_piece == 5:
            value = 0
        pst_mg = _MINICHESS_PST_MG.get(abs_piece)
        pst_eg = _MINICHESS_PST_EG.get(abs_piece)

        if piece > 0:
            mg += value
            eg += value
            if pst_mg:
                mg += pst_mg[sq]
                eg += pst_eg[sq]
            if abs_piece == 1:
                white_pawn_files[sq % 6] += 1
                white_pawn_squares.append(sq)
            elif abs_piece == 3:
                white_rooks.append(sq)
        else:
            mg -= value
            eg -= value
            if pst_mg:
                m = _minichess_mirror_sq(sq)
                mg -= pst_mg[m]
                eg -= pst_eg[m]
            if abs_piece == 1:
                black_pawn_files[sq % 6] += 1
                black_pawn_squares.append(sq)
            elif abs_piece == 3:
                black_rooks.append(sq)

    # Pawn structure (doubled / isolated)
    for f in range(6):
        wc = white_pawn_files[f]
        bc = black_pawn_files[f]
        if wc >= 2:
            mg -= 15 * (wc - 1)
            eg -= 20 * (wc - 1)
        if bc >= 2:
            mg += 15 * (bc - 1)
            eg += 20 * (bc - 1)
        wl = white_pawn_files[f - 1] if f > 0 else 0
        wr = white_pawn_files[f + 1] if f < 5 else 0
        if wc > 0 and wl == 0 and wr == 0:
            mg -= 15
            eg -= 20
        bl = black_pawn_files[f - 1] if f > 0 else 0
        br = black_pawn_files[f + 1] if f < 5 else 0
        if bc > 0 and bl == 0 and br == 0:
            mg += 15
            eg += 20

    # Passed pawn bonus (flat, rank-scaled)
    for sq in white_pawn_squares:
        f, r = sq % 6, sq // 6
        blocked = any(
            abs((bsq % 6) - f) <= 1 and (bsq // 6) > r
            for bsq in black_pawn_squares
        )
        if not blocked:
            advance = max(0, r - 1)
            mg += 15 + advance * 10
            eg += 25 + advance * 15
    for sq in black_pawn_squares:
        f, r = sq % 6, sq // 6
        blocked = any(
            abs((wsq % 6) - f) <= 1 and (wsq // 6) < r
            for wsq in white_pawn_squares
        )
        if not blocked:
            advance = max(0, 4 - r)
            mg -= 15 + advance * 10
            eg -= 25 + advance * 15

    # Rook on open / semi-open file
    for sq in white_rooks:
        f = sq % 6
        if white_pawn_files[f] == 0 and black_pawn_files[f] == 0:
            mg += 15
            eg += 8
        elif white_pawn_files[f] == 0:
            mg += 8
            eg += 4
    for sq in black_rooks:
        f = sq % 6
        if white_pawn_files[f] == 0 and black_pawn_files[f] == 0:
            mg -= 15
            eg -= 8
        elif black_pawn_files[f] == 0:
            mg -= 8
            eg -= 4

    phase = min(phase, _MINICHESS_PHASE_TOTAL)
    mg_w = phase / _MINICHESS_PHASE_TOTAL
    eg_w = 1.0 - mg_w
    score = mg * mg_w + eg * eg_w

    if state.side_to_move() == 1:
        score = -score
    return score


# ===========================================================================
# Minishogi rule-based evaluation (5x5)
# ===========================================================================
#
# Minishogi piece codes (src/games/minishogi/constants.py):
#   1=Pawn 2=Silver 3=Gold 4=Bishop 5=Rook 6=King
#   7=+Pawn(Tokin) 8=+Silver 9=Horse(+B) 10=Dragon(+R)
#   Interface hand indices: PAWN=0 SILVER=1 GOLD=2 BISHOP=3 ROOK=4
#   Board layout: row 0 = gote (black) back rank, row 4 = sente (white)
#   back rank. Sente advances toward row 0.
#
# Board is tiny and drops are powerful, so evaluation tracks: material,
# hand pieces (with a premium for flexibility), piece advancement (with
# a small rank-based bonus), and basic king safety (king-advancement
# penalty + exposed-king hand-threat modifier).

_MINISHOGI_BOARD_VALUES = [
    0,      # 0 empty
    100,    # 1 Pawn
    500,    # 2 Silver
    600,    # 3 Gold
    700,    # 4 Bishop
    900,    # 5 Rook
    0,      # 6 King (handled separately)
    450,    # 7 +Pawn (Tokin) — gold-like
    550,    # 8 +Silver — gold-like
    950,    # 9 Horse — bishop + orthogonal step
    1100,   # 10 Dragon — rook + diagonal step
]

# Hand values, indexed by hand piece type (PAWN=0..ROOK=4). Premium over
# board counterparts because drops can land anywhere and create immediate
# threats on such a small board.
_MINISHOGI_HAND_VALUES = [120, 560, 660, 780, 970]

# Advancement bonus per rank forward, indexed by absolute board code 0..10.
_MINISHOGI_ADVANCE = [
    0,   # 0 empty
    8,   # 1 Pawn — promotion very close
    4,   # 2 Silver
    1,   # 3 Gold — defensive
    1,   # 4 Bishop — diagonal, rank-agnostic
    4,   # 5 Rook — 5th rank raid
    0,   # 6 King
    2,   # 7 +Pawn
    2,   # 8 +Silver
    0,   # 9 Horse
    2,   # 10 Dragon
]


def _minishogi_material(state: GameState) -> float:
    """Minishogi material + hand + advancement, from sente's view."""
    board = state.board_array()
    score = 0.0
    n = len(_MINISHOGI_BOARD_VALUES)
    for sq in range(len(board)):
        piece = int(board[sq])
        if piece == 0:
            continue
        pv = piece if piece > 0 else -piece
        if pv >= n:
            continue
        value = _MINISHOGI_BOARD_VALUES[pv]
        adv = _MINISHOGI_ADVANCE[pv]
        rank = sq // 5
        if piece > 0:
            score += value + adv * (4 - rank)
        else:
            score -= value + adv * rank

    h_len = len(_MINISHOGI_HAND_VALUES)
    for pt, cnt in state.hand_pieces(0).items():
        if 0 <= pt < h_len:
            score += _MINISHOGI_HAND_VALUES[pt] * cnt
    for pt, cnt in state.hand_pieces(1).items():
        if 0 <= pt < h_len:
            score -= _MINISHOGI_HAND_VALUES[pt] * cnt
    return score


def _minishogi_king_safety(state: GameState) -> float:
    """Sente-POV king safety: penalise advancement and exposure.

    The 5x5 board is so tight that a king leaving its home rank is
    usually lethal. We add a hand-threat term because drops make any
    nearby empty square a mate candidate.
    """
    board = state.board_array()
    score = 0.0

    for side, sign, home_rank in ((0, 1, 4), (1, -1, 0)):
        ksq = state.king_square(side)
        if ksq is None or ksq < 0:
            continue
        kr, kf = ksq // 5, ksq % 5

        forward = (home_rank - kr) if side == 0 else (kr - home_rank)
        if forward > 0:
            score += sign * (-60 * forward)
        else:
            score += sign * 20  # on home rank

        # Central-file penalty (file 2 is the true centre)
        if kf == 2:
            score += sign * (-35)
        elif kf in (0, 4):
            score += sign * 15

        # Hand-threat: count opponent's hand value; scale by exposure.
        opp_side = 1 - side
        hand_threat = 0
        for pt, cnt in state.hand_pieces(opp_side).items():
            if 0 <= pt < len(_MINISHOGI_HAND_VALUES):
                hand_threat += _MINISHOGI_HAND_VALUES[pt] * cnt
        exposure = max(0, forward) + (1 if kf == 2 else 0)
        if exposure > 0:
            score -= sign * (hand_threat * exposure) // 15

        # Defender count in the 3x3 ring around the king.
        defender_bonus = 0
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr, nf = kr + dr, kf + df
                if not (0 <= nr < 5 and 0 <= nf < 5):
                    continue
                v = int(board[nr * 5 + nf])
                if v == 0:
                    continue
                is_friend = (v > 0) == (side == 0)
                if not is_friend:
                    continue
                av = abs(v)
                if av == 3 or av in (7, 8):  # gold / tokin / +silver
                    defender_bonus += 25
                elif av == 2:  # silver
                    defender_bonus += 20
                else:
                    defender_bonus += 10
        score += sign * defender_bonus

    return score


def _minishogi_rule_based(state: GameState) -> float:
    score = _minishogi_material(state) + _minishogi_king_safety(state)
    if state.side_to_move() == 1:
        score = -score
    return score


def _minishogi_material_only(state: GameState) -> float:
    score = _minishogi_material(state)
    if state.side_to_move() == 1:
        score = -score
    return score


# ===========================================================================
# Public evaluator classes (dispatch on game name)
# ===========================================================================


def _chess_rule_based_fast(state: GameState) -> float:
    """C-backed chess evaluator. Bit-identical to ``_chess_rule_based``.

    Calls ``chess_c_evaluate`` from the accel extension with the board
    array (via buffer protocol — no bytes() copy needed), the side to
    move, and the two precomputed king squares. The full positional
    calculation happens in C in ~0.3us.
    """
    return _chess_c_evaluate(
        state.board_array(),
        int(state.side_to_move()),
        int(state.king_square(0)),
        int(state.king_square(1)),
    )


_RULE_BASED_DISPATCH = {
    "chess": (
        _chess_rule_based_fast if _HAS_CHESS_C_EVAL else _chess_rule_based
    ),
    "minichess": _minichess_rule_based,
    "shogi":     _shogi_evaluate,
    "minishogi": _minishogi_rule_based,
}

_MATERIAL_DISPATCH = {
    "chess":     _chess_material,
    "minichess": _minichess_material,
    "shogi":     _shogi_material_only,
    "minishogi": _minishogi_material_only,
}


class RuleBasedEvaluator:
    """Material + positional evaluator. Dispatches per game via state.config()."""

    # Kept for backward compatibility with callers that introspect this attr.
    PIECE_VALUES = _CHESS_PIECE_VALUES

    def evaluate(self, state: GameState) -> float:
        if not hasattr(state, "board_array"):
            return 0.0
        name = _game_name(state)
        fn = _RULE_BASED_DISPATCH.get(name, _chess_rule_based)
        return fn(state)

    def set_position(self, state: GameState):
        pass

    def push_move(self, state_before, move, state_after):
        pass

    def pop_move(self):
        pass


class MaterialEvaluator:
    """Pure material evaluator. Dispatches per game via state.config()."""

    PIECE_VALUES = _CHESS_PIECE_VALUES

    def evaluate(self, state: GameState) -> float:
        if not hasattr(state, "board_array"):
            return 0.0
        name = _game_name(state)
        fn = _MATERIAL_DISPATCH.get(name, _chess_material)
        return fn(state)

    def set_position(self, state: GameState):
        pass

    def push_move(self, state_before, move, state_after):
        pass

    def pop_move(self):
        pass
