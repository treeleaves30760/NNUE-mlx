/* ========================================================================
 * Shogi rule-based evaluator in C — 1:1 port of _shogi_material and
 * _shogi_king_safety from src/search/evaluator.py.
 *
 * Score is returned from the side-to-move's perspective in centipawns.
 * Any change to the Python heuristics MUST be mirrored here (and vice-
 * versa), because both paths feed bootstrap training data.
 * ======================================================================= */

#include "_nnue_accel_header.h"
#include "_shogi_position.h"

/* Board piece values keyed by absolute board code 0..14.
 * Matches _SHOGI_BOARD_VALUES in src/search/evaluator.py. */
static const int16_t SH_EVAL_BOARD_VALUES[15] = {
      0,   /* 0 empty */
    100,   /* 1 Pawn */
    430,   /* 2 Lance */
    450,   /* 3 Knight */
    640,   /* 4 Silver */
    690,   /* 5 Gold */
    890,   /* 6 Bishop */
   1040,   /* 7 Rook */
      0,   /* 8 King (game-ending; handled separately) */
    520,   /* 9 +Pawn (Tokin) */
    530,   /* 10 +Lance */
    540,   /* 11 +Knight */
    570,   /* 12 +Silver */
   1150,   /* 13 +Bishop (Horse) */
   1300,   /* 14 +Rook (Dragon) */
};

/* Hand values, indexed by API piece type 0..6. */
static const int16_t SH_EVAL_HAND_VALUES[7] = {
    115, 480, 510, 720, 780, 950, 1100,
};

/* Advance bonus per rank forward, indexed by absolute board code 0..14. */
static const int8_t SH_EVAL_PIECE_ADVANCE[15] = {
    0, /*empty*/
    6, 4, 7, 3, 1, 0, 3,  /* Pawn,Lance,Knight,Silver,Gold,Bishop,Rook */
    0,                    /* King */
    2, 2, 2, 2,           /* +Pawn,+Lance,+Knight,+Silver */
    0, 2,                 /* Horse, Dragon */
};

/* ---- Per-piece PSTs (sente POV) --------------------------------------
 *
 * Must stay in lockstep with the _SHOGI_*_PST tables in evaluator.py.
 * Gote pieces look up via the point-mirror (rotate 180°) defined in
 * sh_mirror_sq, matching the shogi initial position's point-symmetry.
 */

static const int16_t SH_PAWN_PST[81] = {
    72, 72, 72, 74, 74, 74, 72, 72, 72,
    50, 50, 50, 52, 52, 52, 50, 50, 50,
    32, 32, 32, 34, 34, 34, 32, 32, 32,
    18, 18, 18, 20, 20, 20, 18, 18, 18,
     8,  8,  8, 10, 10, 10,  8,  8,  8,
     2,  2,  2,  4,  4,  4,  2,  2,  2,
     0,  0,  0,  2,  2,  2,  0,  0,  0,
     0,  0,  0,  2,  2,  2,  0,  0,  0,
     0,  0,  0,  2,  2,  2,  0,  0,  0,
};

static const int16_t SH_LANCE_PST[81] = {
    19, 19, 16, 16, 16, 16, 16, 19, 19,
    17, 17, 14, 14, 14, 14, 14, 17, 17,
    15, 15, 12, 12, 12, 12, 12, 15, 15,
    13, 13, 10, 10, 10, 10, 10, 13, 13,
    11, 11,  8,  8,  8,  8,  8, 11, 11,
     9,  9,  6,  6,  6,  6,  6,  9,  9,
     7,  7,  4,  4,  4,  4,  4,  7,  7,
     5,  5,  2,  2,  2,  2,  2,  5,  5,
     3,  3,  0,  0,  0,  0,  0,  3,  3,
};

static const int16_t SH_KNIGHT_PST[81] = {
    22, 32, 40, 40, 40, 40, 40, 32, 22,
    18, 28, 36, 36, 36, 36, 36, 28, 18,
    14, 24, 32, 32, 32, 32, 32, 24, 14,
    10, 20, 28, 28, 28, 28, 28, 20, 10,
     6, 16, 24, 24, 24, 24, 24, 16,  6,
     2, 12, 20, 20, 20, 20, 20, 12,  2,
    -2,  8, 16, 16, 16, 16, 16,  8, -2,
    -6,  4, 12, 12, 12, 12, 12,  4, -6,
   -10,  0,  8,  8,  8,  8,  8,  0, -10,
};

static const int16_t SH_SILVER_PST[81] = {
    24, 24, 29, 29, 29, 29, 29, 24, 24,
    21, 21, 26, 26, 26, 26, 26, 21, 21,
    18, 18, 23, 23, 23, 23, 23, 18, 18,
    15, 15, 20, 20, 20, 20, 20, 15, 15,
    12, 12, 17, 17, 17, 17, 17, 12, 12,
     9,  9, 14, 14, 14, 14, 14,  9,  9,
     6,  6, 11, 11, 11, 11, 11,  6,  6,
     3,  3,  8,  8,  8,  8,  8,  3,  3,
     0,  0,  5,  5,  5,  5,  5,  0,  0,
};

static const int16_t SH_GOLD_PST[81] = {
     10,   10,   10,   13,   13,   13,   10,   10,   10,
     15,   15,   15,   18,   18,   18,   15,   15,   15,
      5,    5,    5,    8,    8,    8,    5,    5,    5,
      6,    6,    6,    9,    9,    9,    6,    6,    6,
      8,    8,    8,   11,   11,   11,    8,    8,    8,
     10,   10,   10,   13,   13,   13,   10,   10,   10,
     12,   12,   12,   15,   15,   15,   12,   12,   12,
     14,   14,   14,   17,   17,   17,   14,   14,   14,
     16,   16,   16,   19,   19,   19,   16,   16,   16,
};

static const int16_t SH_BISHOP_PST[81] = {
      5,    3,    6,    9,   12,    9,    6,    3,    5,
      3,   11,    9,   12,   15,   12,    9,   11,    3,
      6,    9,   17,   15,   18,   15,   17,    9,    6,
      9,   12,   15,   23,   21,   23,   15,   12,    9,
     12,   15,   18,   21,   29,   21,   18,   15,   12,
      9,   12,   15,   23,   21,   23,   15,   12,    9,
      6,    9,   17,   15,   18,   15,   17,    9,    6,
      3,   11,    9,   12,   15,   12,    9,   11,    3,
      5,    3,    6,    9,   12,    9,    6,    3,    5,
};

static const int16_t SH_ROOK_PST[81] = {
     40,   40,   40,   40,   40,   40,   40,   43,   40,
     35,   35,   35,   35,   35,   35,   35,   38,   35,
     30,   30,   30,   30,   30,   30,   30,   33,   30,
     10,   10,   10,   10,   10,   10,   10,   13,   10,
     10,   10,   10,   10,   10,   10,   10,   13,   10,
     10,   10,   10,   10,   10,   10,   10,   13,   10,
     10,   10,   10,   10,   10,   10,   10,   13,   10,
      5,    5,    5,    5,    5,    5,    5,    8,    5,
      0,    0,    0,    0,    0,    0,    0,    3,    0,
};

static const int16_t SH_KING_PST[81] = {
    -90, -100, -105, -135, -135, -135, -105, -100,  -90,
    -75,  -85,  -90, -120, -120, -120,  -90,  -85,  -75,
    -60,  -70,  -75, -105, -105, -105,  -75,  -70,  -60,
    -45,  -55,  -60,  -90,  -90,  -90,  -60,  -55,  -45,
    -30,  -40,  -45,  -75,  -75,  -75,  -45,  -40,  -30,
    -15,  -25,  -30,  -60,  -60,  -60,  -30,  -25,  -15,
     -5,  -15,  -20,  -50,  -50,  -50,  -20,  -15,   -5,
     20,   10,    5,  -25,  -25,  -25,    5,   10,   20,
     30,   20,   15,  -15,  -15,  -15,   15,   20,   30,
};

static const int16_t SH_HORSE_PST[81] = {
     24,   16,   16,   16,   16,   16,   16,   16,   24,
     16,   28,   20,   20,   20,   20,   20,   28,   16,
     16,   20,   32,   24,   24,   24,   32,   20,   16,
     16,   20,   24,   36,   28,   36,   24,   20,   16,
     16,   20,   24,   28,   40,   28,   24,   20,   16,
     16,   20,   24,   36,   28,   36,   24,   20,   16,
     16,   20,   32,   24,   24,   24,   32,   20,   16,
     16,   28,   20,   20,   20,   20,   20,   28,   16,
     24,   16,   16,   16,   16,   16,   16,   16,   24,
};

static const int16_t SH_DRAGON_PST[81] = {
     40,   43,   46,   49,   52,   49,   46,   43,   40,
     39,   39,   42,   45,   48,   45,   42,   39,   39,
     38,   38,   38,   41,   44,   41,   38,   38,   38,
     37,   37,   37,   37,   40,   37,   37,   37,   37,
     27,   27,   27,   27,   27,   27,   27,   27,   27,
     24,   24,   24,   24,   27,   24,   24,   24,   24,
     21,   21,   21,   24,   27,   24,   21,   21,   21,
     18,   18,   21,   24,   27,   24,   21,   18,   18,
     15,   18,   21,   24,   27,   24,   21,   18,   15,
};

/* Map absolute piece code 1..14 to its PST. Promoted small pieces
 * (+P/+L/+N/+S) all share the Gold table because they move like gold. */
static const int16_t *sh_pst_for(int abs_p) {
    switch (abs_p) {
        case 1:  return SH_PAWN_PST;
        case 2:  return SH_LANCE_PST;
        case 3:  return SH_KNIGHT_PST;
        case 4:  return SH_SILVER_PST;
        case 5:  return SH_GOLD_PST;
        case 6:  return SH_BISHOP_PST;
        case 7:  return SH_ROOK_PST;
        case 8:  return SH_KING_PST;
        case 9:  return SH_GOLD_PST;   /* tokin */
        case 10: return SH_GOLD_PST;   /* +lance */
        case 11: return SH_GOLD_PST;   /* +knight */
        case 12: return SH_GOLD_PST;   /* +silver */
        case 13: return SH_HORSE_PST;
        case 14: return SH_DRAGON_PST;
    }
    return NULL;
}

/* Point-mirror (180° rotation) for the sente-POV PSTs. Must match
 * _shogi_mirror_sq in evaluator.py. */
static inline int sh_mirror_sq(int sq) {
    int r = sq / 9;
    int f = sq % 9;
    return (8 - r) * 9 + (8 - f);
}

static int32_t shogi_eval_piece_psts(const ShogiPosition *p) {
    int32_t score = 0;
    const int8_t *board = p->board;
    for (int sq = 0; sq < SHOGI_SQUARES; sq++) {
        int piece = board[sq];
        if (piece == 0) continue;
        int abs_p = piece > 0 ? piece : -piece;
        const int16_t *pst = sh_pst_for(abs_p);
        if (pst == NULL) continue;
        if (piece > 0) {
            score += pst[sq];
        } else {
            score -= pst[sh_mirror_sq(sq)];
        }
    }
    return score;
}

/* ------------------------------------------------------------------------
 * _shogi_material: board material + advance + hand pieces (sente view).
 * ------------------------------------------------------------------------ */
static int32_t shogi_eval_material(const ShogiPosition *p) {
    int32_t score = 0;
    const int8_t *board = p->board;
    for (int sq = 0; sq < SHOGI_SQUARES; sq++) {
        int8_t piece = board[sq];
        if (piece == 0) continue;
        int pv = (piece > 0) ? piece : -piece;
        if (pv >= 15) continue;
        int value = SH_EVAL_BOARD_VALUES[pv];
        int adv_w = SH_EVAL_PIECE_ADVANCE[pv];
        int rank = sq / 9;
        if (piece > 0) {
            /* Sente advance = distance toward rank 0. */
            score += value + adv_w * (8 - rank);
        } else {
            score -= value + adv_w * rank;
        }
    }
    /* Hand pieces */
    for (int pt = 0; pt < SHOGI_HAND_TYPES; pt++) {
        score += (int32_t)p->sente_hand[pt] * SH_EVAL_HAND_VALUES[pt];
        score -= (int32_t)p->gote_hand[pt]  * SH_EVAL_HAND_VALUES[pt];
    }
    return score;
}

/* ------------------------------------------------------------------------
 * _shogi_king_safety: advance / central-file / defender-ring / attacker
 * proximity / hand-threat exposure (sente view).
 * ------------------------------------------------------------------------ */
static int32_t shogi_eval_king_safety(const ShogiPosition *p) {
    const int8_t *board = p->board;
    int32_t score = 0;

    for (int persp = 0; persp < 2; persp++) {
        int side = persp;
        int sign = (side == 0) ? 1 : -1;
        int home_rank = (side == 0) ? 8 : 0;
        int ksq = (side == 0) ? p->sente_king_sq : p->gote_king_sq;
        if (ksq < 0 || ksq >= 81) continue;
        int kr = ksq / 9;
        int kf = ksq % 9;

        int32_t side_score = 0;

        /* 1. Advancement penalty */
        int forward = (side == 0) ? (home_rank - kr) : (kr - home_rank);
        if (forward > 0) {
            side_score += -35 * forward;
        } else {
            side_score += 15;
        }

        /* 2. Central-file / wing-file */
        if (kf >= 3 && kf <= 5) {
            side_score += -40;
        } else if (kf == 0 || kf == 8) {
            side_score += 20;
        } else if (kf == 1 || kf == 7) {
            side_score += 10;
        }

        /* 3. Defender count in 3x3, 4. attacker proximity in 5x5 */
        int gold_val   = (side == 0) ?  SP_GOLD   : -SP_GOLD;
        int silver_val = (side == 0) ?  SP_SILVER : -SP_SILVER;
        int defender_bonus  = 0;
        int attacker_penalty = 0;
        for (int dr = -2; dr <= 2; dr++) {
            for (int df = -2; df <= 2; df++) {
                int nr = kr + dr, nf = kf + df;
                if (nr < 0 || nr >= 9 || nf < 0 || nf >= 9) continue;
                int8_t v = board[nr * 9 + nf];
                if (v == 0) continue;
                int is_friend =
                    (side == 0) ? (v > 0) : (v < 0);
                int cheb = (dr < 0 ? -dr : dr);
                int absf = (df < 0 ? -df : df);
                if (absf > cheb) cheb = absf;
                if (is_friend) {
                    if (cheb <= 1) {
                        int av = (v > 0) ? v : -v;
                        if (av == SP_GOLD || av == SP_SILVER) {
                            defender_bonus += 35;
                        } else if (av >= SP_PPAWN && av <= SP_PSILVER) {
                            defender_bonus += 30;
                        } else {
                            defender_bonus += 12;
                        }
                    }
                } else {
                    int av = (v > 0) ? v : -v;
                    int weight;
                    if (av == SP_ROOK || av == SP_DRAGON)        weight = 90;
                    else if (av == SP_BISHOP || av == SP_HORSE)  weight = 70;
                    else if (av == SP_GOLD ||
                             (av >= SP_PPAWN && av <= SP_PSILVER)) weight = 45;
                    else if (av == SP_SILVER)                    weight = 40;
                    else                                         weight = 25;
                    attacker_penalty += weight * (3 - cheb);
                }
            }
        }
        (void)gold_val; (void)silver_val;

        /* Hand-piece threat: opponent's hand weighted by king exposure */
        {
            int hand_threat = 0;
            const int8_t *opp_hand = (side == 0) ? p->gote_hand : p->sente_hand;
            for (int pt = 0; pt < SHOGI_HAND_TYPES; pt++) {
                hand_threat += (int)opp_hand[pt] * SH_EVAL_HAND_VALUES[pt];
            }
            int exposure = (forward > 0 ? forward : 0) +
                           ((kf >= 3 && kf <= 5) ? 1 : 0);
            if (exposure > 0) {
                attacker_penalty += (hand_threat * exposure) / 20;
            }
        }

        side_score += defender_bonus;
        side_score -= attacker_penalty;

        /* Accumulate into sente-view total */
        score += sign * side_score;
    }
    return score;
}

/* ------------------------------------------------------------------------
 * _shogi_castle_bonus: reward king-on-wing with a gold/silver wall.
 * Lockstep with _shogi_castle_bonus in src/search/evaluator.py.
 * ------------------------------------------------------------------------ */
static int32_t shogi_eval_castle_bonus(const ShogiPosition *p) {
    const int8_t *board = p->board;
    int32_t score = 0;

    for (int side = 0; side < 2; side++) {
        int sign = (side == 0) ? 1 : -1;
        int ksq = (side == 0) ? p->sente_king_sq : p->gote_king_sq;
        if (ksq < 0 || ksq >= 81) continue;
        int kr = ksq / 9;
        int kf = ksq % 9;

        /* Gate: king must be on a wing file (not 3..5) AND on its
         * home-side half (sente ranks 7..8, gote ranks 0..1). */
        if (kf >= 3 && kf <= 5) continue;
        if (side == 0) {
            if (kr < 7) continue;
        } else {
            if (kr > 1) continue;
        }

        int wall = 0;
        for (int dr = -2; dr <= 2; dr++) {
            for (int df = -2; df <= 2; df++) {
                if (dr == 0 && df == 0) continue;
                int nr = kr + dr, nf = kf + df;
                if (nr < 0 || nr >= 9 || nf < 0 || nf >= 9) continue;
                int8_t v = board[nr * 9 + nf];
                if (v == 0) continue;
                int is_friend = (side == 0) ? (v > 0) : (v < 0);
                if (!is_friend) continue;
                int av = (v > 0) ? v : -v;
                if (av == SP_SILVER || av == SP_GOLD ||
                    (av >= SP_PPAWN && av <= SP_PSILVER)) {
                    wall++;
                }
            }
        }

        if (wall >= 3) {
            int bonus = 40 + (wall - 3) * 15;
            if (bonus > 85) bonus = 85;
            score += sign * bonus;
        }
    }
    return score;
}


/* ------------------------------------------------------------------------
 * _shogi_attack_cluster: friendly attackers clustered around the ENEMY
 * king (sente view). Lockstep with _shogi_attack_cluster in
 * src/search/evaluator.py — any weight change must be mirrored.
 * ------------------------------------------------------------------------ */
static int32_t shogi_eval_attack_cluster(const ShogiPosition *p) {
    const int8_t *board = p->board;
    int32_t score = 0;

    for (int persp = 0; persp < 2; persp++) {
        int side = persp;              /* 0 = sente */
        int sign = (side == 0) ? 1 : -1;
        int opp = 1 - side;
        int ksq = (opp == 0) ? p->sente_king_sq : p->gote_king_sq;
        if (ksq < 0 || ksq >= 81) continue;
        int kr = ksq / 9;
        int kf = ksq % 9;

        int attack_weight = 0;
        for (int dr = -2; dr <= 2; dr++) {
            for (int df = -2; df <= 2; df++) {
                int nr = kr + dr, nf = kf + df;
                if (nr < 0 || nr >= 9 || nf < 0 || nf >= 9) continue;
                int8_t v = board[nr * 9 + nf];
                if (v == 0) continue;
                int is_friend = (side == 0) ? (v > 0) : (v < 0);
                if (!is_friend) continue;

                int av = (v > 0) ? v : -v;
                int w;
                if (av == SP_ROOK || av == SP_DRAGON)           w = 70;
                else if (av == SP_BISHOP || av == SP_HORSE)     w = 55;
                else if (av == SP_GOLD ||
                         (av >= SP_PPAWN && av <= SP_PSILVER))  w = 30;
                else if (av == SP_SILVER)                       w = 25;
                else if (av == SP_LANCE || av == SP_KNIGHT)     w = 18;
                else if (av == SP_PAWN)                         w = 10;
                else                                            continue;

                int cheb = (dr < 0 ? -dr : dr);
                int af = (df < 0 ? -df : df);
                if (af > cheb) cheb = af;
                attack_weight += w * (3 - cheb);
            }
        }

        /* Drop pressure: mirror of the defender-side hand-threat check. */
        int home_rank_opp = (opp == 0) ? 8 : 0;
        int forward_opp = (opp == 0)
            ? (home_rank_opp - kr)
            : (kr - home_rank_opp);
        int exposure = (forward_opp > 0 ? forward_opp : 0)
                     + ((kf >= 3 && kf <= 5) ? 1 : 0);
        if (exposure > 0) {
            int hand_total = 0;
            const int8_t *my_hand =
                (side == 0) ? p->sente_hand : p->gote_hand;
            for (int pt = 0; pt < SHOGI_HAND_TYPES; pt++) {
                hand_total += (int)my_hand[pt] * SH_EVAL_HAND_VALUES[pt];
            }
            attack_weight += (hand_total * exposure) / 25;
        }

        score += sign * attack_weight;
    }
    return score;
}


/* ------------------------------------------------------------------------
 * _shogi_rook_positional: rook invasion + file bonuses (sente view).
 * Keep in lockstep with _shogi_rook_positional in src/search/evaluator.py.
 * ------------------------------------------------------------------------ */
static int32_t shogi_eval_rook_positional(const ShogiPosition *p) {
    int32_t score = 0;
    const int8_t *board = p->board;
    for (int sq = 0; sq < SHOGI_SQUARES; sq++) {
        int8_t v = board[sq];
        if (v == 0) continue;
        int av = (v > 0) ? v : -v;
        if (av != SP_ROOK && av != SP_DRAGON) continue;
        int rank = sq / 9;
        int f = sq % 9;
        int invasion = (av == SP_DRAGON) ? 50 : 35;
        if (v > 0) {
            if (rank <= 2) score += invasion;
            if (f == 1 || f == 7) score += 8;
        } else {
            if (rank >= 6) score -= invasion;
            if (f == 1 || f == 7) score -= 8;
        }
    }
    return score;
}

/* ------------------------------------------------------------------------
 * shogi_rule_evaluate: total eval, flipped to side-to-move perspective.
 * ------------------------------------------------------------------------ */
int32_t shogi_rule_evaluate(const ShogiPosition *p) {
    int32_t score = shogi_eval_material(p)
                  + shogi_eval_piece_psts(p)
                  + shogi_eval_king_safety(p)
                  + shogi_eval_rook_positional(p)
                  + shogi_eval_attack_cluster(p)
                  + shogi_eval_castle_bonus(p);
    if (p->side == 1) score = -score;
    return score;
}

/* ---- Python wrapper for validation against the Python reference ------- */
static PyObject *accel_shogi_c_evaluate(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj, *sh_obj, *gh_obj;
    int side;
    if (!PyArg_ParseTuple(args, "OOOi", &board_obj, &sh_obj, &gh_obj, &side))
        return NULL;

    ShogiPosition p;
    if (sh_pyapi_parse_position(board_obj, sh_obj, gh_obj, side, &p) != 0)
        return NULL;

    int32_t score = shogi_rule_evaluate(&p);
    return PyFloat_FromDouble((double)score);
}
