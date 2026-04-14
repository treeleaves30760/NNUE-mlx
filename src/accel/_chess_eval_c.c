/* ========================================================================
 * Chess rule-based evaluator in C — line-for-line port of
 * _chess_rule_based in src/search/evaluator.py. Kept in lockstep via
 * tests/test_evaluator.py::test_chess_python_c_parity_25_plies.
 *
 * All arithmetic is integer. The taper is
 *   score = (mg * phase + eg * (24 - phase)) / 24
 * which matches the Python formulation exactly. Scales that would
 * otherwise be floats (passed pawn 1.5x in EG, etc.) are implemented
 * as rational integer factors — ``(x * 3) / 2`` instead of ``x * 1.5``.
 * Integer division in C rounds toward zero just like Python ``//`` for
 * non-negative inputs; for mixed signs the Python and C rounding
 * directions differ, so we use branch-free floor-division helpers
 * where mixed signs are possible.
 * ======================================================================= */

#include "_nnue_accel_header.h"
#include "_chess_position.h"

/* Python-style floor division (rounds toward negative infinity).
 * C's ``/`` truncates toward zero, so for negative numerators we need
 * to subtract one when there's a remainder. Used for the passed-pawn
 * EG scale and the mobility 6/10 scale. */
static inline int32_t cp_fdiv(int32_t num, int32_t den) {
    int32_t q = num / den;
    if ((num % den != 0) && ((num < 0) != (den < 0))) q -= 1;
    return q;
}

/* ------------------------------------------------------------------------
 * Piece values and tapered piece-square tables. Each pair of MG/EG
 * tables mirrors _CHESS_PAWN_MG/_CHESS_PAWN_EG/... in evaluator.py.
 * ------------------------------------------------------------------------ */

static const int16_t CP_PIECE_VALUES[7] = {
    0,       /* empty */
    100,     /* pawn */
    320,     /* knight */
    330,     /* bishop */
    500,     /* rook */
    900,     /* queen */
    0,       /* king (not counted in material swing) */
};

/* Phase weights: how much each piece contributes to middlegame-ness. */
static const int8_t CP_PHASE_WEIGHTS[7] = {
    0, 0, 1, 1, 2, 4, 0
};
#define CP_PHASE_TOTAL 24

/* Tapered piece-square tables (PeSTO). Orientation: ``sq = rank * 8 +
 * file`` with ``rank 0 = white back rank``; first row is rank 0, last
 * row is rank 7. Must stay in lockstep with _CHESS_*_MG/EG in
 * src/search/evaluator.py — enforced by test_chess_python_c_parity. */

static const int16_t CP_PAWN_MG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    -35,  -1, -20, -23, -15,  24,  38, -22,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -14,  13,   6,  21,  23,  12,  17, -23,
     -6,   7,  26,  31,  65,  56,  25, -20,
     98, 134,  61,  95,  68, 126,  34, -11,
      0,   0,   0,   0,   0,   0,   0,   0,
};

static const int16_t CP_PAWN_EG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     13,   8,   8,  10,  13,   0,   2,  -7,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     32,  24,  13,   5,  -2,   4,  17,  17,
     94, 100,  85,  67,  56,  53,  82,  84,
    178, 173, 158, 134, 147, 132, 165, 187,
      0,   0,   0,   0,   0,   0,   0,   0,
};

static const int16_t CP_KNIGHT_MG[64] = {
   -105, -21, -58, -33, -17, -28, -19, -23,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -13,   4,  16,  13,  28,  19,  21,  -8,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -47,  60,  37,  65,  84, 129,  73,  44,
    -73, -41,  72,  36,  23,  62,   7, -17,
   -167, -89, -34, -49,  61, -97, -15,-107,
};

static const int16_t CP_KNIGHT_EG[64] = {
    -29, -51, -23, -15, -22, -18, -50, -64,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -58, -38, -13, -28, -31, -27, -63, -99,
};

static const int16_t CP_BISHOP_MG[64] = {
    -33,  -3, -14, -21, -13, -12, -39, -21,
      4,  15,  16,   0,   7,  21,  33,   1,
      0,  15,  15,  15,  14,  27,  18,  10,
     -6,  13,  13,  26,  34,  12,  10,   4,
     -4,   5,  19,  50,  37,  37,   7,  -2,
    -16,  37,  43,  40,  35,  50,  37,  -2,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -29,   4, -82, -37, -25, -42,   7,  -8,
};

static const int16_t CP_BISHOP_EG[64] = {
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
     -3,   9,  12,   9,  14,  10,   3,   2,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
};

static const int16_t CP_ROOK_MG[64] = {
    -19, -13,   1,  17,  16,   7, -37, -26,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -24, -11,   7,  26,  24,  35,  -8, -20,
     -5,  19,  26,  36,  17,  45,  61,  16,
     27,  32,  58,  62,  80,  67,  26,  44,
     32,  42,  32,  51,  63,   9,  31,  43,
};

static const int16_t CP_ROOK_EG[64] = {
     -9,   2,   3,  -1,  -5, -13,   4, -20,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
      4,   3,  13,   1,   2,   1,  -1,   2,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
     11,  13,  13,  11,  -3,   3,   8,   3,
     13,  10,  18,  15,  12,  12,   8,   5,
};

static const int16_t CP_QUEEN_MG[64] = {
     -1, -18,  -9,  10, -15, -25, -31, -50,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -28,   0,  29,  12,  59,  44,  43,  45,
};

static const int16_t CP_QUEEN_EG[64] = {
    -33, -28, -22, -43,  -5, -32, -20, -41,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -16, -27,  15,   6,   9,  17,  10,   5,
     -1,  15,   2,  12,  17,  15,  20,  10,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -17,  20,  32,  41,  58,  25,  30,   0,
     -9,  22,  22,  27,  27,  19,  10,  20,
};

static const int16_t CP_KING_MG[64] = {
    -15,  36,  12, -54,   8, -28,  24,  14,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -14, -14, -22, -46, -44, -30, -15, -27,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -17, -20, -12, -27, -30, -25, -14, -36,
     -9,  24,   2, -16, -20,   6,  22, -22,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -65,  23,  16, -15, -56, -34,   2,  13,
};

static const int16_t CP_KING_EG[64] = {
    -53, -34, -21, -11, -28, -14, -24, -43,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -18,  -4,  21,  24,  27,  23,   9, -11,
     -8,  22,  24,  27,  26,  33,  26,   3,
     10,  17,  23,  15,  20,  45,  44,  13,
    -12,  17,  14,  17,  17,  38,  23,  11,
    -74, -35, -18, -18, -11,  15,   4, -17,
};

static inline const int16_t *cp_pst_mg(int abs_piece) {
    switch (abs_piece) {
        case CP_PAWN:   return CP_PAWN_MG;
        case CP_KNIGHT: return CP_KNIGHT_MG;
        case CP_BISHOP: return CP_BISHOP_MG;
        case CP_ROOK:   return CP_ROOK_MG;
        case CP_QUEEN:  return CP_QUEEN_MG;
        case CP_KING:   return CP_KING_MG;
    }
    return NULL;
}

static inline const int16_t *cp_pst_eg(int abs_piece) {
    switch (abs_piece) {
        case CP_PAWN:   return CP_PAWN_EG;
        case CP_KNIGHT: return CP_KNIGHT_EG;
        case CP_BISHOP: return CP_BISHOP_EG;
        case CP_ROOK:   return CP_ROOK_EG;
        case CP_QUEEN:  return CP_QUEEN_EG;
        case CP_KING:   return CP_KING_EG;
    }
    return NULL;
}

static inline int cp_mirror_sq(int sq) {
    return (7 - (sq >> 3)) * 8 + (sq & 7);
}

/* ------------------------------------------------------------------------
 * Feature sub-scores (white POV, integer-only). Mirrors the per-function
 * Python helpers in src/search/evaluator.py.
 * ------------------------------------------------------------------------ */

static int32_t chess_eval_pawn_structure(const int8_t *pfw, const int8_t *pfb) {
    int32_t score = 0;
    for (int f = 0; f < 8; f++) {
        int wc = pfw[f];
        int bc = pfb[f];

        if (wc >= 2) score -= 15 * (wc - 1);
        if (bc >= 2) score += 15 * (bc - 1);

        int wl = (f > 0) ? pfw[f - 1] : 0;
        int wr = (f < 7) ? pfw[f + 1] : 0;
        if (wc > 0 && wl == 0 && wr == 0) score -= 20 * wc;

        int bl = (f > 0) ? pfb[f - 1] : 0;
        int br = (f < 7) ? pfb[f + 1] : 0;
        if (bc > 0 && bl == 0 && br == 0) score += 20 * bc;
    }
    return score;
}

static int32_t chess_eval_connected_pawns(const int8_t *wpawns, int nw,
                                           const int8_t *bpawns, int nb) {
    /* Reconstruct occupancy bitsets for quick membership tests. */
    uint64_t w_set = 0, b_set = 0;
    for (int i = 0; i < nw; i++) w_set |= (uint64_t)1 << wpawns[i];
    for (int i = 0; i < nb; i++) b_set |= (uint64_t)1 << bpawns[i];

    int32_t score = 0;
    for (int i = 0; i < nw; i++) {
        int sq = wpawns[i];
        int f = sq & 7;
        int r = sq >> 3;
        if (f < 7 && (w_set >> (sq + 1)) & 1) score += 10;
        if (r > 0) {
            if (f > 0 && (w_set >> ((r - 1) * 8 + (f - 1))) & 1) score += 8;
            if (f < 7 && (w_set >> ((r - 1) * 8 + (f + 1))) & 1) score += 8;
        }
    }
    for (int i = 0; i < nb; i++) {
        int sq = bpawns[i];
        int f = sq & 7;
        int r = sq >> 3;
        if (f < 7 && (b_set >> (sq + 1)) & 1) score -= 10;
        if (r < 7) {
            if (f > 0 && (b_set >> ((r + 1) * 8 + (f - 1))) & 1) score -= 8;
            if (f < 7 && (b_set >> ((r + 1) * 8 + (f + 1))) & 1) score -= 8;
        }
    }
    return score;
}

/* Rank-based passed pawn bonus. Index = white pawn's rank (or mirrored
 * for black). Kept in lockstep with _CHESS_PASSED_PAWN_RANK_BONUS. */
static const int16_t CP_PASSED_PAWN_RANK_BONUS[8] = {
    0, 5, 15, 35, 75, 130, 180, 0,
};

static int32_t chess_eval_passed_pawns(const int8_t *wpawns, int nw,
                                        const int8_t *bpawns, int nb) {
    int32_t score = 0;
    for (int i = 0; i < nw; i++) {
        int sq = wpawns[i];
        int f = sq & 7;
        int r = sq >> 3;
        int blocked = 0;
        for (int j = 0; j < nb; j++) {
            int bsq = bpawns[j];
            int bf = bsq & 7;
            int br = bsq >> 3;
            int df = bf - f;
            if ((df >= -1 && df <= 1) && br > r) { blocked = 1; break; }
        }
        if (!blocked) {
            score += CP_PASSED_PAWN_RANK_BONUS[r];
        }
    }
    for (int i = 0; i < nb; i++) {
        int sq = bpawns[i];
        int f = sq & 7;
        int r = sq >> 3;
        int blocked = 0;
        for (int j = 0; j < nw; j++) {
            int wsq = wpawns[j];
            int wf = wsq & 7;
            int wr = wsq >> 3;
            int df = wf - f;
            if ((df >= -1 && df <= 1) && wr < r) { blocked = 1; break; }
        }
        if (!blocked) {
            score -= CP_PASSED_PAWN_RANK_BONUS[7 - r];
        }
    }
    return score;
}

/* Backward pawn penalty — pawn has no friendly pawn support on
 * adjacent files at or behind its rank, and its advance square is
 * controlled by an enemy pawn. See _chess_backward_pawns in
 * evaluator.py for the precise definition. */
static int32_t chess_eval_backward_pawns(const int8_t *wpawns, int nw,
                                          const int8_t *bpawns, int nb) {
    uint64_t w_set = 0, b_set = 0;
    for (int i = 0; i < nw; i++) w_set |= (uint64_t)1 << wpawns[i];
    for (int i = 0; i < nb; i++) b_set |= (uint64_t)1 << bpawns[i];

    int32_t score = 0;
    for (int i = 0; i < nw; i++) {
        int sq = wpawns[i];
        int f = sq & 7;
        int r = sq >> 3;
        int supported = 0;
        for (int df = -1; df <= 1 && !supported; df += 2) {
            int nf = f + df;
            if (nf < 0 || nf >= 8) continue;
            for (int nr = 0; nr <= r; nr++) {
                if ((w_set >> (nr * 8 + nf)) & 1) { supported = 1; break; }
            }
        }
        if (supported) continue;
        int adv_r = r + 1;
        if (adv_r >= 8) continue;
        int controlled = 0;
        for (int df = -1; df <= 1 && !controlled; df += 2) {
            int nf = f + df;
            if (nf < 0 || nf >= 8) continue;
            int attacker_r = adv_r + 1;
            if (attacker_r < 8 && ((b_set >> (attacker_r * 8 + nf)) & 1))
                controlled = 1;
        }
        if (controlled) score -= 10;
    }
    for (int i = 0; i < nb; i++) {
        int sq = bpawns[i];
        int f = sq & 7;
        int r = sq >> 3;
        int supported = 0;
        for (int df = -1; df <= 1 && !supported; df += 2) {
            int nf = f + df;
            if (nf < 0 || nf >= 8) continue;
            for (int nr = r; nr < 8; nr++) {
                if ((b_set >> (nr * 8 + nf)) & 1) { supported = 1; break; }
            }
        }
        if (supported) continue;
        int adv_r = r - 1;
        if (adv_r < 0) continue;
        int controlled = 0;
        for (int df = -1; df <= 1 && !controlled; df += 2) {
            int nf = f + df;
            if (nf < 0 || nf >= 8) continue;
            int attacker_r = adv_r - 1;
            if (attacker_r >= 0 && ((w_set >> (attacker_r * 8 + nf)) & 1))
                controlled = 1;
        }
        if (controlled) score += 10;
    }
    return score;
}

/* Trapped rook: rook cornered by its own king after a castling mistake. */
static int32_t chess_eval_trapped_rook(const int8_t *wrooks, int nwr,
                                        const int8_t *brooks, int nbr,
                                        int white_king_sq,
                                        int black_king_sq) {
    int32_t score = 0;
    if (white_king_sq >= 0) {
        int wkf = white_king_sq & 7;
        int wkr = white_king_sq >> 3;
        if (wkr == 0) {
            for (int i = 0; i < nwr; i++) {
                int rsq = wrooks[i];
                int rf = rsq & 7;
                int rr = rsq >> 3;
                if (rr != 0) continue;
                if (wkf >= 4 && wkf <= 6 && rf >= 5 && rf <= 7 && rf > wkf) {
                    score -= 40; break;
                }
                if (wkf >= 1 && wkf <= 3 && rf >= 0 && rf <= 2 && rf < wkf) {
                    score -= 40; break;
                }
            }
        }
    }
    if (black_king_sq >= 0) {
        int bkf = black_king_sq & 7;
        int bkr = black_king_sq >> 3;
        if (bkr == 7) {
            for (int i = 0; i < nbr; i++) {
                int rsq = brooks[i];
                int rf = rsq & 7;
                int rr = rsq >> 3;
                if (rr != 7) continue;
                if (bkf >= 4 && bkf <= 6 && rf >= 5 && rf <= 7 && rf > bkf) {
                    score += 40; break;
                }
                if (bkf >= 1 && bkf <= 3 && rf >= 0 && rf <= 2 && rf < bkf) {
                    score += 40; break;
                }
            }
        }
    }
    return score;
}

static int32_t chess_eval_bad_bishop(const int8_t *wbi, int nwb,
                                      const int8_t *bbi, int nbb,
                                      const int8_t *wpawns, int nw,
                                      const int8_t *bpawns, int nb) {
    int32_t score = 0;
    for (int i = 0; i < nwb; i++) {
        int bsq = wbi[i];
        int bc = ((bsq >> 3) + (bsq & 7)) & 1;
        int same = 0;
        for (int j = 0; j < nw; j++) {
            int psq = wpawns[j];
            if ((((psq >> 3) + (psq & 7)) & 1) == bc) same++;
        }
        score -= 4 * same;
    }
    for (int i = 0; i < nbb; i++) {
        int bsq = bbi[i];
        int bc = ((bsq >> 3) + (bsq & 7)) & 1;
        int same = 0;
        for (int j = 0; j < nb; j++) {
            int psq = bpawns[j];
            if ((((psq >> 3) + (psq & 7)) & 1) == bc) same++;
        }
        score += 4 * same;
    }
    return score;
}

static int32_t chess_eval_undeveloped_minors(const int8_t *wkn, int nwk,
                                               const int8_t *wbi, int nwb,
                                               const int8_t *bkn, int nbk,
                                               const int8_t *bbi, int nbb) {
    int32_t penalty = 0;
    for (int i = 0; i < nwk; i++) {
        int sq = wkn[i];
        if (sq == 1 || sq == 6) penalty -= 8;
    }
    for (int i = 0; i < nwb; i++) {
        int sq = wbi[i];
        if (sq == 2 || sq == 5) penalty -= 8;
    }
    for (int i = 0; i < nbk; i++) {
        int sq = bkn[i];
        if (sq == 57 || sq == 62) penalty += 8;
    }
    for (int i = 0; i < nbb; i++) {
        int sq = bbi[i];
        if (sq == 58 || sq == 61) penalty += 8;
    }
    return penalty;
}

static int cn_rooks_see_each_other(int sq0, int sq1, const int8_t *board) {
    int r0 = sq0 >> 3, f0 = sq0 & 7;
    int r1 = sq1 >> 3, f1 = sq1 & 7;
    if (r0 == r1) {
        int lo = f0 < f1 ? f0 : f1;
        int hi = f0 < f1 ? f1 : f0;
        for (int f = lo + 1; f < hi; f++) {
            if (board[r0 * 8 + f] != 0) return 0;
        }
        return 1;
    }
    if (f0 == f1) {
        int lo = r0 < r1 ? r0 : r1;
        int hi = r0 < r1 ? r1 : r0;
        for (int r = lo + 1; r < hi; r++) {
            if (board[r * 8 + f0] != 0) return 0;
        }
        return 1;
    }
    return 0;
}

static int32_t chess_eval_rook_connectivity(const int8_t *wro, int nwr,
                                              const int8_t *bro, int nbr,
                                              const int8_t *board) {
    int32_t score = 0;
    if (nwr == 2 && cn_rooks_see_each_other(wro[0], wro[1], board)) score += 15;
    if (nbr == 2 && cn_rooks_see_each_other(bro[0], bro[1], board)) score -= 15;
    return score;
}

/* Attacker weight table for king-zone pressure. Must stay in lockstep
 * with _CHESS_KING_ATTACK_WEIGHT in evaluator.py. */
static const int16_t CP_KING_ATTACK_WEIGHT[7] = {
    0,   /* empty */
    0,   /* pawn */
    20,  /* knight */
    20,  /* bishop */
    40,  /* rook */
    80,  /* queen */
    0,   /* king */
};

static int cp_piece_attacks_mask(const int8_t *board, int sq, int abs_p,
                                  uint64_t zone_mask) {
    int r0 = sq >> 3, f0 = sq & 7;
    int count = 0;
    if (abs_p == CP_KNIGHT) {
        static const int KNIGHT_OFFS[8][2] = {
            {1,2},{1,-2},{-1,2},{-1,-2},{2,1},{2,-1},{-2,1},{-2,-1}
        };
        for (int k = 0; k < 8; k++) {
            int nr = r0 + KNIGHT_OFFS[k][0];
            int nf = f0 + KNIGHT_OFFS[k][1];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                if (zone_mask & ((uint64_t)1 << (nr * 8 + nf))) count++;
            }
        }
        return count;
    }
    const int (*dirs)[2];
    int ndirs;
    static const int BISHOP_DIRS[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static const int ROOK_DIRS[4][2]   = {{1,0},{-1,0},{0,1},{0,-1}};
    static const int QUEEN_DIRS[8][2]  = {
        {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
    };
    if (abs_p == CP_BISHOP) { dirs = BISHOP_DIRS; ndirs = 4; }
    else if (abs_p == CP_ROOK) { dirs = ROOK_DIRS; ndirs = 4; }
    else if (abs_p == CP_QUEEN) { dirs = QUEEN_DIRS; ndirs = 8; }
    else return 0;
    for (int d = 0; d < ndirs; d++) {
        int dr = dirs[d][0], df = dirs[d][1];
        int nr = r0 + dr, nf = f0 + df;
        while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
            if (zone_mask & ((uint64_t)1 << (nr * 8 + nf))) count++;
            if (board[nr * 8 + nf] != 0) break;
            nr += dr; nf += df;
        }
    }
    return count;
}

static int32_t chess_eval_king_danger_from(const int8_t *board, int king_sq,
                                             int attacker_sign) {
    int kr = king_sq >> 3, kf = king_sq & 7;
    uint64_t zone_mask = 0;
    for (int dr = -1; dr <= 1; dr++) {
        for (int df = -1; df <= 1; df++) {
            int nr = kr + dr, nf = kf + df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                zone_mask |= (uint64_t)1 << (nr * 8 + nf);
            }
        }
    }
    int attackers = 0;
    int weight_sum = 0;
    for (int sq = 0; sq < 64; sq++) {
        int8_t piece = board[sq];
        if (piece == 0) continue;
        if (attacker_sign > 0 && piece < 0) continue;
        if (attacker_sign < 0 && piece > 0) continue;
        int abs_p = piece > 0 ? piece : -piece;
        if (abs_p < CP_KNIGHT || abs_p > CP_QUEEN) continue;
        int attacks = cp_piece_attacks_mask(board, sq, abs_p, zone_mask);
        if (attacks > 0) {
            attackers++;
            weight_sum += CP_KING_ATTACK_WEIGHT[abs_p] * attacks;
        }
    }
    if (attackers == 0) return 0;
    int32_t danger = (weight_sum * attackers) / 8;
    if (danger > 400) danger = 400;
    return danger;
}

static int32_t chess_eval_king_attack_pressure(const int8_t *board,
                                                 int wk_sq, int bk_sq) {
    int32_t score = 0;
    if (wk_sq >= 0) score -= chess_eval_king_danger_from(board, wk_sq, -1);
    if (bk_sq >= 0) score += chess_eval_king_danger_from(board, bk_sq, +1);
    return score;
}

static int32_t chess_eval_knight_outposts(const int8_t *wkn, int nwk,
                                           const int8_t *bkn, int nbk,
                                           const int8_t *wpawns, int nw,
                                           const int8_t *bpawns, int nb) {
    uint64_t w_pawn_set = 0, b_pawn_set = 0;
    for (int i = 0; i < nw; i++) w_pawn_set |= (uint64_t)1 << wpawns[i];
    for (int i = 0; i < nb; i++) b_pawn_set |= (uint64_t)1 << bpawns[i];

    int32_t score = 0;
    for (int i = 0; i < nwk; i++) {
        int sq = wkn[i];
        int f = sq & 7;
        int r = sq >> 3;
        if (r < 4) continue;
        int supported = 0;
        if (r > 0) {
            if (f > 0 && (w_pawn_set >> ((r - 1) * 8 + (f - 1))) & 1)
                supported = 1;
            else if (f < 7 && (w_pawn_set >> ((r - 1) * 8 + (f + 1))) & 1)
                supported = 1;
        }
        if (!supported) continue;
        int safe = 1;
        for (int d = -1; d <= 1 && safe; d += 2) {
            int nf = f + d;
            if (nf < 0 || nf >= 8) continue;
            for (int nr = r + 1; nr < 8; nr++) {
                if ((b_pawn_set >> (nr * 8 + nf)) & 1) { safe = 0; break; }
            }
        }
        if (safe) score += 20 + (r - 4) * 5;
    }
    for (int i = 0; i < nbk; i++) {
        int sq = bkn[i];
        int f = sq & 7;
        int r = sq >> 3;
        if (r > 3) continue;
        int supported = 0;
        if (r < 7) {
            if (f > 0 && (b_pawn_set >> ((r + 1) * 8 + (f - 1))) & 1)
                supported = 1;
            else if (f < 7 && (b_pawn_set >> ((r + 1) * 8 + (f + 1))) & 1)
                supported = 1;
        }
        if (!supported) continue;
        int safe = 1;
        for (int d = -1; d <= 1 && safe; d += 2) {
            int nf = f + d;
            if (nf < 0 || nf >= 8) continue;
            for (int nr = r - 1; nr >= 0; nr--) {
                if ((w_pawn_set >> (nr * 8 + nf)) & 1) { safe = 0; break; }
            }
        }
        if (safe) score -= 20 + (3 - r) * 5;
    }
    return score;
}

static int32_t chess_eval_king_shelter(int king_sq, int side,
                                        const int8_t *friendly_pawns,
                                        int np) {
    if (king_sq < 0) return 0;
    int kf = king_sq & 7;
    int kr = king_sq >> 3;
    int want_r1, want_r2;
    if (side == 0) {
        if (kr > 2) return 0;
        want_r1 = kr + 1;
        want_r2 = kr + 2;
    } else {
        if (kr < 5) return 0;
        want_r1 = kr - 1;
        want_r2 = kr - 2;
    }

    int shelter = 0;
    for (int df = -1; df <= 1; df++) {
        int pf = kf + df;
        if (pf < 0 || pf >= 8) continue;
        int best = 0;
        for (int i = 0; i < np; i++) {
            int psq = friendly_pawns[i];
            if ((psq & 7) != pf) continue;
            int pr = psq >> 3;
            if (pr == want_r1) { if (best < 20) best = 20; }
            else if (pr == want_r2) { if (best < 10) best = 10; }
        }
        shelter += best;
    }
    return shelter;
}

/* Mobility: inlined ray walks; matches the Python _chess_mobility body. */
static int32_t chess_eval_mobility(const int8_t *board,
                                    const int8_t *wkn, int nwk,
                                    const int8_t *bkn, int nbk,
                                    const int8_t *wbi, int nwb,
                                    const int8_t *bbi, int nbb,
                                    const int8_t *wro, int nwr,
                                    const int8_t *bro, int nbr,
                                    const int8_t *wqu, int nwq,
                                    const int8_t *bqu, int nbq) {
    int32_t score = 0;
    static const int KNIGHT_OFFS[8][2] = {
        {1,2},{1,-2},{-1,2},{-1,-2},{2,1},{2,-1},{-2,1},{-2,-1}
    };

    /* Knights */
    int kw_c = 0;
    for (int i = 0; i < nwk; i++) {
        int sq = wkn[i];
        int r0 = sq >> 3;
        int f0 = sq & 7;
        for (int k = 0; k < 8; k++) {
            int r = r0 + KNIGHT_OFFS[k][0];
            int f = f0 + KNIGHT_OFFS[k][1];
            if (r < 0 || r >= 8 || f < 0 || f >= 8) continue;
            int t = board[(r << 3) + f];
            if (t <= 0) kw_c++;
        }
    }
    int kb_c = 0;
    for (int i = 0; i < nbk; i++) {
        int sq = bkn[i];
        int r0 = sq >> 3;
        int f0 = sq & 7;
        for (int k = 0; k < 8; k++) {
            int r = r0 + KNIGHT_OFFS[k][0];
            int f = f0 + KNIGHT_OFFS[k][1];
            if (r < 0 || r >= 8 || f < 0 || f >= 8) continue;
            int t = board[(r << 3) + f];
            if (t >= 0) kb_c++;
        }
    }
    score += (kw_c - kb_c) * 3;

    /* Helper macro: walk one ray direction (dr,df) and accumulate. */
#define CP_RAY(start_sq, dr, df, accum, own_sign) do {                       \
        int __r = ((start_sq) >> 3) + (dr);                                  \
        int __f = ((start_sq) & 7) + (df);                                   \
        while (__r >= 0 && __r < 8 && __f >= 0 && __f < 8) {                 \
            int __t = board[(__r << 3) + __f];                               \
            if (__t == 0) { (accum)++; }                                      \
            else {                                                            \
                if (((own_sign) > 0 && __t < 0) ||                            \
                    ((own_sign) < 0 && __t > 0)) (accum)++;                   \
                break;                                                        \
            }                                                                 \
            __r += (dr);                                                      \
            __f += (df);                                                      \
        }                                                                     \
    } while (0)

    /* Bishops (4 diagonals) */
    int bw_c = 0;
    for (int i = 0; i < nwb; i++) {
        int sq = wbi[i];
        CP_RAY(sq,  1,  1, bw_c, +1);
        CP_RAY(sq,  1, -1, bw_c, +1);
        CP_RAY(sq, -1,  1, bw_c, +1);
        CP_RAY(sq, -1, -1, bw_c, +1);
    }
    int bb_c = 0;
    for (int i = 0; i < nbb; i++) {
        int sq = bbi[i];
        CP_RAY(sq,  1,  1, bb_c, -1);
        CP_RAY(sq,  1, -1, bb_c, -1);
        CP_RAY(sq, -1,  1, bb_c, -1);
        CP_RAY(sq, -1, -1, bb_c, -1);
    }
    score += (bw_c - bb_c) * 4;

    /* Rooks (4 orthogonal) */
    int rw_c = 0;
    for (int i = 0; i < nwr; i++) {
        int sq = wro[i];
        CP_RAY(sq,  1,  0, rw_c, +1);
        CP_RAY(sq, -1,  0, rw_c, +1);
        CP_RAY(sq,  0,  1, rw_c, +1);
        CP_RAY(sq,  0, -1, rw_c, +1);
    }
    int rb_c = 0;
    for (int i = 0; i < nbr; i++) {
        int sq = bro[i];
        CP_RAY(sq,  1,  0, rb_c, -1);
        CP_RAY(sq, -1,  0, rb_c, -1);
        CP_RAY(sq,  0,  1, rb_c, -1);
        CP_RAY(sq,  0, -1, rb_c, -1);
    }
    score += (rw_c - rb_c) * 3;

    /* Queens (8 directions) */
    int qw_c = 0;
    for (int i = 0; i < nwq; i++) {
        int sq = wqu[i];
        CP_RAY(sq,  1,  0, qw_c, +1); CP_RAY(sq, -1,  0, qw_c, +1);
        CP_RAY(sq,  0,  1, qw_c, +1); CP_RAY(sq,  0, -1, qw_c, +1);
        CP_RAY(sq,  1,  1, qw_c, +1); CP_RAY(sq,  1, -1, qw_c, +1);
        CP_RAY(sq, -1,  1, qw_c, +1); CP_RAY(sq, -1, -1, qw_c, +1);
    }
    int qb_c = 0;
    for (int i = 0; i < nbq; i++) {
        int sq = bqu[i];
        CP_RAY(sq,  1,  0, qb_c, -1); CP_RAY(sq, -1,  0, qb_c, -1);
        CP_RAY(sq,  0,  1, qb_c, -1); CP_RAY(sq,  0, -1, qb_c, -1);
        CP_RAY(sq,  1,  1, qb_c, -1); CP_RAY(sq,  1, -1, qb_c, -1);
        CP_RAY(sq, -1,  1, qb_c, -1); CP_RAY(sq, -1, -1, qb_c, -1);
    }
    score += (qw_c - qb_c) * 2;

#undef CP_RAY

    return score;
}

/* ------------------------------------------------------------------------
 * Top-level evaluator.
 * ------------------------------------------------------------------------ */

int32_t chess_rule_evaluate(const ChessPosition *p) {
    const int8_t *board = p->board;

    int32_t mg = 0;
    int32_t eg = 0;
    int phase = 0;

    /* Per-piece bookkeeping. Chess has at most 16 pieces per side. */
    int8_t wpawns[16], bpawns[16];
    int8_t wknights[16], bknights[16];
    int8_t wbishops[16], bbishops[16];
    int8_t wrooks[16], brooks[16];
    int8_t wqueens[16], bqueens[16];
    int nwp = 0, nbp = 0, nwk = 0, nbk = 0;
    int nwb = 0, nbb = 0, nwr = 0, nbr = 0;
    int nwq = 0, nbq = 0;
    int8_t pfw[8] = {0};
    int8_t pfb[8] = {0};

    for (int sq = 0; sq < 64; sq++) {
        int piece = board[sq];
        if (piece == 0) continue;
        int abs_piece = (piece > 0) ? piece : -piece;

        phase += CP_PHASE_WEIGHTS[abs_piece];

        int value = CP_PIECE_VALUES[abs_piece];
        const int16_t *pmg = cp_pst_mg(abs_piece);
        const int16_t *peg = cp_pst_eg(abs_piece);

        if (piece > 0) {
            mg += value;
            eg += value;
            if (pmg) {
                mg += pmg[sq];
                eg += peg[sq];
            }
            if (abs_piece == CP_PAWN) {
                pfw[sq & 7]++;
                wpawns[nwp++] = (int8_t)sq;
            } else if (abs_piece == CP_KNIGHT) {
                wknights[nwk++] = (int8_t)sq;
            } else if (abs_piece == CP_BISHOP) {
                wbishops[nwb++] = (int8_t)sq;
            } else if (abs_piece == CP_ROOK) {
                wrooks[nwr++] = (int8_t)sq;
            } else if (abs_piece == CP_QUEEN) {
                wqueens[nwq++] = (int8_t)sq;
            }
        } else {
            mg -= value;
            eg -= value;
            if (pmg) {
                int m = cp_mirror_sq(sq);
                mg -= pmg[m];
                eg -= peg[m];
            }
            if (abs_piece == CP_PAWN) {
                pfb[sq & 7]++;
                bpawns[nbp++] = (int8_t)sq;
            } else if (abs_piece == CP_KNIGHT) {
                bknights[nbk++] = (int8_t)sq;
            } else if (abs_piece == CP_BISHOP) {
                bbishops[nbb++] = (int8_t)sq;
            } else if (abs_piece == CP_ROOK) {
                brooks[nbr++] = (int8_t)sq;
            } else if (abs_piece == CP_QUEEN) {
                bqueens[nbq++] = (int8_t)sq;
            }
        }
    }

    /* Pawn structure */
    int32_t pawn_struct = chess_eval_pawn_structure(pfw, pfb);
    int32_t passed = chess_eval_passed_pawns(wpawns, nwp, bpawns, nbp);
    int32_t connected = chess_eval_connected_pawns(wpawns, nwp, bpawns, nbp);
    int32_t backward = chess_eval_backward_pawns(wpawns, nwp, bpawns, nbp);
    mg += pawn_struct + passed + connected + backward;
    /* EG scales: passed * 3/2, connected * 7/10, backward * 2. */
    eg += pawn_struct + cp_fdiv(passed * 3, 2)
        + cp_fdiv(connected * 7, 10) + backward * 2;

    /* Knight outposts */
    int32_t outposts = chess_eval_knight_outposts(
        wknights, nwk, bknights, nbk,
        wpawns, nwp, bpawns, nbp
    );
    mg += outposts;
    eg += cp_fdiv(outposts, 2);

    /* Bishop pair */
    if (nwb >= 2) { mg += 30; eg += 50; }
    if (nbb >= 2) { mg -= 30; eg -= 50; }

    /* Rooks on open / semi-open files + 7th rank */
    for (int i = 0; i < nwr; i++) {
        int sq = wrooks[i];
        int f = sq & 7;
        int r = sq >> 3;
        int wp = pfw[f];
        int bp = pfb[f];
        if (wp == 0 && bp == 0) { mg += 20; eg += 10; }
        else if (wp == 0)        { mg += 10; eg += 5; }
        if (r == 6) { mg += 25; eg += 15; }
    }
    for (int i = 0; i < nbr; i++) {
        int sq = brooks[i];
        int f = sq & 7;
        int r = sq >> 3;
        int wp = pfw[f];
        int bp = pfb[f];
        if (wp == 0 && bp == 0) { mg -= 20; eg -= 10; }
        else if (bp == 0)        { mg -= 10; eg -= 5; }
        if (r == 1) { mg -= 25; eg -= 15; }
    }

    /* King shelter (MG only) */
    mg += chess_eval_king_shelter(p->white_king_sq, 0, wpawns, nwp);
    mg -= chess_eval_king_shelter(p->black_king_sq, 1, bpawns, nbp);

    /* King attack pressure (MG only) */
    mg += chess_eval_king_attack_pressure(board, p->white_king_sq,
                                            p->black_king_sq);

    /* Tactical/positional terms */
    int32_t trapped = chess_eval_trapped_rook(wrooks, nwr, brooks, nbr,
                                                p->white_king_sq,
                                                p->black_king_sq);
    mg += trapped;

    int32_t bad_bishop = chess_eval_bad_bishop(wbishops, nwb, bbishops, nbb,
                                                 wpawns, nwp, bpawns, nbp);
    mg += bad_bishop;
    eg += bad_bishop;

    int32_t connectivity = chess_eval_rook_connectivity(wrooks, nwr,
                                                           brooks, nbr, board);
    mg += connectivity;
    eg += cp_fdiv(connectivity, 2);

    int32_t undeveloped = chess_eval_undeveloped_minors(wknights, nwk,
                                                          wbishops, nwb,
                                                          bknights, nbk,
                                                          bbishops, nbb);
    mg += undeveloped;

    /* Mobility (MG 1.0, EG 0.6) */
    int32_t mobility = chess_eval_mobility(
        board,
        wknights, nwk, bknights, nbk,
        wbishops, nwb, bbishops, nbb,
        wrooks, nwr, brooks, nbr,
        wqueens, nwq, bqueens, nbq
    );
    mg += mobility;
    eg += cp_fdiv(mobility * 6, 10);

    /* Tapered interpolation — integer floor division matching Python. */
    if (phase > CP_PHASE_TOTAL) phase = CP_PHASE_TOTAL;
    int32_t score = cp_fdiv(
        mg * phase + eg * (CP_PHASE_TOTAL - phase),
        CP_PHASE_TOTAL
    );

    if (p->side == 1) score = -score;
    return score;
}

/* ------------------------------------------------------------------------
 * Python entry point: accel_chess_c_evaluate(board, side, wk_sq, bk_sq)
 * The caller passes the board as bytes-like (numpy int8 is fine), the
 * side as 0/1, and precomputed king squares (from state.king_square).
 * ------------------------------------------------------------------------ */

static PyObject *accel_chess_c_evaluate(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj;
    int side, wk_sq, bk_sq;
    if (!PyArg_ParseTuple(args, "Oiii", &board_obj, &side, &wk_sq, &bk_sq))
        return NULL;

    Py_buffer buf;
    if (PyObject_GetBuffer(board_obj, &buf, PyBUF_SIMPLE) != 0) return NULL;
    if (buf.len < CHESS_SQUARES) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "chess board must have >= 64 bytes");
        return NULL;
    }

    ChessPosition p;
    memcpy(p.board, buf.buf, CHESS_SQUARES);
    PyBuffer_Release(&buf);

    p.side = (int8_t)(side & 1);
    p.white_king_sq = (int16_t)wk_sq;
    p.black_king_sq = (int16_t)bk_sq;

    int32_t score = chess_rule_evaluate(&p);
    return PyFloat_FromDouble((double)score);
}
