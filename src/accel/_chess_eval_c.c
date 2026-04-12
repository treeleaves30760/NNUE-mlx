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

static const int16_t CP_PAWN_MG[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
};

static const int16_t CP_PAWN_EG[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
};

static const int16_t CP_KNIGHT_MG[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
};

/* Knight EG = MG in the Python reference. */

static const int16_t CP_BISHOP_MG[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10, 10,  5, 10, 10,  5, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
};

static const int16_t CP_ROOK_MG[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
};

static const int16_t CP_QUEEN_MG[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
};

static const int16_t CP_KING_MG[64] = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
};

static const int16_t CP_KING_EG[64] = {
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
};

/* Knight and bishop/rook/queen EG are identical to MG in the Python
 * reference (only pawn and king have explicit EG tables). We look them
 * up through these index helpers so the main loop doesn't branch. */
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
        case CP_KNIGHT: return CP_KNIGHT_MG;   /* EG == MG for N */
        case CP_BISHOP: return CP_BISHOP_MG;   /* EG == MG for B */
        case CP_ROOK:   return CP_ROOK_MG;     /* EG == MG for R */
        case CP_QUEEN:  return CP_QUEEN_MG;    /* EG == MG for Q */
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
            int advance = r - 1;
            score += 10 + advance * 12;
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
            int advance = 6 - r;
            score -= 10 + advance * 12;
        }
    }
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
    mg += pawn_struct + passed + connected;
    /* EG scales: passed * 3/2, connected * 7/10. Floor division. */
    eg += pawn_struct + cp_fdiv(passed * 3, 2) + cp_fdiv(connected * 7, 10);

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
