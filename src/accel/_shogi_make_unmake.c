/* ========================================================================
 * Shogi make_move / unmake_move with incremental Zobrist update.
 *
 * Mirrors src/games/shogi/state.py::make_move semantics exactly, but
 * operates in place on a ShogiPosition and records enough state in a
 * ShogiUndo frame to reverse any move.
 *
 * Returns 0 on success, -1 on error (invalid move).
 * ======================================================================= */

#include "_shogi_position.h"

/* Demote a promoted board value back to its base type (for hand add) */
static inline int sh_demote(int bv) {
    switch (bv) {
        case SP_PPAWN:   return SP_PAWN;
        case SP_PLANCE:  return SP_LANCE;
        case SP_PKNIGHT: return SP_KNIGHT;
        case SP_PSILVER: return SP_SILVER;
        case SP_HORSE:   return SP_BISHOP;
        case SP_DRAGON:  return SP_ROOK;
        default:         return (bv >= 1 && bv <= 7) ? bv : 0;
    }
}

/* Compute full Zobrist hash from scratch (used for init + sanity). */
uint64_t shogi_compute_hash(const ShogiPosition *p) {
    uint64_t h = g_shogi_z_side[p->side];
    for (int sq = 0; sq < SHOGI_SQUARES; sq++) {
        int8_t v = p->board[sq];
        if (v == 0) continue;
        int pv = (v > 0) ? v : -v;
        int color = (v > 0) ? 0 : 1;
        h ^= g_shogi_z_piece[pv][color][sq];
    }
    for (int pt = 0; pt < SHOGI_HAND_TYPES; pt++) {
        int cs = p->sente_hand[pt];
        int cg = p->gote_hand[pt];
        if (cs > 0) h ^= g_shogi_z_hand[pt][0][cs];
        if (cg > 0) h ^= g_shogi_z_hand[pt][1][cg];
    }
    return h;
}

/* ---- Make a move in place. Returns 0 on success. ---------------------- */

int shogi_make_move(ShogiPosition *p, SMove m, ShogiUndo *u) {
    int from = smove_from(m);
    int to   = smove_to(m);
    int promo = smove_promo(m);
    int drop  = smove_drop(m);
    int side  = p->side;
    int mover_color = side;      /* 0=sente, 1=gote */
    int sign = (side == 0) ? 1 : -1;

    u->move = m;
    u->captured = 0;
    u->prev_from_piece = 0;
    u->prev_king_sq = (side == 0) ? p->sente_king_sq : p->gote_king_sq;
    u->prev_hash = p->hash;

    uint64_t h = p->hash;
    /* Flip side */
    h ^= g_shogi_z_side[side];
    h ^= g_shogi_z_side[side ^ 1];

    int8_t *own_hand = (side == 0) ? p->sente_hand : p->gote_hand;

    if (smove_is_drop(m)) {
        /* Drop */
        int pt = drop;
        int bv = pt + 1;
        if (pt < 0 || pt >= SHOGI_HAND_TYPES || own_hand[pt] <= 0)
            return -1;
        /* Place piece */
        p->board[to] = (int8_t)(sign * bv);
        h ^= g_shogi_z_piece[bv][mover_color][to];
        /* Decrement hand count (update hash for old & new count) */
        int old_cnt = own_hand[pt];
        int new_cnt = old_cnt - 1;
        if (old_cnt > 0) h ^= g_shogi_z_hand[pt][mover_color][old_cnt];
        if (new_cnt > 0) h ^= g_shogi_z_hand[pt][mover_color][new_cnt];
        own_hand[pt] = (int8_t)new_cnt;
    } else {
        /* Board move */
        int8_t from_piece = p->board[from];
        int8_t captured   = p->board[to];
        u->captured = captured;
        u->prev_from_piece = from_piece;

        int pv = (from_piece > 0) ? from_piece : -from_piece;
        int new_pv = (promo != 0) ? promo : pv;

        /* Remove from source */
        h ^= g_shogi_z_piece[pv][mover_color][from];

        /* Remove captured and add to hand */
        if (captured != 0) {
            int cv = (captured > 0) ? captured : -captured;
            int cap_color = (captured > 0) ? 0 : 1;
            h ^= g_shogi_z_piece[cv][cap_color][to];
            int demoted = sh_demote(cv);
            if (demoted != 0 && cv != SP_KING) {
                int pt = demoted - 1;  /* board 1..7 -> API 0..6 */
                int old_cnt = own_hand[pt];
                int new_cnt = old_cnt + 1;
                if (old_cnt > 0) h ^= g_shogi_z_hand[pt][mover_color][old_cnt];
                if (new_cnt > 0) h ^= g_shogi_z_hand[pt][mover_color][new_cnt];
                own_hand[pt] = (int8_t)new_cnt;
            }
        }

        /* Place the (possibly promoted) piece on destination */
        p->board[from] = 0;
        p->board[to] = (int8_t)(sign * new_pv);
        h ^= g_shogi_z_piece[new_pv][mover_color][to];

        /* Update king tracking if the king moved */
        if (pv == SP_KING) {
            if (side == 0) p->sente_king_sq = (int16_t)to;
            else           p->gote_king_sq  = (int16_t)to;
        }
    }

    /* Flip side to move */
    p->side = (int8_t)(side ^ 1);
    p->hash = h;
    return 0;
}

/* ---- Unmake: reverse a previously made move using an Undo frame ------- */

void shogi_unmake_move(ShogiPosition *p, const ShogiUndo *u) {
    SMove m = u->move;
    int from = smove_from(m);
    int to   = smove_to(m);
    int promo = smove_promo(m);
    int drop  = smove_drop(m);

    /* Before we called shogi_make_move, the mover was the *other* side. */
    int mover_side = p->side ^ 1;
    int sign = (mover_side == 0) ? 1 : -1;
    int8_t *own_hand = (mover_side == 0) ? p->sente_hand : p->gote_hand;

    if (smove_is_drop(m)) {
        int pt = drop;
        /* Remove dropped piece; restore hand count */
        p->board[to] = 0;
        own_hand[pt] += 1;
    } else {
        /* Restore the mover to its original (possibly unpromoted) form */
        p->board[from] = u->prev_from_piece;
        p->board[to]   = u->captured;  /* may be 0 */

        /* If the mover captured a piece, one item was added to their hand. */
        if (u->captured != 0) {
            int cv = (u->captured > 0) ? u->captured : -u->captured;
            int demoted = sh_demote(cv);
            if (demoted != 0 && cv != SP_KING) {
                int pt = demoted - 1;
                own_hand[pt] -= 1;
            }
        }

        /* Restore king tracking */
        int pv = (u->prev_from_piece > 0) ? u->prev_from_piece : -u->prev_from_piece;
        if (pv == SP_KING) {
            if (mover_side == 0) p->sente_king_sq = u->prev_king_sq;
            else                 p->gote_king_sq  = u->prev_king_sq;
        }
    }

    p->side = (int8_t)mover_side;
    p->hash = u->prev_hash;
    (void)sign; (void)promo;
}

/* ---- Initial position -------------------------------------------------- */

void shogi_position_init_startpos(ShogiPosition *p) {
    /* Back rank: Lance Knight Silver Gold King Gold Silver Knight Lance */
    static const int8_t back[9] = {
        SP_LANCE, SP_KNIGHT, SP_SILVER, SP_GOLD, SP_KING, SP_GOLD, SP_SILVER, SP_KNIGHT, SP_LANCE,
    };
    memset(p, 0, sizeof(*p));
    /* Gote back rank (row 0, negative) */
    for (int f = 0; f < 9; f++) p->board[sh_sq(0, f)] = (int8_t)(-back[f]);
    /* Gote bishop (1,1) and rook (1,7) */
    p->board[sh_sq(1, 1)] = -SP_BISHOP;
    p->board[sh_sq(1, 7)] = -SP_ROOK;
    /* Gote pawns row 2 */
    for (int f = 0; f < 9; f++) p->board[sh_sq(2, f)] = -SP_PAWN;
    /* Sente pawns row 6 */
    for (int f = 0; f < 9; f++) p->board[sh_sq(6, f)] = SP_PAWN;
    /* Sente rook (7,1) and bishop (7,7) */
    p->board[sh_sq(7, 1)] = SP_ROOK;
    p->board[sh_sq(7, 7)] = SP_BISHOP;
    /* Sente back rank row 8 */
    for (int f = 0; f < 9; f++) p->board[sh_sq(8, f)] = (int8_t)back[f];

    p->side = 0;  /* sente to move */
    p->sente_king_sq = sh_sq(8, 4);
    p->gote_king_sq  = sh_sq(0, 4);
    p->hash = shogi_compute_hash(p);
}

/* ---- Adapter: reuse existing attack-detection helper from _accel_attacks.c
 *
 * The helper there is `accel_is_square_attacked_shogi` but it takes a
 * Py_buffer and parses arguments — too heavy for inner loops. We re-
 * implement the same logic here on a raw int8_t* for speed.
 * ---------------------------------------------------------------------- */

int shogi_is_square_attacked(const int8_t *board, int sq, int by_side) {
    int rank = sq / 9, file = sq % 9;
    int sign = (by_side == 0) ? 1 : -1;

    /* 1. Pawn */
    {
        int pr = rank + (by_side == 0 ? 1 : -1);
        if (pr >= 0 && pr < 9 && board[pr * 9 + file] == sign * SP_PAWN)
            return 1;
    }
    /* 2. Lance */
    {
        int ldr = (by_side == 0) ? 1 : -1;
        for (int r = rank + ldr; r >= 0 && r < 9; r += ldr) {
            int8_t v = board[r * 9 + file];
            if (v != 0) {
                if (v == sign * SP_LANCE) return 1;
                break;
            }
        }
    }
    /* 3. Knight */
    {
        int kdr = (by_side == 0) ? 2 : -2;
        for (int df = -1; df <= 1; df += 2) {
            int nr = rank + kdr, nf = file + df;
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                if (board[nr * 9 + nf] == sign * SP_KNIGHT) return 1;
            }
        }
    }
    /* 4. Silver (5 reverse directions) */
    {
        static const int8_t S_REV_S[5][2] = {{1,1},{1,0},{1,-1},{-1,1},{-1,-1}};
        static const int8_t S_REV_G[5][2] = {{-1,-1},{-1,0},{-1,1},{1,-1},{1,1}};
        const int8_t (*dirs)[2] = (by_side == 0) ? S_REV_S : S_REV_G;
        for (int d = 0; d < 5; d++) {
            int nr = rank + dirs[d][0], nf = file + dirs[d][1];
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                if (board[nr * 9 + nf] == sign * SP_SILVER) return 1;
            }
        }
    }
    /* 5. Gold-like: gold, +pawn, +lance, +knight, +silver */
    {
        static const int8_t G_REV_S[6][2] = {{1,1},{1,0},{1,-1},{0,1},{0,-1},{-1,0}};
        static const int8_t G_REV_G[6][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,0}};
        const int8_t (*dirs)[2] = (by_side == 0) ? G_REV_S : G_REV_G;
        for (int d = 0; d < 6; d++) {
            int nr = rank + dirs[d][0], nf = file + dirs[d][1];
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                int8_t v = board[nr * 9 + nf];
                if (v != 0 && ((v > 0) == (sign > 0))) {
                    int pv = (v > 0) ? v : -v;
                    if (pv == SP_GOLD || pv == SP_PPAWN || pv == SP_PLANCE ||
                        pv == SP_PKNIGHT || pv == SP_PSILVER)
                        return 1;
                }
            }
        }
    }
    /* 6. Bishop / Horse diagonal slide */
    {
        static const int8_t BD[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
        for (int d = 0; d < 4; d++) {
            int cr = rank + BD[d][0], cf = file + BD[d][1];
            while (cr >= 0 && cr < 9 && cf >= 0 && cf < 9) {
                int8_t v = board[cr * 9 + cf];
                if (v != 0) {
                    if (v == sign * SP_BISHOP || v == sign * SP_HORSE) return 1;
                    break;
                }
                cr += BD[d][0]; cf += BD[d][1];
            }
        }
    }
    /* 7. Rook / Dragon orthogonal slide */
    {
        static const int8_t RD[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        for (int d = 0; d < 4; d++) {
            int cr = rank + RD[d][0], cf = file + RD[d][1];
            while (cr >= 0 && cr < 9 && cf >= 0 && cf < 9) {
                int8_t v = board[cr * 9 + cf];
                if (v != 0) {
                    if (v == sign * SP_ROOK || v == sign * SP_DRAGON) return 1;
                    break;
                }
                cr += RD[d][0]; cf += RD[d][1];
            }
        }
    }
    /* 8. Horse extra orthogonal one-step */
    {
        static const int8_t RD[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        for (int d = 0; d < 4; d++) {
            int nr = rank + RD[d][0], nf = file + RD[d][1];
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                if (board[nr * 9 + nf] == sign * SP_HORSE) return 1;
            }
        }
    }
    /* 9. Dragon extra diagonal one-step */
    {
        static const int8_t BD[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
        for (int d = 0; d < 4; d++) {
            int nr = rank + BD[d][0], nf = file + BD[d][1];
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                if (board[nr * 9 + nf] == sign * SP_DRAGON) return 1;
            }
        }
    }
    /* 10. King adjacency */
    {
        static const int8_t KD[8][2] = {
            {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1},
        };
        for (int d = 0; d < 8; d++) {
            int nr = rank + KD[d][0], nf = file + KD[d][1];
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                if (board[nr * 9 + nf] == sign * SP_KING) return 1;
            }
        }
    }
    return 0;
}
