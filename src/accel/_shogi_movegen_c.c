/* ========================================================================
 * Shogi pseudo-legal and legal move generation in C.
 *
 * Mirrors the semantics of src/games/shogi/movegen.py but operates directly
 * on a ShogiPosition struct (no Python callbacks) for use inside the C
 * alpha-beta search hot loop.
 * ======================================================================= */

#include "_shogi_position.h"

/* Forward decls provided by _shogi_make_unmake.c */
int shogi_make_move(ShogiPosition *p, SMove m, ShogiUndo *u);
void shogi_unmake_move(ShogiPosition *p, const ShogiUndo *u);

/* ---- Piece-specific pseudo-attack helpers ------------------------------ *
 *
 * These write to out[n] and advance *n_out, never exceeding SHOGI_MAX_MOVES.
 * They only emit moves to empty squares or enemy-occupied squares (no
 * friendly captures). Promotion expansion is handled in the generator.
 */

static inline int sh_is_friend(int8_t v, int side) {
    if (v == 0) return 0;
    return (side == 0) ? (v > 0) : (v < 0);
}
static inline int sh_is_enemy(int8_t v, int side) {
    if (v == 0) return 0;
    return (side == 0) ? (v < 0) : (v > 0);
}

/* One-step moves: sente uses the raw direction, gote flips dr. */
static void sh_one_step_moves(const int8_t *board, int sq, int side,
                               const int8_t dirs[][2], int n_dirs,
                               int *targets, int *n_out) {
    int r = sh_rank(sq), f = sh_file(sq);
    int flip = (side == 0) ? 1 : -1;
    for (int i = 0; i < n_dirs; i++) {
        int nr = r + dirs[i][0] * flip;
        int nf = f + dirs[i][1];
        if (nr < 0 || nr >= 9 || nf < 0 || nf >= 9) continue;
        int tsq = sh_sq(nr, nf);
        int8_t v = board[tsq];
        if (sh_is_friend(v, side)) continue;
        targets[(*n_out)++] = tsq;
    }
}

static void sh_sliding_moves(const int8_t *board, int sq, int side,
                              const int8_t dirs[][2], int n_dirs,
                              int *targets, int *n_out) {
    int r = sh_rank(sq), f = sh_file(sq);
    for (int i = 0; i < n_dirs; i++) {
        int dr = dirs[i][0];
        int df = dirs[i][1];
        int cr = r + dr, cf = f + df;
        while (cr >= 0 && cr < 9 && cf >= 0 && cf < 9) {
            int tsq = sh_sq(cr, cf);
            int8_t v = board[tsq];
            if (v != 0) {
                if (!sh_is_friend(v, side))
                    targets[(*n_out)++] = tsq;
                break;
            }
            targets[(*n_out)++] = tsq;
            cr += dr; cf += df;
        }
    }
}

static void sh_lance_moves(const int8_t *board, int sq, int side,
                            int *targets, int *n_out) {
    int r = sh_rank(sq), f = sh_file(sq);
    int dr = (side == 0) ? -1 : 1;
    int cr = r + dr;
    while (cr >= 0 && cr < 9) {
        int tsq = sh_sq(cr, f);
        int8_t v = board[tsq];
        if (v != 0) {
            if (!sh_is_friend(v, side))
                targets[(*n_out)++] = tsq;
            break;
        }
        targets[(*n_out)++] = tsq;
        cr += dr;
    }
}

static void sh_knight_moves(const int8_t *board, int sq, int side,
                             int *targets, int *n_out) {
    int r = sh_rank(sq), f = sh_file(sq);
    int dr = (side == 0) ? -2 : 2;
    for (int df = -1; df <= 1; df += 2) {
        int nr = r + dr, nf = f + df;
        if (nr < 0 || nr >= 9 || nf < 0 || nf >= 9) continue;
        int tsq = sh_sq(nr, nf);
        int8_t v = board[tsq];
        if (sh_is_friend(v, side)) continue;
        targets[(*n_out)++] = tsq;
    }
}

/* Raw targets for a piece at sq. Returns target count (≤ 20). */
static int sh_raw_targets(const int8_t *board, int sq, int side,
                           int piece_val, int *targets) {
    int n = 0;
    switch (piece_val) {
        case SP_PAWN: {
            int r = sh_rank(sq), f = sh_file(sq);
            int dr = (side == 0) ? -1 : 1;
            int nr = r + dr;
            if (nr >= 0 && nr < 9) {
                int tsq = sh_sq(nr, f);
                if (!sh_is_friend(board[tsq], side))
                    targets[n++] = tsq;
            }
            break;
        }
        case SP_LANCE:
            sh_lance_moves(board, sq, side, targets, &n); break;
        case SP_KNIGHT:
            sh_knight_moves(board, sq, side, targets, &n); break;
        case SP_SILVER:
            sh_one_step_moves(board, sq, side, SH_SILVER_DIRS, 5, targets, &n); break;
        case SP_GOLD:
        case SP_PPAWN: case SP_PLANCE: case SP_PKNIGHT: case SP_PSILVER:
            sh_one_step_moves(board, sq, side, SH_GOLD_DIRS, 6, targets, &n); break;
        case SP_BISHOP:
            sh_sliding_moves(board, sq, side, SH_BISHOP_DIRS, 4, targets, &n); break;
        case SP_ROOK:
            sh_sliding_moves(board, sq, side, SH_ROOK_DIRS, 4, targets, &n); break;
        case SP_KING:
            sh_one_step_moves(board, sq, side, SH_KING_DIRS, 8, targets, &n); break;
        case SP_HORSE:
            sh_sliding_moves(board, sq, side, SH_BISHOP_DIRS, 4, targets, &n);
            sh_one_step_moves(board, sq, side, SH_ROOK_DIRS,   4, targets, &n); break;
        case SP_DRAGON:
            sh_sliding_moves(board, sq, side, SH_ROOK_DIRS,   4, targets, &n);
            sh_one_step_moves(board, sq, side, SH_BISHOP_DIRS, 4, targets, &n); break;
        default: break;
    }
    return n;
}

/* ---- Promotion logic --------------------------------------------------- */

static inline int sh_can_promote_type(int pv) {
    /* 1 Pawn, 2 Lance, 3 Knight, 4 Silver, 6 Bishop, 7 Rook */
    return (pv == 1 || pv == 2 || pv == 3 || pv == 4 || pv == 6 || pv == 7);
}

static inline int sh_promoted_of(int pv) {
    switch (pv) {
        case SP_PAWN:   return SP_PPAWN;
        case SP_LANCE:  return SP_PLANCE;
        case SP_KNIGHT: return SP_PKNIGHT;
        case SP_SILVER: return SP_PSILVER;
        case SP_BISHOP: return SP_HORSE;
        case SP_ROOK:   return SP_DRAGON;
        default:        return 0;
    }
}

static inline int sh_must_promote(int pv, int to_sq, int side) {
    int r = sh_rank(to_sq);
    if (side == 0) {
        if (pv == SP_PAWN || pv == SP_LANCE) return r == 0;
        if (pv == SP_KNIGHT) return r <= 1;
    } else {
        if (pv == SP_PAWN || pv == SP_LANCE) return r == 8;
        if (pv == SP_KNIGHT) return r >= 7;
    }
    return 0;
}

static inline int sh_promotion_possible(int pv, int from_sq, int to_sq, int side) {
    if (!sh_can_promote_type(pv)) return 0;
    return sh_in_promo_zone(from_sq, side) || sh_in_promo_zone(to_sq, side);
}

/* ---- Check detection (reuses attack routine below) -------------------- */

/* Bring in the existing C reverse-attack routine. We reuse the body by
 * calling the module-level helper; declared extern so we can use it here. */
int shogi_is_square_attacked(const int8_t *board, int sq, int by_side);

int shogi_is_in_check(const ShogiPosition *p, int side) {
    int ksq = (side == 0) ? p->sente_king_sq : p->gote_king_sq;
    int opp = side ^ 1;
    return shogi_is_square_attacked(p->board, ksq, opp);
}

/* ---- Pseudo-legal generation ------------------------------------------ */

int shogi_expand_pseudo_moves(const ShogiPosition *p, SMove *out) {
    int n = 0;
    int side = p->side;
    int sign = (side == 0) ? 1 : -1;
    const int8_t *board = p->board;

    /* --- Board moves --- */
    for (int sq = 0; sq < SHOGI_SQUARES; sq++) {
        int8_t v = board[sq];
        if (v == 0) continue;
        if ((side == 0 && v < 0) || (side == 1 && v > 0)) continue;
        int pv = (v > 0) ? v : -v;

        int targets[24];
        int nt;
        if (pv == SP_KING) {
            nt = 0;
            sh_one_step_moves(board, sq, side, SH_KING_DIRS, 8, targets, &nt);
            for (int i = 0; i < nt; i++)
                out[n++] = smove_encode(sq, targets[i], 0, SMOVE_NONE_DROP);
            continue;
        }
        nt = sh_raw_targets(board, sq, side, pv, targets);
        for (int i = 0; i < nt; i++) {
            int tsq = targets[i];
            int must  = sh_must_promote(pv, tsq, side);
            int canp  = sh_promotion_possible(pv, sq, tsq, side);
            if (must) {
                out[n++] = smove_encode(sq, tsq, sh_promoted_of(pv), SMOVE_NONE_DROP);
            } else if (canp) {
                out[n++] = smove_encode(sq, tsq, sh_promoted_of(pv), SMOVE_NONE_DROP);
                out[n++] = smove_encode(sq, tsq, 0, SMOVE_NONE_DROP);
            } else {
                out[n++] = smove_encode(sq, tsq, 0, SMOVE_NONE_DROP);
            }
        }
    }
    (void)sign;

    /* --- Drop moves --- */
    const int8_t *hand = (side == 0) ? p->sente_hand : p->gote_hand;
    int any_in_hand = 0;
    for (int i = 0; i < SHOGI_HAND_TYPES; i++) {
        if (hand[i] > 0) { any_in_hand = 1; break; }
    }
    if (any_in_hand) {
        /* Precompute files that already contain a friendly unpromoted pawn
         * (for nifu: can't drop a pawn on a file with an existing pawn). */
        uint16_t pawn_files = 0;
        for (int sq = 0; sq < SHOGI_SQUARES; sq++) {
            int8_t v = board[sq];
            if (v == (side == 0 ? SP_PAWN : -SP_PAWN)) {
                pawn_files |= (uint16_t)(1u << sh_file(sq));
            }
        }
        for (int pt = 0; pt < SHOGI_HAND_TYPES; pt++) {
            if (hand[pt] == 0) continue;
            int bv = pt + 1;  /* API 0..6 -> board 1..7 */
            for (int sq = 0; sq < SHOGI_SQUARES; sq++) {
                if (board[sq] != 0) continue;
                int r = sh_rank(sq);
                int f = sh_file(sq);
                /* Pawn / Lance cannot land on last rank */
                if (bv == SP_PAWN || bv == SP_LANCE) {
                    if (side == 0 && r == 0) continue;
                    if (side == 1 && r == 8) continue;
                }
                /* Knight cannot land on last 2 ranks */
                if (bv == SP_KNIGHT) {
                    if (side == 0 && r <= 1) continue;
                    if (side == 1 && r >= 7) continue;
                }
                /* Nifu */
                if (bv == SP_PAWN && (pawn_files & (1u << f))) continue;
                out[n++] = smove_encode(SMOVE_NONE_FROM, sq, 0, pt);
            }
        }
    }
    return n;
}

/* ---- Uchifuzume check (expensive — only on pawn drops) ---------------- */

static int shogi_is_uchifuzume(ShogiPosition *p, int drop_sq) {
    /* Apply the pawn drop tentatively */
    SMove drop = smove_encode(SMOVE_NONE_FROM, drop_sq, 0, 0 /* PAWN */);
    ShogiUndo undo;
    if (shogi_make_move(p, drop, &undo) != 0) return 0;
    int opp = p->side;  /* side to move has flipped */
    int opp_check = shogi_is_in_check(p, opp);
    if (!opp_check) {
        shogi_unmake_move(p, &undo);
        return 0;
    }
    /* Opponent must have no legal reply */
    SMove buf[SHOGI_MAX_MOVES];
    int nm = shogi_expand_pseudo_moves(p, buf);
    int escape = 0;
    for (int i = 0; i < nm; i++) {
        ShogiUndo u2;
        if (shogi_make_move(p, buf[i], &u2) != 0) continue;
        /* After opp's move, we check if *they* are still in check. */
        if (!shogi_is_in_check(p, opp)) {
            shogi_unmake_move(p, &u2);
            escape = 1;
            break;
        }
        shogi_unmake_move(p, &u2);
    }
    shogi_unmake_move(p, &undo);
    return !escape;
}

/* ---- Legal move generation -------------------------------------------- */

int shogi_expand_legal_moves(ShogiPosition *p, SMove *out) {
    SMove pseudo[SHOGI_MAX_MOVES];
    int np = shogi_expand_pseudo_moves(p, pseudo);
    int n = 0;
    int mover = p->side;
    for (int i = 0; i < np; i++) {
        SMove m = pseudo[i];
        /* Uchifuzume check for pawn drops */
        if (smove_is_drop(m) && smove_drop(m) == 0 /* PAWN */) {
            if (shogi_is_uchifuzume(p, smove_to(m))) continue;
        }
        ShogiUndo u;
        if (shogi_make_move(p, m, &u) != 0) continue;
        int illegal = shogi_is_in_check(p, mover);
        shogi_unmake_move(p, &u);
        if (!illegal) out[n++] = m;
    }
    return n;
}
