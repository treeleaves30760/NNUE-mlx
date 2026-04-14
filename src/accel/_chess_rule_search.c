/* ========================================================================
 * Chess rule-based alpha-beta search (pure C).
 *
 * Minimal viable port of the Python AlphaBetaSearch + RuleBasedEvaluator
 * for chess. Reuses chess_rule_evaluate() from _chess_eval_c.c as the
 * static eval, chess_expand_legal_moves from _chess_movegen_c.c, and
 * chess_make_move / chess_unmake_move from _chess_make_unmake.c.
 *
 * Features:
 *   - Iterative deepening
 *   - Alpha-beta with a 1M-entry TT (depth-preferred replacement)
 *   - MVV-LVA + TT move ordering, plus killers and history
 *   - Quiescence search on captures and promotions
 *   - Check extension
 *   - Three-fold repetition detection via a lightweight hash ring
 *
 * Deliberately omitted (defer to follow-up): null-move pruning, LMR,
 * PVS, aspiration windows, futility pruning. Each of those has sharp
 * edges that need their own validation pass.
 * ======================================================================= */

#include <stdlib.h>
#include <time.h>
#include "_nnue_accel_header.h"
#include "_chess_position.h"

extern int32_t chess_rule_evaluate(const ChessPosition *p);

/* ---- Constants --------------------------------------------------------- */

#define CRS_INF          1000000
#define CRS_MATE         100000
#define CRS_MAX_PLY      64
#define CRS_TT_SIZE      (1u << 20)
#define CRS_MAX_QDEPTH   8
#define CRS_HISTORY_SIZE (64 * 64)
#define CRS_HISTORY_MAX  64   /* max hashes in the repetition ring */

/* MVV-LVA values (matches CP_PIECE_VALUES, plus a dominant king stub). */
static const int CRS_MVV_LVA[7] = {
    0, 100, 320, 330, 500, 900, 20000,
};

/* ---- Data structures --------------------------------------------------- */

typedef struct {
    uint64_t key;
    int32_t  score;
    ChMove   best_move;
    int16_t  depth;
    uint8_t  flag;      /* 0=exact, 1=alpha-upper, 2=beta-lower */
    uint8_t  _pad;
} CRSTTEntry;

#define CRS_TT_EXACT 0
#define CRS_TT_ALPHA 1
#define CRS_TT_BETA  2

typedef struct {
    CRSTTEntry *tt;
    uint32_t    tt_mask;

    /* Killer moves: [ply][2] */
    ChMove killers[CRS_MAX_PLY][2];

    /* History heuristic: from*64 + to */
    int32_t history[CRS_HISTORY_SIZE];

    /* Counter-move heuristic: counter[from_prev * 64 + to_prev] is the
     * move that refuted the previous move. Cheap, cuts maybe 5% of
     * nodes in quiet middlegame positions. */
    ChMove counter[64 * 64];
    ChMove prev_move;

    /* Game history of hashes (for 3-fold detection). Copied in at
     * search-start from the Python ChessState history tuple. */
    uint64_t hist[CRS_HISTORY_MAX];
    int      hist_n;

    /* LMR table: reduction amount indexed by [depth][move_index]. */
    int8_t lmr[64][64];

    int    max_depth;
    double time_limit_ms;
    double start_time_ms;
    int    time_up;
    long   nodes;
} CRSContext;

/* Futility margin per depth (for shallow futility pruning). */
static const int32_t CRS_FUTILITY_MARGIN[4] = { 0, 150, 300, 500 };

/* Single global context — bootstrap workers are processes, so thread
 * safety is not a concern. */
static CRSContext g_crs;
static int g_crs_initialized = 0;

static double crs_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void crs_lmr_init(CRSContext *ctx) {
    for (int d = 0; d < 64; d++) {
        for (int m = 0; m < 64; m++) {
            if (d < 2 || m < 2) { ctx->lmr[d][m] = 0; continue; }
            /* Stockfish-lite formula: log(d) * log(m) / 2. Keep the
             * reduction modest — chess rule search is less aggressive
             * than an NNUE search. */
            int r = (int)(1.0 + log((double)d) * log((double)m) / 2.5);
            if (r < 0) r = 0;
            if (r > 6) r = 6;
            ctx->lmr[d][m] = (int8_t)r;
        }
    }
}

static void crs_context_init(CRSContext *ctx) {
    if (!ctx->tt) {
        ctx->tt = (CRSTTEntry *)calloc(CRS_TT_SIZE, sizeof(CRSTTEntry));
        ctx->tt_mask = CRS_TT_SIZE - 1;
    }
    crs_lmr_init(ctx);
    g_crs_initialized = 1;
}

static void crs_reset_for_search(CRSContext *ctx, int max_depth,
                                  double time_limit_ms) {
    memset(ctx->killers, 0xFF, sizeof(ctx->killers));
    memset(ctx->history, 0, sizeof(ctx->history));
    memset(ctx->counter, 0xFF, sizeof(ctx->counter));
    ctx->prev_move = CRS_INF;
    if (ctx->tt) memset(ctx->tt, 0, CRS_TT_SIZE * sizeof(CRSTTEntry));
    ctx->max_depth = (max_depth <= 0) ? (CRS_MAX_PLY - 1) : max_depth;
    ctx->time_limit_ms = (time_limit_ms <= 0.0) ? 1e12 : time_limit_ms;
    ctx->start_time_ms = crs_now_ms();
    ctx->time_up = 0;
    ctx->nodes = 0;
    ctx->hist_n = 0;
}

static inline int crs_time_check(CRSContext *ctx) {
    if ((ctx->nodes & 4095) == 0) {
        double now = crs_now_ms();
        if (now - ctx->start_time_ms >= ctx->time_limit_ms) {
            ctx->time_up = 1;
        }
    }
    return ctx->time_up;
}

/* ---- Three-fold repetition check ------------------------------------- */

static int crs_is_repetition(CRSContext *ctx, uint64_t hash) {
    int count = 0;
    for (int i = 0; i < ctx->hist_n; i++) {
        if (ctx->hist[i] == hash) count++;
        if (count >= 2) return 1;   /* 2 previous + the current one = 3 */
    }
    return 0;
}

/* ---- Mate score ply adjustment (Stockfish convention) ----------------- *
 * See the shogi rule search for the rationale: transpositions can reach
 * the same position at different plies, so mate distances stored in the
 * TT must be rooted at the stored node (not the search root) and
 * de-rooted on probe. Without this, mate scores can leak across the
 * root and surface as spurious near-mate evals at unrelated positions. */
static inline int32_t crs_score_to_tt(int32_t s, int ply) {
    if (s >=  CRS_MATE - CRS_MAX_PLY) return s + ply;
    if (s <= -CRS_MATE + CRS_MAX_PLY) return s - ply;
    return s;
}
static inline int32_t crs_score_from_tt(int32_t s, int ply) {
    if (s >=  CRS_MATE - CRS_MAX_PLY) return s - ply;
    if (s <= -CRS_MATE + CRS_MAX_PLY) return s + ply;
    return s;
}

/* ---- TT helpers ------------------------------------------------------- */

static inline CRSTTEntry *crs_tt_probe(CRSContext *ctx, uint64_t key) {
    CRSTTEntry *e = &ctx->tt[key & ctx->tt_mask];
    if (e->key == key) return e;
    return NULL;
}

static inline void crs_tt_store(CRSContext *ctx, uint64_t key, int depth,
                                 int32_t score, uint8_t flag, ChMove best) {
    CRSTTEntry *e = &ctx->tt[key & ctx->tt_mask];
    if (e->key == 0 || depth >= e->depth) {
        e->key = key;
        e->score = score;
        e->best_move = best;
        e->depth = (int16_t)depth;
        e->flag = flag;
    }
}

/* ---- Move scoring ----------------------------------------------------- */

typedef struct {
    ChMove move;
    int32_t score;
} CRSScoredMove;

static void crs_score_moves(CRSContext *ctx, const ChessPosition *p,
                             const ChMove *moves, int n, CRSScoredMove *out,
                             int ply, ChMove tt_move) {
    const int8_t *board = p->board;
    /* Counter-move lookup key from the last move played. */
    ChMove counter_hint = CRS_INF;
    if (ctx->prev_move != CRS_INF) {
        int pf = chmove_from(ctx->prev_move);
        int pt = chmove_to(ctx->prev_move);
        counter_hint = ctx->counter[pf * 64 + pt];
    }
    for (int i = 0; i < n; i++) {
        ChMove m = moves[i];
        int32_t score = 0;
        int to = chmove_to(m);
        int from = chmove_from(m);
        int is_cap = (board[to] != 0) || chmove_is_ep(m);
        int is_promo = (chmove_promo(m) != 0);

        if (m == tt_move && tt_move != CRS_INF) {
            score = 100000000;
        } else if (is_cap) {
            int victim = chmove_is_ep(m) ? CP_PAWN :
                         ((board[to] > 0) ? board[to] : -board[to]);
            int attacker = (board[from] > 0) ? board[from] : -board[from];
            score = 10000000 + CRS_MVV_LVA[victim] * 10 - CRS_MVV_LVA[attacker];
        } else {
            if (ply < CRS_MAX_PLY) {
                if (ctx->killers[ply][0] == m) score = 5000000;
                else if (ctx->killers[ply][1] == m) score = 4000000;
            }
            if (m == counter_hint && counter_hint != CRS_INF) {
                score += 3500000;
            }
            score += ctx->history[from * 64 + to];
        }
        if (is_promo) score += 3000000 + chmove_promo(m) * 100;

        out[i].move = m;
        out[i].score = score;
    }
}

static void crs_sort_moves(CRSScoredMove *arr, int n) {
    for (int i = 1; i < n; i++) {
        CRSScoredMove cur = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].score < cur.score) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = cur;
    }
}

/* ---- Forward decls ---------------------------------------------------- */

static int32_t crs_alphabeta(CRSContext *ctx, ChessPosition *p,
                              int depth, int32_t alpha, int32_t beta,
                              int ply, int allow_null);
static int32_t crs_quiescence(CRSContext *ctx, ChessPosition *p,
                               int32_t alpha, int32_t beta, int qdepth);

/* Returns 1 if side ``side`` has at least one non-king, non-pawn piece
 * left on the board. Used to gate null-move pruning — in pawn-only
 * endings null moves cause zugzwang errors. */
static int crs_has_pieces(const ChessPosition *p, int side) {
    const int8_t *board = p->board;
    for (int sq = 0; sq < 64; sq++) {
        int8_t v = board[sq];
        if (v == 0) continue;
        int pv = v > 0 ? v : -v;
        int is_side = (v > 0) ? 0 : 1;
        if (is_side != side) continue;
        if (pv >= CP_KNIGHT && pv <= CP_QUEEN) return 1;
    }
    return 0;
}

/* ---- Quiescence: captures + promotions ------------------------------ */

static int32_t crs_quiescence(CRSContext *ctx, ChessPosition *p,
                               int32_t alpha, int32_t beta, int qdepth) {
    ctx->nodes++;
    if (crs_time_check(ctx)) return 0;

    int32_t stand = chess_rule_evaluate(p);
    if (stand >= beta) return beta;
    if (stand > alpha) alpha = stand;
    if (qdepth >= CRS_MAX_QDEPTH) return alpha;

    ChMove moves[CHESS_MAX_MOVES];
    int nm = chess_expand_legal_moves(p, moves);
    const int8_t *board = p->board;
    ChMove cap_moves[CHESS_MAX_MOVES];
    int nc = 0;
    for (int i = 0; i < nm; i++) {
        ChMove m = moves[i];
        int to = chmove_to(m);
        int is_cap = (board[to] != 0) || chmove_is_ep(m);
        int is_promo = (chmove_promo(m) != 0);
        if (is_cap || is_promo) cap_moves[nc++] = m;
    }
    if (nc == 0) return alpha;

    CRSScoredMove scored[CHESS_MAX_MOVES];
    crs_score_moves(ctx, p, cap_moves, nc, scored, 0, CRS_INF);
    crs_sort_moves(scored, nc);

    for (int i = 0; i < nc; i++) {
        ChessUndo u;
        if (chess_make_move(p, scored[i].move, &u) != 0) continue;
        int32_t score = -crs_quiescence(ctx, p, -beta, -alpha, qdepth + 1);
        chess_unmake_move(p, &u);
        if (ctx->time_up) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

/* ---- Main alpha-beta ------------------------------------------------- */

static int32_t crs_alphabeta(CRSContext *ctx, ChessPosition *p,
                              int depth, int32_t alpha, int32_t beta,
                              int ply, int allow_null) {
    ctx->nodes++;
    if (crs_time_check(ctx)) return 0;

    /* Three-fold repetition inside the search tree counts as a draw. */
    if (ply > 0 && crs_is_repetition(ctx, p->hash)) return 0;

    /* 50-move rule */
    if (p->halfmove >= 100) return 0;

    int in_check = chess_is_in_check(p, p->side);
    if (in_check) depth += 1;  /* check extension */

    if (depth <= 0) {
        return crs_quiescence(ctx, p, alpha, beta, 0);
    }

    /* TT probe */
    uint64_t key = p->hash;
    CRSTTEntry *tt = crs_tt_probe(ctx, key);
    ChMove tt_move = CRS_INF;
    if (tt && tt->depth >= depth) {
        int32_t tt_score = crs_score_from_tt(tt->score, ply);
        if (tt->flag == CRS_TT_EXACT) return tt_score;
        if (tt->flag == CRS_TT_ALPHA && tt_score <= alpha) return alpha;
        if (tt->flag == CRS_TT_BETA  && tt_score >= beta)  return beta;
    }
    if (tt) tt_move = tt->best_move;

    /* ---- Static eval for pruning decisions -------------------------- */
    int32_t static_eval = 0;
    int have_static = 0;
    if (!in_check) {
        static_eval = chess_rule_evaluate(p);
        have_static = 1;
    }

    /* ---- Null-move pruning -------------------------------------------
     * Give the opponent a free move. If they still can't bring our eval
     * below beta, our position is so good that the current move will
     * cut off — return beta. Gated on non-pk material (zugzwang) and
     * shallow depth. */
    if (allow_null && depth >= 3 && !in_check && have_static
        && static_eval >= beta
        && crs_has_pieces(p, p->side)) {
        int R = 2 + (depth >= 6 ? 1 : 0);
        /* Make null move: flip side, clear ep square. */
        ChessPosition save = *p;
        if (p->ep_square >= 0) {
            p->hash ^= g_chess_z_ep_file[cp_file(p->ep_square)];
        }
        p->hash ^= g_chess_z_side;
        p->ep_square = -1;
        p->side ^= 1;
        int32_t nscore = -crs_alphabeta(ctx, p, depth - 1 - R,
                                          -beta, -beta + 1, ply + 1, 0);
        *p = save;
        if (ctx->time_up) return 0;
        if (nscore >= beta) return beta;
    }

    ChMove moves[CHESS_MAX_MOVES];
    int nm = chess_expand_legal_moves(p, moves);
    if (nm == 0) {
        /* No legal moves: checkmate in check, else stalemate = draw. */
        if (in_check) return -CRS_MATE + ply;
        return 0;
    }

    /* ---- Futility pruning setup ------------------------------------- */
    int futile = 0;
    if (depth <= 3 && !in_check && have_static) {
        if (static_eval + CRS_FUTILITY_MARGIN[depth] <= alpha) futile = 1;
    }

    CRSScoredMove scored[CHESS_MAX_MOVES];
    crs_score_moves(ctx, p, moves, nm, scored, ply, tt_move);
    crs_sort_moves(scored, nm);

    const int8_t *board = p->board;
    int32_t orig_alpha = alpha;
    int32_t best_score = -CRS_INF;
    ChMove best_move = scored[0].move;

    /* Push current hash into history for repetition detection inside
     * recursive children. Save prev_move so we can restore it on exit. */
    ChMove saved_prev = ctx->prev_move;
    if (ctx->hist_n < CRS_HISTORY_MAX) {
        ctx->hist[ctx->hist_n++] = p->hash;
    }

    for (int i = 0; i < nm; i++) {
        ChMove m = scored[i].move;
        int is_cap = (board[chmove_to(m)] != 0) || chmove_is_ep(m);
        int is_promo = (chmove_promo(m) != 0);
        int is_quiet = !is_cap && !is_promo;

        /* Futility pruning: skip quiet moves past the first when the
         * static eval is hopelessly below alpha. */
        if (futile && i > 0 && is_quiet) continue;

        ChessUndo u;
        if (chess_make_move(p, m, &u) != 0) continue;

        /* LMR: reduce depth for late quiet moves in non-check positions.
         * Re-search at full depth if the reduced search exceeds alpha. */
        int reduction = 0;
        if (i >= 3 && depth >= 3 && !in_check && is_quiet) {
            int di = depth < 64 ? depth : 63;
            int mi = i < 64 ? i : 63;
            reduction = ctx->lmr[di][mi];
            if (reduction >= depth - 1) reduction = depth - 2;
            if (reduction < 0) reduction = 0;
        }

        ctx->prev_move = m;
        int32_t score;
        if (i == 0) {
            score = -crs_alphabeta(ctx, p, depth - 1, -beta, -alpha,
                                     ply + 1, 1);
        } else {
            score = -crs_alphabeta(ctx, p, depth - 1 - reduction,
                                     -alpha - 1, -alpha, ply + 1, 1);
            if (!ctx->time_up && reduction > 0 && score > alpha) {
                score = -crs_alphabeta(ctx, p, depth - 1,
                                         -alpha - 1, -alpha, ply + 1, 1);
            }
            if (!ctx->time_up && score > alpha && score < beta) {
                score = -crs_alphabeta(ctx, p, depth - 1,
                                         -beta, -alpha, ply + 1, 1);
            }
        }
        ctx->prev_move = saved_prev;
        chess_unmake_move(p, &u);
        if (ctx->time_up) {
            if (ctx->hist_n > 0) ctx->hist_n--;
            return 0;
        }

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            /* Beta cutoff */
            if (is_quiet && ply < CRS_MAX_PLY) {
                if (ctx->killers[ply][0] != m) {
                    ctx->killers[ply][1] = ctx->killers[ply][0];
                    ctx->killers[ply][0] = m;
                }
                int from = chmove_from(m);
                int to = chmove_to(m);
                ctx->history[from * 64 + to] += depth * depth;
                /* Counter-move: the move that refuted prev_move. */
                if (saved_prev != CRS_INF) {
                    int pf = chmove_from(saved_prev);
                    int pt = chmove_to(saved_prev);
                    ctx->counter[pf * 64 + pt] = m;
                }
            }
            crs_tt_store(ctx, key, depth,
                          crs_score_to_tt(beta, ply), CRS_TT_BETA, m);
            if (ctx->hist_n > 0) ctx->hist_n--;
            return beta;
        }
    }

    if (ctx->hist_n > 0) ctx->hist_n--;

    uint8_t flag = (best_score <= orig_alpha) ? CRS_TT_ALPHA : CRS_TT_EXACT;
    crs_tt_store(ctx, key, depth,
                  crs_score_to_tt(best_score, ply), flag, best_move);
    return best_score;
}

/* ---- Iterative deepening root search --------------------------------- */

static void crs_search_root(CRSContext *ctx, ChessPosition *p,
                             ChMove *out_best, int32_t *out_score) {
    *out_best = CRS_INF;
    *out_score = 0;

    ChMove moves[CHESS_MAX_MOVES];
    int nm = chess_expand_legal_moves(p, moves);
    if (nm == 0) return;

    /* Seed fallback to the first legal move. */
    *out_best = moves[0];

    for (int d = 1; d <= ctx->max_depth; d++) {
        int32_t alpha = -CRS_INF;
        int32_t beta = CRS_INF;
        int32_t best_score = -CRS_INF;
        ChMove best_move = moves[0];

        CRSScoredMove scored[CHESS_MAX_MOVES];
        ChMove tt_move = CRS_INF;
        CRSTTEntry *tt = crs_tt_probe(ctx, p->hash);
        if (tt) tt_move = tt->best_move;
        crs_score_moves(ctx, p, moves, nm, scored, 0, tt_move);
        crs_sort_moves(scored, nm);

        /* Push root hash for repetition inside child trees. */
        if (ctx->hist_n < CRS_HISTORY_MAX) {
            ctx->hist[ctx->hist_n++] = p->hash;
        }

        for (int i = 0; i < nm; i++) {
            ChMove m = scored[i].move;
            ChessUndo u;
            if (chess_make_move(p, m, &u) != 0) continue;
            int32_t score = -crs_alphabeta(ctx, p, d - 1, -beta, -alpha, 1, 1);
            chess_unmake_move(p, &u);
            if (ctx->time_up) break;
            if (score > best_score) {
                best_score = score;
                best_move = m;
            }
            if (score > alpha) alpha = score;
        }

        if (ctx->hist_n > 0) ctx->hist_n--;

        if (ctx->time_up && d > 1) break;
        if (best_score > -CRS_INF) {
            *out_best = best_move;
            *out_score = best_score;
        }
        if (ctx->time_up) break;
    }
}

/* ========================================================================
 * Python wrapper: chess_c_rule_search
 *
 * Args (all ints unless noted):
 *   board (bytes-like, >= 64 bytes)
 *   side, castling, ep_sq, halfmove, wk_sq, bk_sq
 *   history_bytes (bytes, length 8*N, uint64 hashes for 3-fold detection)
 *   max_depth
 *   time_limit_ms (float)
 *
 * Returns ((from, to, promo_base_or_None), score, nodes) or None when
 * no legal moves exist.
 * ======================================================================= */

static PyObject *accel_chess_c_rule_search(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj;
    PyObject *history_obj;
    int side, castling, ep_sq, halfmove, wk_sq, bk_sq, max_depth;
    double time_limit_ms;
    if (!PyArg_ParseTuple(args, "OiiiiiiOid",
                          &board_obj, &side, &castling, &ep_sq, &halfmove,
                          &wk_sq, &bk_sq, &history_obj, &max_depth,
                          &time_limit_ms)) {
        return NULL;
    }

    chess_init_tables();

    Py_buffer bbuf;
    if (PyObject_GetBuffer(board_obj, &bbuf, PyBUF_SIMPLE) != 0) return NULL;
    if (bbuf.len < CHESS_SQUARES) {
        PyBuffer_Release(&bbuf);
        PyErr_SetString(PyExc_ValueError, "chess board must have >= 64 bytes");
        return NULL;
    }
    ChessPosition p;
    memset(&p, 0, sizeof(p));
    memcpy(p.board, bbuf.buf, CHESS_SQUARES);
    PyBuffer_Release(&bbuf);
    p.side = (int8_t)(side & 1);
    p.castling = (int8_t)(castling & 0xF);
    p.ep_square = (int16_t)ep_sq;
    p.halfmove = (int16_t)halfmove;
    p.white_king_sq = (int16_t)wk_sq;
    p.black_king_sq = (int16_t)bk_sq;
    p.hash = chess_compute_hash(&p);

    /* Parse history bytes into the repetition ring. */
    Py_buffer hbuf;
    int hist_n = 0;
    uint64_t hist_copy[CRS_HISTORY_MAX];
    if (history_obj != Py_None) {
        if (PyObject_GetBuffer(history_obj, &hbuf, PyBUF_SIMPLE) != 0) return NULL;
        size_t nn = (size_t)(hbuf.len / 8);
        if (nn > CRS_HISTORY_MAX) nn = CRS_HISTORY_MAX;
        memcpy(hist_copy, hbuf.buf, nn * 8);
        hist_n = (int)nn;
        PyBuffer_Release(&hbuf);
    }

    if (!g_crs_initialized) crs_context_init(&g_crs);
    crs_reset_for_search(&g_crs, max_depth, time_limit_ms);
    /* Seed history ring with Python-side game history */
    for (int i = 0; i < hist_n; i++) {
        g_crs.hist[i] = hist_copy[i];
    }
    g_crs.hist_n = hist_n;

    ChMove best_move = CRS_INF;
    int32_t best_score = 0;
    crs_search_root(&g_crs, &p, &best_move, &best_score);

    if (best_move == CRS_INF) {
        Py_RETURN_NONE;
    }

    int promo_internal = chmove_promo(best_move);
    int promo_base;
    switch (promo_internal) {
        case CP_QUEEN:  promo_base = 4; break;
        case CP_ROOK:   promo_base = 3; break;
        case CP_BISHOP: promo_base = 2; break;
        case CP_KNIGHT: promo_base = 1; break;
        default:        promo_base = -1; break;
    }
    PyObject *promo_obj;
    if (promo_base < 0) {
        Py_INCREF(Py_None);
        promo_obj = Py_None;
    } else {
        promo_obj = PyLong_FromLong(promo_base);
    }

    PyObject *res = Py_BuildValue(
        "((iiO)id)",
        chmove_from(best_move), chmove_to(best_move), promo_obj,
        (int)best_score, (double)g_crs.nodes);
    Py_DECREF(promo_obj);
    return res;
}
