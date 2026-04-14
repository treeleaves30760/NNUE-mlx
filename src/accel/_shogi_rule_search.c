/* ========================================================================
 * Self-contained shogi rule-based alpha-beta search (pure C).
 *
 * No Python callbacks in the hot loop. Operates directly on ShogiPosition.
 * Uses shogi_rule_evaluate() as the static eval.
 *
 * Enhancements: iterative deepening, TT, killers, history heuristic,
 * null-move pruning, LMR, PVS, aspiration windows (shallow), futility,
 * quiescence (captures + promotions).
 *
 * Reuses the move-encoding and ShogiUndo machinery from
 * _shogi_movegen_c.c / _shogi_make_unmake.c.
 * ======================================================================= */

#include "_nnue_accel_header.h"
#include "_shogi_position.h"

/* Forward declaration: defined in _shogi_eval_c.c (same translation unit). */
extern int32_t shogi_rule_evaluate(const ShogiPosition *p);

/* ---- Constants --------------------------------------------------------- */

#define SRS_INF            1000000
#define SRS_MATE           100000
#define SRS_MAX_PLY        128
/* Mate threshold and non-mate cap. Any score in [-MATE_THRESHOLD,
 * MATE_THRESHOLD] is treated as a regular eval and never gets the mate
 * ply adjustment. NON_MATE_CAP bounds aspiration windows and
 * fail-high/fail-low returns so they can never enter the mate range
 * and therefore can't be drifted by the mate adjustment through TT
 * transpositions. */
#define SRS_MATE_THRESHOLD (SRS_MATE - SRS_MAX_PLY)   /* 99872 */
#define SRS_NON_MATE_CAP   (SRS_MATE_THRESHOLD - 1)   /* 99871 */
#define SRS_TT_SIZE        (1u << 20)  /* ~1M entries */
#define SRS_HISTORY_SIZE   (82 * 82)   /* from [0..80] + drop(81), to [0..80] */
#define SRS_MAX_QDEPTH     8

/* MVV-LVA piece values for shogi (board code 0..14). Matches
 * SHOGI_MVV_LVA_VALUES in src/search/evaluator.py. */
static const int SRS_MVV_LVA[15] = {
       0,    /* empty */
     100,    /* Pawn */
     430,    /* Lance */
     450,    /* Knight */
     640,    /* Silver */
     690,    /* Gold */
     890,    /* Bishop */
    1040,    /* Rook */
   20000,    /* King */
     520,    /* +Pawn */
     530,    /* +Lance */
     540,    /* +Knight */
     570,    /* +Silver */
    1150,    /* Horse */
    1300,    /* Dragon */
};

/* Futility margins by depth */
static const int SRS_FUTILITY_MARGIN[3] = {0, 200, 500};

/* ---- Data structures --------------------------------------------------- */

typedef struct {
    uint64_t key;
    int32_t  score;
    SMove    best_move;
    int16_t  depth;
    uint8_t  flag;       /* 0=exact, 1=alpha (upper), 2=beta (lower) */
    uint8_t  _pad;
} SRSTTEntry;

#define SRS_TT_EXACT  0
#define SRS_TT_ALPHA  1
#define SRS_TT_BETA   2

typedef struct {
    /* Transposition table */
    SRSTTEntry *tt;
    uint32_t    tt_mask;

    /* Killer moves: [ply][2] */
    SMove killers[SRS_MAX_PLY][2];

    /* History heuristic: idx = from * 82 + to. Drop moves use from = 81. */
    int32_t history[SRS_HISTORY_SIZE];

    /* Search state */
    int   max_depth;
    double time_limit_ms;
    double start_time_ms;
    int   time_up;
    int   stop;
    long  nodes;

    /* LMR table */
    int8_t lmr[64][64];

    /* ----- Live progress reporting ----------------------------------- */
    /* Python callback called every `cb_interval` nodes, and after each
     * root iteration. NULL disables reporting. Holds a borrowed reference
     * (caller keeps the callable alive for the duration of the search). */
    PyObject *progress_cb;
    long      cb_interval;
    long      nodes_last_cb;
    int       abort_requested;
    int       published_depth;
    int32_t   published_best_score;
    int       published_has_best_move;
    SMove     published_best_move;

    /* Current-iteration root move list. search_root populates these and
     * updates scores as each root move's subtree is searched. The callback
     * reads this snapshot, sorts by score, and reports the top N. */
    SMove   root_moves[SHOGI_MAX_MOVES];
    int32_t root_scores[SHOGI_MAX_MOVES];
    int     root_n_moves;
    int     root_current_depth;   /* depth of the iteration being built */
    int     root_completed_depth; /* deepest iteration fully completed */
} SRSContext;

/* Global context (one at a time — bootstrap workers are processes, not
 * threads, so this is safe). */
static SRSContext g_srs;
static int g_srs_initialized = 0;

/* ---- Helpers ----------------------------------------------------------- */

static double srs_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void srs_lmr_init(SRSContext *ctx) {
    for (int d = 0; d < 64; d++) {
        for (int m = 0; m < 64; m++) {
            if (d == 0 || m == 0) { ctx->lmr[d][m] = 0; continue; }
            int v = (int)(1.0 + log((double)d) * log((double)m) / 2.0);
            if (v < 0) v = 0;
            if (v > 8) v = 8;
            ctx->lmr[d][m] = (int8_t)v;
        }
    }
}

static void srs_context_init(SRSContext *ctx) {
    if (!ctx->tt) {
        ctx->tt = (SRSTTEntry *)calloc(SRS_TT_SIZE, sizeof(SRSTTEntry));
        ctx->tt_mask = SRS_TT_SIZE - 1;
    }
    srs_lmr_init(ctx);
    g_srs_initialized = 1;
}

static void srs_reset_for_search(SRSContext *ctx, int max_depth,
                                  double time_limit_ms) {
    memset(ctx->killers, 0xFF, sizeof(ctx->killers));
    memset(ctx->history, 0, sizeof(ctx->history));
    if (ctx->tt) memset(ctx->tt, 0, SRS_TT_SIZE * sizeof(SRSTTEntry));
    /* Sentinel handling:
     *   max_depth <= 0   -> "effectively infinite" (cap at SRS_MAX_PLY/2)
     *   time_limit <= 0  -> "no time limit" (huge value)
     * Both together give an infinite-analysis mode bounded only by the
     * external stop flag set from the Python progress callback.
     */
    ctx->max_depth = (max_depth <= 0) ? (SRS_MAX_PLY / 2) : max_depth;
    ctx->time_limit_ms = (time_limit_ms <= 0.0) ? 1e12 : time_limit_ms;
    ctx->start_time_ms = srs_now_ms();
    ctx->time_up = 0;
    ctx->stop = 0;
    ctx->nodes = 0;

    ctx->progress_cb = NULL;
    ctx->cb_interval = 1000000;
    ctx->nodes_last_cb = 0;
    ctx->abort_requested = 0;
    ctx->published_depth = 0;
    ctx->published_best_score = -SRS_INF;
    ctx->published_has_best_move = 0;
    ctx->published_best_move = SMOVE_NULL;
    ctx->root_n_moves = 0;
    ctx->root_current_depth = 0;
    ctx->root_completed_depth = 0;
    for (int i = 0; i < SHOGI_MAX_MOVES; i++) {
        ctx->root_scores[i] = -SRS_INF;
    }
}

/* ---- Live progress callback -------------------------------------------- *
 *
 * Builds a Python tuple of the current top-N root moves (sorted by score)
 * and calls the user-supplied callback. The callback receives:
 *   (completed_depth, max_depth, [(move_tuple, score), ...], done)
 * and returns a bool: True -> abort the search.
 *
 * Mid-iteration heartbeats are only published if they still belong to the
 * same completed depth and the current principal move is strictly better
 * than the last published snapshot for that depth. Completed iterations are
 * always published.
 *
 * The caller (search root / alpha-beta node check) must hold the GIL via
 * PyGILState_Ensure when entering this function. We briefly ensure the
 * GIL state so it is safe to call even after Py_BEGIN_ALLOW_THREADS. */
static void srs_fire_progress(SRSContext *ctx, int done) {
    if (!ctx->progress_cb) return;

    PyGILState_STATE gstate = PyGILState_Ensure();

    /* Find top-3 (or up to MAX_REPORT for safety margin) by partial selection. */
#define SRS_MAX_REPORT 10
    SMove top_moves[SRS_MAX_REPORT];
    int32_t top_scores[SRS_MAX_REPORT];
    int count = 0;
    const int MAX_REPORT = SRS_MAX_REPORT;

    int n = ctx->root_n_moves;
    for (int i = 0; i < n; i++) {
        int32_t sc = ctx->root_scores[i];
        if (sc == -SRS_INF) continue;  /* not yet searched at current depth */
        if (count < MAX_REPORT) {
            /* Insert in sorted order */
            int k = count;
            while (k > 0 && top_scores[k - 1] < sc) {
                top_scores[k] = top_scores[k - 1];
                top_moves[k] = top_moves[k - 1];
                k--;
            }
            top_scores[k] = sc;
            top_moves[k] = ctx->root_moves[i];
            count++;
        } else if (sc > top_scores[MAX_REPORT - 1]) {
            int k = MAX_REPORT - 1;
            while (k > 0 && top_scores[k - 1] < sc) {
                top_scores[k] = top_scores[k - 1];
                top_moves[k] = top_moves[k - 1];
                k--;
            }
            top_scores[k] = sc;
            top_moves[k] = ctx->root_moves[i];
        }
    }

    if (count == 0) {
        PyGILState_Release(gstate);
        return;
    }

    if (!done) {
        int improved_depth = (ctx->root_completed_depth > ctx->published_depth);
        int same_completed_depth = (ctx->root_completed_depth == ctx->published_depth);
        int improved_pv = 0;
        if (same_completed_depth) {
            if (!ctx->published_has_best_move) {
                improved_pv = 1;
            } else {
                const SMove current_best = top_moves[0];
                improved_pv = (top_scores[0] > ctx->published_best_score) ||
                              (top_scores[0] == ctx->published_best_score &&
                               current_best != ctx->published_best_move);
            }
        }
        if (!improved_depth && !improved_pv) {
            PyGILState_Release(gstate);
            return;
        }
    }

    /* Build Python list of (move_tuple, score) */
    PyObject *moves_list = PyList_New(count);
    if (!moves_list) { PyErr_Clear(); PyGILState_Release(gstate); return; }
    for (int i = 0; i < count; i++) {
        PyObject *mv_tup = sh_pyapi_move_to_tuple(top_moves[i]);
        PyObject *pair = PyTuple_Pack(2, mv_tup, PyFloat_FromDouble((double)top_scores[i]));
        Py_DECREF(mv_tup);
        if (!pair) { Py_DECREF(moves_list); PyErr_Clear(); PyGILState_Release(gstate); return; }
        PyList_SET_ITEM(moves_list, i, pair);
    }

    PyObject *args = Py_BuildValue("(iiOO)",
        ctx->root_completed_depth,
        ctx->max_depth,
        moves_list,
        done ? Py_True : Py_False);
    Py_DECREF(moves_list);
    if (!args) { PyErr_Clear(); PyGILState_Release(gstate); return; }

    PyObject *res = PyObject_Call(ctx->progress_cb, args, NULL);
    Py_DECREF(args);
    if (!res) {
        PyErr_Clear();
    } else {
        if (PyObject_IsTrue(res)) ctx->abort_requested = 1;
        Py_DECREF(res);
    }
    ctx->nodes_last_cb = ctx->nodes;
    ctx->published_depth = ctx->root_completed_depth;
    ctx->published_best_score = top_scores[0];
    ctx->published_has_best_move = 1;
    ctx->published_best_move = top_moves[0];

    PyGILState_Release(gstate);
}

static inline int srs_time_check(SRSContext *ctx) {
    if ((ctx->nodes & 4095) == 0) {
        double now = srs_now_ms();
        if (now - ctx->start_time_ms >= ctx->time_limit_ms) {
            ctx->time_up = 1;
            return 1;
        }
        /* Progress heartbeat: fire callback every cb_interval nodes so the
         * GUI can refresh top-N during long searches. */
        if (ctx->progress_cb && ctx->cb_interval > 0 &&
            (ctx->nodes - ctx->nodes_last_cb) >= ctx->cb_interval) {
            srs_fire_progress(ctx, 0);
            if (ctx->abort_requested) {
                ctx->time_up = 1;
                return 1;
            }
        }
    }
    return ctx->time_up;
}

/* ---- Mate score ply adjustment (Stockfish convention) ----------------- *
 * A "mate in N from root" must be stored as "mate in (N - current_ply)"
 * so the distance-to-mate is measured relative to the stored node, not
 * the root. When read back at a different ply we reverse the shift so
 * the returned distance is again relative to the current root.
 *
 * Without this adjustment, a mate stored at ply=10 and later probed at
 * ply=2 via a transposition would report "mate in N-10" from the wrong
 * origin, and could surface at the search root as a near-mate score
 * even at a perfectly balanced opening position. */
static inline int32_t srs_score_to_tt(int32_t s, int ply) {
    if (s >=  SRS_MATE - SRS_MAX_PLY) return s + ply;
    if (s <= -SRS_MATE + SRS_MAX_PLY) return s - ply;
    return s;
}
static inline int32_t srs_score_from_tt(int32_t s, int ply) {
    if (s >=  SRS_MATE - SRS_MAX_PLY) return s - ply;
    if (s <= -SRS_MATE + SRS_MAX_PLY) return s + ply;
    return s;
}

/* ---- TT probe / store -------------------------------------------------- */

static inline SRSTTEntry *srs_tt_probe(SRSContext *ctx, uint64_t key) {
    SRSTTEntry *e = &ctx->tt[key & ctx->tt_mask];
    if (e->key == key) return e;
    return NULL;
}

static inline void srs_tt_store(SRSContext *ctx, uint64_t key, int depth,
                                 int32_t score, uint8_t flag, SMove best) {
    SRSTTEntry *e = &ctx->tt[key & ctx->tt_mask];
    if (e->key == 0 || depth >= e->depth) {
        e->key = key;
        e->score = score;
        e->best_move = best;
        e->depth = (int16_t)depth;
        e->flag = flag;
    }
}

/* ---- Move scoring (MVV-LVA + killers + history + TT) ------------------ */

typedef struct {
    SMove move;
    int32_t score;
} SRSScoredMove;

static inline int srs_history_idx(SMove m) {
    int from = smove_is_drop(m) ? 81 : smove_from(m);
    int to = smove_to(m);
    return from * 82 + to;
}

static inline int srs_abs_i(int x) {
    return x < 0 ? -x : x;
}

static inline int srs_king_sq(const ShogiPosition *p, int side) {
    return side == 0 ? p->sente_king_sq : p->gote_king_sq;
}

static inline int srs_is_king_zone_move(const ShogiPosition *p, SMove m) {
    int opp = p->side ^ 1;
    int king_sq = srs_king_sq(p, opp);
    int to = smove_to(m);
    int dr = srs_abs_i(sh_rank(to) - sh_rank(king_sq));
    int df = srs_abs_i(sh_file(to) - sh_file(king_sq));
    return dr <= 1 && df <= 1;
}

static inline int srs_gives_check(const ShogiPosition *p, SMove m) {
    ShogiPosition tmp = *p;
    ShogiUndo u;
    int gives_check = 0;
    if (shogi_make_move(&tmp, m, &u) == 0) {
        gives_check = shogi_is_in_check(&tmp, tmp.side);
    }
    return gives_check;
}

static void srs_score_moves(SRSContext *ctx, const ShogiPosition *p,
                             const SMove *moves, int n_moves,
                             SRSScoredMove *out, int ply,
                             SMove tt_move) {
    const int8_t *board = p->board;
    for (int i = 0; i < n_moves; i++) {
        SMove m = moves[i];
        int32_t score = 0;
        int is_drop = smove_is_drop(m);
        int to = smove_to(m);
        int is_cap = !is_drop && board[to] != 0;

        if (m == tt_move && tt_move != SMOVE_NULL) {
            score = 100000000;
        } else if (is_cap) {
            /* Capture: MVV-LVA. Attacker is the piece on from_sq,
             * victim is the piece on to_sq (their absolute values). */
            int victim_v = board[to];
            int attacker_v = board[smove_from(m)];
            int va = victim_v > 0 ? victim_v : -victim_v;
            int aa = attacker_v > 0 ? attacker_v : -attacker_v;
            score = 10000000 + SRS_MVV_LVA[va] * 10 - SRS_MVV_LVA[aa];
        } else {
            /* Killer / history for quiet moves (including drops). */
            if (ply < SRS_MAX_PLY) {
                if (ctx->killers[ply][0] == m) score = 5000000;
                else if (ctx->killers[ply][1] == m) score = 4000000;
            }
            score += ctx->history[srs_history_idx(m)];
        }
        if (smove_promo(m) != 0) score += 3000000;
        if (srs_is_king_zone_move(p, m)) {
            score += is_drop ? 1800000 : 1200000;
        }
        if (srs_gives_check(p, m)) {
            score += 2500000;
        }

        out[i].move = m;
        out[i].score = score;
    }
}

/* Simple insertion sort by descending score. N is small (<= 593). */
static void srs_sort_moves(SRSScoredMove *arr, int n) {
    for (int i = 1; i < n; i++) {
        SRSScoredMove cur = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].score < cur.score) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = cur;
    }
}

/* ---- Forward decls ---------------------------------------------------- */

static int32_t srs_alphabeta(SRSContext *ctx, ShogiPosition *p,
                              int depth, int32_t alpha, int32_t beta,
                              int ply, int allow_null);
static int32_t srs_quiescence(SRSContext *ctx, ShogiPosition *p,
                               int32_t alpha, int32_t beta, int qdepth);

/* ---- Quiescence: captures + promotions only -------------------------- */

static int32_t srs_quiescence(SRSContext *ctx, ShogiPosition *p,
                               int32_t alpha, int32_t beta, int qdepth) {
    ctx->nodes++;
    if (srs_time_check(ctx)) return 0;

    int32_t stand = shogi_rule_evaluate(p);
    if (stand >= beta) return beta;
    if (stand > alpha) alpha = stand;

    if (qdepth >= SRS_MAX_QDEPTH) return alpha;

    /* Generate only captures & promotions (reuse legal_moves filter to
     * avoid illegal-self-check replies). */
    SMove moves[SHOGI_MAX_MOVES];
    int nm = shogi_expand_legal_moves(p, moves);
    const int8_t *board = p->board;
    SMove cap_moves[SHOGI_MAX_MOVES];
    int nc = 0;
    for (int i = 0; i < nm; i++) {
        SMove m = moves[i];
        int is_cap = !smove_is_drop(m) && board[smove_to(m)] != 0;
        int is_promo = smove_promo(m) != 0;
        if (is_cap || is_promo) cap_moves[nc++] = m;
    }
    if (nc == 0) return alpha;

    SRSScoredMove scored[SHOGI_MAX_MOVES];
    srs_score_moves(ctx, p, cap_moves, nc, scored, 0, SMOVE_NULL);
    srs_sort_moves(scored, nc);

    for (int i = 0; i < nc; i++) {
        ShogiUndo u;
        if (shogi_make_move(p, scored[i].move, &u) != 0) continue;
        int32_t score = -srs_quiescence(ctx, p, -beta, -alpha, qdepth + 1);
        shogi_unmake_move(p, &u);
        if (ctx->time_up) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

/* ---- Main alpha-beta --------------------------------------------------- */

static int32_t srs_alphabeta(SRSContext *ctx, ShogiPosition *p,
                              int depth, int32_t alpha, int32_t beta,
                              int ply, int allow_null) {
    ctx->nodes++;
    if (srs_time_check(ctx)) return 0;

    /* Terminal check: no legal moves. Also handle king-captured (shouldn't
     * happen in legal play, but serves as a safeguard). */
    int in_check = shogi_is_in_check(p, p->side);
    if (in_check) depth += 1;  /* check extension */

    if (depth <= 0) {
        return srs_quiescence(ctx, p, alpha, beta, 0);
    }

    /* TT probe */
    uint64_t key = p->hash;
    SRSTTEntry *tt = srs_tt_probe(ctx, key);
    SMove tt_move = SMOVE_NULL;
    if (tt && tt->depth >= depth) {
        int32_t tt_score = srs_score_from_tt(tt->score, ply);
        /* Safety net: after the aspiration-window clamp, any non-mate
         * score should stay inside [-NON_MATE_CAP, NON_MATE_CAP]. An
         * out-of-range TT value means the entry drifted through
         * repeated mate-ply adjustments across transpositions — reject
         * it and re-search to avoid propagating garbage. Mate leaves
         * (-SRS_MATE + ply) correctly stay inside the mate range and
         * are still usable for move ordering only. */
        if (tt_score > -SRS_NON_MATE_CAP && tt_score < SRS_NON_MATE_CAP) {
            if (tt->flag == SRS_TT_EXACT) return tt_score;
            if (tt->flag == SRS_TT_ALPHA && tt_score <= alpha) return alpha;
            if (tt->flag == SRS_TT_BETA  && tt_score >= beta)  return beta;
        }
    }
    if (tt) tt_move = tt->best_move;

    /* Static eval for pruning */
    int32_t static_eval = shogi_rule_evaluate(p);

    /* Null move pruning */
    if (allow_null && depth > 2 && !in_check && static_eval >= beta) {
        int R = 2 + (depth > 6 ? 1 : 0);
        ShogiPosition saved = *p;
        p->hash ^= g_shogi_z_side[p->side];
        p->side ^= 1;
        p->hash ^= g_shogi_z_side[p->side];
        int32_t s = -srs_alphabeta(ctx, p, depth - 1 - R, -beta, -beta + 1,
                                    ply + 1, 0);
        *p = saved;
        if (ctx->time_up) return 0;
        if (s >= beta) return beta;
    }

    /* Futility */
    int futile = 0;
    if (depth <= 2 && !in_check) {
        if (static_eval + SRS_FUTILITY_MARGIN[depth] <= alpha) futile = 1;
    }

    /* Generate legal moves */
    SMove moves[SHOGI_MAX_MOVES];
    int nm = shogi_expand_legal_moves(p, moves);
    if (nm == 0) {
        /* No legal moves: checkmate if in check, otherwise stalemate.
         * Shogi convention: stalemate is also a loss. Ply-adjust so the
         * root sees "mate distance" (closer mates score higher). */
        return -SRS_MATE + ply;
    }

    SRSScoredMove scored[SHOGI_MAX_MOVES];
    srs_score_moves(ctx, p, moves, nm, scored, ply, tt_move);
    srs_sort_moves(scored, nm);

    const int8_t *board = p->board;
    int32_t orig_alpha = alpha;
    int32_t best_score = -SRS_INF;
    SMove best_move = scored[0].move;

    for (int i = 0; i < nm; i++) {
        SMove m = scored[i].move;
        int is_cap = !smove_is_drop(m) && board[smove_to(m)] != 0;
        int is_promo = smove_promo(m) != 0;
        int attacks_king_zone = srs_is_king_zone_move(p, m);
        int is_quiet = !is_cap && !is_promo;
        int tactical = is_cap || is_promo || attacks_king_zone;

        if (futile && i > 0 && is_quiet) continue;

        int reduction = 0;
        if (i >= 3 && depth >= 3 && !in_check && is_quiet && !attacks_king_zone) {
            int di = depth < 64 ? depth : 63;
            int mi = i < 64 ? i : 63;
            reduction = ctx->lmr[di][mi];
            if (reduction >= depth - 1) reduction = depth - 2;
            if (reduction < 0) reduction = 0;
        }

        ShogiUndo u;
        if (shogi_make_move(p, m, &u) != 0) continue;
        int gives_check = shogi_is_in_check(p, p->side);
        if (gives_check || tactical) reduction = 0;
        int32_t score;
        if (i == 0) {
            score = -srs_alphabeta(ctx, p, depth - 1, -beta, -alpha,
                                    ply + 1, 1);
        } else {
            score = -srs_alphabeta(ctx, p, depth - 1 - reduction,
                                    -alpha - 1, -alpha, ply + 1, 1);
            if (!ctx->time_up && reduction > 0 && score > alpha) {
                score = -srs_alphabeta(ctx, p, depth - 1,
                                        -alpha - 1, -alpha, ply + 1, 1);
            }
            if (!ctx->time_up && score > alpha && score < beta) {
                score = -srs_alphabeta(ctx, p, depth - 1, -beta, -alpha,
                                        ply + 1, 1);
            }
        }
        shogi_unmake_move(p, &u);
        if (ctx->time_up) return 0;

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            /* Beta cutoff: update killers and history for quiet moves. */
            if (is_quiet && ply < SRS_MAX_PLY) {
                if (ctx->killers[ply][0] != m) {
                    ctx->killers[ply][1] = ctx->killers[ply][0];
                    ctx->killers[ply][0] = m;
                }
                ctx->history[srs_history_idx(m)] += depth * depth;
            }
            srs_tt_store(ctx, key, depth,
                          srs_score_to_tt(beta, ply), SRS_TT_BETA, m);
            return beta;
        }
    }

    uint8_t flag = (best_score <= orig_alpha) ? SRS_TT_ALPHA : SRS_TT_EXACT;
    srs_tt_store(ctx, key, depth,
                  srs_score_to_tt(best_score, ply), flag, best_move);
    return best_score;
}

/* ---- Root search with iterative deepening ---------------------------- */

static void srs_search_root(SRSContext *ctx, ShogiPosition *p,
                             SMove *best_move, int32_t *best_score) {
    *best_move = SMOVE_NULL;
    *best_score = 0;

    int32_t last_score = 0;

    /* Generate the root move list once; the ordering may be refined each
     * iteration by previous scores, but the move set itself is fixed for
     * the whole search. Store it in the context so the live callback can
     * report stable move indices. */
    {
        SMove moves[SHOGI_MAX_MOVES];
        int nm = shogi_expand_legal_moves(p, moves);
        if (nm == 0) return;
        ctx->root_n_moves = nm;
        for (int i = 0; i < nm; i++) {
            ctx->root_moves[i] = moves[i];
            ctx->root_scores[i] = -SRS_INF;
        }
    }

    for (int d = 1; d <= ctx->max_depth; d++) {
        ctx->root_current_depth = d;

        int32_t alpha, beta;
        int32_t delta = 25;

        if (d <= 2) {
            /* Initial search: use the widest NON-MATE window so fail-
             * high/fail-low returns don't enter the mate-score range
             * and get accidentally ply-adjusted in the TT. */
            alpha = -SRS_NON_MATE_CAP;
            beta  = SRS_NON_MATE_CAP;
        } else {
            alpha = last_score - delta;
            beta  = last_score + delta;
        }

        int32_t score = 0;
        SMove iter_best = SMOVE_NULL;

        while (1) {
            /* Score and sort the root move list for this iteration. We use
             * the previous iteration's scores (stored in ctx->root_scores)
             * as the ordering key when available, which gives an excellent
             * first-move hit rate. Fresh iterations (no prior scores) fall
             * back to MVV-LVA / killers / history. */
            SRSScoredMove scored[SHOGI_MAX_MOVES];
            int nm = ctx->root_n_moves;
            int has_prev = 0;
            for (int i = 0; i < nm; i++) {
                if (ctx->root_scores[i] != -SRS_INF) { has_prev = 1; break; }
            }
            if (has_prev) {
                for (int i = 0; i < nm; i++) {
                    scored[i].move = ctx->root_moves[i];
                    /* Boost by previous score; -INF moves go to the back. */
                    int32_t prev = ctx->root_scores[i];
                    scored[i].score = (prev == -SRS_INF) ? -1000000 : prev;
                }
            } else {
                SMove tt_move = SMOVE_NULL;
                SRSTTEntry *tt = srs_tt_probe(ctx, p->hash);
                if (tt) tt_move = tt->best_move;
                srs_score_moves(ctx, p, ctx->root_moves, nm, scored, 0, tt_move);
            }
            srs_sort_moves(scored, nm);

            /* Synchronise root_moves[] with the sorted iteration order so
             * root_scores[i] directly corresponds to root_moves[i]. This
             * also ensures the live-callback top-N has the same first
             * move as the search's returned best on score ties. */
            for (int i = 0; i < nm; i++) {
                ctx->root_moves[i] = scored[i].move;
                ctx->root_scores[i] = -SRS_INF;
            }

            int32_t root_alpha = alpha;
            int32_t root_best = -SRS_INF;
            SMove root_best_move = ctx->root_moves[0];

            for (int i = 0; i < nm; i++) {
                SMove m = ctx->root_moves[i];
                ShogiUndo u;
                if (shogi_make_move(p, m, &u) != 0) continue;
                int32_t s;
                if (i == 0) {
                    s = -srs_alphabeta(ctx, p, d - 1, -beta, -root_alpha, 1, 1);
                } else {
                    s = -srs_alphabeta(ctx, p, d - 1,
                                        -root_alpha - 1, -root_alpha, 1, 1);
                    if (!ctx->time_up && s > root_alpha && s < beta) {
                        s = -srs_alphabeta(ctx, p, d - 1, -beta, -root_alpha, 1, 1);
                    }
                }
                shogi_unmake_move(p, &u);
                if (ctx->time_up) break;

                /* root_moves[i] == m by construction, so store directly. */
                ctx->root_scores[i] = s;

                if (s > root_best) {
                    root_best = s;
                    root_best_move = m;
                }
                if (s > root_alpha) root_alpha = s;
            }

            if (ctx->time_up) { score = root_best; iter_best = root_best_move; break; }

            if (root_best <= alpha) {
                alpha = alpha - delta * 2;
                delta *= 2;
                /* Clamp alpha above -MATE_THRESHOLD so fail-low returns
                 * don't enter the mate-score range and drift through
                 * TT transpositions via the ply adjustment. */
                if (alpha < -SRS_NON_MATE_CAP) alpha = -SRS_NON_MATE_CAP;
                continue;
            }
            if (root_best >= beta) {
                beta = beta + delta * 2;
                delta *= 2;
                /* Clamp beta below MATE_THRESHOLD so fail-high returns
                 * don't get mate-adjusted in TT. If the true value is
                 * above the cap, the search correctly treats it as
                 * "mate or very-high-eval" and returns ~NON_MATE_CAP. */
                if (beta > SRS_NON_MATE_CAP) beta = SRS_NON_MATE_CAP;
                continue;
            }
            score = root_best;
            iter_best = root_best_move;
            break;
        }

        if (ctx->time_up && d > 1) break;

        if (iter_best != SMOVE_NULL) {
            *best_move = iter_best;
            *best_score = score;
            last_score = score;
            ctx->root_completed_depth = d;
            /* Fire callback after each completed iteration so the GUI sees
             * depth progression even if cb_interval nodes haven't elapsed. */
            if (ctx->progress_cb) {
                srs_fire_progress(ctx, (d == ctx->max_depth) ? 1 : 0);
                if (ctx->abort_requested) break;
            }
        }
    }

    /* Final callback — mark as done even if the search was time-limited. */
    if (ctx->progress_cb) {
        srs_fire_progress(ctx, 1);
    }
}

/* ---- Python wrapper ---------------------------------------------------- */

static PyObject *accel_shogi_rule_search(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj, *sh_obj, *gh_obj;
    int side, max_depth;
    double time_ms = 1000.0;
    if (!PyArg_ParseTuple(args, "OOOii|d",
                          &board_obj, &sh_obj, &gh_obj, &side,
                          &max_depth, &time_ms)) {
        return NULL;
    }

    ShogiPosition p;
    if (sh_pyapi_parse_position(board_obj, sh_obj, gh_obj, side, &p) != 0)
        return NULL;

    if (!g_srs_initialized) srs_context_init(&g_srs);
    srs_reset_for_search(&g_srs, max_depth, time_ms);

    SMove best = SMOVE_NULL;
    int32_t score = 0;

    /* Release the GIL for the duration of the search — the hot loop is
     * pure C on a local ShogiPosition and does not touch any Python
     * objects, so it's safe to let other Python threads run. The GUI
     * thread (if any) stays responsive. */
    Py_BEGIN_ALLOW_THREADS
    srs_search_root(&g_srs, &p, &best, &score);
    Py_END_ALLOW_THREADS

    if (best == SMOVE_NULL) {
        Py_RETURN_NONE;
    }

    /* Return ((from, to, promo, drop), score, nodes) */
    PyObject *move_tuple = sh_pyapi_move_to_tuple(best);
    if (!move_tuple) return NULL;
    PyObject *result = PyTuple_Pack(3,
        move_tuple,
        PyFloat_FromDouble((double)score),
        PyLong_FromLong(g_srs.nodes));
    Py_DECREF(move_tuple);
    return result;
}

/* ------------------------------------------------------------------------
 *  shogi_rule_search_live(board, sh, gh, side, max_depth, time_ms,
 *                         callback, cb_interval=1_000_000)
 *
 *  Like shogi_rule_search but fires a Python callback periodically with
 *  the current root top-N moves so a GUI thread can render live progress.
 *
 *  The callback signature:
 *      callback(completed_depth, max_depth, [(move_tuple, score), ...], done)
 *          -> truthy to request abort, falsy to continue
 *
 *  Returns the same ((from, to, promo, drop), score, nodes) tuple as the
 *  non-live variant, or None if no best move found.
 * ----------------------------------------------------------------------- */
static PyObject *accel_shogi_rule_search_live(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj, *sh_obj, *gh_obj, *cb_obj;
    int side, max_depth;
    double time_ms = 60000.0;
    long cb_interval = 1000000;
    if (!PyArg_ParseTuple(args, "OOOiidO|l",
                          &board_obj, &sh_obj, &gh_obj, &side,
                          &max_depth, &time_ms, &cb_obj, &cb_interval)) {
        return NULL;
    }
    if (!PyCallable_Check(cb_obj)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable");
        return NULL;
    }

    ShogiPosition p;
    if (sh_pyapi_parse_position(board_obj, sh_obj, gh_obj, side, &p) != 0)
        return NULL;

    if (!g_srs_initialized) srs_context_init(&g_srs);
    srs_reset_for_search(&g_srs, max_depth, time_ms);

    /* Wire up the live-progress callback. We borrow a reference — the
     * caller is expected to keep the callable alive for the duration of
     * this call, which is trivially true because this function is
     * synchronous. */
    Py_INCREF(cb_obj);
    g_srs.progress_cb = cb_obj;
    g_srs.cb_interval = (cb_interval > 0) ? cb_interval : 1000000;
    g_srs.nodes_last_cb = 0;

    SMove best = SMOVE_NULL;
    int32_t score = 0;

    Py_BEGIN_ALLOW_THREADS
    srs_search_root(&g_srs, &p, &best, &score);
    Py_END_ALLOW_THREADS

    Py_DECREF(cb_obj);
    g_srs.progress_cb = NULL;

    if (best == SMOVE_NULL) {
        Py_RETURN_NONE;
    }

    PyObject *move_tuple = sh_pyapi_move_to_tuple(best);
    if (!move_tuple) return NULL;
    PyObject *result = PyTuple_Pack(3,
        move_tuple,
        PyFloat_FromDouble((double)score),
        PyLong_FromLong(g_srs.nodes));
    Py_DECREF(move_tuple);
    return result;
}
