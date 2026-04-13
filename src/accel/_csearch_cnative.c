/* ========================================================================
 * _csearch_cnative.c — C-native, multi-threaded (Lazy SMP) NNUE search.
 *
 * Phase 1 + 2 delivery: the hot alpha-beta loop runs entirely in C on a
 * local ChessPosition / ShogiPosition struct. Movegen, make/unmake,
 * repetition detection, TT probes and the NNUE eval all happen with the
 * Python GIL released, so a pthread worker pool can do real parallel
 * work. Lazy SMP: N workers search the same root through a shared
 * atomic TT with Stockfish's key-XOR-data torn-read detection.
 *
 * Exposes two methods on CSearchObject (defined in _csearch_types.c):
 *
 *     CSearch.search_cnative(game_type, ..., max_depth, time_limit_ms)
 *         -> ((from, to, promo, drop), score, nodes) or None
 *
 *     CSearch.search_top_n_live_cnative(game_type, ..., n, max_depth,
 *                                        time_limit_ms, live_ref, stop_event)
 *         -> list of ((from, to, promo, drop), score)
 *
 * The Python wrapper (src/search/alphabeta.py) unpacks a GameState once
 * at the entry, so we never touch PyObject inside the parallel section.
 * ======================================================================= */

#include <pthread.h>
#include <stdatomic.h>

/* ---- Constants --------------------------------------------------------- */
#define CN_INF              1000000
#define CN_MATE             100000     /* half-ply units */
#define CN_MATE_THRESHOLD    90000
#define CN_MAX_PLY              64
#define CN_MAX_QDEPTH            8
#define CN_HIST_RING            64
#define CN_MAX_THREADS          16
#define CN_MAX_MOVES           600     /* shogi upper bound */
#define CN_HISTORY_SIZE      (81 * 81)

#define CN_TT_EXACT  0
#define CN_TT_ALPHA  1
#define CN_TT_BETA   2

/* ---- Game dispatch ----------------------------------------------------- */
typedef enum {
    CN_GAME_CHESS = 0,
    CN_GAME_SHOGI = 1,
} CNGameType;

/* A CNMove is a unified 32-bit handle that can encode either a ChMove
 * (chess, 18 bits used) or an SMove (shogi, 24 bits used). Both fit. */
typedef uint32_t CNMove;
#define CN_MOVE_NULL ((CNMove)0xFFFFFFFFu)

/* ---- Atomic TT with Stockfish XOR trick -------------------------------- *
 *
 * Single 64-bit data word: bits 0..15 = clamped int16 score,
 *                          bits 16..47 = 32-bit move,
 *                          bits 48..55 = depth (0..255),
 *                          bits 56..57 = flag.
 *
 * key_xor_data = full 64-bit position key XOR data. Readers check that
 * (key_xor_data XOR data) == key. Torn reads — where `data` is from a
 * different store than `key_xor_data` — detected with probability 1 -
 * 2^-64, which is plenty for a heuristic TT. No mutex needed.
 */
typedef struct CNTTEntry {
    _Atomic uint64_t key_xor_data;
    _Atomic uint64_t data;
} CNTTEntry;

static inline int16_t cn_clamp_score(int32_t s) {
    if (s >  32000) return  32000;
    if (s < -32000) return -32000;
    return (int16_t)s;
}

static inline uint64_t cn_tt_pack(int32_t score, CNMove move, int depth, int flag) {
    uint16_t s = (uint16_t)(int16_t)cn_clamp_score(score);
    uint64_t d = (uint64_t)(depth & 0xFF) << 48;
    uint64_t f = (uint64_t)(flag  & 0x03) << 56;
    uint64_t m = (uint64_t)move << 16;
    return (uint64_t)s | m | d | f;
}

static inline void cn_tt_unpack(uint64_t d, int32_t *score, CNMove *move,
                                 int *depth, int *flag) {
    *score = (int32_t)(int16_t)(d & 0xFFFF);
    *move  = (CNMove)((d >> 16) & 0xFFFFFFFFu);
    *depth = (int)((d >> 48) & 0xFF);
    *flag  = (int)((d >> 56) & 0x03);
}

static inline void cn_tt_store(CNTTEntry *e, uint64_t key, uint64_t data) {
    atomic_store_explicit(&e->data,         data,        memory_order_relaxed);
    atomic_store_explicit(&e->key_xor_data, key ^ data,  memory_order_relaxed);
}

static inline int cn_tt_probe(CNTTEntry *e, uint64_t key, uint64_t *out_data) {
    uint64_t d    = atomic_load_explicit(&e->data,         memory_order_relaxed);
    uint64_t kxor = atomic_load_explicit(&e->key_xor_data, memory_order_relaxed);
    if ((kxor ^ d) == key) { *out_data = d; return 1; }
    return 0;
}

/* Mate-score ply adjustment (Stockfish convention). */
static inline int32_t cn_tt_score_to_tt(int32_t s, int ply) {
    if (s >=  CN_MATE_THRESHOLD) return s + ply;
    if (s <= -CN_MATE_THRESHOLD) return s - ply;
    return s;
}
static inline int32_t cn_tt_score_from_tt(int32_t s, int ply) {
    if (s >=  CN_MATE_THRESHOLD) return s - ply;
    if (s <= -CN_MATE_THRESHOLD) return s + ply;
    return s;
}

/* ---- Position/move unified types --------------------------------------- */
typedef union {
    ChessPosition chess;
    ShogiPosition shogi;
} CNPosition;

typedef union {
    ChessUndo chess;
    ShogiUndo shogi;
} CNUndo;

/* ---- Feature set config (cached once at search start) ----------------- */
typedef struct {
    int      num_squares;         /* 64 chess, 81 shogi */
    int      num_piece_types;     /* num_board_piece_types for shogi */
    int      king_board_val;      /* 6 chess, 8 shogi */
    int      bv2type_len;
    int8_t   bv2type[32];
    /* Shogi-only */
    int      num_hand_piece_types;
    int      max_hand_count;
    int      board_features;      /* = ns * (npt * 2 * ns) */
} CNFeatureCfg;

/* ---- Pure-C feature computation (mirrors halfkp_active_features) ------- */
static int cn_halfkp_chess_features(const CNFeatureCfg *cfg,
                                     const int8_t *board, int king_sq,
                                     int perspective, int *out) {
    int count = 0;
    int piece_sq_combos = cfg->num_piece_types * 2 * cfg->num_squares;
    for (int sq = 0; sq < cfg->num_squares; sq++) {
        int8_t val = board[sq];
        if (val == 0) continue;
        int abs_val = (val > 0) ? val : -val;
        if (abs_val == cfg->king_board_val) continue;
        if (abs_val >= cfg->bv2type_len) continue;
        int pt = cfg->bv2type[abs_val];
        if (pt < 0) continue;
        int color = (val > 0) ? 0 : 1;
        int rel = (color == perspective) ? 0 : 1;
        int idx = king_sq * piece_sq_combos
                + (rel * cfg->num_piece_types + pt) * cfg->num_squares
                + sq;
        out[count++] = idx;
    }
    return count;
}

static int cn_halfkp_shogi_features(const CNFeatureCfg *cfg,
                                     const ShogiPosition *p,
                                     int perspective, int *out) {
    int count = 0;
    int king_sq = (perspective == 0) ? p->sente_king_sq : p->gote_king_sq;
    int piece_sq_combos = cfg->num_piece_types * 2 * cfg->num_squares;
    int hand_combos     = cfg->num_hand_piece_types * 2 * cfg->max_hand_count;

    /* Board pieces */
    for (int sq = 0; sq < cfg->num_squares; sq++) {
        int8_t val = p->board[sq];
        if (val == 0) continue;
        int abs_val = (val > 0) ? val : -val;
        if (abs_val == cfg->king_board_val) continue;
        if (abs_val >= cfg->bv2type_len) continue;
        int pt = cfg->bv2type[abs_val];
        if (pt < 0) continue;
        int color = (val > 0) ? 0 : 1;
        int rel = (color == perspective) ? 0 : 1;
        int idx = king_sq * piece_sq_combos
                + (rel * cfg->num_piece_types + pt) * cfg->num_squares
                + sq;
        out[count++] = idx;
    }

    /* Hand pieces: side 0 (sente) and side 1 (gote). Each hand piece type
     * with count N produces N features: "has at least k of this piece". */
    const int8_t *hands[2] = { p->sente_hand, p->gote_hand };
    for (int h = 0; h < 2; h++) {
        int rel = (h == perspective) ? 0 : 1;
        for (int pt = 0; pt < cfg->num_hand_piece_types && pt < SHOGI_HAND_TYPES; pt++) {
            int cnt = hands[h][pt];
            int lim = (cnt < cfg->max_hand_count) ? cnt : cfg->max_hand_count;
            for (int k = 0; k < lim; k++) {
                int idx = cfg->board_features
                        + king_sq * hand_combos
                        + (rel * cfg->num_hand_piece_types + pt) * cfg->max_hand_count
                        + k;
                out[count++] = idx;
            }
        }
    }
    return count;
}

/* ---- Per-thread accumulator view -------------------------------------- *
 *
 * The shared AccelAccumObject holds the weight tables. Each thread gets
 * its own white_acc / black_acc buffers (so their refreshes don't race),
 * but they reference the same ft_weight / ft_bias / layer weights.
 */
typedef struct {
    AccelAccumObject *weights;  /* borrowed: shared weights */
    int      acc_size;
    int      use_int16;
    float    inv_quant_scale;
    /* Thread-local accumulator buffers (one active pair) */
    float   *white_acc;
    float   *black_acc;
    int16_t *white_acc_q;
    int16_t *black_acc_q;
} CNAccumView;

static void cn_accum_refresh_perspective(CNAccumView *a, int perspective,
                                          const int *indices, int count) {
    int sz = a->acc_size;
    AccelAccumObject *w = a->weights;
    if (a->use_int16) {
        int16_t *dst = (perspective == 0) ? a->white_acc_q : a->black_acc_q;
        memcpy(dst, w->ft_bias_q, (size_t)sz * sizeof(int16_t));
        for (int i = 0; i < count; i++) {
            int idx = indices[i];
            if (idx >= 0 && idx < w->num_features) {
                neon_vec_add_i16(dst, w->ft_weight_q + (size_t)idx * sz, sz);
            }
        }
    } else {
        float *dst = (perspective == 0) ? a->white_acc : a->black_acc;
        memcpy(dst, w->ft_bias, (size_t)sz * sizeof(float));
        for (int i = 0; i < count; i++) {
            int idx = indices[i];
            if (idx >= 0 && idx < w->num_features) {
                neon_vec_add(dst, w->ft_weight + (size_t)idx * sz, sz);
            }
        }
    }
}

static float cn_accum_evaluate(CNAccumView *a, int side_to_move) {
    AccelAccumObject *w = a->weights;
    int acc_size = a->acc_size;
    int l1       = w->l1_size;
    int l2       = w->l2_size;

    float input[1024];
    float l1_out[256];
    float l2_out[256];

    if (a->use_int16) {
        int16_t *first  = (side_to_move == 0) ? a->white_acc_q : a->black_acc_q;
        int16_t *second = (side_to_move == 0) ? a->black_acc_q : a->white_acc_q;
        neon_dequant_clipped_relu(input,            first,  acc_size, a->inv_quant_scale);
        neon_dequant_clipped_relu(input + acc_size, second, acc_size, a->inv_quant_scale);
    } else {
        float *first  = (side_to_move == 0) ? a->white_acc : a->black_acc;
        float *second = (side_to_move == 0) ? a->black_acc : a->white_acc;
        neon_clipped_relu_copy(input,            first,  acc_size);
        neon_clipped_relu_copy(input + acc_size, second, acc_size);
    }

    memcpy(l1_out, w->l1_bias, (size_t)l1 * sizeof(float));
    sgemv(l1, acc_size * 2, 1.0f, w->l1_weight, acc_size * 2, input, 1.0f, l1_out);
    neon_screlu_inplace(l1_out, l1);

    memcpy(l2_out, w->l2_bias, (size_t)l2 * sizeof(float));
    sgemv(l2, l1, 1.0f, w->l2_weight, l1, l1_out, 1.0f, l2_out);
    neon_screlu_inplace(l2_out, l2);

    return sdot(l2, w->out_weight, l2_out) + w->out_bias[0];
}

/* Refresh the accumulator from the current position. Called at every
 * node that needs a static eval. */
static void cn_refresh_full(CNAccumView *acc, const CNFeatureCfg *cfg,
                             CNGameType gt, const CNPosition *pos) {
    int idx_w[256], idx_b[256];
    int n_w, n_b;
    if (gt == CN_GAME_CHESS) {
        const ChessPosition *p = &pos->chess;
        n_w = cn_halfkp_chess_features(cfg, p->board, p->white_king_sq, 0, idx_w);
        n_b = cn_halfkp_chess_features(cfg, p->board, p->black_king_sq, 1, idx_b);
    } else {
        const ShogiPosition *p = &pos->shogi;
        n_w = cn_halfkp_shogi_features(cfg, p, 0, idx_w);
        n_b = cn_halfkp_shogi_features(cfg, p, 1, idx_b);
    }
    cn_accum_refresh_perspective(acc, 0, idx_w, n_w);
    cn_accum_refresh_perspective(acc, 1, idx_b, n_b);
}

/* ---- Shared context shared across all worker threads ------------------ */
typedef struct {
    CNGameType       game_type;
    CNFeatureCfg     cfg;
    AccelAccumObject *acc_weights;   /* shared weights, borrowed */
    float            eval_scale;

    CNTTEntry       *tt;
    uint32_t         tt_mask;

    CNPosition       root_pos;
    uint64_t         root_history[CN_HIST_RING];
    int              root_history_n;

    int              max_depth;       /* 0 = infinite */
    double           time_limit_ms;   /* 0 = infinite */
    double           start_time_ms;
    _Atomic int      time_up;
    _Atomic int      abort_flag;

    int              n_threads;

    /* Best result aggregated from all threads; updated under best_mutex. */
    pthread_mutex_t  best_mutex;
    _Atomic int      global_best_depth;
    int32_t          global_best_score;
    CNMove           global_best_move;
    long             global_total_nodes;
    /* Top-N state for live callbacks (updated by primary thread only) */
    int              top_n;
    CNMove           top_moves[8];
    int32_t          top_scores[8];
    _Atomic int      top_version;
} CNShared;

/* ---- Per-thread context ----------------------------------------------- */
typedef struct CNThread {
    CNShared   *shared;
    int         tid;
    CNPosition  pos;

    /* Accumulator buffers (thread-local) */
    CNAccumView acc;
    float      *acc_bufs_f[2];   /* allocation handles for free() */
    int16_t    *acc_bufs_q[2];

    /* Search state (thread-local) */
    CNMove      killers[CN_MAX_PLY][2];
    int32_t     history[CN_HISTORY_SIZE];
    uint64_t    hist[CN_HIST_RING];
    int         hist_n;
    long        nodes;
    int         completed_depth;
    int32_t     best_score;
    CNMove      best_move;

    pthread_t   pthread;
} CNThread;

/* ---- LMR table --------------------------------------------------------- */
static int cn_lmr_table[CN_MAX_PLY][CN_MAX_MOVES];
static int cn_lmr_ready = 0;
static void cn_init_lmr(void) {
    if (cn_lmr_ready) return;
    for (int d = 0; d < CN_MAX_PLY; d++) {
        for (int m = 0; m < CN_MAX_MOVES; m++) {
            if (d < 2 || m < 2) { cn_lmr_table[d][m] = 0; continue; }
            int r = (int)(0.5 + log((double)d) * log((double)m) / 2.3);
            cn_lmr_table[d][m] = r;
        }
    }
    cn_lmr_ready = 1;
}

/* ---- Time helpers ----------------------------------------------------- */
static double cn_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static inline int cn_time_check(CNThread *t) {
    if ((t->nodes & 4095) == 0) {
        CNShared *s = t->shared;
        if (atomic_load_explicit(&s->abort_flag, memory_order_relaxed)) return 1;
        if (s->time_limit_ms > 0.0) {
            double now = cn_now_ms();
            if (now - s->start_time_ms >= s->time_limit_ms) {
                atomic_store_explicit(&s->time_up, 1, memory_order_relaxed);
                return 1;
            }
        }
        if (atomic_load_explicit(&s->time_up, memory_order_relaxed)) return 1;
    }
    return 0;
}

static inline int cn_should_stop(CNThread *t) {
    CNShared *s = t->shared;
    return atomic_load_explicit(&s->abort_flag, memory_order_relaxed)
        || atomic_load_explicit(&s->time_up,    memory_order_relaxed);
}

/* ---- Repetition + 50-move draw check --------------------------------- */
static int cn_is_repetition(const CNThread *t, uint64_t hash) {
    int cnt = 0;
    for (int i = 0; i < t->hist_n; i++) {
        if (t->hist[i] == hash) { if (++cnt >= 2) return 1; }
    }
    return 0;
}

/* ---- Game-type dispatch wrappers ------------------------------------- */
static int cn_expand_legal(CNGameType gt, CNPosition *p, CNMove *out) {
    if (gt == CN_GAME_CHESS) {
        ChMove buf[CHESS_MAX_MOVES];
        int n = chess_expand_legal_moves(&p->chess, buf);
        for (int i = 0; i < n; i++) out[i] = (CNMove)buf[i];
        return n;
    } else {
        SMove buf[SHOGI_MAX_MOVES];
        int n = shogi_expand_legal_moves(&p->shogi, buf);
        for (int i = 0; i < n; i++) out[i] = (CNMove)buf[i];
        return n;
    }
}

static int cn_make_move(CNGameType gt, CNPosition *p, CNMove m, CNUndo *u) {
    if (gt == CN_GAME_CHESS) return chess_make_move(&p->chess, (ChMove)m, &u->chess);
    else                      return shogi_make_move(&p->shogi, (SMove)m, &u->shogi);
}

static void cn_unmake_move(CNGameType gt, CNPosition *p, CNUndo *u) {
    if (gt == CN_GAME_CHESS) chess_unmake_move(&p->chess, &u->chess);
    else                      shogi_unmake_move(&p->shogi, &u->shogi);
}

static int cn_is_in_check(CNGameType gt, const CNPosition *p) {
    if (gt == CN_GAME_CHESS) return chess_is_in_check(&p->chess, p->chess.side);
    else                      return shogi_is_in_check(&p->shogi, p->shogi.side);
}

static int cn_side_to_move(CNGameType gt, const CNPosition *p) {
    return (gt == CN_GAME_CHESS) ? p->chess.side : p->shogi.side;
}

static uint64_t cn_hash(CNGameType gt, const CNPosition *p) {
    return (gt == CN_GAME_CHESS) ? p->chess.hash : p->shogi.hash;
}

static int cn_halfmove_draw(CNGameType gt, const CNPosition *p) {
    return (gt == CN_GAME_CHESS) && p->chess.halfmove >= 100;
}

/* ---- Move encoding helpers for the two games ------------------------- */
static int cn_move_is_capture(CNGameType gt, const CNPosition *p, CNMove m) {
    if (gt == CN_GAME_CHESS) {
        ChMove ch = (ChMove)m;
        int to = chmove_to(ch);
        return (p->chess.board[to] != 0) || chmove_is_ep(ch);
    } else {
        SMove sm = (SMove)m;
        if (smove_is_drop(sm)) return 0;
        int to = smove_to(sm);
        return p->shogi.board[to] != 0;
    }
}

static int cn_move_is_promo(CNGameType gt, CNMove m) {
    if (gt == CN_GAME_CHESS) return chmove_promo((ChMove)m) != 0;
    else                      return smove_promo((SMove)m) != 0;
}

static int cn_move_from(CNGameType gt, CNMove m) {
    return (gt == CN_GAME_CHESS) ? chmove_from((ChMove)m) : smove_from((SMove)m);
}

static int cn_move_to(CNGameType gt, CNMove m) {
    return (gt == CN_GAME_CHESS) ? chmove_to((ChMove)m) : smove_to((SMove)m);
}

/* MVV-LVA with default piece values. */
static const int CN_PIECE_VAL[16] = {
    /* 0..7 */ 0, 100, 300, 300, 500, 900, 20000, 600,
    /* 8..15 (shogi gold-like promos) */ 20000, 150, 250, 300, 400, 700, 800, 0,
};

/* ---- Move ordering --------------------------------------------------- */
typedef struct { CNMove move; int32_t score; } CNScoredMove;

static void cn_score_moves(CNThread *t, CNGameType gt, const CNPosition *p,
                            const CNMove *moves, int n, CNScoredMove *out,
                            int ply, CNMove tt_move) {
    for (int i = 0; i < n; i++) {
        CNMove m = moves[i];
        int32_t s = 0;
        if (m == tt_move && tt_move != CN_MOVE_NULL) {
            s = 100000000;
        } else if (cn_move_is_capture(gt, p, m)) {
            int to = cn_move_to(gt, m);
            int from = cn_move_from(gt, m);
            int victim_abs = 0;
            int attacker_abs = 0;
            if (gt == CN_GAME_CHESS) {
                int8_t vb = p->chess.board[to];
                int8_t ab = (from >= 0) ? p->chess.board[from] : 0;
                if (chmove_is_ep((ChMove)m)) victim_abs = CP_PAWN;
                else                          victim_abs = (vb > 0) ? vb : -vb;
                attacker_abs = (ab > 0) ? ab : -ab;
            } else {
                int8_t vb = p->shogi.board[to];
                int8_t ab = (from >= 0 && from < SHOGI_SQUARES) ? p->shogi.board[from] : 0;
                victim_abs = (vb > 0) ? vb : -vb;
                attacker_abs = (ab > 0) ? ab : -ab;
            }
            int vi = victim_abs & 0xF;
            int ai = attacker_abs & 0xF;
            s = 10000000 + CN_PIECE_VAL[vi] * 10 - CN_PIECE_VAL[ai];
        } else {
            if (ply < CN_MAX_PLY) {
                if (t->killers[ply][0] == m) s = 5000000;
                else if (t->killers[ply][1] == m) s = 4000000;
            }
            int from = cn_move_from(gt, m);
            int to   = cn_move_to(gt, m);
            if (from >= 0 && from < 81 && to >= 0 && to < 81) {
                s += t->history[from * 81 + to];
            }
        }
        if (cn_move_is_promo(gt, m)) s += 3000000;
        out[i].move  = m;
        out[i].score = s;
    }
}

static void cn_sort_moves(CNScoredMove *arr, int n) {
    for (int i = 1; i < n; i++) {
        CNScoredMove cur = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].score < cur.score) {
            arr[j + 1] = arr[j]; j--;
        }
        arr[j + 1] = cur;
    }
}

/* ---- Forward decls ---------------------------------------------------- */
static int32_t cn_alphabeta(CNThread *t, int depth, int32_t alpha, int32_t beta,
                             int ply, int allow_null);
static int32_t cn_quiescence(CNThread *t, int32_t alpha, int32_t beta, int qdepth);

/* ---- Quiescence search ----------------------------------------------- */
static int32_t cn_quiescence(CNThread *t, int32_t alpha, int32_t beta, int qdepth) {
    t->nodes++;
    if (cn_time_check(t)) return 0;

    CNShared *s = t->shared;
    CNGameType gt = s->game_type;

    cn_refresh_full(&t->acc, &s->cfg, gt, &t->pos);
    int stm = cn_side_to_move(gt, &t->pos);
    int32_t stand = (int32_t)(cn_accum_evaluate(&t->acc, stm) * s->eval_scale);

    if (stand >= beta)  return beta;
    if (stand > alpha)  alpha = stand;
    if (qdepth >= CN_MAX_QDEPTH) return alpha;

    CNMove moves[CN_MAX_MOVES];
    int nm = cn_expand_legal(gt, &t->pos, moves);
    if (nm == 0) return alpha;

    CNMove caps[CN_MAX_MOVES];
    int nc = 0;
    for (int i = 0; i < nm; i++) {
        CNMove m = moves[i];
        if (cn_move_is_capture(gt, &t->pos, m) || cn_move_is_promo(gt, m))
            caps[nc++] = m;
    }
    if (nc == 0) return alpha;

    CNScoredMove scored[CN_MAX_MOVES];
    cn_score_moves(t, gt, &t->pos, caps, nc, scored, 0, CN_MOVE_NULL);
    cn_sort_moves(scored, nc);

    for (int i = 0; i < nc; i++) {
        CNUndo u;
        if (cn_make_move(gt, &t->pos, scored[i].move, &u) != 0) continue;
        int32_t score = -cn_quiescence(t, -beta, -alpha, qdepth + 1);
        cn_unmake_move(gt, &t->pos, &u);
        if (cn_should_stop(t)) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

/* ---- Main alpha-beta -------------------------------------------------- */
static int32_t cn_alphabeta(CNThread *t, int depth, int32_t alpha, int32_t beta,
                             int ply, int allow_null) {
    t->nodes++;
    if (cn_time_check(t)) return 0;

    CNShared *s = t->shared;
    CNGameType gt = s->game_type;

    /* Repetition / 50-move draw inside the tree */
    if (ply > 0 && cn_is_repetition(t, cn_hash(gt, &t->pos))) return 0;
    if (cn_halfmove_draw(gt, &t->pos)) return 0;

    int in_check = cn_is_in_check(gt, &t->pos);
    if (in_check) depth += 1;  /* check extension */

    if (depth <= 0) return cn_quiescence(t, alpha, beta, 0);

    /* TT probe */
    uint64_t key = cn_hash(gt, &t->pos);
    uint64_t tt_data;
    CNMove tt_move = CN_MOVE_NULL;
    int32_t tt_score = 0;
    int tt_depth = 0, tt_flag = 0;
    int has_tt = cn_tt_probe(&s->tt[key & s->tt_mask], key, &tt_data);
    if (has_tt) {
        cn_tt_unpack(tt_data, &tt_score, &tt_move, &tt_depth, &tt_flag);
        tt_score = cn_tt_score_from_tt(tt_score, ply);
        if (tt_depth >= depth && ply > 0) {
            if (tt_flag == CN_TT_EXACT) return tt_score;
            if (tt_flag == CN_TT_ALPHA && tt_score <= alpha) return alpha;
            if (tt_flag == CN_TT_BETA  && tt_score >= beta)  return beta;
        }
    }

    CNMove moves[CN_MAX_MOVES];
    int nm = cn_expand_legal(gt, &t->pos, moves);
    if (nm == 0) {
        /* No legal moves: mate if in check, stalemate = draw otherwise */
        if (in_check) return -CN_MATE + ply;
        return 0;
    }

    /* Push current position into history ring for child-level repetition
     * detection. */
    if (t->hist_n < CN_HIST_RING) t->hist[t->hist_n++] = key;

    /* ---- Futility pruning setup ------------------------------------ */
    int futility_ok = 0;
    int32_t static_eval = 0;
    if (depth <= 2 && !in_check) {
        cn_refresh_full(&t->acc, &s->cfg, gt, &t->pos);
        int stm = cn_side_to_move(gt, &t->pos);
        static_eval = (int32_t)(cn_accum_evaluate(&t->acc, stm) * s->eval_scale);
        int margin = (depth == 1) ? 200 : 500;
        if (static_eval + margin <= alpha) futility_ok = 1;
    }

    /* ---- Null-move pruning ----------------------------------------- */
    if (allow_null && depth >= 3 && !in_check) {
        /* Only if we have a positive static_eval wrt beta */
        if (static_eval == 0) {
            cn_refresh_full(&t->acc, &s->cfg, gt, &t->pos);
            int stm = cn_side_to_move(gt, &t->pos);
            static_eval = (int32_t)(cn_accum_evaluate(&t->acc, stm) * s->eval_scale);
        }
        if (static_eval >= beta) {
            /* Make null move: flip side, clear ep */
            CNPosition save = t->pos;
            if (gt == CN_GAME_CHESS) {
                ChessPosition *p = &t->pos.chess;
                /* Hash XOR: remove old ep file, flip side */
                if (p->ep_square >= 0) p->hash ^= g_chess_z_ep_file[cp_file(p->ep_square)];
                p->hash ^= g_chess_z_side;
                p->ep_square = -1;
                p->side ^= 1;
            } else {
                ShogiPosition *p = &t->pos.shogi;
                p->hash ^= g_shogi_z_side[0];
                p->hash ^= g_shogi_z_side[1];
                p->side ^= 1;
            }
            int R = 2 + (depth >= 6 ? 1 : 0);
            int32_t nscore = -cn_alphabeta(t, depth - 1 - R, -beta, -beta + 1, ply + 1, 0);
            t->pos = save;
            if (cn_should_stop(t)) { if (t->hist_n > 0) t->hist_n--; return 0; }
            if (nscore >= beta) { if (t->hist_n > 0) t->hist_n--; return beta; }
        }
    }

    /* ---- Main move loop ------------------------------------------- */
    CNScoredMove scored[CN_MAX_MOVES];
    cn_score_moves(t, gt, &t->pos, moves, nm, scored, ply, tt_move);
    cn_sort_moves(scored, nm);

    int32_t best_score = -CN_INF;
    CNMove  best_move  = scored[0].move;
    int32_t orig_alpha = alpha;

    for (int i = 0; i < nm; i++) {
        if (cn_should_stop(t)) break;
        CNMove m = scored[i].move;
        int is_cap = cn_move_is_capture(gt, &t->pos, m);
        int is_pro = cn_move_is_promo(gt, m);
        int is_quiet = !is_cap && !is_pro;

        /* Futility pruning: skip quiet moves past the first */
        if (futility_ok && i > 0 && is_quiet) continue;

        CNUndo u;
        if (cn_make_move(gt, &t->pos, m, &u) != 0) continue;

        int32_t score;
        if (i == 0) {
            score = -cn_alphabeta(t, depth - 1, -beta, -alpha, ply + 1, 1);
        } else {
            /* LMR + PVS */
            int reduction = 0;
            if (i >= 3 && depth >= 3 && !in_check && is_quiet) {
                int di = (depth < CN_MAX_PLY) ? depth : CN_MAX_PLY - 1;
                int mi = (i < CN_MAX_MOVES) ? i : CN_MAX_MOVES - 1;
                reduction = cn_lmr_table[di][mi];
            }
            score = -cn_alphabeta(t, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, 1);
            if (!cn_should_stop(t) && reduction > 0 && score > alpha) {
                score = -cn_alphabeta(t, depth - 1, -alpha - 1, -alpha, ply + 1, 1);
            }
            if (!cn_should_stop(t) && score > alpha && score < beta) {
                score = -cn_alphabeta(t, depth - 1, -beta, -alpha, ply + 1, 1);
            }
        }
        cn_unmake_move(gt, &t->pos, &u);

        if (cn_should_stop(t)) { if (t->hist_n > 0) t->hist_n--; return 0; }

        if (score > best_score) { best_score = score; best_move = m; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            /* Beta cutoff */
            if (is_quiet && ply < CN_MAX_PLY) {
                if (t->killers[ply][0] != m) {
                    t->killers[ply][1] = t->killers[ply][0];
                    t->killers[ply][0] = m;
                }
                int from = cn_move_from(gt, m);
                int to   = cn_move_to(gt, m);
                if (from >= 0 && from < 81 && to >= 0 && to < 81) {
                    t->history[from * 81 + to] += depth * depth;
                }
            }
            cn_tt_store(&s->tt[key & s->tt_mask], key,
                         cn_tt_pack(cn_tt_score_to_tt(beta, ply), m, depth, CN_TT_BETA));
            if (t->hist_n > 0) t->hist_n--;
            return beta;
        }
    }

    if (t->hist_n > 0) t->hist_n--;

    int flag = (best_score <= orig_alpha) ? CN_TT_ALPHA : CN_TT_EXACT;
    cn_tt_store(&s->tt[key & s->tt_mask], key,
                 cn_tt_pack(cn_tt_score_to_tt(best_score, ply), best_move, depth, flag));
    return best_score;
}

/* ---- Root search: iterative deepening for one thread ----------------- */
static void cn_search_root(CNThread *t) {
    CNShared *s = t->shared;
    CNGameType gt = s->game_type;
    cn_init_lmr();

    CNMove moves[CN_MAX_MOVES];
    int nm = cn_expand_legal(gt, &t->pos, moves);
    if (nm == 0) {
        t->best_move = CN_MOVE_NULL;
        t->best_score = 0;
        t->completed_depth = 0;
        return;
    }

    /* Skip-depth heuristic: each worker uses a different starting depth
     * pattern so they diverge through the TT. Stockfish-lite. */
    static const int SKIP_START[CN_MAX_THREADS] = {
        0, 1, 0, 1, 2, 3, 2, 3, 4, 0, 5, 1, 6, 2, 7, 3,
    };
    int skip_start = SKIP_START[t->tid % CN_MAX_THREADS];

    /* Seed default best = first legal move so we always have an answer. */
    t->best_move  = moves[0];
    t->best_score = 0;
    t->completed_depth = 0;

    int max_depth = (s->max_depth > 0) ? s->max_depth : 63;

    for (int depth = 1; depth <= max_depth; depth++) {
        if (cn_should_stop(t)) break;
        /* Helper threads start a couple plies deeper to diversify. */
        int d = depth + (t->tid > 0 ? (skip_start & 1) : 0);
        if (d > max_depth) d = max_depth;

        /* Reset killers/history at depth boundaries */
        if (depth == 1) {
            memset(t->killers, 0xFF, sizeof(t->killers));
            memset(t->history, 0, sizeof(t->history));
        }

        /* Seed repetition ring from the root history at the start of the
         * iteration. The ring is then mutated by cn_alphabeta as we
         * descend. */
        t->hist_n = s->root_history_n;
        for (int i = 0; i < t->hist_n; i++) t->hist[i] = s->root_history[i];

        CNScoredMove scored[CN_MAX_MOVES];
        uint64_t root_key = cn_hash(gt, &t->pos);
        CNMove tt_move = CN_MOVE_NULL;
        uint64_t tt_data;
        if (cn_tt_probe(&s->tt[root_key & s->tt_mask], root_key, &tt_data)) {
            int tscore, tdepth, tflag;
            CNMove tm;
            cn_tt_unpack(tt_data, &tscore, &tm, &tdepth, &tflag);
            tt_move = tm;
        }
        cn_score_moves(t, gt, &t->pos, moves, nm, scored, 0, tt_move);
        cn_sort_moves(scored, nm);

        int32_t alpha = -CN_INF, beta = CN_INF;
        int32_t iter_best = -CN_INF;
        CNMove  iter_move = scored[0].move;

        for (int i = 0; i < nm; i++) {
            if (cn_should_stop(t)) break;
            CNUndo u;
            if (cn_make_move(gt, &t->pos, scored[i].move, &u) != 0) continue;
            /* Push root key into history ring so children see repetition. */
            int pushed = 0;
            if (t->hist_n < CN_HIST_RING) { t->hist[t->hist_n++] = root_key; pushed = 1; }

            int32_t score;
            if (i == 0) {
                score = -cn_alphabeta(t, d - 1, -beta, -alpha, 1, 1);
            } else {
                score = -cn_alphabeta(t, d - 1, -alpha - 1, -alpha, 1, 1);
                if (!cn_should_stop(t) && score > alpha && score < beta) {
                    score = -cn_alphabeta(t, d - 1, -beta, -alpha, 1, 1);
                }
            }
            if (pushed && t->hist_n > 0) t->hist_n--;
            cn_unmake_move(gt, &t->pos, &u);

            if (cn_should_stop(t)) break;
            if (score > iter_best) {
                iter_best = score;
                iter_move = scored[i].move;
                if (score > alpha) alpha = score;
            }
        }

        if (cn_should_stop(t) && depth > 1) break;
        if (iter_best > -CN_INF) {
            t->best_score      = iter_best;
            t->best_move       = iter_move;
            t->completed_depth = d;

            /* Write back to TT with a root entry so sibling threads can
             * see the PV move. */
            cn_tt_store(&s->tt[root_key & s->tt_mask], root_key,
                         cn_tt_pack(cn_tt_score_to_tt(iter_best, 0),
                                     iter_move, d, CN_TT_EXACT));

            /* Primary thread publishes top-N for live callbacks. */
            if (t->tid == 0) {
                pthread_mutex_lock(&s->best_mutex);
                int n = (s->top_n < 8) ? s->top_n : 8;
                /* Resort the root moves by score from THIS iteration —
                 * reuse scored[] which we already scored. We'll just
                 * collect iter_best as the first, then the rest in the
                 * order they were searched (which, for a well-ordered
                 * root, is close to ranked). */
                s->top_moves[0]  = iter_move;
                s->top_scores[0] = iter_best;
                int k = 1;
                for (int i = 0; i < nm && k < n; i++) {
                    if (scored[i].move != iter_move) {
                        s->top_moves[k]  = scored[i].move;
                        s->top_scores[k] = 0;  /* unknown without re-search */
                        k++;
                    }
                }
                s->global_best_score = iter_best;
                s->global_best_move  = iter_move;
                atomic_store_explicit(&s->global_best_depth, d, memory_order_relaxed);
                atomic_fetch_add_explicit(&s->top_version, 1, memory_order_release);
                pthread_mutex_unlock(&s->best_mutex);
            } else {
                /* Helper thread: update global best if we completed a
                 * deeper iteration with a PV move. */
                int cur = atomic_load_explicit(&s->global_best_depth, memory_order_relaxed);
                if (d > cur) {
                    pthread_mutex_lock(&s->best_mutex);
                    if (d > s->global_best_depth) {
                        s->global_best_depth = d;
                        s->global_best_score = iter_best;
                        s->global_best_move  = iter_move;
                    }
                    pthread_mutex_unlock(&s->best_mutex);
                }
            }
        }

        /* Stop if this iteration exhausted the depth budget and time is
         * already up — don't spin on the next iteration. */
        if (cn_should_stop(t)) break;
    }
}

static void *cn_worker_entry(void *arg) {
    CNThread *t = (CNThread *)arg;
    cn_search_root(t);
    return NULL;
}

/* ---- Thread setup / teardown ----------------------------------------- */
static int cn_thread_alloc_accum(CNThread *t) {
    CNShared *s = t->shared;
    int sz = s->acc_weights->accumulator_size;
    t->acc.weights          = s->acc_weights;
    t->acc.acc_size         = sz;
    t->acc.use_int16        = s->acc_weights->use_int16;
    t->acc.inv_quant_scale  = s->acc_weights->inv_quant_scale;
    if (t->acc.use_int16) {
        t->acc_bufs_q[0] = (int16_t *)aligned_alloc(64, (size_t)sz * sizeof(int16_t) + 64);
        t->acc_bufs_q[1] = (int16_t *)aligned_alloc(64, (size_t)sz * sizeof(int16_t) + 64);
        if (!t->acc_bufs_q[0] || !t->acc_bufs_q[1]) return -1;
        t->acc.white_acc_q = t->acc_bufs_q[0];
        t->acc.black_acc_q = t->acc_bufs_q[1];
        t->acc.white_acc = NULL;
        t->acc.black_acc = NULL;
    } else {
        t->acc_bufs_f[0] = (float *)aligned_alloc(64, (size_t)sz * sizeof(float) + 64);
        t->acc_bufs_f[1] = (float *)aligned_alloc(64, (size_t)sz * sizeof(float) + 64);
        if (!t->acc_bufs_f[0] || !t->acc_bufs_f[1]) return -1;
        t->acc.white_acc = t->acc_bufs_f[0];
        t->acc.black_acc = t->acc_bufs_f[1];
        t->acc.white_acc_q = NULL;
        t->acc.black_acc_q = NULL;
    }
    return 0;
}

static void cn_thread_free_accum(CNThread *t) {
    free(t->acc_bufs_f[0]); t->acc_bufs_f[0] = NULL;
    free(t->acc_bufs_f[1]); t->acc_bufs_f[1] = NULL;
    free(t->acc_bufs_q[0]); t->acc_bufs_q[0] = NULL;
    free(t->acc_bufs_q[1]); t->acc_bufs_q[1] = NULL;
}

/* ---- Feature-set config extraction (runs while holding GIL) ---------- */
static int cn_extract_feature_cfg(PyObject *py_fs, CNFeatureCfg *cfg,
                                    CNGameType gt) {
    memset(cfg, 0, sizeof(*cfg));
    PyObject *v;

#define GETINT(field, attr, dflt) do {                                   \
        v = PyObject_GetAttrString(py_fs, attr);                         \
        if (v && v != Py_None) { cfg->field = (int)PyLong_AsLong(v); }   \
        else                    { cfg->field = (dflt); PyErr_Clear(); }  \
        Py_XDECREF(v);                                                   \
    } while (0)

    GETINT(num_squares,    "_num_squares",    (gt == CN_GAME_CHESS) ? 64 : 81);
    GETINT(king_board_val, "_king_board_val", (gt == CN_GAME_CHESS) ? 6 : 8);

    if (gt == CN_GAME_CHESS) {
        GETINT(num_piece_types, "_num_piece_types", 5);
    } else {
        GETINT(num_piece_types,      "_num_board_piece_types", 13);
        GETINT(num_hand_piece_types, "_num_hand_piece_types",   7);
        GETINT(max_hand_count,       "_max_hand_count",         18);
        GETINT(board_features,       "_board_features",          0);
        if (cfg->board_features == 0) {
            cfg->board_features = cfg->num_squares
                                 * cfg->num_piece_types * 2 * cfg->num_squares;
        }
    }
#undef GETINT

    /* Pull bv2type from the numpy int8 array via the buffer protocol. */
    v = PyObject_GetAttrString(py_fs, "_bv2type_arr");
    if (v && v != Py_None) {
        Py_buffer buf;
        if (PyObject_GetBuffer(v, &buf, PyBUF_SIMPLE) == 0) {
            int n = (int)buf.len;
            if (n > 32) n = 32;
            memcpy(cfg->bv2type, buf.buf, (size_t)n);
            cfg->bv2type_len = n;
            PyBuffer_Release(&buf);
        } else {
            PyErr_Clear();
        }
    } else {
        PyErr_Clear();
    }
    Py_XDECREF(v);
    if (cfg->bv2type_len == 0) {
        /* Fallback identity mapping for chess: piece code k -> type k-1. */
        cfg->bv2type_len = 16;
        memset(cfg->bv2type, -1, sizeof(cfg->bv2type));
        if (gt == CN_GAME_CHESS) {
            cfg->bv2type[1] = 0;  /* P */
            cfg->bv2type[2] = 1;  /* N */
            cfg->bv2type[3] = 2;  /* B */
            cfg->bv2type[4] = 3;  /* R */
            cfg->bv2type[5] = 4;  /* Q */
        } else {
            cfg->bv2type[1]  = 0; cfg->bv2type[2]  = 1; cfg->bv2type[3] = 2;
            cfg->bv2type[4]  = 3; cfg->bv2type[5]  = 4; cfg->bv2type[6] = 5;
            cfg->bv2type[7]  = 6;
            cfg->bv2type[9]  = 7; cfg->bv2type[10] = 8; cfg->bv2type[11] = 9;
            cfg->bv2type[12] = 10; cfg->bv2type[13] = 11; cfg->bv2type[14] = 12;
        }
    }

    /* Sanity: need acc_weights->num_features big enough. Caller verifies. */
    return 0;
}

/* ---- Position unpack from Python args -------------------------------- */
static int cn_unpack_chess_pos(ChessPosition *p,
                                 PyObject *board_obj, int side, int castling,
                                 int ep_sq, int halfmove, int wk_sq, int bk_sq) {
    Py_buffer bb;
    if (PyObject_GetBuffer(board_obj, &bb, PyBUF_SIMPLE) != 0) return -1;
    if (bb.len < CHESS_SQUARES) { PyBuffer_Release(&bb); return -1; }
    memset(p, 0, sizeof(*p));
    memcpy(p->board, bb.buf, CHESS_SQUARES);
    PyBuffer_Release(&bb);
    p->side     = (int8_t)(side & 1);
    p->castling = (int8_t)(castling & 0xF);
    p->ep_square     = (int16_t)ep_sq;
    p->halfmove      = (int16_t)halfmove;
    p->white_king_sq = (int16_t)wk_sq;
    p->black_king_sq = (int16_t)bk_sq;
    p->hash = chess_compute_hash(p);
    return 0;
}

static int cn_unpack_shogi_pos(ShogiPosition *p,
                                 PyObject *board_obj,
                                 PyObject *sh_obj, PyObject *gh_obj,
                                 int side) {
    return sh_pyapi_parse_position(board_obj, sh_obj, gh_obj, side, p);
}

/* ---- Public: run a Lazy-SMP search ----------------------------------- */
static int cn_run_search(CNShared *shared, CNThread *threads,
                          int n_threads, int *out_nodes) {
    cn_init_lmr();
    shared->start_time_ms = cn_now_ms();
    atomic_store(&shared->time_up, 0);
    atomic_store(&shared->abort_flag, 0);
    atomic_store(&shared->global_best_depth, 0);
    atomic_store(&shared->top_version, 0);
    shared->global_best_move  = CN_MOVE_NULL;
    shared->global_best_score = 0;
    shared->global_total_nodes = 0;

    /* Kick off helper workers; run the primary inline. */
    for (int i = 1; i < n_threads; i++) {
        threads[i].shared = shared;
        threads[i].tid    = i;
        threads[i].nodes  = 0;
        threads[i].completed_depth = 0;
        threads[i].best_move = CN_MOVE_NULL;
        threads[i].best_score = 0;
        threads[i].hist_n = 0;
        memset(threads[i].killers, 0xFF, sizeof(threads[i].killers));
        memset(threads[i].history, 0, sizeof(threads[i].history));
        /* Each worker starts from a copy of the root position. */
        threads[i].pos = shared->root_pos;
        if (pthread_create(&threads[i].pthread, NULL, cn_worker_entry, &threads[i]) != 0) {
            /* If we can't spawn a helper, degrade gracefully. */
            atomic_store(&shared->abort_flag, 1);
            return -1;
        }
    }

    /* Primary thread runs in the foreground so this function is
     * synchronous from the caller's point of view. */
    threads[0].shared = shared;
    threads[0].tid    = 0;
    threads[0].nodes  = 0;
    threads[0].completed_depth = 0;
    threads[0].best_move = CN_MOVE_NULL;
    threads[0].best_score = 0;
    threads[0].hist_n = 0;
    memset(threads[0].killers, 0xFF, sizeof(threads[0].killers));
    memset(threads[0].history, 0, sizeof(threads[0].history));
    threads[0].pos = shared->root_pos;
    cn_search_root(&threads[0]);

    /* Tell helpers to wrap up. */
    atomic_store(&shared->abort_flag, 1);
    for (int i = 1; i < n_threads; i++) {
        pthread_join(threads[i].pthread, NULL);
    }

    long total = 0;
    int best_depth = 0;
    int32_t best_score = 0;
    CNMove best_move = CN_MOVE_NULL;
    for (int i = 0; i < n_threads; i++) {
        total += threads[i].nodes;
        if (threads[i].completed_depth > best_depth
            || (threads[i].completed_depth == best_depth && i == 0)) {
            best_depth = threads[i].completed_depth;
            best_score = threads[i].best_score;
            best_move  = threads[i].best_move;
        }
    }
    if (best_move == CN_MOVE_NULL && threads[0].best_move != CN_MOVE_NULL) {
        best_move  = threads[0].best_move;
        best_score = threads[0].best_score;
    }
    shared->global_best_move  = best_move;
    shared->global_best_score = best_score;
    shared->global_total_nodes = total;
    if (out_nodes) *out_nodes = (int)total;
    return 0;
}

/* ======================================================================== *
 *                          Python-facing methods                           *
 * ======================================================================== */

/* CSearch.search_cnative_chess(board_bytes, side, castling, ep_sq, halfmove,
 *                               wk_sq, bk_sq, history_bytes,
 *                               max_depth, time_limit_ms, n_threads)
 *   -> ((from, to, promo_or_None), score, nodes) or None
 *
 * The history_bytes buffer is packed uint64 Zobrist hashes of previous
 * positions for 3-fold repetition detection. max_depth=0 or time<=0
 * means "infinite" — useful for GUI analysis. */
static PyObject *
CSearch_search_cnative_chess(CSearchObject *self, PyObject *args)
{
    PyObject *board_obj, *history_obj;
    int side, castling, ep_sq, halfmove, wk_sq, bk_sq, max_depth;
    double time_limit_ms;
    int n_threads = 1;
    if (!PyArg_ParseTuple(args, "OiiiiiiOid|i",
                          &board_obj, &side, &castling, &ep_sq, &halfmove,
                          &wk_sq, &bk_sq, &history_obj,
                          &max_depth, &time_limit_ms, &n_threads))
        return NULL;
    if (!self->accumulator || !self->py_feature_set) {
        PyErr_SetString(PyExc_RuntimeError,
                        "CSearch has no accumulator/feature_set bound");
        return NULL;
    }
    if (n_threads < 1) n_threads = 1;
    if (n_threads > CN_MAX_THREADS) n_threads = CN_MAX_THREADS;

    CNShared shared;
    memset(&shared, 0, sizeof(shared));
    shared.game_type    = CN_GAME_CHESS;
    shared.acc_weights  = self->accumulator;
    shared.eval_scale   = self->eval_scale;
    shared.tt           = (CNTTEntry *)self->tt;  /* reinterpret */
    shared.tt_mask      = self->tt_mask;
    shared.max_depth    = max_depth;
    shared.time_limit_ms = time_limit_ms;
    shared.n_threads    = n_threads;
    shared.top_n        = 4;
    pthread_mutex_init(&shared.best_mutex, NULL);

    if (cn_extract_feature_cfg(self->py_feature_set, &shared.cfg,
                                CN_GAME_CHESS) != 0) {
        pthread_mutex_destroy(&shared.best_mutex);
        return NULL;
    }
    if (cn_unpack_chess_pos(&shared.root_pos.chess, board_obj, side, castling,
                              ep_sq, halfmove, wk_sq, bk_sq) != 0) {
        pthread_mutex_destroy(&shared.best_mutex);
        PyErr_SetString(PyExc_ValueError, "invalid chess board bytes");
        return NULL;
    }

    /* History buffer */
    if (history_obj != Py_None) {
        Py_buffer hb;
        if (PyObject_GetBuffer(history_obj, &hb, PyBUF_SIMPLE) != 0) {
            pthread_mutex_destroy(&shared.best_mutex);
            return NULL;
        }
        size_t nn = (size_t)(hb.len / 8);
        if (nn > CN_HIST_RING) nn = CN_HIST_RING;
        memcpy(shared.root_history, hb.buf, nn * 8);
        shared.root_history_n = (int)nn;
        PyBuffer_Release(&hb);
    }

    /* Allocate worker threads and their per-thread accumulators. */
    CNThread *threads = (CNThread *)calloc((size_t)n_threads, sizeof(CNThread));
    if (!threads) {
        pthread_mutex_destroy(&shared.best_mutex);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < n_threads; i++) {
        threads[i].shared = &shared;
        if (cn_thread_alloc_accum(&threads[i]) != 0) {
            for (int j = 0; j < i; j++) cn_thread_free_accum(&threads[j]);
            free(threads);
            pthread_mutex_destroy(&shared.best_mutex);
            PyErr_NoMemory();
            return NULL;
        }
    }

    int total_nodes = 0;
    Py_BEGIN_ALLOW_THREADS
    cn_run_search(&shared, threads, n_threads, &total_nodes);
    Py_END_ALLOW_THREADS

    CNMove best = shared.global_best_move;
    int32_t best_score = shared.global_best_score;
    for (int i = 0; i < n_threads; i++) cn_thread_free_accum(&threads[i]);
    free(threads);
    pthread_mutex_destroy(&shared.best_mutex);

    if (best == CN_MOVE_NULL) Py_RETURN_NONE;

    ChMove ch = (ChMove)best;
    int promo_internal = chmove_promo(ch);
    int promo_base = -1;
    switch (promo_internal) {
        case CP_QUEEN:  promo_base = 4; break;
        case CP_ROOK:   promo_base = 3; break;
        case CP_BISHOP: promo_base = 2; break;
        case CP_KNIGHT: promo_base = 1; break;
        default: break;
    }
    PyObject *promo_obj;
    if (promo_base < 0) { Py_INCREF(Py_None); promo_obj = Py_None; }
    else                { promo_obj = PyLong_FromLong(promo_base); }

    PyObject *move_tuple = PyTuple_Pack(3,
        PyLong_FromLong(chmove_from(ch)),
        PyLong_FromLong(chmove_to(ch)),
        promo_obj);
    Py_DECREF(promo_obj);

    PyObject *result = PyTuple_Pack(3,
        move_tuple,
        PyFloat_FromDouble((double)best_score),
        PyLong_FromLong(total_nodes));
    Py_DECREF(move_tuple);
    return result;
}

/* CSearch.search_cnative_shogi(board_bytes, sente_hand, gote_hand, side,
 *                               history_bytes, max_depth, time_limit_ms,
 *                               n_threads)
 *   -> ((from, to, promo, drop), score, nodes) or None */
static PyObject *
CSearch_search_cnative_shogi(CSearchObject *self, PyObject *args)
{
    PyObject *board_obj, *sh_obj, *gh_obj, *history_obj;
    int side, max_depth;
    double time_limit_ms;
    int n_threads = 1;
    if (!PyArg_ParseTuple(args, "OOOiOid|i",
                          &board_obj, &sh_obj, &gh_obj, &side,
                          &history_obj, &max_depth, &time_limit_ms, &n_threads))
        return NULL;
    if (!self->accumulator || !self->py_feature_set) {
        PyErr_SetString(PyExc_RuntimeError,
                        "CSearch has no accumulator/feature_set bound");
        return NULL;
    }
    if (n_threads < 1) n_threads = 1;
    if (n_threads > CN_MAX_THREADS) n_threads = CN_MAX_THREADS;

    CNShared shared;
    memset(&shared, 0, sizeof(shared));
    shared.game_type    = CN_GAME_SHOGI;
    shared.acc_weights  = self->accumulator;
    shared.eval_scale   = self->eval_scale;
    shared.tt           = (CNTTEntry *)self->tt;
    shared.tt_mask      = self->tt_mask;
    shared.max_depth    = max_depth;
    shared.time_limit_ms = time_limit_ms;
    shared.n_threads    = n_threads;
    shared.top_n        = 4;
    pthread_mutex_init(&shared.best_mutex, NULL);

    if (cn_extract_feature_cfg(self->py_feature_set, &shared.cfg,
                                CN_GAME_SHOGI) != 0) {
        pthread_mutex_destroy(&shared.best_mutex);
        return NULL;
    }
    if (cn_unpack_shogi_pos(&shared.root_pos.shogi,
                              board_obj, sh_obj, gh_obj, side) != 0) {
        pthread_mutex_destroy(&shared.best_mutex);
        return NULL;  /* sh_pyapi_parse_position sets an error */
    }

    if (history_obj != Py_None) {
        Py_buffer hb;
        if (PyObject_GetBuffer(history_obj, &hb, PyBUF_SIMPLE) != 0) {
            pthread_mutex_destroy(&shared.best_mutex);
            return NULL;
        }
        size_t nn = (size_t)(hb.len / 8);
        if (nn > CN_HIST_RING) nn = CN_HIST_RING;
        memcpy(shared.root_history, hb.buf, nn * 8);
        shared.root_history_n = (int)nn;
        PyBuffer_Release(&hb);
    }

    CNThread *threads = (CNThread *)calloc((size_t)n_threads, sizeof(CNThread));
    if (!threads) {
        pthread_mutex_destroy(&shared.best_mutex);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < n_threads; i++) {
        threads[i].shared = &shared;
        if (cn_thread_alloc_accum(&threads[i]) != 0) {
            for (int j = 0; j < i; j++) cn_thread_free_accum(&threads[j]);
            free(threads);
            pthread_mutex_destroy(&shared.best_mutex);
            PyErr_NoMemory();
            return NULL;
        }
    }

    int total_nodes = 0;
    Py_BEGIN_ALLOW_THREADS
    cn_run_search(&shared, threads, n_threads, &total_nodes);
    Py_END_ALLOW_THREADS

    CNMove best = shared.global_best_move;
    int32_t best_score = shared.global_best_score;
    for (int i = 0; i < n_threads; i++) cn_thread_free_accum(&threads[i]);
    free(threads);
    pthread_mutex_destroy(&shared.best_mutex);

    if (best == CN_MOVE_NULL) Py_RETURN_NONE;

    SMove sm = (SMove)best;
    int from = smove_from(sm);
    int drop = smove_drop(sm);
    PyObject *from_obj = (from == SMOVE_NONE_FROM) ? (Py_INCREF(Py_None), Py_None)
                                                    : PyLong_FromLong(from);
    PyObject *to_obj   = PyLong_FromLong(smove_to(sm));
    int promo = smove_promo(sm);
    PyObject *promo_obj = (promo == 0) ? (Py_INCREF(Py_None), Py_None)
                                        : PyLong_FromLong(promo);
    PyObject *drop_obj  = (drop == SMOVE_NONE_DROP) ? (Py_INCREF(Py_None), Py_None)
                                                     : PyLong_FromLong(drop);
    PyObject *move_tuple = PyTuple_Pack(4, from_obj, to_obj, promo_obj, drop_obj);
    Py_DECREF(from_obj); Py_DECREF(to_obj);
    Py_DECREF(promo_obj); Py_DECREF(drop_obj);

    PyObject *result = PyTuple_Pack(3,
        move_tuple,
        PyFloat_FromDouble((double)best_score),
        PyLong_FromLong(total_nodes));
    Py_DECREF(move_tuple);
    return result;
}

/* ---- Live-progress wrapper: iterative deepening with callback updates -- *
 *
 * Python caller passes ``live_ref`` (a mutable list) and ``stop_event``
 * (a threading.Event with is_set()). We run the search in background on
 * a helper thread, then poll from this thread periodically. When we see
 * a new top_version we acquire the GIL, update live_ref, and check if
 * stop_event is set.
 */
static PyObject *
_cn_build_live_move_tuple(CNGameType gt, CNMove m) {
    if (gt == CN_GAME_CHESS) {
        ChMove ch = (ChMove)m;
        int promo_internal = chmove_promo(ch);
        int promo_base = -1;
        switch (promo_internal) {
            case CP_QUEEN:  promo_base = 4; break;
            case CP_ROOK:   promo_base = 3; break;
            case CP_BISHOP: promo_base = 2; break;
            case CP_KNIGHT: promo_base = 1; break;
            default: break;
        }
        PyObject *promo_obj = (promo_base < 0)
            ? (Py_INCREF(Py_None), Py_None)
            : PyLong_FromLong(promo_base);
        /* Shape: (from, to, promo, None) for uniformity with shogi. */
        PyObject *t = PyTuple_Pack(4,
            PyLong_FromLong(chmove_from(ch)),
            PyLong_FromLong(chmove_to(ch)),
            promo_obj,
            (Py_INCREF(Py_None), Py_None));
        Py_DECREF(promo_obj);
        return t;
    } else {
        SMove sm = (SMove)m;
        int from = smove_from(sm);
        int drop = smove_drop(sm);
        int promo = smove_promo(sm);
        PyObject *from_obj = (from == SMOVE_NONE_FROM) ? (Py_INCREF(Py_None), Py_None)
                                                        : PyLong_FromLong(from);
        PyObject *to_obj   = PyLong_FromLong(smove_to(sm));
        PyObject *promo_obj = (promo == 0) ? (Py_INCREF(Py_None), Py_None)
                                            : PyLong_FromLong(promo);
        PyObject *drop_obj  = (drop == SMOVE_NONE_DROP) ? (Py_INCREF(Py_None), Py_None)
                                                         : PyLong_FromLong(drop);
        PyObject *t = PyTuple_Pack(4, from_obj, to_obj, promo_obj, drop_obj);
        Py_DECREF(from_obj); Py_DECREF(to_obj);
        Py_DECREF(promo_obj); Py_DECREF(drop_obj);
        return t;
    }
}

static void
_cn_publish_live(CNShared *shared, PyObject *live_ref, PyObject *stop_event,
                  int done) {
    /* Must hold the GIL. */
    int depth = atomic_load(&shared->global_best_depth);
    int n     = shared->top_n;
    if (n > 8) n = 8;
    PyObject *top_list = PyList_New(0);
    if (!top_list) { PyErr_Clear(); return; }
    for (int i = 0; i < n; i++) {
        CNMove m = shared->top_moves[i];
        if (m == CN_MOVE_NULL) continue;
        PyObject *mt = _cn_build_live_move_tuple(shared->game_type, m);
        PyObject *sc = PyFloat_FromDouble((double)shared->top_scores[i]);
        PyObject *entry = PyTuple_Pack(2, mt, sc);
        Py_DECREF(mt); Py_DECREF(sc);
        PyList_Append(top_list, entry);
        Py_DECREF(entry);
    }
    PyObject *snap = PyTuple_Pack(4,
        PyLong_FromLong(depth),
        PyLong_FromLong(shared->max_depth > 0 ? shared->max_depth : 64),
        top_list,
        PyBool_FromLong(done));
    Py_DECREF(top_list);
    if (live_ref && live_ref != Py_None && PyList_Check(live_ref)
        && PyList_Size(live_ref) >= 1) {
        PyList_SetItem(live_ref, 0, snap);  /* steals ref */
    } else {
        Py_DECREF(snap);
    }
    if (stop_event && stop_event != Py_None) {
        PyObject *is_set = PyObject_GetAttrString(stop_event, "is_set");
        if (is_set) {
            PyObject *r = PyObject_CallNoArgs(is_set);
            Py_DECREF(is_set);
            if (r && PyObject_IsTrue(r)) {
                atomic_store(&shared->abort_flag, 1);
            }
            Py_XDECREF(r);
        }
        if (PyErr_Occurred()) PyErr_Clear();
    }
}

static PyObject *
_cn_run_live(CNShared *shared, CNThread *threads, int n_threads,
              PyObject *live_ref, PyObject *stop_event) {
    cn_init_lmr();
    shared->start_time_ms = cn_now_ms();
    atomic_store(&shared->time_up, 0);
    atomic_store(&shared->abort_flag, 0);
    atomic_store(&shared->global_best_depth, 0);
    atomic_store(&shared->top_version, 0);
    shared->global_best_move  = CN_MOVE_NULL;
    shared->global_best_score = 0;
    shared->global_total_nodes = 0;

    /* Init all thread state (same as cn_run_search) but do NOT inline the
     * primary; we want all workers in background so this function can
     * release the GIL and poll. */
    for (int i = 0; i < n_threads; i++) {
        threads[i].shared = shared;
        threads[i].tid    = i;
        threads[i].nodes  = 0;
        threads[i].completed_depth = 0;
        threads[i].best_move  = CN_MOVE_NULL;
        threads[i].best_score = 0;
        threads[i].hist_n = 0;
        memset(threads[i].killers, 0xFF, sizeof(threads[i].killers));
        memset(threads[i].history, 0, sizeof(threads[i].history));
        threads[i].pos = shared->root_pos;
    }

    /* Start all workers. */
    for (int i = 0; i < n_threads; i++) {
        if (pthread_create(&threads[i].pthread, NULL, cn_worker_entry, &threads[i]) != 0) {
            atomic_store(&shared->abort_flag, 1);
            for (int j = 0; j < i; j++) pthread_join(threads[j].pthread, NULL);
            PyErr_SetString(PyExc_RuntimeError, "pthread_create failed");
            return NULL;
        }
    }

    /* Poll loop: release GIL, sleep briefly, reacquire, publish snapshot. */
    int last_version = 0;
    while (1) {
        int any_alive = 0;
        /* We can't query pthread liveness directly. Use time_up/abort
         * as the termination signals — helper thread runs until one
         * of those fires. */

        /* Check progress: if version bumped, publish. */
        int ver = atomic_load(&shared->top_version);
        if (ver != last_version) {
            last_version = ver;
            _cn_publish_live(shared, live_ref, stop_event, 0);
        }

        /* Check if done: if time limit elapsed OR stop flag set. */
        if (atomic_load(&shared->abort_flag) || atomic_load(&shared->time_up))
            any_alive = 0;
        else
            any_alive = 1;

        if (shared->time_limit_ms > 0.0) {
            double now = cn_now_ms();
            if (now - shared->start_time_ms >= shared->time_limit_ms) {
                atomic_store(&shared->time_up, 1);
                any_alive = 0;
            }
        }

        /* Poll stop_event */
        if (stop_event && stop_event != Py_None) {
            PyObject *is_set = PyObject_GetAttrString(stop_event, "is_set");
            if (is_set) {
                PyObject *r = PyObject_CallNoArgs(is_set);
                Py_DECREF(is_set);
                if (r && PyObject_IsTrue(r)) {
                    atomic_store(&shared->abort_flag, 1);
                    any_alive = 0;
                }
                Py_XDECREF(r);
            }
            if (PyErr_Occurred()) PyErr_Clear();
        }

        if (!any_alive) break;

        /* Sleep 50ms without holding the GIL. */
        Py_BEGIN_ALLOW_THREADS
        struct timespec ts = { 0, 50 * 1000 * 1000 };
        nanosleep(&ts, NULL);
        Py_END_ALLOW_THREADS

        /* Absolute sanity: if no depth progress in 10s with no time
         * limit, something's wrong. Bail. */
        if (shared->time_limit_ms <= 0.0) {
            int best_d = atomic_load(&shared->global_best_depth);
            if (best_d >= 63) {   /* depth saturated */
                atomic_store(&shared->abort_flag, 1);
                break;
            }
        }
    }

    /* Join all workers. */
    atomic_store(&shared->abort_flag, 1);
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i].pthread, NULL);

    long total = 0;
    int best_depth = 0;
    int32_t best_score = 0;
    CNMove best_move = CN_MOVE_NULL;
    for (int i = 0; i < n_threads; i++) {
        total += threads[i].nodes;
        if (threads[i].completed_depth > best_depth) {
            best_depth = threads[i].completed_depth;
            best_score = threads[i].best_score;
            best_move  = threads[i].best_move;
        }
    }
    if (best_move == CN_MOVE_NULL) best_move = threads[0].best_move;
    shared->global_best_move  = best_move;
    shared->global_best_score = best_score;
    shared->global_total_nodes = total;

    /* Final publish with done=True. */
    _cn_publish_live(shared, live_ref, stop_event, 1);

    /* Build final result */
    if (best_move == CN_MOVE_NULL) Py_RETURN_NONE;
    PyObject *mt = _cn_build_live_move_tuple(shared->game_type, best_move);
    PyObject *result = PyTuple_Pack(3,
        mt,
        PyFloat_FromDouble((double)best_score),
        PyLong_FromLong((long)total));
    Py_DECREF(mt);
    return result;
}

static PyObject *
CSearch_search_cnative_live_chess(CSearchObject *self, PyObject *args)
{
    PyObject *board_obj, *history_obj, *live_ref, *stop_event;
    int side, castling, ep_sq, halfmove, wk_sq, bk_sq, max_depth, n_threads;
    double time_limit_ms;
    if (!PyArg_ParseTuple(args, "OiiiiiiOidiOO",
                          &board_obj, &side, &castling, &ep_sq, &halfmove,
                          &wk_sq, &bk_sq, &history_obj,
                          &max_depth, &time_limit_ms, &n_threads,
                          &live_ref, &stop_event))
        return NULL;
    if (!self->accumulator || !self->py_feature_set) {
        PyErr_SetString(PyExc_RuntimeError,
                        "CSearch has no accumulator/feature_set bound");
        return NULL;
    }
    if (n_threads < 1) n_threads = 1;
    if (n_threads > CN_MAX_THREADS) n_threads = CN_MAX_THREADS;

    CNShared shared;
    memset(&shared, 0, sizeof(shared));
    shared.game_type    = CN_GAME_CHESS;
    shared.acc_weights  = self->accumulator;
    shared.eval_scale   = self->eval_scale;
    shared.tt           = (CNTTEntry *)self->tt;
    shared.tt_mask      = self->tt_mask;
    shared.max_depth    = max_depth;
    shared.time_limit_ms = time_limit_ms;
    shared.n_threads    = n_threads;
    shared.top_n        = 4;
    pthread_mutex_init(&shared.best_mutex, NULL);

    if (cn_extract_feature_cfg(self->py_feature_set, &shared.cfg,
                                CN_GAME_CHESS) != 0) goto fail;
    if (cn_unpack_chess_pos(&shared.root_pos.chess, board_obj, side, castling,
                              ep_sq, halfmove, wk_sq, bk_sq) != 0) {
        PyErr_SetString(PyExc_ValueError, "invalid chess board bytes");
        goto fail;
    }
    if (history_obj != Py_None) {
        Py_buffer hb;
        if (PyObject_GetBuffer(history_obj, &hb, PyBUF_SIMPLE) != 0) goto fail;
        size_t nn = (size_t)(hb.len / 8);
        if (nn > CN_HIST_RING) nn = CN_HIST_RING;
        memcpy(shared.root_history, hb.buf, nn * 8);
        shared.root_history_n = (int)nn;
        PyBuffer_Release(&hb);
    }

    CNThread *threads = (CNThread *)calloc((size_t)n_threads, sizeof(CNThread));
    if (!threads) { PyErr_NoMemory(); goto fail; }
    for (int i = 0; i < n_threads; i++) {
        threads[i].shared = &shared;
        if (cn_thread_alloc_accum(&threads[i]) != 0) {
            for (int j = 0; j < i; j++) cn_thread_free_accum(&threads[j]);
            free(threads);
            PyErr_NoMemory();
            goto fail;
        }
    }

    PyObject *result = _cn_run_live(&shared, threads, n_threads, live_ref, stop_event);

    for (int i = 0; i < n_threads; i++) cn_thread_free_accum(&threads[i]);
    free(threads);
    pthread_mutex_destroy(&shared.best_mutex);
    return result;

fail:
    pthread_mutex_destroy(&shared.best_mutex);
    return NULL;
}

static PyObject *
CSearch_search_cnative_live_shogi(CSearchObject *self, PyObject *args)
{
    PyObject *board_obj, *sh_obj, *gh_obj, *history_obj, *live_ref, *stop_event;
    int side, max_depth, n_threads;
    double time_limit_ms;
    if (!PyArg_ParseTuple(args, "OOOiOidiOO",
                          &board_obj, &sh_obj, &gh_obj, &side,
                          &history_obj, &max_depth, &time_limit_ms, &n_threads,
                          &live_ref, &stop_event))
        return NULL;
    if (!self->accumulator || !self->py_feature_set) {
        PyErr_SetString(PyExc_RuntimeError,
                        "CSearch has no accumulator/feature_set bound");
        return NULL;
    }
    if (n_threads < 1) n_threads = 1;
    if (n_threads > CN_MAX_THREADS) n_threads = CN_MAX_THREADS;

    CNShared shared;
    memset(&shared, 0, sizeof(shared));
    shared.game_type    = CN_GAME_SHOGI;
    shared.acc_weights  = self->accumulator;
    shared.eval_scale   = self->eval_scale;
    shared.tt           = (CNTTEntry *)self->tt;
    shared.tt_mask      = self->tt_mask;
    shared.max_depth    = max_depth;
    shared.time_limit_ms = time_limit_ms;
    shared.n_threads    = n_threads;
    shared.top_n        = 4;
    pthread_mutex_init(&shared.best_mutex, NULL);

    if (cn_extract_feature_cfg(self->py_feature_set, &shared.cfg,
                                CN_GAME_SHOGI) != 0) goto fail;
    if (cn_unpack_shogi_pos(&shared.root_pos.shogi, board_obj, sh_obj, gh_obj, side) != 0)
        goto fail;
    if (history_obj != Py_None) {
        Py_buffer hb;
        if (PyObject_GetBuffer(history_obj, &hb, PyBUF_SIMPLE) != 0) goto fail;
        size_t nn = (size_t)(hb.len / 8);
        if (nn > CN_HIST_RING) nn = CN_HIST_RING;
        memcpy(shared.root_history, hb.buf, nn * 8);
        shared.root_history_n = (int)nn;
        PyBuffer_Release(&hb);
    }

    CNThread *threads = (CNThread *)calloc((size_t)n_threads, sizeof(CNThread));
    if (!threads) { PyErr_NoMemory(); goto fail; }
    for (int i = 0; i < n_threads; i++) {
        threads[i].shared = &shared;
        if (cn_thread_alloc_accum(&threads[i]) != 0) {
            for (int j = 0; j < i; j++) cn_thread_free_accum(&threads[j]);
            free(threads);
            PyErr_NoMemory();
            goto fail;
        }
    }

    PyObject *result = _cn_run_live(&shared, threads, n_threads, live_ref, stop_event);

    for (int i = 0; i < n_threads; i++) cn_thread_free_accum(&threads[i]);
    free(threads);
    pthread_mutex_destroy(&shared.best_mutex);
    return result;

fail:
    pthread_mutex_destroy(&shared.best_mutex);
    return NULL;
}
