/* ======================================================================== */
/* ---- CSearch data structures ------------------------------------------- */
/* ======================================================================== */

#define CSEARCH_INF         1000000.0f
#define CSEARCH_MATE_SCORE   100000.0f
#define TT_EXACT  0
#define TT_ALPHA  1
#define TT_BETA   2

typedef struct {
    int16_t from_sq;    /* -1 for drops */
    int16_t to_sq;
    int8_t  promotion;  /* -1 for none */
    int8_t  drop_piece; /* -1 for none */
    int16_t _pad;
} CMove;

typedef struct {
    uint64_t key;
    float    score;
    CMove    best_move;
    int16_t  depth;
    int8_t   flag;      /* EXACT=0, ALPHA=1, BETA=2 */
    int8_t   _pad;
} CTTEntry;

typedef struct {
    CMove    move;
    int32_t  score;
} CScoredMove;

/* Sentinel / null move */
static inline CMove cmove_null(void) {
    CMove m;
    m.from_sq = -1; m.to_sq = -1;
    m.promotion = -1; m.drop_piece = -1;
    m._pad = 0;
    return m;
}

static inline int cmove_is_null(const CMove *m) {
    return (m->from_sq == -1 && m->to_sq == -1);
}

/* Forward declaration for the C-native TT entry, defined in
 * _csearch_cnative.c (included later in the same translation unit). */
struct CNTTEntry;

typedef struct {
    PyObject_HEAD

    /* Transposition table */
    CTTEntry *tt;
    uint32_t  tt_size;
    uint32_t  tt_mask;

    /* Killer moves: indexed [ply][0|1] */
    CMove     killers[128][2];

    /* History heuristic: indexed [from_sq * max_sq + to_sq] */
    int32_t  *history;   /* heap-allocated, size max_sq*max_sq */
    int       max_sq;    /* 64 for chess, 81 for shogi */

    /* Direct C references (borrowed during search) */
    AccelAccumObject *accumulator;
    PyObject         *py_feature_set;

    /* Eval scaling */
    float eval_scale;

    /* Search state */
    int   max_depth;
    float time_limit_ms;
    int   nodes_searched;
    int   time_up;
    double start_time;   /* seconds, from clock_gettime */

    /* Interned attribute/method name strings */
    PyObject *str_legal_moves;
    PyObject *str_make_move_inplace;
    PyObject *str_unmake_move;
    PyObject *str_make_null_move;
    PyObject *str_is_terminal;
    PyObject *str_result;
    PyObject *str_side_to_move;
    PyObject *str_zobrist_hash;
    PyObject *str_is_check;
    PyObject *str_board_array;
    PyObject *str_active_features;

    /* Cached Move class */
    PyObject *Move_class;

    /* C-native (Lazy SMP) transposition table — allocated lazily on the
     * first call to search_cnative_*. Uses a different entry layout
     * (atomic with Stockfish XOR trick) than the legacy `tt` field, so
     * they can't share storage. */
    struct CNTTEntry *cn_tt;
    uint32_t          cn_tt_size;
    uint32_t          cn_tt_mask;

} CSearchObject;

/* ---- LMR table ---------------------------------------------------------- */
#define LMR_MAX_DEPTH 64
#define LMR_MAX_MOVES 64
static int lmr_table[LMR_MAX_DEPTH][LMR_MAX_MOVES];
static int lmr_initialized = 0;

static void init_lmr_table(void) {
    if (lmr_initialized) return;
    for (int d = 0; d < LMR_MAX_DEPTH; d++) {
        for (int m = 0; m < LMR_MAX_MOVES; m++) {
            if (d == 0 || m == 0) { lmr_table[d][m] = 0; continue; }
            lmr_table[d][m] = (int)(1.0 + log((double)d) * log((double)m) / 2.0);
            if (lmr_table[d][m] < 0) lmr_table[d][m] = 0;
        }
    }
    lmr_initialized = 1;
}

/* Futility margins by depth */
static const float FUTILITY_MARGINS[] = {0.0f, 200.0f, 500.0f};
#define MAX_QDEPTH 8
