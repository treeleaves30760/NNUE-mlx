/* ======================================================================== */
/* ---- Move ordering ------------------------------------------------------ */
/* ======================================================================== */

/* MVV-LVA piece value by abs(board_val) */
static inline int piece_value(int abs_board_val) {
    switch (abs_board_val) {
        case 1: return 100;
        case 2: return 300;
        case 3: return 300;
        case 4: return 500;
        case 5: return 900;
        default: return 100;
    }
}

static int cmove_score_cmp(const void *a, const void *b) {
    const CScoredMove *sa = (const CScoredMove *)a;
    const CScoredMove *sb = (const CScoredMove *)b;
    if (sb->score > sa->score) return 1;
    if (sb->score < sa->score) return -1;
    return 0;
}

/*
 * Score all moves in py_moves[] into scored[].
 * board_array: bytes buffer (borrowed, may be NULL — skips MVV-LVA).
 */
static int
csearch_score_moves(CSearchObject *self, PyObject *py_state,
                    CScoredMove *scored, PyObject **py_moves, int n_moves,
                    int depth, CTTEntry *tt_entry)
{
    /* Get board array for MVV-LVA (optional) */
    PyObject *board_obj = PyObject_GetAttr(py_state, self->str_board_array);
    const int8_t *board = NULL;
    Py_buffer board_buf;
    int has_buf = 0;
    if (board_obj && board_obj != Py_None) {
        if (PyObject_GetBuffer(board_obj, &board_buf, PyBUF_SIMPLE) == 0) {
            board = (const int8_t *)board_buf.buf;
            has_buf = 1;
        } else {
            PyErr_Clear();
        }
    } else if (board_obj) {
        /* board_obj is Py_None */
    } else {
        PyErr_Clear();
    }
    Py_XDECREF(board_obj);

    for (int i = 0; i < n_moves; i++) {
        scored[i].move  = pyobj_to_cmove(py_moves[i]);
        scored[i].score = 0;
        CMove *m = &scored[i].move;

        /* 1. TT best move: highest priority */
        if (tt_entry && !cmove_is_null(&tt_entry->best_move)) {
            CTTEntry *te = tt_entry;
            if (te->best_move.from_sq    == m->from_sq &&
                te->best_move.to_sq      == m->to_sq   &&
                te->best_move.promotion  == m->promotion &&
                te->best_move.drop_piece == m->drop_piece) {
                scored[i].score = 2000000;
                continue;
            }
        }

        /* 2. Captures: MVV-LVA using board array */
        if (has_buf && board && m->to_sq >= 0 && m->drop_piece < 0) {
            int to = m->to_sq;
            if (to >= 0 && to < (int)board_buf.len) {
                int8_t victim = board[to];
                if (victim != 0) {
                    int vval = piece_value((victim > 0) ? victim : -victim);
                    int aval = 0;
                    if (m->from_sq >= 0 && m->from_sq < (int)board_buf.len) {
                        int8_t attacker = board[m->from_sq];
                        aval = piece_value((attacker > 0) ? attacker : -attacker);
                    }
                    scored[i].score = 1000000 + vval * 10 - aval;
                    continue;
                }
            }
        }

        /* 3. Promotions */
        if (m->promotion >= 0) {
            scored[i].score = 900000;
            continue;
        }

        /* 4. Killer moves */
        if (depth >= 0 && depth < 128) {
            if (!cmove_is_null(&self->killers[depth][0]) &&
                self->killers[depth][0].from_sq    == m->from_sq &&
                self->killers[depth][0].to_sq      == m->to_sq   &&
                self->killers[depth][0].drop_piece == m->drop_piece) {
                scored[i].score = 800000;
                continue;
            }
            if (!cmove_is_null(&self->killers[depth][1]) &&
                self->killers[depth][1].from_sq    == m->from_sq &&
                self->killers[depth][1].to_sq      == m->to_sq   &&
                self->killers[depth][1].drop_piece == m->drop_piece) {
                scored[i].score = 700000;
                continue;
            }
        }

        /* 5. History score */
        if (m->from_sq >= 0 && m->to_sq >= 0 &&
            m->from_sq < self->max_sq && m->to_sq < self->max_sq) {
            scored[i].score = (int32_t)self->history[m->from_sq * self->max_sq + m->to_sq];
        }
    }

    if (has_buf) PyBuffer_Release(&board_buf);

    qsort(scored, (size_t)n_moves, sizeof(CScoredMove), cmove_score_cmp);
    return n_moves;
}
