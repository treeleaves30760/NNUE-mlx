/* ======================================================================== */
/* ---- Core alpha-beta with all search enhancements ----------------------- */
/* ======================================================================== */

/* Forward declarations */
static float csearch_alphabeta(CSearchObject *self, PyObject *py_state,
                                int depth, float alpha, float beta,
                                int ply, int allow_null);

/* Helper: check if side to move is in check */
static int csearch_is_check(CSearchObject *self, PyObject *py_state) {
    PyObject *py_check = PyObject_CallMethodNoArgs(py_state, self->str_is_check);
    if (!py_check) { PyErr_Clear(); return 0; }
    int result = PyObject_IsTrue(py_check);
    Py_DECREF(py_check);
    return result;
}

/* Helper: is this move a capture? (uses board array) */
static int csearch_is_capture(const int8_t *board, int to_sq, int max_sq) {
    if (!board || to_sq < 0 || to_sq >= max_sq) return 0;
    return board[to_sq] != 0;
}

static float
csearch_alphabeta(CSearchObject *self, PyObject *py_state,
                  int depth, float alpha, float beta,
                  int ply, int allow_null)
{
    self->nodes_searched++;

    /* Time check every 4096 nodes */
    if ((self->nodes_searched & 4095) == 0) {
        double now = csearch_now_ms();
        if (now - self->start_time >= (double)self->time_limit_ms)
            self->time_up = 1;
    }
    if (self->time_up) return 0.0f;

    /* Terminal check */
    PyObject *py_terminal = PyObject_CallMethodNoArgs(py_state, self->str_is_terminal);
    if (!py_terminal) return 0.0f;
    int is_terminal = PyObject_IsTrue(py_terminal);
    Py_DECREF(py_terminal);

    if (is_terminal) {
        PyObject *py_res = PyObject_CallMethodNoArgs(py_state, self->str_result);
        if (!py_res) return 0.0f;
        float score = (float)PyFloat_AsDouble(py_res) * CSEARCH_MATE_SCORE;
        Py_DECREF(py_res);
        return score;
    }

    /* Check extension: search one ply deeper when in check */
    int in_check = csearch_is_check(self, py_state);
    if (in_check) depth += 1;

    /* Leaf node: quiescence search */
    if (depth <= 0) {
        return csearch_quiescence(self, py_state, alpha, beta, 0);
    }

    /* TT probe */
    PyObject *py_hash = PyObject_CallMethodNoArgs(py_state, self->str_zobrist_hash);
    if (!py_hash) return 0.0f;
    uint64_t key = PyLong_AsUnsignedLongLong(py_hash);
    Py_DECREF(py_hash);
    if (key == (uint64_t)-1 && PyErr_Occurred()) { PyErr_Clear(); key = 0; }

    CTTEntry *tt_hit = csearch_tt_probe(self, key);
    if (tt_hit && tt_hit->depth >= depth) {
        if (tt_hit->flag == TT_EXACT) return tt_hit->score;
        if (tt_hit->flag == TT_ALPHA && tt_hit->score <= alpha) return alpha;
        if (tt_hit->flag == TT_BETA  && tt_hit->score >= beta)  return beta;
    }

    /* Static eval for pruning decisions */
    PyObject *py_stm = PyObject_CallMethodNoArgs(py_state, self->str_side_to_move);
    if (!py_stm) return 0.0f;
    int stm = (int)PyLong_AsLong(py_stm);
    Py_DECREF(py_stm);
    float static_eval = _accel_evaluate_direct(self->accumulator, stm) * self->eval_scale;

    /* ---- Null Move Pruning ---- */
    if (allow_null && depth > 2 && !in_check && static_eval >= beta) {
        /* Try passing: if opponent can't beat beta even with a free move, prune */
        int R = 2 + (depth > 6 ? 1 : 0);
        PyObject *py_null = PyObject_CallMethodNoArgs(py_state, self->str_make_null_move);
        if (py_null && py_null != Py_None) {
            /* Null move: just push/pop accumulator, eval new state */
            _accel_push_direct(self->accumulator);
            /* Refresh accumulator for null-move state */
            PyObject *fs = self->py_feature_set;
            for (int persp = 0; persp < 2; persp++) {
                PyObject *feats = PyObject_CallMethod(fs, "active_features", "Oi",
                                                       py_null, persp);
                if (feats) {
                    int nf = 0;
                    int feat_idx[64];
                    Py_ssize_t flen = PyList_Size(feats);
                    for (Py_ssize_t fi = 0; fi < flen && nf < 64; fi++) {
                        feat_idx[nf++] = (int)PyLong_AsLong(PyList_GET_ITEM(feats, fi));
                    }
                    _accel_refresh_perspective_direct(self->accumulator, persp, feat_idx, nf);
                    Py_DECREF(feats);
                }
            }

            float null_score = -csearch_alphabeta(self, py_null,
                                                   depth - 1 - R, -beta, -beta + 1,
                                                   ply + 1, 0);
            _accel_pop_direct(self->accumulator);
            Py_DECREF(py_null);

            if (!self->time_up && null_score >= beta) {
                return beta;  /* Null move cutoff */
            }
        } else {
            Py_XDECREF(py_null);
            PyErr_Clear();
        }
    }

    /* ---- Futility Pruning decision ---- */
    int futility_ok = 0;
    if (depth <= 2 && !in_check) {
        float margin = FUTILITY_MARGINS[depth];
        if (static_eval + margin <= alpha) {
            futility_ok = 1;
        }
    }

    /* Generate legal moves */
    PyObject *py_moves_list = PyObject_CallMethodNoArgs(py_state, self->str_legal_moves);
    if (!py_moves_list) return 0.0f;

    Py_ssize_t n_moves = PyList_Size(py_moves_list);
    if (n_moves == 0) {
        Py_DECREF(py_moves_list);
        return 0.0f;
    }

    /* Get board for capture detection */
    PyObject *board_obj = PyObject_GetAttr(py_state, self->str_board_array);
    const int8_t *board = NULL;
    Py_buffer board_buf;
    int has_buf = 0;
    if (board_obj) {
        if (PyObject_GetBuffer(board_obj, &board_buf, PyBUF_SIMPLE) == 0) {
            board = (const int8_t *)board_buf.buf;
            has_buf = 1;
        }
        Py_DECREF(board_obj);
    }

    int nm = (int)n_moves;
    PyObject **py_moves_arr = (PyObject **)malloc((size_t)nm * sizeof(PyObject *));
    if (!py_moves_arr) {
        if (has_buf) PyBuffer_Release(&board_buf);
        Py_DECREF(py_moves_list);
        PyErr_NoMemory();
        return 0.0f;
    }
    for (int i = 0; i < nm; i++)
        py_moves_arr[i] = PyList_GET_ITEM(py_moves_list, i);

    CScoredMove *scored = (CScoredMove *)malloc((size_t)nm * sizeof(CScoredMove));
    if (!scored) {
        free(py_moves_arr);
        if (has_buf) PyBuffer_Release(&board_buf);
        Py_DECREF(py_moves_list);
        PyErr_NoMemory();
        return 0.0f;
    }
    csearch_score_moves(self, py_state, scored, py_moves_arr, nm, ply, tt_hit);
    free(py_moves_arr);

    float orig_alpha = alpha;
    float best_score = -CSEARCH_INF;
    CMove best_move  = cmove_null();

    for (int i = 0; i < nm; i++) {
        if (self->time_up) break;

        CMove *mv = &scored[i].move;
        int is_cap = csearch_is_capture(board, mv->to_sq, self->max_sq);
        int is_promo = (mv->promotion >= 0);
        int is_quiet = (!is_cap && !is_promo);

        /* ---- Futility Pruning: skip quiet moves in futile positions ---- */
        if (futility_ok && i > 0 && is_quiet) {
            continue;
        }

        PyObject *py_move = cmove_to_pyobj(self, mv);
        if (!py_move) { PyErr_Clear(); continue; }

        PyObject *undo = csearch_make_move(self, py_state, py_move);
        Py_DECREF(py_move);
        if (!undo) { PyErr_Clear(); continue; }

        float score;

        if (i == 0) {
            /* First move: full window (PV move) */
            score = -csearch_alphabeta(self, py_state, depth - 1,
                                        -beta, -alpha, ply + 1, 1);
        } else {
            /* ---- Late Move Reduction ---- */
            int reduction = 0;
            if (i >= 3 && depth >= 3 && !in_check && is_quiet) {
                int di = depth < LMR_MAX_DEPTH ? depth : LMR_MAX_DEPTH - 1;
                int mi = i < LMR_MAX_MOVES ? i : LMR_MAX_MOVES - 1;
                reduction = lmr_table[di][mi];
            }

            /* PVS: zero-width search with possible LMR */
            score = -csearch_alphabeta(self, py_state,
                                        depth - 1 - reduction,
                                        -alpha - 1, -alpha, ply + 1, 1);

            /* Re-search at full depth if LMR reduced and score raises alpha */
            if (!self->time_up && reduction > 0 && score > alpha) {
                score = -csearch_alphabeta(self, py_state, depth - 1,
                                            -alpha - 1, -alpha, ply + 1, 1);
            }
            /* Re-search with full window if score is in (alpha, beta) */
            if (!self->time_up && score > alpha && score < beta) {
                score = -csearch_alphabeta(self, py_state, depth - 1,
                                            -beta, -alpha, ply + 1, 1);
            }
        }

        csearch_unmake_move(self, py_state, undo);
        Py_DECREF(undo);

        if (self->time_up) break;

        if (score > best_score) {
            best_score = score;
            best_move  = *mv;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            /* Beta cutoff: update killers and history for quiet moves */
            if (is_quiet && ply >= 0 && ply < 128) {
                if (!(self->killers[ply][0].from_sq == mv->from_sq &&
                      self->killers[ply][0].to_sq   == mv->to_sq)) {
                    self->killers[ply][1] = self->killers[ply][0];
                    self->killers[ply][0] = *mv;
                }
            }
            if (mv->from_sq >= 0 && mv->to_sq >= 0 &&
                mv->from_sq < self->max_sq && mv->to_sq < self->max_sq) {
                self->history[mv->from_sq * self->max_sq + mv->to_sq] += depth * depth;
            }
            break;
        }
    }

    free(scored);
    if (has_buf) PyBuffer_Release(&board_buf);
    Py_DECREF(py_moves_list);

    if (!self->time_up) {
        int8_t flag;
        if (best_score <= orig_alpha)     flag = TT_ALPHA;
        else if (best_score >= beta)      flag = TT_BETA;
        else                              flag = TT_EXACT;
        csearch_tt_store(self, key, depth, best_score, flag, &best_move);
    }

    return best_score;
}
