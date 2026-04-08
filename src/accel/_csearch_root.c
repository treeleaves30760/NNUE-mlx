/*
 * csearch_search_root
 * Iterative deepening with aspiration windows.
 * Returns best score, writes best move to *out_best.
 */
static float
csearch_search_root(CSearchObject *self, PyObject *py_state,
                    int max_depth, CMove *out_best)
{
    init_lmr_table();
    *out_best = cmove_null();
    float best_score = -CSEARCH_INF;

    for (int depth = 1; depth <= max_depth; depth++) {
        if (self->time_up) break;

        memset(self->killers, 0xFF, sizeof(self->killers));

        PyObject *py_moves_list = PyObject_CallMethodNoArgs(py_state, self->str_legal_moves);
        if (!py_moves_list) break;

        Py_ssize_t n_moves = PyList_Size(py_moves_list);
        if (n_moves == 0) { Py_DECREF(py_moves_list); break; }

        int nm = (int)n_moves;
        PyObject **py_moves_arr = (PyObject **)malloc((size_t)nm * sizeof(PyObject *));
        if (!py_moves_arr) { Py_DECREF(py_moves_list); break; }
        for (int i = 0; i < nm; i++)
            py_moves_arr[i] = PyList_GET_ITEM(py_moves_list, i);

        PyObject *py_hash = PyObject_CallMethodNoArgs(py_state, self->str_zobrist_hash);
        uint64_t root_key = 0;
        if (py_hash) {
            root_key = PyLong_AsUnsignedLongLong(py_hash);
            Py_DECREF(py_hash);
            if (root_key == (uint64_t)-1 && PyErr_Occurred()) { PyErr_Clear(); root_key = 0; }
        }
        CTTEntry *tt_root = csearch_tt_probe(self, root_key);

        CScoredMove *scored = (CScoredMove *)malloc((size_t)nm * sizeof(CScoredMove));
        if (!scored) { free(py_moves_arr); Py_DECREF(py_moves_list); break; }
        csearch_score_moves(self, py_state, scored, py_moves_arr, nm, 0, tt_root);
        free(py_moves_arr);

        /* Aspiration window: narrow search around previous best */
        float asp_delta = 25.0f;
        float alpha, beta;
        if (depth >= 4 && best_score > -CSEARCH_MATE_SCORE + 1000) {
            alpha = best_score - asp_delta;
            beta  = best_score + asp_delta;
        } else {
            alpha = -CSEARCH_INF;
            beta  =  CSEARCH_INF;
        }

        int asp_retries = 0;
asp_retry:;

        float iter_best = -CSEARCH_INF;
        CMove iter_move = cmove_null();
        float local_alpha = alpha;

        for (int i = 0; i < nm; i++) {
            if (self->time_up) break;

            PyObject *py_move = cmove_to_pyobj(self, &scored[i].move);
            if (!py_move) { PyErr_Clear(); continue; }

            PyObject *undo = csearch_make_move(self, py_state, py_move);
            Py_DECREF(py_move);
            if (!undo) { PyErr_Clear(); continue; }

            float score;
            if (i == 0) {
                score = -csearch_alphabeta(self, py_state, depth - 1,
                                            -beta, -local_alpha, 1, 1);
            } else {
                /* PVS: zero-width then re-search */
                score = -csearch_alphabeta(self, py_state, depth - 1,
                                            -local_alpha - 1, -local_alpha, 1, 1);
                if (!self->time_up && score > local_alpha && score < beta) {
                    score = -csearch_alphabeta(self, py_state, depth - 1,
                                                -beta, -local_alpha, 1, 1);
                }
            }

            csearch_unmake_move(self, py_state, undo);
            Py_DECREF(undo);

            if (self->time_up) break;

            if (score > iter_best) {
                iter_best = score;
                iter_move = scored[i].move;
                if (score > local_alpha) local_alpha = score;
            }
        }

        /* Aspiration window fail: widen and retry */
        if (!self->time_up && asp_retries < 2) {
            if (iter_best <= alpha || iter_best >= beta) {
                asp_delta *= 4;
                alpha = best_score - asp_delta;
                beta  = best_score + asp_delta;
                if (alpha < -CSEARCH_INF + 1000) alpha = -CSEARCH_INF;
                if (beta  >  CSEARCH_INF - 1000) beta  =  CSEARCH_INF;
                asp_retries++;
                goto asp_retry;
            }
        }

        free(scored);
        Py_DECREF(py_moves_list);

        if (!self->time_up && !cmove_is_null(&iter_move)) {
            best_score = iter_best;
            *out_best  = iter_move;
            csearch_tt_store(self, root_key, depth, best_score, TT_EXACT, out_best);
        }
    }

    return best_score;
}
