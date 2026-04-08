/* ======================================================================== */
/* ---- Quiescence search -------------------------------------------------- */
/* ======================================================================== */

static float csearch_quiescence(CSearchObject *self, PyObject *py_state,
                                 float alpha, float beta, int qdepth);

static float
csearch_quiescence(CSearchObject *self, PyObject *py_state,
                    float alpha, float beta, int qdepth)
{
    self->nodes_searched++;

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

    /* Stand-pat: static eval as lower bound */
    PyObject *py_stm = PyObject_CallMethodNoArgs(py_state, self->str_side_to_move);
    if (!py_stm) return 0.0f;
    int stm = (int)PyLong_AsLong(py_stm);
    Py_DECREF(py_stm);
    float stand_pat = _accel_evaluate_direct(self->accumulator, stm) * self->eval_scale;

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;
    if (qdepth >= MAX_QDEPTH) return alpha;

    /* Generate legal moves, filter to captures and promotions */
    PyObject *py_moves_list = PyObject_CallMethodNoArgs(py_state, self->str_legal_moves);
    if (!py_moves_list) return alpha;

    Py_ssize_t n_moves = PyList_Size(py_moves_list);
    if (n_moves == 0) { Py_DECREF(py_moves_list); return alpha; }

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
    for (int i = 0; i < nm; i++) {
        if (self->time_up) break;

        PyObject *py_move = PyList_GET_ITEM(py_moves_list, i);  /* borrowed */

        /* Filter: only captures and promotions */
        PyObject *to_obj = PyObject_GetAttrString(py_move, "to_sq");
        PyObject *promo_obj = PyObject_GetAttrString(py_move, "promotion");
        int to_sq = to_obj ? (int)PyLong_AsLong(to_obj) : -1;
        int has_promo = (promo_obj && promo_obj != Py_None);
        Py_XDECREF(to_obj);
        Py_XDECREF(promo_obj);

        int is_capture = 0;
        if (board && to_sq >= 0 && to_sq < self->max_sq) {
            is_capture = (board[to_sq] != 0);
        }
        if (!is_capture && !has_promo) continue;

        Py_INCREF(py_move);
        PyObject *undo = csearch_make_move(self, py_state, py_move);
        Py_DECREF(py_move);
        if (!undo) { PyErr_Clear(); continue; }

        float score = -csearch_quiescence(self, py_state, -beta, -alpha, qdepth + 1);

        csearch_unmake_move(self, py_state, undo);
        Py_DECREF(undo);

        if (self->time_up) break;

        if (score > alpha) {
            alpha = score;
            if (alpha >= beta) break;
        }
    }

    if (has_buf) PyBuffer_Release(&board_buf);
    Py_DECREF(py_moves_list);
    return alpha;
}
