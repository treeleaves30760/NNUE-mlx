/* ======================================================================== */
/* ---- CSearch Python-exposed methods ------------------------------------ */
/* ======================================================================== */

/*
 * CSearch.clear_tt()
 */
static PyObject *
CSearch_clear_tt(CSearchObject *self, PyObject *Py_UNUSED(args))
{
    memset(self->tt, 0, self->tt_size * sizeof(CTTEntry));
    Py_RETURN_NONE;
}

/*
 * CSearch.search(game_state, max_depth, time_limit_ms)
 * Returns ((from_sq, to_sq, promotion, drop_piece), score, nodes) or None.
 */
static PyObject *
CSearch_search(CSearchObject *self, PyObject *args)
{
    PyObject *py_state;
    int       max_depth;
    float     time_limit_ms = 1000.0f;

    if (!PyArg_ParseTuple(args, "Oi|f", &py_state, &max_depth, &time_limit_ms))
        return NULL;

    /* Reset search state */
    memset(self->killers, 0xFF, sizeof(self->killers));
    memset(self->history, 0, (size_t)self->max_sq * self->max_sq * sizeof(int32_t));
    self->nodes_searched = 0;
    self->time_up        = 0;
    self->time_limit_ms  = time_limit_ms;
    self->start_time     = csearch_now_ms();

    /* Refresh accumulator for the root position */
    if (csearch_refresh_accumulator(self, py_state) < 0) {
        PyErr_Clear();
        /* Non-fatal: search without a fresh accumulator */
    }

    CMove best_move = cmove_null();
    float best_score = csearch_search_root(self, py_state, max_depth, &best_move);

    if (cmove_is_null(&best_move)) {
        Py_RETURN_NONE;
    }

    /* Build result tuple: ((from,to,promo,drop), score, nodes) */
    PyObject *move_tuple = PyTuple_New(4);
    if (!move_tuple) return NULL;

    PyTuple_SET_ITEM(move_tuple, 0,
        (best_move.from_sq  >= 0) ? PyLong_FromLong(best_move.from_sq)  : (Py_INCREF(Py_None), Py_None));
    PyTuple_SET_ITEM(move_tuple, 1,
        (best_move.to_sq    >= 0) ? PyLong_FromLong(best_move.to_sq)    : (Py_INCREF(Py_None), Py_None));
    PyTuple_SET_ITEM(move_tuple, 2,
        (best_move.promotion >= 0) ? PyLong_FromLong(best_move.promotion) : (Py_INCREF(Py_None), Py_None));
    PyTuple_SET_ITEM(move_tuple, 3,
        (best_move.drop_piece >= 0) ? PyLong_FromLong(best_move.drop_piece) : (Py_INCREF(Py_None), Py_None));

    PyObject *result = PyTuple_Pack(3,
        move_tuple,
        PyFloat_FromDouble((double)best_score),
        PyLong_FromLong(self->nodes_searched));
    Py_DECREF(move_tuple);
    return result;
}

/*
 * CSearch.search_top_n(game_state, n, max_depth, time_limit_ms)
 * Returns list of ((from_sq, to_sq, promotion, drop_piece), score).
 */
static PyObject *
CSearch_search_top_n(CSearchObject *self, PyObject *args)
{
    PyObject *py_state;
    int       n;
    int       max_depth;
    float     time_limit_ms = 1000.0f;

    if (!PyArg_ParseTuple(args, "Oii|f", &py_state, &n, &max_depth, &time_limit_ms))
        return NULL;

    /* Reset search state */
    memset(self->killers, 0xFF, sizeof(self->killers));
    memset(self->history, 0, (size_t)self->max_sq * self->max_sq * sizeof(int32_t));
    self->nodes_searched = 0;
    self->time_up        = 0;
    self->time_limit_ms  = time_limit_ms;
    self->start_time     = csearch_now_ms();

    if (csearch_refresh_accumulator(self, py_state) < 0)
        PyErr_Clear();

    /* Get root legal moves */
    PyObject *py_moves_list = PyObject_CallMethodNoArgs(py_state, self->str_legal_moves);
    if (!py_moves_list) return NULL;

    Py_ssize_t n_moves = PyList_Size(py_moves_list);
    if (n_moves == 0) {
        Py_DECREF(py_moves_list);
        return PyList_New(0);
    }

    int nm = (int)n_moves;
    if (n > nm) n = nm;

    /* Get root key for TT probe */
    PyObject *py_hash = PyObject_CallMethodNoArgs(py_state, self->str_zobrist_hash);
    uint64_t root_key = 0;
    if (py_hash) {
        root_key = PyLong_AsUnsignedLongLong(py_hash);
        Py_DECREF(py_hash);
        if (root_key == (uint64_t)-1 && PyErr_Occurred()) { PyErr_Clear(); root_key = 0; }
    }
    CTTEntry *tt_root = csearch_tt_probe(self, root_key);

    PyObject **py_moves_arr = (PyObject **)malloc((size_t)nm * sizeof(PyObject *));
    CScoredMove *scored     = (CScoredMove *)malloc((size_t)nm * sizeof(CScoredMove));
    if (!py_moves_arr || !scored) {
        free(py_moves_arr); free(scored);
        Py_DECREF(py_moves_list);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < nm; i++)
        py_moves_arr[i] = PyList_GET_ITEM(py_moves_list, i);

    /* Score for ordering (just use one depth pass) */
    csearch_score_moves(self, py_state, scored, py_moves_arr, nm, 0, tt_root);
    free(py_moves_arr);

    /* Evaluate each of the top n moves at max_depth */
    PyObject *result_list = PyList_New(0);
    if (!result_list) {
        free(scored);
        Py_DECREF(py_moves_list);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        if (self->time_up) break;

        PyObject *py_move = cmove_to_pyobj(self, &scored[i].move);
        if (!py_move) { PyErr_Clear(); continue; }

        PyObject *undo = csearch_make_move(self, py_state, py_move);
        Py_DECREF(py_move);
        if (!undo) { PyErr_Clear(); continue; }

        float score = -csearch_alphabeta(self, py_state, max_depth - 1,
                                         -CSEARCH_INF, CSEARCH_INF, 1, 1);
        csearch_unmake_move(self, py_state, undo);
        Py_DECREF(undo);

        if (self->time_up) break;

        PyObject *move_tuple = PyTuple_New(4);
        if (!move_tuple) break;

        CMove *m = &scored[i].move;
        PyTuple_SET_ITEM(move_tuple, 0,
            (m->from_sq   >= 0) ? PyLong_FromLong(m->from_sq)   : (Py_INCREF(Py_None), Py_None));
        PyTuple_SET_ITEM(move_tuple, 1,
            (m->to_sq     >= 0) ? PyLong_FromLong(m->to_sq)     : (Py_INCREF(Py_None), Py_None));
        PyTuple_SET_ITEM(move_tuple, 2,
            (m->promotion  >= 0) ? PyLong_FromLong(m->promotion)  : (Py_INCREF(Py_None), Py_None));
        PyTuple_SET_ITEM(move_tuple, 3,
            (m->drop_piece >= 0) ? PyLong_FromLong(m->drop_piece) : (Py_INCREF(Py_None), Py_None));

        PyObject *entry = PyTuple_Pack(2, move_tuple, PyFloat_FromDouble((double)score));
        Py_DECREF(move_tuple);
        if (!entry) break;
        PyList_Append(result_list, entry);
        Py_DECREF(entry);
    }

    free(scored);
    Py_DECREF(py_moves_list);
    return result_list;
}

/* ---- CSearch method table ---------------------------------------------- */

/* Forward declarations for the C-native (multi-threaded Lazy SMP) search
 * methods defined in _csearch_cnative.c — same translation unit so no
 * linker bridge is needed. */
static PyObject *CSearch_search_cnative_chess(CSearchObject *self, PyObject *args);
static PyObject *CSearch_search_cnative_shogi(CSearchObject *self, PyObject *args);
static PyObject *CSearch_search_cnative_live_chess(CSearchObject *self, PyObject *args);
static PyObject *CSearch_search_cnative_live_shogi(CSearchObject *self, PyObject *args);

static PyMethodDef CSearch_methods[] = {
    {"search", (PyCFunction)CSearch_search, METH_VARARGS,
     "search(game_state, max_depth, time_limit_ms=1000.0) -> ((from_sq, to_sq, promo, drop), score, nodes) or None"},
    {"search_top_n", (PyCFunction)CSearch_search_top_n, METH_VARARGS,
     "search_top_n(game_state, n, max_depth, time_limit_ms=1000.0) -> list of ((from_sq, to_sq, promo, drop), score)"},
    {"clear_tt", (PyCFunction)CSearch_clear_tt, METH_NOARGS,
     "Clear the transposition table."},
    {"search_cnative_chess", (PyCFunction)CSearch_search_cnative_chess, METH_VARARGS,
     "search_cnative_chess(board_bytes, side, castling, ep_sq, halfmove, "
     "wk_sq, bk_sq, history_bytes, max_depth, time_limit_ms, n_threads=1) "
     "-> ((from, to, promo_or_None), score, nodes) or None. "
     "C-native multi-threaded (Lazy SMP) NNUE search. max_depth=0 or "
     "time_limit_ms=0 means infinite; caller must use stop_event."},
    {"search_cnative_shogi", (PyCFunction)CSearch_search_cnative_shogi, METH_VARARGS,
     "search_cnative_shogi(board_bytes, sente_hand, gote_hand, side, "
     "history_bytes, max_depth, time_limit_ms, n_threads=1) "
     "-> ((from, to, promo, drop), score, nodes) or None."},
    {"search_cnative_live_chess", (PyCFunction)CSearch_search_cnative_live_chess, METH_VARARGS,
     "search_cnative_live_chess(board_bytes, side, castling, ep_sq, halfmove, "
     "wk_sq, bk_sq, history_bytes, max_depth, time_limit_ms, n_threads, "
     "live_ref, stop_event) -> final result tuple or None. "
     "Publishes (depth, max_depth, top_moves, done) to live_ref[0] and "
     "polls stop_event.is_set() to abort."},
    {"search_cnative_live_shogi", (PyCFunction)CSearch_search_cnative_live_shogi, METH_VARARGS,
     "search_cnative_live_shogi(board_bytes, sente_hand, gote_hand, side, "
     "history_bytes, max_depth, time_limit_ms, n_threads, "
     "live_ref, stop_event) -> final result tuple or None."},
    {NULL}
};

/* ---- CSearch type object ------------------------------------------------ */

static PyTypeObject CSearchType = {
    .ob_base   = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name   = "src.accel._nnue_accel.CSearch",
    .tp_doc    = "C-native alpha-beta search with TT, killers, history and NNUE evaluation.",
    .tp_basicsize = sizeof(CSearchObject),
    .tp_itemsize  = 0,
    .tp_flags  = Py_TPFLAGS_DEFAULT,
    .tp_new    = PyType_GenericNew,
    .tp_init   = (initproc)CSearch_init,
    .tp_dealloc = (destructor)CSearch_dealloc,
    .tp_methods = CSearch_methods,
};
