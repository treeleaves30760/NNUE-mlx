/* ======================================================================== */
/* ---- Accumulator refresh via Python feature_set ------------------------- */
/* ======================================================================== */

/*
 * Ask py_feature_set for active features for both perspectives and refresh
 * the accumulator. py_state must have a board_array attribute.
 * Returns 0 on success, -1 on Python error.
 */
static int
csearch_refresh_accumulator(CSearchObject *self, PyObject *py_state)
{
    if (!self->accumulator || !self->py_feature_set) return 0;

    int indices_w[MAX_INDICES], indices_b[MAX_INDICES];
    int n_w = 0, n_b = 0;

    /* Call feature_set.active_features(state, perspective) for each perspective */
    for (int persp = 0; persp < 2; persp++) {
        PyObject *py_persp = PyLong_FromLong(persp);
        if (!py_persp) return -1;

        PyObject *feats = PyObject_CallMethodObjArgs(
            self->py_feature_set, self->str_active_features,
            py_state, py_persp, NULL);
        Py_DECREF(py_persp);
        if (!feats) return -1;

        int n = extract_indices(feats, (persp == 0 ? indices_w : indices_b), MAX_INDICES);
        Py_DECREF(feats);
        if (n < 0) return -1;
        if (persp == 0) n_w = n; else n_b = n;
    }

    if (self->accumulator->use_int16) {
        _do_refresh_i16(self->accumulator, self->accumulator->white_acc_q, indices_w, n_w);
        _do_refresh_i16(self->accumulator, self->accumulator->black_acc_q, indices_b, n_b);
    } else {
        _do_refresh(self->accumulator, self->accumulator->white_acc, indices_w, n_w);
        _do_refresh(self->accumulator, self->accumulator->black_acc, indices_b, n_b);
    }

    return 0;
}

/* ======================================================================== */
/* ---- Make / unmake with accumulator ------------------------------------- */
/* ======================================================================== */

/*
 * csearch_make_move
 * Pushes accumulator, calls state.make_move_inplace(move),
 * then refreshes accumulator from feature_set.
 * Returns the undo object (borrowed from Python), or NULL on error.
 * Caller owns the returned reference (must Py_DECREF when done).
 */
static PyObject *
csearch_make_move(CSearchObject *self, PyObject *py_state, PyObject *py_move)
{
    _accel_push_direct(self->accumulator);

    PyObject *undo = PyObject_CallMethodOneArg(
        py_state, self->str_make_move_inplace, py_move);
    if (!undo) {
        _accel_pop_direct(self->accumulator);
        return NULL;
    }

    /* Refresh accumulator for new position */
    if (csearch_refresh_accumulator(self, py_state) < 0) {
        /* Try to undo the move */
        PyObject *res = PyObject_CallMethodOneArg(
            py_state, self->str_unmake_move, undo);
        Py_XDECREF(res);
        Py_DECREF(undo);
        _accel_pop_direct(self->accumulator);
        return NULL;
    }

    return undo;
}

/*
 * csearch_unmake_move
 * Pops accumulator, calls state.unmake_move(undo).
 */
static void
csearch_unmake_move(CSearchObject *self, PyObject *py_state, PyObject *undo)
{
    _accel_pop_direct(self->accumulator);

    PyObject *res = PyObject_CallMethodOneArg(
        py_state, self->str_unmake_move, undo);
    Py_XDECREF(res);
}
