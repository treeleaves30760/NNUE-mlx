/* ======================================================================== */
/* ---- TT helpers --------------------------------------------------------- */
/* ======================================================================== */

static CTTEntry *
csearch_tt_probe(CSearchObject *self, uint64_t key)
{
    uint32_t idx = (uint32_t)(key & self->tt_mask);
    CTTEntry *e = &self->tt[idx];
    if (e->key == key) return e;
    return NULL;
}

static void
csearch_tt_store(CSearchObject *self, uint64_t key, int depth,
                 float score, int8_t flag, CMove *best_move)
{
    uint32_t idx = (uint32_t)(key & self->tt_mask);
    CTTEntry *e = &self->tt[idx];
    /* Always-replace: simple and effective */
    e->key   = key;
    e->score = score;
    e->depth = (int16_t)depth;
    e->flag  = flag;
    if (best_move) e->best_move = *best_move;
    else           e->best_move = cmove_null();
}

/* ======================================================================== */
/* ---- Move conversion helpers ------------------------------------------- */
/* ======================================================================== */

static CMove
pyobj_to_cmove(PyObject *py_move)
{
    CMove m;
    m._pad = 0;

    PyObject *v;

    v = PyObject_GetAttrString(py_move, "from_sq");
    m.from_sq = (v && v != Py_None) ? (int16_t)PyLong_AsLong(v) : (int16_t)-1;
    Py_XDECREF(v);

    v = PyObject_GetAttrString(py_move, "to_sq");
    m.to_sq = (v && v != Py_None) ? (int16_t)PyLong_AsLong(v) : (int16_t)-1;
    Py_XDECREF(v);

    v = PyObject_GetAttrString(py_move, "promotion");
    m.promotion = (v && v != Py_None) ? (int8_t)PyLong_AsLong(v) : (int8_t)-1;
    Py_XDECREF(v);

    v = PyObject_GetAttrString(py_move, "drop_piece");
    m.drop_piece = (v && v != Py_None) ? (int8_t)PyLong_AsLong(v) : (int8_t)-1;
    Py_XDECREF(v);

    /* Clear any attribute errors from None checks */
    if (PyErr_Occurred()) PyErr_Clear();

    return m;
}

static PyObject *
cmove_to_pyobj(CSearchObject *self, const CMove *m)
{
    /* Build Move(from_sq=..., to_sq=..., promotion=..., drop_piece=...) */
    PyObject *kwargs = PyDict_New();
    if (!kwargs) return NULL;

    PyObject *v;

    if (m->from_sq >= 0) {
        v = PyLong_FromLong(m->from_sq);
        if (!v) { Py_DECREF(kwargs); return NULL; }
        PyDict_SetItemString(kwargs, "from_sq", v);
        Py_DECREF(v);
    } else {
        PyDict_SetItemString(kwargs, "from_sq", Py_None);
    }

    if (m->to_sq >= 0) {
        v = PyLong_FromLong(m->to_sq);
        if (!v) { Py_DECREF(kwargs); return NULL; }
        PyDict_SetItemString(kwargs, "to_sq", v);
        Py_DECREF(v);
    } else {
        PyDict_SetItemString(kwargs, "to_sq", Py_None);
    }

    if (m->promotion >= 0) {
        v = PyLong_FromLong(m->promotion);
        if (!v) { Py_DECREF(kwargs); return NULL; }
        PyDict_SetItemString(kwargs, "promotion", v);
        Py_DECREF(v);
    } else {
        PyDict_SetItemString(kwargs, "promotion", Py_None);
    }

    if (m->drop_piece >= 0) {
        v = PyLong_FromLong(m->drop_piece);
        if (!v) { Py_DECREF(kwargs); return NULL; }
        PyDict_SetItemString(kwargs, "drop_piece", v);
        Py_DECREF(v);
    } else {
        PyDict_SetItemString(kwargs, "drop_piece", Py_None);
    }

    PyObject *empty_args = PyTuple_New(0);
    if (!empty_args) { Py_DECREF(kwargs); return NULL; }
    PyObject *result = PyObject_Call(self->Move_class, empty_args, kwargs);
    Py_DECREF(empty_args);
    Py_DECREF(kwargs);
    return result;
}

/* ======================================================================== */
/* ---- Timing helper ------------------------------------------------------ */
/* ======================================================================== */

static double
csearch_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
