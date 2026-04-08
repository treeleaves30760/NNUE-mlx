/* ======================================================================== */
/* ---- CSearchObject lifecycle ------------------------------------------- */
/* ======================================================================== */

static void
CSearch_dealloc(CSearchObject *self)
{
    free(self->tt);
    free(self->history);
    Py_XDECREF(self->str_legal_moves);
    Py_XDECREF(self->str_make_move_inplace);
    Py_XDECREF(self->str_unmake_move);
    Py_XDECREF(self->str_is_terminal);
    Py_XDECREF(self->str_make_null_move);
    Py_XDECREF(self->str_result);
    Py_XDECREF(self->str_side_to_move);
    Py_XDECREF(self->str_zobrist_hash);
    Py_XDECREF(self->str_is_check);
    Py_XDECREF(self->str_board_array);
    Py_XDECREF(self->str_active_features);
    Py_XDECREF(self->Move_class);
    /* accumulator and py_feature_set are borrowed refs: do not DECREF */
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
CSearch_init(CSearchObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {
        "accumulator", "feature_set",
        "tt_size", "eval_scale", "max_sq",
        NULL
    };

    PyObject *py_accumulator = NULL;
    PyObject *py_feature_set = NULL;
    uint32_t  tt_size        = (uint32_t)(1 << 20);
    float     eval_scale     = 128.0f;
    int       max_sq         = 64;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|Ifi", kwlist,
                                     &py_accumulator, &py_feature_set,
                                     &tt_size, &eval_scale, &max_sq))
        return -1;

    if (!PyObject_TypeCheck(py_accumulator, &AccelAccumType)) {
        PyErr_SetString(PyExc_TypeError,
                        "accumulator must be an AcceleratedAccumulator instance");
        return -1;
    }

    /* Round tt_size down to power of two */
    if (tt_size == 0) tt_size = 1;
    uint32_t pot = 1;
    while (pot * 2 <= tt_size) pot *= 2;
    tt_size = pot;

    /* Allocate TT */
    self->tt = (CTTEntry *)calloc(tt_size, sizeof(CTTEntry));
    if (!self->tt) {
        PyErr_NoMemory();
        return -1;
    }
    self->tt_size = tt_size;
    self->tt_mask = tt_size - 1;

    /* Allocate history table */
    self->history = (int32_t *)calloc((size_t)max_sq * max_sq, sizeof(int32_t));
    if (!self->history) {
        free(self->tt); self->tt = NULL;
        PyErr_NoMemory();
        return -1;
    }
    self->max_sq = max_sq;

    /* Zero killers */
    memset(self->killers, 0xFF, sizeof(self->killers));

    /* Store borrowed references */
    self->accumulator    = (AccelAccumObject *)py_accumulator;
    self->py_feature_set = py_feature_set;
    self->eval_scale     = eval_scale;

    /* Search state */
    self->max_depth     = 0;
    self->time_limit_ms = 1000.0f;
    self->nodes_searched = 0;
    self->time_up       = 0;
    self->start_time    = 0.0;

    /* Intern method name strings */
#define INTERN(field, name) \
    self->field = PyUnicode_InternFromString(name); \
    if (!self->field) return -1;

    INTERN(str_legal_moves,        "legal_moves")
    INTERN(str_make_move_inplace,  "make_move_inplace")
    INTERN(str_unmake_move,        "unmake_move")
    INTERN(str_is_terminal,        "is_terminal")
    INTERN(str_result,             "result")
    INTERN(str_side_to_move,       "side_to_move")
    INTERN(str_zobrist_hash,       "zobrist_hash")
    INTERN(str_make_null_move,     "make_null_move")
    INTERN(str_is_check,           "is_check")
    INTERN(str_board_array,        "board_array")
    INTERN(str_active_features,    "active_features")
#undef INTERN

    /* Import Move class */
    PyObject *base_mod = PyImport_ImportModule("src.games.base");
    if (!base_mod) {
        /* Try alternate path if package not installed as src.games.base */
        PyErr_Clear();
        base_mod = PyImport_ImportModule("games.base");
    }
    if (base_mod) {
        self->Move_class = PyObject_GetAttrString(base_mod, "Move");
        Py_DECREF(base_mod);
        if (!self->Move_class) {
            PyErr_Clear();
            self->Move_class = NULL;
        }
    } else {
        PyErr_Clear();
        self->Move_class = NULL;
    }

    /* If Move_class is unavailable we will fall back to tuples in cmove_to_pyobj */

    return 0;
}
