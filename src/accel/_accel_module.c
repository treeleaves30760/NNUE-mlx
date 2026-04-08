static PyMethodDef module_methods[] = {
    {"halfkp_active_features", accel_halfkp_active_features, METH_VARARGS,
     "Compute HalfKP active feature indices from board array (C accelerated)."},
    {"halfkp_shogi_active_features", accel_halfkp_shogi_active_features, METH_VARARGS,
     "Compute HalfKPShogi active feature indices from board + hand arrays (C accelerated)."},
    {"is_square_attacked_shogi", accel_is_square_attacked_shogi, METH_VARARGS,
     "Check if a square is attacked on a 9x9 shogi board (C accelerated)."},
    {"is_square_attacked_minishogi", accel_is_square_attacked_minishogi, METH_VARARGS,
     "Check if a square is attacked on a 5x5 minishogi board (C accelerated)."},
    {NULL, NULL, 0, NULL},
};

/* ---- Module definition -------------------------------------------------- */

static struct PyModuleDef accel_module = {
    PyModuleDef_HEAD_INIT,
    "_nnue_accel",
    "Accelerated NNUE inference with NEON SIMD and Apple Accelerate.",
    -1,
    module_methods,
};

PyMODINIT_FUNC
PyInit__nnue_accel(void)
{
    if (PyType_Ready(&AccelAccumType) < 0)
        return NULL;
    if (PyType_Ready(&CSearchType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&accel_module);
    if (!m) return NULL;

    Py_INCREF(&AccelAccumType);
    if (PyModule_AddObject(m, "AcceleratedAccumulator",
                           (PyObject *)&AccelAccumType) < 0) {
        Py_DECREF(&AccelAccumType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&CSearchType);
    if (PyModule_AddObject(m, "CSearch",
                           (PyObject *)&CSearchType) < 0) {
        Py_DECREF(&CSearchType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
