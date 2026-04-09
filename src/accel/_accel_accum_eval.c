/* ---- evaluate ----------------------------------------------------------- */

static PyObject *
AccelAccum_evaluate(AccelAccumObject *self, PyObject *args)
{
    int side_to_move;
    if (!PyArg_ParseTuple(args, "i", &side_to_move))
        return NULL;

    int acc_size = self->accumulator_size;
    int l1 = self->l1_size;
    int l2 = self->l2_size;

    /* Stack-allocated scratch buffers -- no heap allocation. */
    float input[1024];  /* acc_size * 2, max 512*2 = 1024 */
    float l1_out[256];  /* l1_size, max 256 */
    float l2_out[256];  /* l2_size, max 256 */

    /* 1. ClippedReLU + perspective-ordered concat.
     * For int16 mode: dequantize + ClippedReLU in one pass. */
    if (self->use_int16) {
        int16_t *first  = (side_to_move == 0) ? self->white_acc_q : self->black_acc_q;
        int16_t *second = (side_to_move == 0) ? self->black_acc_q : self->white_acc_q;
        neon_dequant_clipped_relu(input, first, acc_size, self->inv_quant_scale);
        neon_dequant_clipped_relu(input + acc_size, second, acc_size, self->inv_quant_scale);
    } else {
        float *first  = (side_to_move == 0) ? self->white_acc : self->black_acc;
        float *second = (side_to_move == 0) ? self->black_acc : self->white_acc;
        neon_clipped_relu_copy(input, first, acc_size);
        neon_clipped_relu_copy(input + acc_size, second, acc_size);
    }

    /* 2. L1: y = SCReLU(W*x + b) = clamp(W*x + b, 0, 1)^2. */
    memcpy(l1_out, self->l1_bias, l1 * sizeof(float));
    sgemv(l1, acc_size * 2, 1.0f, self->l1_weight, acc_size * 2, input, 1.0f, l1_out);
    neon_screlu_inplace(l1_out, l1);

    /* 3. L2: y = SCReLU(W*x + b). */
    memcpy(l2_out, self->l2_bias, l2 * sizeof(float));
    sgemv(l2, l1, 1.0f, self->l2_weight, l1, l1_out, 1.0f, l2_out);
    neon_screlu_inplace(l2_out, l2);

    /* 4. Output: dot product + bias. */
    float result = sdot(l2, self->out_weight, l2_out) + self->out_bias;

    return PyFloat_FromDouble((double)result);
}

/* ---- from_model classmethod --------------------------------------------- */

static PyObject *
AccelAccum_from_model(PyTypeObject *type, PyObject *args)
{
    PyObject *model;
    if (!PyArg_ParseTuple(args, "O", &model))
        return NULL;

    /* Extract weights via mlx.utils.tree_flatten, convert to numpy.
     * {k: np.array(v) for k, v in tree_flatten(model.parameters())} */
    PyObject *mlx_utils = PyImport_ImportModule("mlx.utils");
    if (!mlx_utils) return NULL;
    PyObject *tree_flatten = PyObject_GetAttrString(mlx_utils, "tree_flatten");
    Py_DECREF(mlx_utils);
    if (!tree_flatten) return NULL;

    PyObject *params_method = PyObject_GetAttrString(model, "parameters");
    if (!params_method) { Py_DECREF(tree_flatten); return NULL; }
    PyObject *params = PyObject_CallNoArgs(params_method);
    Py_DECREF(params_method);
    if (!params) { Py_DECREF(tree_flatten); return NULL; }

    PyObject *flat_list = PyObject_CallOneArg(tree_flatten, params);
    Py_DECREF(params);
    Py_DECREF(tree_flatten);
    if (!flat_list) return NULL;

    /* Convert list of (key, mx.array) tuples to dict of numpy arrays. */
    PyObject *np_mod = PyImport_ImportModule("numpy");
    if (!np_mod) { Py_DECREF(flat_list); return NULL; }
    PyObject *np_array_fn = PyObject_GetAttrString(np_mod, "array");
    Py_DECREF(np_mod);

    PyObject *numpy_state = PyDict_New();
    Py_ssize_t n = PyList_Size(flat_list);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *pair = PyList_GetItem(flat_list, i);
        PyObject *key = PyTuple_GetItem(pair, 0);
        PyObject *mx_val = PyTuple_GetItem(pair, 1);
        PyObject *np_val = PyObject_CallOneArg(np_array_fn, mx_val);
        if (!np_val) {
            Py_DECREF(flat_list); Py_DECREF(np_array_fn);
            Py_DECREF(numpy_state); return NULL;
        }
        PyDict_SetItem(numpy_state, key, np_val);
        Py_DECREF(np_val);
    }
    Py_DECREF(flat_list);
    Py_DECREF(np_array_fn);

    /* Look up weight arrays. */
    PyObject *ft_w = PyDict_GetItemString(numpy_state, "feature_table.weight");
    PyObject *ft_b = PyDict_GetItemString(numpy_state, "ft_bias");
    PyObject *l1_w = PyDict_GetItemString(numpy_state, "l1.weight");
    PyObject *l1_b = PyDict_GetItemString(numpy_state, "l1.bias");
    PyObject *l2_w = PyDict_GetItemString(numpy_state, "l2.weight");
    PyObject *l2_b = PyDict_GetItemString(numpy_state, "l2.bias");
    PyObject *out_w = PyDict_GetItemString(numpy_state, "output.weight");
    PyObject *out_b = PyDict_GetItemString(numpy_state, "output.bias");

    if (!ft_w || !ft_b || !l1_w || !l1_b || !l2_w || !l2_b || !out_w || !out_b) {
        PyErr_SetString(PyExc_KeyError, "Missing expected weight keys in model state dict");
        Py_DECREF(numpy_state);
        return NULL;
    }

    /* Flatten out_weight from (1, l2_size) to (l2_size,). */
    PyObject *flatten_method = PyObject_GetAttrString(out_w, "flatten");
    PyObject *out_w_flat = PyObject_CallNoArgs(flatten_method);
    Py_DECREF(flatten_method);

    PyObject *flatten_method2 = PyObject_GetAttrString(out_b, "flatten");
    PyObject *out_b_flat = PyObject_CallNoArgs(flatten_method2);
    Py_DECREF(flatten_method2);

    /* Create the AccelAccumObject. */
    PyObject *init_args = PyTuple_Pack(8, ft_w, ft_b, l1_w, l1_b,
                                          l2_w, l2_b, out_w_flat, out_b_flat);
    Py_DECREF(out_w_flat);
    Py_DECREF(out_b_flat);

    PyObject *obj = type->tp_new(type, PyTuple_New(0), NULL);
    if (!obj) {
        Py_DECREF(init_args);
        Py_DECREF(numpy_state);
        return NULL;
    }

    if (AccelAccum_init((AccelAccumObject *)obj, init_args, NULL) < 0) {
        Py_DECREF(obj);
        Py_DECREF(init_args);
        Py_DECREF(numpy_state);
        return NULL;
    }

    Py_DECREF(init_args);
    Py_DECREF(numpy_state);
    return obj;
}

/* ---- Method table ------------------------------------------------------- */

static PyMethodDef AccelAccum_methods[] = {
    {"refresh", (PyCFunction)AccelAccum_refresh, METH_VARARGS,
     "Full recomputation from feature lists for both perspectives."},
    {"refresh_perspective", (PyCFunction)AccelAccum_refresh_perspective, METH_VARARGS,
     "Full recomputation for a single perspective (0=white, 1=black)."},
    {"update", (PyCFunction)AccelAccum_update, METH_VARARGS,
     "Incremental update: add/remove feature indices for one perspective."},
    {"push", (PyCFunction)AccelAccum_push, METH_NOARGS,
     "Save accumulator state before making a move."},
    {"pop", (PyCFunction)AccelAccum_pop, METH_NOARGS,
     "Restore accumulator state after unmaking a move."},
    {"evaluate", (PyCFunction)AccelAccum_evaluate, METH_VARARGS,
     "Forward pass through hidden layers. Returns evaluation score."},
    {"from_model", (PyCFunction)AccelAccum_from_model,
     METH_VARARGS | METH_CLASS,
     "Create an AcceleratedAccumulator from a trained NNUEModel."},
    {NULL}
};

/* ---- Type object -------------------------------------------------------- */

static PyTypeObject AccelAccumType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "src.accel._nnue_accel.AcceleratedAccumulator",
    .tp_doc = "Accelerated NNUE accumulator with NEON SIMD + Accelerate framework.",
    .tp_basicsize = sizeof(AccelAccumObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)AccelAccum_init,
    .tp_dealloc = (destructor)AccelAccum_dealloc,
    .tp_methods = AccelAccum_methods,
};
