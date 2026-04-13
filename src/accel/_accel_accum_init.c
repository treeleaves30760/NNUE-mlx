/* ---- Type methods ------------------------------------------------------- */

static void
AccelAccum_dealloc(AccelAccumObject *self)
{
    free(self->ft_weight);
    free(self->ft_bias);
    free(self->l1_weight);
    free(self->l1_bias);
    free(self->l2_weight);
    free(self->l2_bias);
    free(self->out_weight);
    free(self->out_bias);
    free(self->white_acc);
    free(self->black_acc);
    free(self->stack_buf);
    free(self->ft_weight_q);
    free(self->ft_bias_q);
    free(self->white_acc_q);
    free(self->black_acc_q);
    free(self->stack_buf_q);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
AccelAccum_init(AccelAccumObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {
        "ft_weight", "ft_bias",
        "l1_weight", "l1_bias",
        "l2_weight", "l2_bias",
        "out_weight", "out_bias",
        "quant_scale",
        NULL
    };

    PyObject *py_ft_weight, *py_ft_bias;
    PyObject *py_l1_weight, *py_l1_bias;
    PyObject *py_l2_weight, *py_l2_bias;
    PyObject *py_out_weight, *py_out_bias;
    float quant_scale = 0.0f;  /* 0 = auto-detect from dtype */

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOO|f", kwlist,
            &py_ft_weight, &py_ft_bias,
            &py_l1_weight, &py_l1_bias,
            &py_l2_weight, &py_l2_bias,
            &py_out_weight, &py_out_bias,
            &quant_scale))
        return -1;

    /* Determine dimensions from ft_weight shape.
     * ft_weight is (num_features, accumulator_size). */
    PyObject *shape = PyObject_GetAttrString(py_ft_weight, "shape");
    if (!shape) return -1;
    PyObject *dim0 = PyTuple_GetItem(shape, 0);
    PyObject *dim1 = PyTuple_GetItem(shape, 1);
    self->num_features = (int)PyLong_AsLong(dim0);
    self->accumulator_size = (int)PyLong_AsLong(dim1);
    Py_DECREF(shape);

    /* l1_weight is (l1_size, accumulator_size * 2). */
    shape = PyObject_GetAttrString(py_l1_weight, "shape");
    if (!shape) return -1;
    self->l1_size = (int)PyLong_AsLong(PyTuple_GetItem(shape, 0));
    Py_DECREF(shape);

    /* l2_weight is (l2_size, l1_size). */
    shape = PyObject_GetAttrString(py_l2_weight, "shape");
    if (!shape) return -1;
    self->l2_size = (int)PyLong_AsLong(PyTuple_GetItem(shape, 0));
    Py_DECREF(shape);

    int acc = self->accumulator_size;
    int nf  = self->num_features;
    int l1  = self->l1_size;
    int l2  = self->l2_size;

    /* Detect int16 quantized mode */
    int is_int16 = is_numpy_int16(py_ft_weight);
    self->use_int16 = is_int16;

    /* Parse num_buckets from out_weight shape (ndim==1 -> legacy, ndim==2 -> bucketed). */
    int num_buckets = 1;
    {
        PyObject *ow_shape = PyObject_GetAttrString(py_out_weight, "shape");
        if (!ow_shape) return -1;
        Py_ssize_t ow_ndim = PyTuple_Size(ow_shape);
        if (ow_ndim == 2) {
            num_buckets = (int)PyLong_AsLong(PyTuple_GetItem(ow_shape, 0));
        }
        Py_DECREF(ow_shape);
        if (num_buckets <= 0) num_buckets = 1;
    }

    /* Defensive dtype check: l1_weight, l2_weight, out_weight must be float32. */
    if (is_numpy_int16(py_l1_weight) || is_numpy_int16(py_l2_weight) ||
        is_numpy_int16(py_out_weight)) {
        PyErr_SetString(PyExc_TypeError,
            "l1_weight, l2_weight, and out_weight must be float32; "
            "only ft_weight may be int16");
        return -1;
    }

    /* Initialize pointers to NULL for clean dealloc */
    self->ft_weight = NULL; self->ft_bias = NULL;
    self->l1_weight = NULL; self->l1_bias = NULL;
    self->l2_weight = NULL; self->l2_bias = NULL;
    self->out_weight = NULL; self->out_bias = NULL;
    self->num_buckets = num_buckets;
    self->white_acc = NULL; self->black_acc = NULL;
    self->stack_buf = NULL;
    self->ft_weight_q = NULL; self->ft_bias_q = NULL;
    self->white_acc_q = NULL; self->black_acc_q = NULL;
    self->stack_buf_q = NULL;

    self->max_stack = DEFAULT_MAX_STACK;
    self->stack_top = 0;

    if (is_int16) {
        /* ---- Int16 quantized path ---- */
        if (quant_scale <= 0.0f) quant_scale = 512.0f;
        self->quant_scale = quant_scale;
        self->inv_quant_scale = 1.0f / quant_scale;

        self->ft_weight_q = aligned_alloc_i16((size_t)nf * acc);
        self->ft_bias_q   = aligned_alloc_i16(acc);
        self->white_acc_q = aligned_alloc_i16(acc);
        self->black_acc_q = aligned_alloc_i16(acc);
        self->stack_buf_q = aligned_alloc_i16((size_t)self->max_stack * 2 * acc);

        if (!self->ft_weight_q || !self->ft_bias_q ||
            !self->white_acc_q || !self->black_acc_q || !self->stack_buf_q) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate int16 aligned memory");
            return -1;
        }

        if (copy_numpy_to_aligned_i16(py_ft_weight, self->ft_weight_q, (Py_ssize_t)nf * acc) < 0)
            return -1;
        if (copy_numpy_to_aligned_i16(py_ft_bias, self->ft_bias_q, acc) < 0)
            return -1;

        /* L1/L2/output stay float32 */
        self->l1_weight  = aligned_alloc_f32((size_t)l1 * acc * 2);
        self->l1_bias    = aligned_alloc_f32(l1);
        self->l2_weight  = aligned_alloc_f32((size_t)l2 * l1);
        self->l2_bias    = aligned_alloc_f32(l2);
        self->out_weight = aligned_alloc_f32((size_t)num_buckets * l2);
        self->out_bias   = aligned_alloc_f32(num_buckets);

        if (!self->l1_weight || !self->l1_bias || !self->l2_weight ||
            !self->l2_bias || !self->out_weight || !self->out_bias) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate float32 aligned memory");
            return -1;
        }

        if (copy_numpy_to_aligned(py_l1_weight, self->l1_weight, (Py_ssize_t)l1 * acc * 2) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_l1_bias, self->l1_bias, l1) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_l2_weight, self->l2_weight, (Py_ssize_t)l2 * l1) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_l2_bias, self->l2_bias, l2) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_out_weight, self->out_weight, (Py_ssize_t)num_buckets * l2) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_out_bias, self->out_bias, num_buckets) < 0)
            return -1;

        /* Initialize int16 accumulators to bias */
        memcpy(self->white_acc_q, self->ft_bias_q, acc * sizeof(int16_t));
        memcpy(self->black_acc_q, self->ft_bias_q, acc * sizeof(int16_t));

    } else {
        /* ---- Float32 path (existing, unchanged) ---- */
        self->quant_scale = 0.0f;
        self->inv_quant_scale = 0.0f;

        self->ft_weight  = aligned_alloc_f32((size_t)nf * acc);
        self->ft_bias    = aligned_alloc_f32(acc);
        self->l1_weight  = aligned_alloc_f32((size_t)l1 * acc * 2);
        self->l1_bias    = aligned_alloc_f32(l1);
        self->l2_weight  = aligned_alloc_f32((size_t)l2 * l1);
        self->l2_bias    = aligned_alloc_f32(l2);
        self->out_weight = aligned_alloc_f32((size_t)num_buckets * l2);
        self->out_bias   = aligned_alloc_f32(num_buckets);
        self->white_acc  = aligned_alloc_f32(acc);
        self->black_acc  = aligned_alloc_f32(acc);
        self->stack_buf  = aligned_alloc_f32((size_t)self->max_stack * 2 * acc);

        if (!self->ft_weight || !self->ft_bias || !self->l1_weight ||
            !self->l1_bias || !self->l2_weight || !self->l2_bias ||
            !self->out_weight || !self->out_bias || !self->white_acc ||
            !self->black_acc || !self->stack_buf) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate aligned memory");
            return -1;
        }

        if (copy_numpy_to_aligned(py_ft_weight, self->ft_weight, (Py_ssize_t)nf * acc) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_ft_bias, self->ft_bias, acc) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_l1_weight, self->l1_weight, (Py_ssize_t)l1 * acc * 2) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_l1_bias, self->l1_bias, l1) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_l2_weight, self->l2_weight, (Py_ssize_t)l2 * l1) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_l2_bias, self->l2_bias, l2) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_out_weight, self->out_weight, (Py_ssize_t)num_buckets * l2) < 0)
            return -1;
        if (copy_numpy_to_aligned(py_out_bias, self->out_bias, num_buckets) < 0)
            return -1;

        memcpy(self->white_acc, self->ft_bias, acc * sizeof(float));
        memcpy(self->black_acc, self->ft_bias, acc * sizeof(float));
    }

    return 0;
}
