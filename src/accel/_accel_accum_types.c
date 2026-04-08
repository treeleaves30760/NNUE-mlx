/* ---- AccelAccumObject -------------------------------------------------- */

typedef struct {
    PyObject_HEAD

    int num_features;
    int accumulator_size;    /* 256 */
    int l1_size;             /* 32  */
    int l2_size;             /* 32  */

    /* Float32 weight storage (64-byte aligned) */
    float *ft_weight;        /* [num_features * accumulator_size] */
    float *ft_bias;          /* [accumulator_size] */
    float *l1_weight;        /* [l1_size * accumulator_size * 2] */
    float *l1_bias;          /* [l1_size] */
    float *l2_weight;        /* [l2_size * l1_size] */
    float *l2_bias;          /* [l2_size] */
    float *out_weight;       /* [l2_size] */
    float  out_bias;

    /* Float32 accumulator state */
    float *white_acc;        /* [accumulator_size] */
    float *black_acc;        /* [accumulator_size] */

    /* Pre-allocated stack for push/pop (zero allocation during search) */
    float *stack_buf;        /* [max_stack * 2 * accumulator_size] (float32 mode) */
    int    stack_top;
    int    max_stack;

    /* Int16 quantized mode (set by detecting int16 dtype on ft_weight) */
    int    use_int16;
    float  quant_scale;      /* Q (e.g. 512) */
    float  inv_quant_scale;  /* 1/Q */

    int16_t *ft_weight_q;    /* [num_features * accumulator_size] */
    int16_t *ft_bias_q;      /* [accumulator_size] */
    int16_t *white_acc_q;    /* [accumulator_size] */
    int16_t *black_acc_q;    /* [accumulator_size] */
    int16_t *stack_buf_q;    /* [max_stack * 2 * accumulator_size] (int16 mode) */

} AccelAccumObject;

/* ---- Helper: copy numpy array data to aligned float32 buffer ----------- */

/*
 * Import a numpy array, convert to float32, and copy into aligned buffer.
 * Caller must have allocated `dst` with aligned_alloc_f32(count).
 * Returns 0 on success, -1 on error.
 */
static int copy_numpy_to_aligned(PyObject *np_array, float *dst, Py_ssize_t expected) {
    /* Import numpy C API lazily. We use the buffer protocol instead for
     * simplicity -- numpy arrays support the buffer protocol and we can
     * just iterate. However, the most robust approach is to call
     * numpy.ascontiguousarray(arr, dtype=float32) from C. Let's do that. */

    PyObject *np_mod = PyImport_ImportModule("numpy");
    if (!np_mod) return -1;

    PyObject *as_contig = PyObject_GetAttrString(np_mod, "ascontiguousarray");
    Py_DECREF(np_mod);
    if (!as_contig) return -1;

    /* Get numpy.float32 dtype object. */
    PyObject *np_mod2 = PyImport_ImportModule("numpy");
    PyObject *f32_dtype = PyObject_GetAttrString(np_mod2, "float32");
    Py_DECREF(np_mod2);

    /* Call numpy.ascontiguousarray(arr, dtype=numpy.float32). */
    PyObject *args = PyTuple_Pack(1, np_array);
    PyObject *kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "dtype", f32_dtype);
    Py_DECREF(f32_dtype);

    PyObject *result = PyObject_Call(as_contig, args, kwargs);
    Py_DECREF(as_contig);
    Py_DECREF(args);
    Py_DECREF(kwargs);

    if (!result) return -1;

    /* Get buffer from the float32 contiguous array. */
    Py_buffer buf;
    if (PyObject_GetBuffer(result, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        Py_DECREF(result);
        return -1;
    }

    Py_ssize_t n_elements = buf.len / sizeof(float);
    if (n_elements < expected) {
        PyErr_Format(PyExc_ValueError,
                     "Array has %zd elements, expected at least %zd",
                     n_elements, expected);
        PyBuffer_Release(&buf);
        Py_DECREF(result);
        return -1;
    }

    memcpy(dst, buf.buf, expected * sizeof(float));
    PyBuffer_Release(&buf);
    Py_DECREF(result);
    return 0;
}

/* Copy numpy int16 array data to aligned int16 buffer. */
static int copy_numpy_to_aligned_i16(PyObject *np_array, int16_t *dst, Py_ssize_t expected) {
    PyObject *np_mod = PyImport_ImportModule("numpy");
    if (!np_mod) return -1;
    PyObject *as_contig = PyObject_GetAttrString(np_mod, "ascontiguousarray");
    Py_DECREF(np_mod);
    if (!as_contig) return -1;

    PyObject *np_mod2 = PyImport_ImportModule("numpy");
    PyObject *i16_dtype = PyObject_GetAttrString(np_mod2, "int16");
    Py_DECREF(np_mod2);

    PyObject *args = PyTuple_Pack(1, np_array);
    PyObject *kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "dtype", i16_dtype);
    Py_DECREF(i16_dtype);

    PyObject *result = PyObject_Call(as_contig, args, kwargs);
    Py_DECREF(as_contig);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (!result) return -1;

    Py_buffer buf;
    if (PyObject_GetBuffer(result, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        Py_DECREF(result);
        return -1;
    }
    Py_ssize_t n_elements = buf.len / sizeof(int16_t);
    if (n_elements < expected) {
        PyErr_Format(PyExc_ValueError,
                     "Array has %zd elements, expected at least %zd",
                     n_elements, expected);
        PyBuffer_Release(&buf);
        Py_DECREF(result);
        return -1;
    }
    memcpy(dst, buf.buf, expected * sizeof(int16_t));
    PyBuffer_Release(&buf);
    Py_DECREF(result);
    return 0;
}

/* Check if a numpy array has int16 dtype. Returns 1 if int16, 0 otherwise. */
static int is_numpy_int16(PyObject *np_array) {
    PyObject *dtype = PyObject_GetAttrString(np_array, "dtype");
    if (!dtype) { PyErr_Clear(); return 0; }
    PyObject *name = PyObject_GetAttrString(dtype, "name");
    Py_DECREF(dtype);
    if (!name) { PyErr_Clear(); return 0; }
    const char *s = PyUnicode_AsUTF8(name);
    int result = (s && strcmp(s, "int16") == 0);
    Py_DECREF(name);
    return result;
}
