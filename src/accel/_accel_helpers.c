/* ---- helpers ----------------------------------------------------------- */

/* Allocate 64-byte-aligned memory (cache line + NEON friendly). */
static float *aligned_alloc_f32(size_t count) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, count * sizeof(float)) != 0)
        return NULL;
    return (float *)ptr;
}

static int16_t *aligned_alloc_i16(size_t count) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, count * sizeof(int16_t)) != 0)
        return NULL;
    return (int16_t *)ptr;
}

/* Extract int indices from a Python list[int] into a C array.
 * Returns the number of indices extracted. */
static int extract_indices(PyObject *list, int *out, int max_len) {
    if (!PyList_Check(list)) return -1;
    Py_ssize_t n = PyList_GET_SIZE(list);
    if (n > max_len) n = max_len;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(list, i);
        long val = PyLong_AsLong(item);
        if (val == -1 && PyErr_Occurred()) return -1;
        out[i] = (int)val;
    }
    return (int)n;
}
