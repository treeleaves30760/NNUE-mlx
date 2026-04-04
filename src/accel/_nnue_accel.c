/*
 * _nnue_accel.c -- Accelerated NNUE inference for Apple Silicon.
 *
 * Drop-in replacement for IncrementalAccumulator using:
 *   - ARM NEON SIMD intrinsics for vector operations
 *   - Apple Accelerate framework (cblas_sgemv) for matrix-vector multiply
 *   - Pre-allocated stack buffers for zero-allocation search
 *   - float32 throughout (vs numpy float64) for 2x bandwidth
 *
 * Build:  python setup.py build_ext --inplace
 * Requires: macOS on Apple Silicon (arm64)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_NEON
#include <arm_neon.h>
#endif

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

/* ---- helpers ----------------------------------------------------------- */

/* Allocate 64-byte-aligned memory (cache line + NEON friendly). */
static float *aligned_alloc_f32(size_t count) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, count * sizeof(float)) != 0)
        return NULL;
    return (float *)ptr;
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

/* ---- NEON vector operations -------------------------------------------- */

#ifdef USE_NEON

/* dst[i] = clamp(src[i], 0.0, 1.0) for n floats (n must be multiple of 4). */
static inline void neon_clipped_relu_copy(float *dst, const float *src, int n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t one  = vdupq_n_f32(1.0f);
    for (int i = 0; i < n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        v = vmaxq_f32(v, zero);
        v = vminq_f32(v, one);
        vst1q_f32(dst + i, v);
    }
}

/* In-place clamp to [0, 1] for n floats (n must be multiple of 4). */
static inline void neon_clipped_relu_inplace(float *data, int n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t one  = vdupq_n_f32(1.0f);
    for (int i = 0; i < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        v = vmaxq_f32(v, zero);
        v = vminq_f32(v, one);
        vst1q_f32(data + i, v);
    }
}

/* acc += weight_row, using NEON 16-wide unrolled loop.
 * n must be a multiple of 16. */
static inline void neon_vec_add(float *acc, const float *row, int n) {
    for (int j = 0; j < n; j += 16) {
        float32x4_t a0 = vld1q_f32(acc + j);
        float32x4_t a1 = vld1q_f32(acc + j + 4);
        float32x4_t a2 = vld1q_f32(acc + j + 8);
        float32x4_t a3 = vld1q_f32(acc + j + 12);
        a0 = vaddq_f32(a0, vld1q_f32(row + j));
        a1 = vaddq_f32(a1, vld1q_f32(row + j + 4));
        a2 = vaddq_f32(a2, vld1q_f32(row + j + 8));
        a3 = vaddq_f32(a3, vld1q_f32(row + j + 12));
        vst1q_f32(acc + j, a0);
        vst1q_f32(acc + j + 4, a1);
        vst1q_f32(acc + j + 8, a2);
        vst1q_f32(acc + j + 12, a3);
    }
}

/* acc -= weight_row, using NEON 16-wide unrolled loop.
 * n must be a multiple of 16. */
static inline void neon_vec_sub(float *acc, const float *row, int n) {
    for (int j = 0; j < n; j += 16) {
        float32x4_t a0 = vld1q_f32(acc + j);
        float32x4_t a1 = vld1q_f32(acc + j + 4);
        float32x4_t a2 = vld1q_f32(acc + j + 8);
        float32x4_t a3 = vld1q_f32(acc + j + 12);
        a0 = vsubq_f32(a0, vld1q_f32(row + j));
        a1 = vsubq_f32(a1, vld1q_f32(row + j + 4));
        a2 = vsubq_f32(a2, vld1q_f32(row + j + 8));
        a3 = vsubq_f32(a3, vld1q_f32(row + j + 12));
        vst1q_f32(acc + j, a0);
        vst1q_f32(acc + j + 4, a1);
        vst1q_f32(acc + j + 8, a2);
        vst1q_f32(acc + j + 12, a3);
    }
}

#else
/* Scalar fallbacks for non-NEON platforms. */

static inline void neon_clipped_relu_copy(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) {
        float v = src[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        dst[i] = v;
    }
}

static inline void neon_clipped_relu_inplace(float *data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
        if (data[i] > 1.0f) data[i] = 1.0f;
    }
}

static inline void neon_vec_add(float *acc, const float *row, int n) {
    for (int j = 0; j < n; j++) acc[j] += row[j];
}

static inline void neon_vec_sub(float *acc, const float *row, int n) {
    for (int j = 0; j < n; j++) acc[j] -= row[j];
}

#endif /* USE_NEON */

/* ---- Matrix-vector multiply -------------------------------------------- */

/* y = alpha * A * x + beta * y   (row-major A: m x n). */
static inline void sgemv(int m, int n, float alpha,
                         const float *A, int lda,
                         const float *x,
                         float beta, float *y) {
#ifdef USE_ACCELERATE
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n,
                alpha, A, lda, x, 1, beta, y, 1);
#else
    /* Scalar fallback. */
    for (int i = 0; i < m; i++) {
        float sum = (beta != 0.0f) ? beta * y[i] : 0.0f;
        const float *row = A + i * lda;
        for (int j = 0; j < n; j++)
            sum += alpha * row[j] * x[j];
        y[i] = sum;
    }
#endif
}

/* dot = x . y  for n floats. */
static inline float sdot(int n, const float *x, const float *y) {
#ifdef USE_ACCELERATE
    return cblas_sdot(n, x, 1, y, 1);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * y[i];
    return sum;
#endif
}

/* ---- AccelAccumObject -------------------------------------------------- */

#define MAX_INDICES 64   /* Max features per perspective (well above 30 for chess) */
#define DEFAULT_MAX_STACK 128

typedef struct {
    PyObject_HEAD

    int num_features;
    int accumulator_size;    /* 256 */
    int l1_size;             /* 32  */
    int l2_size;             /* 32  */

    /* Weight storage (64-byte aligned) */
    float *ft_weight;        /* [num_features * accumulator_size] */
    float *ft_bias;          /* [accumulator_size] */
    float *l1_weight;        /* [l1_size * accumulator_size * 2] */
    float *l1_bias;          /* [l1_size] */
    float *l2_weight;        /* [l2_size * l1_size] */
    float *l2_bias;          /* [l2_size] */
    float *out_weight;       /* [l2_size] */
    float  out_bias;

    /* Accumulator state */
    float *white_acc;        /* [accumulator_size] */
    float *black_acc;        /* [accumulator_size] */

    /* Pre-allocated stack for push/pop (zero allocation during search) */
    float *stack_buf;        /* [max_stack * 2 * accumulator_size] */
    int    stack_top;
    int    max_stack;

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
    free(self->white_acc);
    free(self->black_acc);
    free(self->stack_buf);
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
        NULL
    };

    PyObject *py_ft_weight, *py_ft_bias;
    PyObject *py_l1_weight, *py_l1_bias;
    PyObject *py_l2_weight, *py_l2_bias;
    PyObject *py_out_weight, *py_out_bias;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOO", kwlist,
            &py_ft_weight, &py_ft_bias,
            &py_l1_weight, &py_l1_bias,
            &py_l2_weight, &py_l2_bias,
            &py_out_weight, &py_out_bias))
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

    /* Allocate aligned buffers. */
    self->ft_weight  = aligned_alloc_f32((size_t)nf * acc);
    self->ft_bias    = aligned_alloc_f32(acc);
    self->l1_weight  = aligned_alloc_f32((size_t)l1 * acc * 2);
    self->l1_bias    = aligned_alloc_f32(l1);
    self->l2_weight  = aligned_alloc_f32((size_t)l2 * l1);
    self->l2_bias    = aligned_alloc_f32(l2);
    self->out_weight = aligned_alloc_f32(l2);
    self->white_acc  = aligned_alloc_f32(acc);
    self->black_acc  = aligned_alloc_f32(acc);

    self->max_stack = DEFAULT_MAX_STACK;
    self->stack_buf = aligned_alloc_f32((size_t)self->max_stack * 2 * acc);
    self->stack_top = 0;

    if (!self->ft_weight || !self->ft_bias || !self->l1_weight ||
        !self->l1_bias || !self->l2_weight || !self->l2_bias ||
        !self->out_weight || !self->white_acc || !self->black_acc ||
        !self->stack_buf) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate aligned memory");
        return -1;
    }

    /* Copy weight data from numpy arrays into aligned buffers. */
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
    if (copy_numpy_to_aligned(py_out_weight, self->out_weight, l2) < 0)
        return -1;

    /* out_bias is a scalar from a (1,) array. */
    {
        float tmp[1];
        if (copy_numpy_to_aligned(py_out_bias, tmp, 1) < 0) return -1;
        self->out_bias = tmp[0];
    }

    /* Initialize accumulators to bias. */
    memcpy(self->white_acc, self->ft_bias, acc * sizeof(float));
    memcpy(self->black_acc, self->ft_bias, acc * sizeof(float));

    return 0;
}

/* ---- refresh ------------------------------------------------------------ */

static void
_do_refresh(AccelAccumObject *self, float *acc, int *indices, int count)
{
    int acc_size = self->accumulator_size;
    memcpy(acc, self->ft_bias, acc_size * sizeof(float));
    for (int i = 0; i < count; i++) {
        int idx = indices[i];
        if (idx < 0 || idx >= self->num_features) continue;
        const float *row = self->ft_weight + (size_t)idx * acc_size;
        neon_vec_add(acc, row, acc_size);
    }
}

static PyObject *
AccelAccum_refresh(AccelAccumObject *self, PyObject *args)
{
    PyObject *py_white, *py_black;
    if (!PyArg_ParseTuple(args, "OO", &py_white, &py_black))
        return NULL;

    int w_indices[MAX_INDICES], b_indices[MAX_INDICES];
    int w_count = extract_indices(py_white, w_indices, MAX_INDICES);
    if (w_count < 0) return NULL;
    int b_count = extract_indices(py_black, b_indices, MAX_INDICES);
    if (b_count < 0) return NULL;

    _do_refresh(self, self->white_acc, w_indices, w_count);
    _do_refresh(self, self->black_acc, b_indices, b_count);

    Py_RETURN_NONE;
}

/* ---- refresh_perspective ------------------------------------------------ */

static PyObject *
AccelAccum_refresh_perspective(AccelAccumObject *self, PyObject *args)
{
    int perspective;
    PyObject *py_features;
    if (!PyArg_ParseTuple(args, "iO", &perspective, &py_features))
        return NULL;

    int indices[MAX_INDICES];
    int count = extract_indices(py_features, indices, MAX_INDICES);
    if (count < 0) return NULL;

    float *acc = (perspective == 0) ? self->white_acc : self->black_acc;
    _do_refresh(self, acc, indices, count);

    Py_RETURN_NONE;
}

/* ---- update ------------------------------------------------------------- */

static PyObject *
AccelAccum_update(AccelAccumObject *self, PyObject *args)
{
    int perspective;
    PyObject *py_added, *py_removed;
    if (!PyArg_ParseTuple(args, "iOO", &perspective, &py_added, &py_removed))
        return NULL;

    int added[MAX_INDICES], removed[MAX_INDICES];
    int n_added = extract_indices(py_added, added, MAX_INDICES);
    if (n_added < 0) return NULL;
    int n_removed = extract_indices(py_removed, removed, MAX_INDICES);
    if (n_removed < 0) return NULL;

    float *acc = (perspective == 0) ? self->white_acc : self->black_acc;
    int acc_size = self->accumulator_size;

    for (int i = 0; i < n_removed; i++) {
        int idx = removed[i];
        if (idx < 0 || idx >= self->num_features) continue;
        neon_vec_sub(acc, self->ft_weight + (size_t)idx * acc_size, acc_size);
    }
    for (int i = 0; i < n_added; i++) {
        int idx = added[i];
        if (idx < 0 || idx >= self->num_features) continue;
        neon_vec_add(acc, self->ft_weight + (size_t)idx * acc_size, acc_size);
    }

    Py_RETURN_NONE;
}

/* ---- push / pop --------------------------------------------------------- */

static PyObject *
AccelAccum_push(AccelAccumObject *self, PyObject *Py_UNUSED(args))
{
    if (self->stack_top >= self->max_stack) {
        PyErr_SetString(PyExc_RuntimeError, "Accumulator stack overflow");
        return NULL;
    }

    int acc_size = self->accumulator_size;
    size_t offset = (size_t)self->stack_top * 2 * acc_size;
    memcpy(self->stack_buf + offset, self->white_acc, acc_size * sizeof(float));
    memcpy(self->stack_buf + offset + acc_size, self->black_acc, acc_size * sizeof(float));
    self->stack_top++;

    Py_RETURN_NONE;
}

static PyObject *
AccelAccum_pop(AccelAccumObject *self, PyObject *Py_UNUSED(args))
{
    if (self->stack_top <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Accumulator stack underflow");
        return NULL;
    }

    self->stack_top--;
    int acc_size = self->accumulator_size;
    size_t offset = (size_t)self->stack_top * 2 * acc_size;
    memcpy(self->white_acc, self->stack_buf + offset, acc_size * sizeof(float));
    memcpy(self->black_acc, self->stack_buf + offset + acc_size, acc_size * sizeof(float));

    Py_RETURN_NONE;
}

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
    float input[512];   /* acc_size * 2, max 256*2 = 512 */
    float l1_out[64];   /* l1_size, max 64 */
    float l2_out[64];   /* l2_size, max 64 */

    /* 1. ClippedReLU + perspective-ordered concat. */
    float *first  = (side_to_move == 0) ? self->white_acc : self->black_acc;
    float *second = (side_to_move == 0) ? self->black_acc : self->white_acc;
    neon_clipped_relu_copy(input, first, acc_size);
    neon_clipped_relu_copy(input + acc_size, second, acc_size);

    /* 2. L1: y = ClippedReLU(W*x + b). */
    memcpy(l1_out, self->l1_bias, l1 * sizeof(float));
    sgemv(l1, acc_size * 2, 1.0f, self->l1_weight, acc_size * 2, input, 1.0f, l1_out);
    neon_clipped_relu_inplace(l1_out, l1);

    /* 3. L2: y = ClippedReLU(W*x + b). */
    memcpy(l2_out, self->l2_bias, l2 * sizeof(float));
    sgemv(l2, l1, 1.0f, self->l2_weight, l1, l1_out, 1.0f, l2_out);
    neon_clipped_relu_inplace(l2_out, l2);

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

/* ---- Feature extraction (module-level functions) ----------------------- */

/*
 * halfkp_active_features(board, king_sq, perspective,
 *                        num_squares, num_piece_types,
 *                        king_board_val, bv2type_arr)
 *
 * Compute HalfKP feature indices directly from the int8 board array.
 * Returns a Python list of ints.
 */
static PyObject *
accel_halfkp_active_features(PyObject *self, PyObject *args)
{
    Py_buffer board_buf, bv2type_buf;
    int king_sq, perspective, num_squares, num_piece_types, king_board_val;

    if (!PyArg_ParseTuple(args, "y*iiiiiy*",
                          &board_buf, &king_sq, &perspective,
                          &num_squares, &num_piece_types,
                          &king_board_val, &bv2type_buf))
        return NULL;

    const int8_t *board = (const int8_t *)board_buf.buf;
    const int8_t *bv2type = (const int8_t *)bv2type_buf.buf;
    int bv2type_len = (int)bv2type_buf.len;
    int piece_sq_combos = num_piece_types * 2 * num_squares;

    /* Pre-allocate list with generous upper bound */
    int max_pieces = num_squares;  /* can't have more pieces than squares */
    PyObject *result = PyList_New(max_pieces);
    if (!result) {
        PyBuffer_Release(&board_buf);
        PyBuffer_Release(&bv2type_buf);
        return NULL;
    }

    int count = 0;
    for (int sq = 0; sq < num_squares; sq++) {
        int8_t val = board[sq];
        if (val == 0) continue;
        int abs_val = (val > 0) ? val : -val;
        if (abs_val == king_board_val) continue;
        if (abs_val >= bv2type_len) continue;
        int pt = bv2type[abs_val];
        if (pt < 0) continue;

        int color = (val > 0) ? 0 : 1;
        int rel = (color == perspective) ? 0 : 1;
        int idx = king_sq * piece_sq_combos
                + (rel * num_piece_types + pt) * num_squares
                + sq;
        PyObject *py_idx = PyLong_FromLong(idx);
        if (!py_idx) {
            Py_DECREF(result);
            PyBuffer_Release(&board_buf);
            PyBuffer_Release(&bv2type_buf);
            return NULL;
        }
        PyList_SET_ITEM(result, count, py_idx);  /* steals ref */
        count++;
    }

    /* Truncate list to actual count */
    if (count < max_pieces) {
        if (PyList_SetSlice(result, count, max_pieces, NULL) < 0) {
            Py_DECREF(result);
            PyBuffer_Release(&board_buf);
            PyBuffer_Release(&bv2type_buf);
            return NULL;
        }
    }

    PyBuffer_Release(&board_buf);
    PyBuffer_Release(&bv2type_buf);
    return result;
}

/*
 * halfkp_shogi_active_features(board, king_sq, perspective,
 *                               num_squares, num_board_piece_types,
 *                               num_hand_piece_types, max_hand_count,
 *                               king_board_val, bv2type_arr,
 *                               board_features,
 *                               hand0_types, hand0_counts, hand0_len,
 *                               hand1_types, hand1_counts, hand1_len)
 *
 * Compute HalfKPShogi feature indices (board + hand) from raw arrays.
 */
static PyObject *
accel_halfkp_shogi_active_features(PyObject *self, PyObject *args)
{
    Py_buffer board_buf, bv2type_buf;
    Py_buffer h0t_buf, h0c_buf, h1t_buf, h1c_buf;
    int king_sq, perspective, num_squares, num_board_pt, num_hand_pt;
    int max_hand_count, king_board_val, board_features;
    int h0_len, h1_len;

    if (!PyArg_ParseTuple(args, "y*iiiiiiiy*iy*y*iy*y*i",
                          &board_buf, &king_sq, &perspective,
                          &num_squares, &num_board_pt,
                          &num_hand_pt, &max_hand_count,
                          &king_board_val, &bv2type_buf,
                          &board_features,
                          &h0t_buf, &h0c_buf, &h0_len,
                          &h1t_buf, &h1c_buf, &h1_len))
        return NULL;

    const int8_t *board = (const int8_t *)board_buf.buf;
    const int8_t *bv2type = (const int8_t *)bv2type_buf.buf;
    int bv2type_len = (int)bv2type_buf.len;
    const int32_t *h0_types  = (const int32_t *)h0t_buf.buf;
    const int32_t *h0_counts = (const int32_t *)h0c_buf.buf;
    const int32_t *h1_types  = (const int32_t *)h1t_buf.buf;
    const int32_t *h1_counts = (const int32_t *)h1c_buf.buf;

    int piece_sq_combos = num_board_pt * 2 * num_squares;
    int hand_combos = num_hand_pt * 2 * max_hand_count;

    /* Upper bound: pieces on board + hand features */
    int max_features = num_squares + 2 * num_hand_pt * max_hand_count;
    PyObject *result = PyList_New(max_features);
    if (!result) goto fail_clean;

    int count = 0;

    /* Board piece features */
    for (int sq = 0; sq < num_squares; sq++) {
        int8_t val = board[sq];
        if (val == 0) continue;
        int abs_val = (val > 0) ? val : -val;
        if (abs_val == king_board_val) continue;
        if (abs_val >= bv2type_len) continue;
        int pt = bv2type[abs_val];
        if (pt < 0) continue;

        int color = (val > 0) ? 0 : 1;
        int rel = (color == perspective) ? 0 : 1;
        int idx = king_sq * piece_sq_combos
                + (rel * num_board_pt + pt) * num_squares
                + sq;
        PyList_SET_ITEM(result, count++, PyLong_FromLong(idx));
    }

    /* Hand piece features for side 0 and side 1 */
    struct { const int32_t *types; const int32_t *counts; int len; int side; }
    hands[2] = {
        { h0_types, h0_counts, h0_len, 0 },
        { h1_types, h1_counts, h1_len, 1 },
    };
    for (int h = 0; h < 2; h++) {
        int side = hands[h].side;
        int rel = (side == perspective) ? 0 : 1;
        for (int i = 0; i < hands[h].len; i++) {
            int pt = hands[h].types[i];
            int cnt = hands[h].counts[i];
            int lim = (cnt < max_hand_count) ? cnt : max_hand_count;
            for (int k = 0; k < lim; k++) {
                int idx = board_features
                        + king_sq * hand_combos
                        + (rel * num_hand_pt + pt) * max_hand_count
                        + k;
                PyList_SET_ITEM(result, count++, PyLong_FromLong(idx));
            }
        }
    }

    /* Truncate list */
    if (count < max_features) {
        PyList_SetSlice(result, count, max_features, NULL);
    }

    PyBuffer_Release(&board_buf);
    PyBuffer_Release(&bv2type_buf);
    PyBuffer_Release(&h0t_buf);
    PyBuffer_Release(&h0c_buf);
    PyBuffer_Release(&h1t_buf);
    PyBuffer_Release(&h1c_buf);
    return result;

fail_clean:
    PyBuffer_Release(&board_buf);
    PyBuffer_Release(&bv2type_buf);
    PyBuffer_Release(&h0t_buf);
    PyBuffer_Release(&h0c_buf);
    PyBuffer_Release(&h1t_buf);
    PyBuffer_Release(&h1c_buf);
    return NULL;
}

static PyMethodDef module_methods[] = {
    {"halfkp_active_features", accel_halfkp_active_features, METH_VARARGS,
     "Compute HalfKP active feature indices from board array (C accelerated)."},
    {"halfkp_shogi_active_features", accel_halfkp_shogi_active_features, METH_VARARGS,
     "Compute HalfKPShogi active feature indices from board + hand arrays (C accelerated)."},
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

    PyObject *m = PyModule_Create(&accel_module);
    if (!m) return NULL;

    Py_INCREF(&AccelAccumType);
    if (PyModule_AddObject(m, "AcceleratedAccumulator",
                           (PyObject *)&AccelAccumType) < 0) {
        Py_DECREF(&AccelAccumType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
