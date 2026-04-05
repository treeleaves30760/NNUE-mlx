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

/* ---- Int16 NEON vector operations --------------------------------------- */

/* acc += row for int16, 32-wide unrolled (8 per int16x8_t × 4). */
static inline void neon_vec_add_i16(int16_t *acc, const int16_t *row, int n) {
    for (int j = 0; j < n; j += 32) {
        int16x8_t a0 = vld1q_s16(acc + j);
        int16x8_t a1 = vld1q_s16(acc + j + 8);
        int16x8_t a2 = vld1q_s16(acc + j + 16);
        int16x8_t a3 = vld1q_s16(acc + j + 24);
        a0 = vaddq_s16(a0, vld1q_s16(row + j));
        a1 = vaddq_s16(a1, vld1q_s16(row + j + 8));
        a2 = vaddq_s16(a2, vld1q_s16(row + j + 16));
        a3 = vaddq_s16(a3, vld1q_s16(row + j + 24));
        vst1q_s16(acc + j, a0);
        vst1q_s16(acc + j + 8, a1);
        vst1q_s16(acc + j + 16, a2);
        vst1q_s16(acc + j + 24, a3);
    }
}

static inline void neon_vec_sub_i16(int16_t *acc, const int16_t *row, int n) {
    for (int j = 0; j < n; j += 32) {
        int16x8_t a0 = vld1q_s16(acc + j);
        int16x8_t a1 = vld1q_s16(acc + j + 8);
        int16x8_t a2 = vld1q_s16(acc + j + 16);
        int16x8_t a3 = vld1q_s16(acc + j + 24);
        a0 = vsubq_s16(a0, vld1q_s16(row + j));
        a1 = vsubq_s16(a1, vld1q_s16(row + j + 8));
        a2 = vsubq_s16(a2, vld1q_s16(row + j + 16));
        a3 = vsubq_s16(a3, vld1q_s16(row + j + 24));
        vst1q_s16(acc + j, a0);
        vst1q_s16(acc + j + 8, a1);
        vst1q_s16(acc + j + 16, a2);
        vst1q_s16(acc + j + 24, a3);
    }
}

/* Dequantize int16 accumulator to float32 with ClippedReLU in one pass.
 * dst[i] = clamp(src[i] * inv_scale, 0.0, 1.0)
 * Processes 8 elements per iteration using vmovl + vcvt + vmul + clamp. */
static inline void neon_dequant_clipped_relu(float *dst, const int16_t *src,
                                              int n, float inv_scale) {
    float32x4_t scale = vdupq_n_f32(inv_scale);
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t one  = vdupq_n_f32(1.0f);
    for (int i = 0; i < n; i += 8) {
        int16x8_t s = vld1q_s16(src + i);
        /* Widen low/high halves to int32, convert to float32 */
        int32x4_t lo = vmovl_s16(vget_low_s16(s));
        int32x4_t hi = vmovl_s16(vget_high_s16(s));
        float32x4_t flo = vmulq_f32(vcvtq_f32_s32(lo), scale);
        float32x4_t fhi = vmulq_f32(vcvtq_f32_s32(hi), scale);
        flo = vmaxq_f32(flo, zero);
        flo = vminq_f32(flo, one);
        fhi = vmaxq_f32(fhi, zero);
        fhi = vminq_f32(fhi, one);
        vst1q_f32(dst + i, flo);
        vst1q_f32(dst + i + 4, fhi);
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

static inline void neon_vec_add_i16(int16_t *acc, const int16_t *row, int n) {
    for (int j = 0; j < n; j++) acc[j] += row[j];
}

static inline void neon_vec_sub_i16(int16_t *acc, const int16_t *row, int n) {
    for (int j = 0; j < n; j++) acc[j] -= row[j];
}

static inline void neon_dequant_clipped_relu(float *dst, const int16_t *src,
                                              int n, float inv_scale) {
    for (int i = 0; i < n; i++) {
        float v = (float)src[i] * inv_scale;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        dst[i] = v;
    }
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

    /* Initialize pointers to NULL for clean dealloc */
    self->ft_weight = NULL; self->ft_bias = NULL;
    self->l1_weight = NULL; self->l1_bias = NULL;
    self->l2_weight = NULL; self->l2_bias = NULL;
    self->out_weight = NULL;
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
        self->out_weight = aligned_alloc_f32(l2);

        if (!self->l1_weight || !self->l1_bias || !self->l2_weight ||
            !self->l2_bias || !self->out_weight) {
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
        if (copy_numpy_to_aligned(py_out_weight, self->out_weight, l2) < 0)
            return -1;

        { float tmp[1]; if (copy_numpy_to_aligned(py_out_bias, tmp, 1) < 0) return -1; self->out_bias = tmp[0]; }

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
        self->out_weight = aligned_alloc_f32(l2);
        self->white_acc  = aligned_alloc_f32(acc);
        self->black_acc  = aligned_alloc_f32(acc);
        self->stack_buf  = aligned_alloc_f32((size_t)self->max_stack * 2 * acc);

        if (!self->ft_weight || !self->ft_bias || !self->l1_weight ||
            !self->l1_bias || !self->l2_weight || !self->l2_bias ||
            !self->out_weight || !self->white_acc || !self->black_acc ||
            !self->stack_buf) {
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
        if (copy_numpy_to_aligned(py_out_weight, self->out_weight, l2) < 0)
            return -1;

        { float tmp[1]; if (copy_numpy_to_aligned(py_out_bias, tmp, 1) < 0) return -1; self->out_bias = tmp[0]; }

        memcpy(self->white_acc, self->ft_bias, acc * sizeof(float));
        memcpy(self->black_acc, self->ft_bias, acc * sizeof(float));
    }

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

static void
_do_refresh_i16(AccelAccumObject *self, int16_t *acc, int *indices, int count)
{
    int acc_size = self->accumulator_size;
    memcpy(acc, self->ft_bias_q, acc_size * sizeof(int16_t));
    for (int i = 0; i < count; i++) {
        int idx = indices[i];
        if (idx < 0 || idx >= self->num_features) continue;
        const int16_t *row = self->ft_weight_q + (size_t)idx * acc_size;
        neon_vec_add_i16(acc, row, acc_size);
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

    if (self->use_int16) {
        _do_refresh_i16(self, self->white_acc_q, w_indices, w_count);
        _do_refresh_i16(self, self->black_acc_q, b_indices, b_count);
    } else {
        _do_refresh(self, self->white_acc, w_indices, w_count);
        _do_refresh(self, self->black_acc, b_indices, b_count);
    }

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

    if (self->use_int16) {
        int16_t *acc = (perspective == 0) ? self->white_acc_q : self->black_acc_q;
        _do_refresh_i16(self, acc, indices, count);
    } else {
        float *acc = (perspective == 0) ? self->white_acc : self->black_acc;
        _do_refresh(self, acc, indices, count);
    }

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

    int acc_size = self->accumulator_size;

    if (self->use_int16) {
        int16_t *acc = (perspective == 0) ? self->white_acc_q : self->black_acc_q;
        for (int i = 0; i < n_removed; i++) {
            int idx = removed[i];
            if (idx < 0 || idx >= self->num_features) continue;
            neon_vec_sub_i16(acc, self->ft_weight_q + (size_t)idx * acc_size, acc_size);
        }
        for (int i = 0; i < n_added; i++) {
            int idx = added[i];
            if (idx < 0 || idx >= self->num_features) continue;
            neon_vec_add_i16(acc, self->ft_weight_q + (size_t)idx * acc_size, acc_size);
        }
    } else {
        float *acc = (perspective == 0) ? self->white_acc : self->black_acc;
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
    if (self->use_int16) {
        size_t offset = (size_t)self->stack_top * 2 * acc_size;
        memcpy(self->stack_buf_q + offset, self->white_acc_q, acc_size * sizeof(int16_t));
        memcpy(self->stack_buf_q + offset + acc_size, self->black_acc_q, acc_size * sizeof(int16_t));
    } else {
        size_t offset = (size_t)self->stack_top * 2 * acc_size;
        memcpy(self->stack_buf + offset, self->white_acc, acc_size * sizeof(float));
        memcpy(self->stack_buf + offset + acc_size, self->black_acc, acc_size * sizeof(float));
    }
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
    if (self->use_int16) {
        size_t offset = (size_t)self->stack_top * 2 * acc_size;
        memcpy(self->white_acc_q, self->stack_buf_q + offset, acc_size * sizeof(int16_t));
        memcpy(self->black_acc_q, self->stack_buf_q + offset + acc_size, acc_size * sizeof(int16_t));
    } else {
        size_t offset = (size_t)self->stack_top * 2 * acc_size;
        memcpy(self->white_acc, self->stack_buf + offset, acc_size * sizeof(float));
        memcpy(self->black_acc, self->stack_buf + offset + acc_size, acc_size * sizeof(float));
    }

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

/* ---- Shogi / Minishogi attack detection --------------------------------- */

/* Shared slider check: from (rank, file), slide in direction (dr, df).
 * If the first piece encountered is attacker_sign * val1 or * val2, return 1. */
static inline int
slide_check(const int8_t *board, int rank, int file, int dr, int df,
            int sign, int val1, int val2, int board_dim)
{
    int nr = rank + dr, nf = file + df;
    while (nr >= 0 && nr < board_dim && nf >= 0 && nf < board_dim) {
        int8_t v = board[nr * board_dim + nf];
        if (v != 0) {
            return (v == sign * val1 || v == sign * val2);
        }
        nr += dr; nf += df;
    }
    return 0;
}

/* One-step check: is there an attacker_sign * piece_val at (rank+dr, file+df)? */
static inline int
step_check(const int8_t *board, int rank, int file, int dr, int df,
           int sign, int val, int board_dim)
{
    int nr = rank + dr, nf = file + df;
    if (nr < 0 || nr >= board_dim || nf < 0 || nf >= board_dim) return 0;
    return board[nr * board_dim + nf] == sign * val;
}

/*
 * is_square_attacked_shogi(board, target_sq, attacker_side)
 *
 * Reverse attack detection on a 9x9 shogi board.
 * Pieces: 1=Pawn 2=Lance 3=Knight 4=Silver 5=Gold 6=Bishop 7=Rook 8=King
 *         9=+Pawn 10=+Lance 11=+Knight 12=+Silver 13=+Bishop(Horse) 14=+Rook(Dragon)
 */
static PyObject *
accel_is_square_attacked_shogi(PyObject *self, PyObject *args)
{
    Py_buffer board_buf;
    int target_sq, attacker_side;

    if (!PyArg_ParseTuple(args, "y*ii", &board_buf, &target_sq, &attacker_side))
        return NULL;

    if (board_buf.len < 81) {
        PyBuffer_Release(&board_buf);
        PyErr_SetString(PyExc_ValueError, "Board must have at least 81 elements");
        return NULL;
    }

    const int8_t *board = (const int8_t *)board_buf.buf;
    int rank = target_sq / 9, file = target_sq % 9;
    int sign = (attacker_side == 0) ? 1 : -1;  /* +1=sente, -1=gote */

    /* Static direction tables for reverse attack */
    static const int BISHOP_DIRS[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
    static const int ROOK_DIRS[4][2]   = {{-1,0},{1,0},{0,-1},{0,1}};

    int found = 0;

    /* 1. Pawn: attacker's pawn one step behind target */
    {
        int pr = rank + (attacker_side == 0 ? 1 : -1);
        if (pr >= 0 && pr < 9 && board[pr * 9 + file] == sign * 1)
            { found = 1; goto done; }
    }

    /* 2. Lance: slide in reverse of attacker's forward direction */
    {
        int lance_dr = (attacker_side == 0) ? 1 : -1;
        for (int r = rank + lance_dr; r >= 0 && r < 9; r += lance_dr) {
            int8_t v = board[r * 9 + file];
            if (v != 0) {
                if (v == sign * 2) found = 1;
                break;
            }
        }
        if (found) goto done;
    }

    /* 3. Knight: reverse jump (2 back, 1 side) */
    {
        int kdr = (attacker_side == 0) ? 2 : -2;
        for (int df = -1; df <= 1; df += 2) {
            int nr = rank + kdr, nf = file + df;
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                if (board[nr * 9 + nf] == sign * 3) { found = 1; goto done; }
            }
        }
    }

    /* 4. Silver: 5 reverse directions */
    {
        /* Sente silver moves: (-1,-1),(-1,0),(-1,1),(1,-1),(1,1) from piece
         * Reverse from target: (1,1),(1,0),(1,-1),(-1,1),(-1,-1) for sente attacker */
        static const int S_REV_SENTE[5][2] = {{1,1},{1,0},{1,-1},{-1,1},{-1,-1}};
        static const int S_REV_GOTE[5][2]  = {{-1,-1},{-1,0},{-1,1},{1,-1},{1,1}};
        const int (*dirs)[2] = (attacker_side == 0) ? S_REV_SENTE : S_REV_GOTE;
        for (int d = 0; d < 5; d++) {
            int nr = rank + dirs[d][0], nf = file + dirs[d][1];
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                if (board[nr * 9 + nf] == sign * 4) { found = 1; goto done; }
            }
        }
    }

    /* 5. Gold-like: gold(5), +pawn(9), +lance(10), +knight(11), +silver(12) */
    {
        static const int G_REV_SENTE[6][2] = {{1,1},{1,0},{1,-1},{0,1},{0,-1},{-1,0}};
        static const int G_REV_GOTE[6][2]  = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,0}};
        const int (*dirs)[2] = (attacker_side == 0) ? G_REV_SENTE : G_REV_GOTE;
        for (int d = 0; d < 6; d++) {
            int nr = rank + dirs[d][0], nf = file + dirs[d][1];
            if (nr >= 0 && nr < 9 && nf >= 0 && nf < 9) {
                int8_t v = board[nr * 9 + nf];
                if (v != 0 && ((v > 0) == (sign > 0))) {
                    int pv = (v > 0) ? v : -v;
                    if (pv == 5 || pv == 9 || pv == 10 || pv == 11 || pv == 12)
                        { found = 1; goto done; }
                }
            }
        }
    }

    /* 6. Bishop / Horse diagonal slide */
    for (int d = 0; d < 4; d++) {
        if (slide_check(board, rank, file, BISHOP_DIRS[d][0], BISHOP_DIRS[d][1],
                        sign, 6, 13, 9))
            { found = 1; goto done; }
    }

    /* 7. Rook / Dragon orthogonal slide */
    for (int d = 0; d < 4; d++) {
        if (slide_check(board, rank, file, ROOK_DIRS[d][0], ROOK_DIRS[d][1],
                        sign, 7, 14, 9))
            { found = 1; goto done; }
    }

    /* 8. Horse (+Bishop) extra one-step orthogonal */
    for (int d = 0; d < 4; d++) {
        if (step_check(board, rank, file, ROOK_DIRS[d][0], ROOK_DIRS[d][1],
                       sign, 13, 9))
            { found = 1; goto done; }
    }

    /* 9. Dragon (+Rook) extra one-step diagonal */
    for (int d = 0; d < 4; d++) {
        if (step_check(board, rank, file, BISHOP_DIRS[d][0], BISHOP_DIRS[d][1],
                       sign, 14, 9))
            { found = 1; goto done; }
    }

    /* 10. King one-step all 8 directions */
    {
        static const int KING_DIRS[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        for (int d = 0; d < 8; d++) {
            if (step_check(board, rank, file, KING_DIRS[d][0], KING_DIRS[d][1],
                           sign, 8, 9))
                { found = 1; goto done; }
        }
    }

done:
    PyBuffer_Release(&board_buf);
    if (found) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/*
 * is_square_attacked_minishogi(board, target_sq, attacker_side)
 *
 * Reverse attack detection on a 5x5 minishogi board.
 * Pieces: 1=Pawn 2=Silver 3=Gold 4=Bishop 5=Rook 6=King
 *         7=Tokin(+Pawn) 8=+Silver 9=Horse(+Bishop) 10=Dragon(+Rook)
 * No lance, no knight.
 */
static PyObject *
accel_is_square_attacked_minishogi(PyObject *self, PyObject *args)
{
    Py_buffer board_buf;
    int target_sq, attacker_side;

    if (!PyArg_ParseTuple(args, "y*ii", &board_buf, &target_sq, &attacker_side))
        return NULL;

    if (board_buf.len < 25) {
        PyBuffer_Release(&board_buf);
        PyErr_SetString(PyExc_ValueError, "Board must have at least 25 elements");
        return NULL;
    }

    const int8_t *board = (const int8_t *)board_buf.buf;
    int rank = target_sq / 5, file = target_sq % 5;
    int sign = (attacker_side == 0) ? 1 : -1;

    static const int BISHOP_DIRS[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
    static const int ROOK_DIRS[4][2]   = {{-1,0},{1,0},{0,-1},{0,1}};

    int found = 0;

    /* 1. Pawn */
    {
        int pr = rank + (attacker_side == 0 ? 1 : -1);
        if (pr >= 0 && pr < 5 && board[pr * 5 + file] == sign * 1)
            { found = 1; goto done2; }
    }

    /* 2. Silver: 5 reverse directions */
    {
        static const int S_REV_SENTE[5][2] = {{1,1},{1,0},{1,-1},{-1,1},{-1,-1}};
        static const int S_REV_GOTE[5][2]  = {{-1,-1},{-1,0},{-1,1},{1,-1},{1,1}};
        const int (*dirs)[2] = (attacker_side == 0) ? S_REV_SENTE : S_REV_GOTE;
        for (int d = 0; d < 5; d++) {
            int nr = rank + dirs[d][0], nf = file + dirs[d][1];
            if (nr >= 0 && nr < 5 && nf >= 0 && nf < 5) {
                if (board[nr * 5 + nf] == sign * 2) { found = 1; goto done2; }
            }
        }
    }

    /* 3. Gold-like: gold(3), tokin(7), +silver(8) */
    {
        static const int G_REV_SENTE[6][2] = {{1,1},{1,0},{1,-1},{0,1},{0,-1},{-1,0}};
        static const int G_REV_GOTE[6][2]  = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,0}};
        const int (*dirs)[2] = (attacker_side == 0) ? G_REV_SENTE : G_REV_GOTE;
        for (int d = 0; d < 6; d++) {
            int nr = rank + dirs[d][0], nf = file + dirs[d][1];
            if (nr >= 0 && nr < 5 && nf >= 0 && nf < 5) {
                int8_t v = board[nr * 5 + nf];
                if (v != 0 && ((v > 0) == (sign > 0))) {
                    int pv = (v > 0) ? v : -v;
                    if (pv == 3 || pv == 7 || pv == 8)
                        { found = 1; goto done2; }
                }
            }
        }
    }

    /* 4. Bishop / Horse diagonal slide */
    for (int d = 0; d < 4; d++) {
        if (slide_check(board, rank, file, BISHOP_DIRS[d][0], BISHOP_DIRS[d][1],
                        sign, 4, 9, 5))
            { found = 1; goto done2; }
    }

    /* 5. Rook / Dragon orthogonal slide */
    for (int d = 0; d < 4; d++) {
        if (slide_check(board, rank, file, ROOK_DIRS[d][0], ROOK_DIRS[d][1],
                        sign, 5, 10, 5))
            { found = 1; goto done2; }
    }

    /* 6. Horse (+Bishop) extra one-step orthogonal */
    for (int d = 0; d < 4; d++) {
        if (step_check(board, rank, file, ROOK_DIRS[d][0], ROOK_DIRS[d][1],
                       sign, 9, 5))
            { found = 1; goto done2; }
    }

    /* 7. Dragon (+Rook) extra one-step diagonal */
    for (int d = 0; d < 4; d++) {
        if (step_check(board, rank, file, BISHOP_DIRS[d][0], BISHOP_DIRS[d][1],
                       sign, 10, 5))
            { found = 1; goto done2; }
    }

    /* 8. King one-step all 8 directions */
    {
        static const int KING_DIRS[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        for (int d = 0; d < 8; d++) {
            if (step_check(board, rank, file, KING_DIRS[d][0], KING_DIRS[d][1],
                           sign, 6, 5))
                { found = 1; goto done2; }
        }
    }

done2:
    PyBuffer_Release(&board_buf);
    if (found) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* ======================================================================== */
/* ---- Direct accumulator helpers (no PyObject boxing) ------------------- */
/* ======================================================================== */

/*
 * _accel_evaluate_direct
 * Mirrors AccelAccum_evaluate but returns float directly.
 * Handles both use_int16 and float32 modes.
 */
static float
_accel_evaluate_direct(AccelAccumObject *acc, int side_to_move)
{
    int acc_size = acc->accumulator_size;
    int l1       = acc->l1_size;
    int l2       = acc->l2_size;

    float input[512];   /* acc_size * 2, max 256*2 */
    float l1_out[64];   /* l1_size, max 64 */
    float l2_out[64];   /* l2_size, max 64 */

    if (acc->use_int16) {
        int16_t *first  = (side_to_move == 0) ? acc->white_acc_q : acc->black_acc_q;
        int16_t *second = (side_to_move == 0) ? acc->black_acc_q : acc->white_acc_q;
        neon_dequant_clipped_relu(input,            first,  acc_size, acc->inv_quant_scale);
        neon_dequant_clipped_relu(input + acc_size, second, acc_size, acc->inv_quant_scale);
    } else {
        float *first  = (side_to_move == 0) ? acc->white_acc : acc->black_acc;
        float *second = (side_to_move == 0) ? acc->black_acc : acc->white_acc;
        neon_clipped_relu_copy(input,            first,  acc_size);
        neon_clipped_relu_copy(input + acc_size, second, acc_size);
    }

    memcpy(l1_out, acc->l1_bias, l1 * sizeof(float));
    sgemv(l1, acc_size * 2, 1.0f, acc->l1_weight, acc_size * 2, input, 1.0f, l1_out);
    neon_clipped_relu_inplace(l1_out, l1);

    memcpy(l2_out, acc->l2_bias, l2 * sizeof(float));
    sgemv(l2, l1, 1.0f, acc->l2_weight, l1, l1_out, 1.0f, l2_out);
    neon_clipped_relu_inplace(l2_out, l2);

    return sdot(l2, acc->out_weight, l2_out) + acc->out_bias;
}

static inline void
_accel_push_direct(AccelAccumObject *acc)
{
    /* Silently cap at max_stack to avoid a crash; caller should never exceed */
    if (acc->stack_top >= acc->max_stack) return;

    int acc_size = acc->accumulator_size;
    if (acc->use_int16) {
        size_t offset = (size_t)acc->stack_top * 2 * acc_size;
        memcpy(acc->stack_buf_q + offset,            acc->white_acc_q, acc_size * sizeof(int16_t));
        memcpy(acc->stack_buf_q + offset + acc_size, acc->black_acc_q, acc_size * sizeof(int16_t));
    } else {
        size_t offset = (size_t)acc->stack_top * 2 * acc_size;
        memcpy(acc->stack_buf + offset,            acc->white_acc, acc_size * sizeof(float));
        memcpy(acc->stack_buf + offset + acc_size, acc->black_acc, acc_size * sizeof(float));
    }
    acc->stack_top++;
}

static inline void
_accel_pop_direct(AccelAccumObject *acc)
{
    if (acc->stack_top <= 0) return;
    acc->stack_top--;

    int acc_size = acc->accumulator_size;
    if (acc->use_int16) {
        size_t offset = (size_t)acc->stack_top * 2 * acc_size;
        memcpy(acc->white_acc_q, acc->stack_buf_q + offset,            acc_size * sizeof(int16_t));
        memcpy(acc->black_acc_q, acc->stack_buf_q + offset + acc_size, acc_size * sizeof(int16_t));
    } else {
        size_t offset = (size_t)acc->stack_top * 2 * acc_size;
        memcpy(acc->white_acc, acc->stack_buf + offset,            acc_size * sizeof(float));
        memcpy(acc->black_acc, acc->stack_buf + offset + acc_size, acc_size * sizeof(float));
    }
}

/* ======================================================================== */
/* ---- CSearch data structures ------------------------------------------- */
/* ======================================================================== */

#define CSEARCH_INF         1000000.0f
#define CSEARCH_MATE_SCORE   100000.0f
#define TT_EXACT  0
#define TT_ALPHA  1
#define TT_BETA   2

typedef struct {
    int16_t from_sq;    /* -1 for drops */
    int16_t to_sq;
    int8_t  promotion;  /* -1 for none */
    int8_t  drop_piece; /* -1 for none */
    int16_t _pad;
} CMove;

typedef struct {
    uint64_t key;
    float    score;
    CMove    best_move;
    int16_t  depth;
    int8_t   flag;      /* EXACT=0, ALPHA=1, BETA=2 */
    int8_t   _pad;
} CTTEntry;

typedef struct {
    CMove    move;
    int32_t  score;
} CScoredMove;

/* Sentinel / null move */
static inline CMove cmove_null(void) {
    CMove m;
    m.from_sq = -1; m.to_sq = -1;
    m.promotion = -1; m.drop_piece = -1;
    m._pad = 0;
    return m;
}

static inline int cmove_is_null(const CMove *m) {
    return (m->from_sq == -1 && m->to_sq == -1);
}

typedef struct {
    PyObject_HEAD

    /* Transposition table */
    CTTEntry *tt;
    uint32_t  tt_size;
    uint32_t  tt_mask;

    /* Killer moves: indexed [ply][0|1] */
    CMove     killers[128][2];

    /* History heuristic: indexed [from_sq * max_sq + to_sq] */
    int32_t  *history;   /* heap-allocated, size max_sq*max_sq */
    int       max_sq;    /* 64 for chess, 81 for shogi */

    /* Direct C references (borrowed during search) */
    AccelAccumObject *accumulator;
    PyObject         *py_feature_set;

    /* Eval scaling */
    float eval_scale;

    /* Search state */
    int   max_depth;
    float time_limit_ms;
    int   nodes_searched;
    int   time_up;
    double start_time;   /* seconds, from clock_gettime */

    /* Interned attribute/method name strings */
    PyObject *str_legal_moves;
    PyObject *str_make_move_inplace;
    PyObject *str_unmake_move;
    PyObject *str_is_terminal;
    PyObject *str_result;
    PyObject *str_side_to_move;
    PyObject *str_zobrist_hash;
    PyObject *str_is_check;
    PyObject *str_board_array;
    PyObject *str_active_features;

    /* Cached Move class */
    PyObject *Move_class;

} CSearchObject;

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

/* ======================================================================== */
/* ---- Move ordering ------------------------------------------------------ */
/* ======================================================================== */

/* MVV-LVA piece value by abs(board_val) */
static inline int piece_value(int abs_board_val) {
    switch (abs_board_val) {
        case 1: return 100;
        case 2: return 300;
        case 3: return 300;
        case 4: return 500;
        case 5: return 900;
        default: return 100;
    }
}

static int cmove_score_cmp(const void *a, const void *b) {
    const CScoredMove *sa = (const CScoredMove *)a;
    const CScoredMove *sb = (const CScoredMove *)b;
    if (sb->score > sa->score) return 1;
    if (sb->score < sa->score) return -1;
    return 0;
}

/*
 * Score all moves in py_moves[] into scored[].
 * board_array: bytes buffer (borrowed, may be NULL — skips MVV-LVA).
 */
static int
csearch_score_moves(CSearchObject *self, PyObject *py_state,
                    CScoredMove *scored, PyObject **py_moves, int n_moves,
                    int depth, CTTEntry *tt_entry)
{
    /* Get board array for MVV-LVA (optional) */
    PyObject *board_obj = PyObject_GetAttr(py_state, self->str_board_array);
    const int8_t *board = NULL;
    Py_buffer board_buf;
    int has_buf = 0;
    if (board_obj && board_obj != Py_None) {
        if (PyObject_GetBuffer(board_obj, &board_buf, PyBUF_SIMPLE) == 0) {
            board = (const int8_t *)board_buf.buf;
            has_buf = 1;
        } else {
            PyErr_Clear();
        }
    } else if (board_obj) {
        /* board_obj is Py_None */
    } else {
        PyErr_Clear();
    }
    Py_XDECREF(board_obj);

    for (int i = 0; i < n_moves; i++) {
        scored[i].move  = pyobj_to_cmove(py_moves[i]);
        scored[i].score = 0;
        CMove *m = &scored[i].move;

        /* 1. TT best move: highest priority */
        if (tt_entry && !cmove_is_null(&tt_entry->best_move)) {
            CTTEntry *te = tt_entry;
            if (te->best_move.from_sq    == m->from_sq &&
                te->best_move.to_sq      == m->to_sq   &&
                te->best_move.promotion  == m->promotion &&
                te->best_move.drop_piece == m->drop_piece) {
                scored[i].score = 2000000;
                continue;
            }
        }

        /* 2. Captures: MVV-LVA using board array */
        if (has_buf && board && m->to_sq >= 0 && m->drop_piece < 0) {
            int to = m->to_sq;
            if (to >= 0 && to < (int)board_buf.len) {
                int8_t victim = board[to];
                if (victim != 0) {
                    int vval = piece_value((victim > 0) ? victim : -victim);
                    int aval = 0;
                    if (m->from_sq >= 0 && m->from_sq < (int)board_buf.len) {
                        int8_t attacker = board[m->from_sq];
                        aval = piece_value((attacker > 0) ? attacker : -attacker);
                    }
                    scored[i].score = 1000000 + vval * 10 - aval;
                    continue;
                }
            }
        }

        /* 3. Promotions */
        if (m->promotion >= 0) {
            scored[i].score = 900000;
            continue;
        }

        /* 4. Killer moves */
        if (depth >= 0 && depth < 128) {
            if (!cmove_is_null(&self->killers[depth][0]) &&
                self->killers[depth][0].from_sq    == m->from_sq &&
                self->killers[depth][0].to_sq      == m->to_sq   &&
                self->killers[depth][0].drop_piece == m->drop_piece) {
                scored[i].score = 800000;
                continue;
            }
            if (!cmove_is_null(&self->killers[depth][1]) &&
                self->killers[depth][1].from_sq    == m->from_sq &&
                self->killers[depth][1].to_sq      == m->to_sq   &&
                self->killers[depth][1].drop_piece == m->drop_piece) {
                scored[i].score = 700000;
                continue;
            }
        }

        /* 5. History score */
        if (m->from_sq >= 0 && m->to_sq >= 0 &&
            m->from_sq < self->max_sq && m->to_sq < self->max_sq) {
            scored[i].score = (int32_t)self->history[m->from_sq * self->max_sq + m->to_sq];
        }
    }

    if (has_buf) PyBuffer_Release(&board_buf);

    qsort(scored, (size_t)n_moves, sizeof(CScoredMove), cmove_score_cmp);
    return n_moves;
}

/* ======================================================================== */
/* ---- Core alpha-beta ---------------------------------------------------- */
/* ======================================================================== */

/* Forward declaration */
static float csearch_alphabeta(CSearchObject *self, PyObject *py_state,
                                int depth, float alpha, float beta, int ply);

static float
csearch_alphabeta(CSearchObject *self, PyObject *py_state,
                  int depth, float alpha, float beta, int ply)
{
    self->nodes_searched++;

    /* Time check every 4096 nodes */
    if ((self->nodes_searched & 4095) == 0) {
        double now = csearch_now_ms();
        if (now - self->start_time >= (double)self->time_limit_ms) {
            self->time_up = 1;
        }
    }
    if (self->time_up) return 0.0f;

    /* Terminal check */
    PyObject *py_terminal = PyObject_CallMethodNoArgs(py_state, self->str_is_terminal);
    if (!py_terminal) return 0.0f;
    int is_terminal = PyObject_IsTrue(py_terminal);
    Py_DECREF(py_terminal);

    if (is_terminal) {
        /* Get result from Python: +1.0 = current player wins, -1.0 = loss, 0 = draw
         * We expect side_to_move perspective from result(). */
        PyObject *py_res = PyObject_CallMethodNoArgs(py_state, self->str_result);
        if (!py_res) return 0.0f;
        float score = (float)PyFloat_AsDouble(py_res) * CSEARCH_MATE_SCORE;
        Py_DECREF(py_res);
        return score;
    }

    /* Leaf node: evaluate */
    if (depth <= 0) {
        PyObject *py_stm = PyObject_CallMethodNoArgs(py_state, self->str_side_to_move);
        if (!py_stm) return 0.0f;
        int stm = (int)PyLong_AsLong(py_stm);
        Py_DECREF(py_stm);
        return _accel_evaluate_direct(self->accumulator, stm) * self->eval_scale;
    }

    /* TT probe */
    PyObject *py_hash = PyObject_CallMethodNoArgs(py_state, self->str_zobrist_hash);
    if (!py_hash) return 0.0f;
    uint64_t key = PyLong_AsUnsignedLongLong(py_hash);
    Py_DECREF(py_hash);
    if (key == (uint64_t)-1 && PyErr_Occurred()) { PyErr_Clear(); key = 0; }

    CTTEntry *tt_hit = csearch_tt_probe(self, key);
    if (tt_hit && tt_hit->depth >= depth) {
        if (tt_hit->flag == TT_EXACT) return tt_hit->score;
        if (tt_hit->flag == TT_ALPHA && tt_hit->score <= alpha) return alpha;
        if (tt_hit->flag == TT_BETA  && tt_hit->score >= beta)  return beta;
    }

    /* Generate legal moves */
    PyObject *py_moves_list = PyObject_CallMethodNoArgs(py_state, self->str_legal_moves);
    if (!py_moves_list) return 0.0f;

    Py_ssize_t n_moves = PyList_Size(py_moves_list);
    if (n_moves == 0) {
        Py_DECREF(py_moves_list);
        /* No moves and not terminal -- treat as draw */
        return 0.0f;
    }

    /* Build C array of Python move objects (borrowed) */
    int nm = (int)n_moves;
    PyObject **py_moves_arr = (PyObject **)malloc((size_t)nm * sizeof(PyObject *));
    if (!py_moves_arr) {
        Py_DECREF(py_moves_list);
        PyErr_NoMemory();
        return 0.0f;
    }
    for (int i = 0; i < nm; i++)
        py_moves_arr[i] = PyList_GET_ITEM(py_moves_list, i);  /* borrowed */

    /* Score and sort moves */
    CScoredMove *scored = (CScoredMove *)malloc((size_t)nm * sizeof(CScoredMove));
    if (!scored) {
        free(py_moves_arr);
        Py_DECREF(py_moves_list);
        PyErr_NoMemory();
        return 0.0f;
    }
    csearch_score_moves(self, py_state, scored, py_moves_arr, nm, ply, tt_hit);
    free(py_moves_arr);

    float orig_alpha = alpha;
    float best_score = -CSEARCH_INF;
    CMove best_move  = cmove_null();

    for (int i = 0; i < nm; i++) {
        if (self->time_up) break;

        /* Re-build Python move object for this iteration */
        PyObject *py_move = cmove_to_pyobj(self, &scored[i].move);
        if (!py_move) {
            /* Fallback: use original list item */
            /* Find the matching list item by index -- scored is sorted so we
               need to match by content. For robustness, just skip on error. */
            PyErr_Clear();
            continue;
        }

        PyObject *undo = csearch_make_move(self, py_state, py_move);
        Py_DECREF(py_move);
        if (!undo) {
            PyErr_Clear();
            continue;
        }

        float score = -csearch_alphabeta(self, py_state, depth - 1, -beta, -alpha, ply + 1);

        csearch_unmake_move(self, py_state, undo);
        Py_DECREF(undo);

        if (self->time_up) break;

        if (score > best_score) {
            best_score = score;
            best_move  = scored[i].move;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            /* Beta cutoff: update killers and history */
            if (ply >= 0 && ply < 128) {
                CMove *km = &scored[i].move;
                /* Only quiet moves as killers */
                if (km->drop_piece < 0) {
                    if (!(self->killers[ply][0].from_sq == km->from_sq &&
                          self->killers[ply][0].to_sq   == km->to_sq)) {
                        self->killers[ply][1] = self->killers[ply][0];
                        self->killers[ply][0] = *km;
                    }
                }
            }
            /* History update */
            if (scored[i].move.from_sq >= 0 &&
                scored[i].move.to_sq   >= 0 &&
                scored[i].move.from_sq < self->max_sq &&
                scored[i].move.to_sq   < self->max_sq) {
                self->history[scored[i].move.from_sq * self->max_sq
                              + scored[i].move.to_sq] += depth * depth;
            }
            break;
        }
    }

    free(scored);
    Py_DECREF(py_moves_list);

    if (!self->time_up) {
        int8_t flag;
        if (best_score <= orig_alpha)     flag = TT_ALPHA;
        else if (best_score >= beta)      flag = TT_BETA;
        else                              flag = TT_EXACT;
        csearch_tt_store(self, key, depth, best_score, flag, &best_move);
    }

    return best_score;
}

/*
 * csearch_search_root
 * Iterative deepening root. Returns best score, writes best move to *out_best.
 */
static float
csearch_search_root(CSearchObject *self, PyObject *py_state,
                    int max_depth, CMove *out_best)
{
    *out_best = cmove_null();
    float best_score = -CSEARCH_INF;

    for (int depth = 1; depth <= max_depth; depth++) {
        if (self->time_up) break;

        /* Reset per-search state (keep TT, killers, history across iterations) */
        /* Clear killers for this depth iteration */
        memset(self->killers, 0xFF, sizeof(self->killers)); /* -1 everywhere */

        PyObject *py_moves_list = PyObject_CallMethodNoArgs(py_state, self->str_legal_moves);
        if (!py_moves_list) break;

        Py_ssize_t n_moves = PyList_Size(py_moves_list);
        if (n_moves == 0) { Py_DECREF(py_moves_list); break; }

        int nm = (int)n_moves;
        PyObject **py_moves_arr = (PyObject **)malloc((size_t)nm * sizeof(PyObject *));
        if (!py_moves_arr) { Py_DECREF(py_moves_list); break; }
        for (int i = 0; i < nm; i++)
            py_moves_arr[i] = PyList_GET_ITEM(py_moves_list, i);

        /* Probe TT for the root position */
        PyObject *py_hash = PyObject_CallMethodNoArgs(py_state, self->str_zobrist_hash);
        uint64_t root_key = 0;
        if (py_hash) {
            root_key = PyLong_AsUnsignedLongLong(py_hash);
            Py_DECREF(py_hash);
            if (root_key == (uint64_t)-1 && PyErr_Occurred()) { PyErr_Clear(); root_key = 0; }
        }
        CTTEntry *tt_root = csearch_tt_probe(self, root_key);

        CScoredMove *scored = (CScoredMove *)malloc((size_t)nm * sizeof(CScoredMove));
        if (!scored) { free(py_moves_arr); Py_DECREF(py_moves_list); break; }
        csearch_score_moves(self, py_state, scored, py_moves_arr, nm, 0, tt_root);
        free(py_moves_arr);

        float iter_best = -CSEARCH_INF;
        CMove iter_move = cmove_null();
        float alpha = -CSEARCH_INF;
        float beta  =  CSEARCH_INF;

        for (int i = 0; i < nm; i++) {
            if (self->time_up) break;

            PyObject *py_move = cmove_to_pyobj(self, &scored[i].move);
            if (!py_move) { PyErr_Clear(); continue; }

            PyObject *undo = csearch_make_move(self, py_state, py_move);
            Py_DECREF(py_move);
            if (!undo) { PyErr_Clear(); continue; }

            float score = -csearch_alphabeta(self, py_state, depth - 1, -beta, -alpha, 1);

            csearch_unmake_move(self, py_state, undo);
            Py_DECREF(undo);

            if (self->time_up) break;

            if (score > iter_best) {
                iter_best = score;
                iter_move = scored[i].move;
                if (score > alpha) alpha = score;
            }
        }

        free(scored);
        Py_DECREF(py_moves_list);

        if (!self->time_up && !cmove_is_null(&iter_move)) {
            best_score = iter_best;
            *out_best  = iter_move;
            /* Store root result in TT */
            csearch_tt_store(self, root_key, depth, best_score, TT_EXACT, out_best);
        }
    }

    return best_score;
}

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
                                         -CSEARCH_INF, CSEARCH_INF, 1);
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

static PyMethodDef CSearch_methods[] = {
    {"search", (PyCFunction)CSearch_search, METH_VARARGS,
     "search(game_state, max_depth, time_limit_ms=1000.0) -> ((from_sq, to_sq, promo, drop), score, nodes) or None"},
    {"search_top_n", (PyCFunction)CSearch_search_top_n, METH_VARARGS,
     "search_top_n(game_state, n, max_depth, time_limit_ms=1000.0) -> list of ((from_sq, to_sq, promo, drop), score)"},
    {"clear_tt", (PyCFunction)CSearch_clear_tt, METH_NOARGS,
     "Clear the transposition table."},
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

/* ======================================================================== */
/* ---- End of CSearch ----------------------------------------------------- */
/* ======================================================================== */

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
