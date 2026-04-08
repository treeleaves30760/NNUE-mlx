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

/* In-place SCReLU: clamp(x, 0, 1)^2 for n floats (n must be multiple of 4). */
static inline void neon_screlu_inplace(float *data, int n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t one  = vdupq_n_f32(1.0f);
    for (int i = 0; i < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        v = vmaxq_f32(v, zero);
        v = vminq_f32(v, one);
        v = vmulq_f32(v, v);  /* square */
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

static inline void neon_screlu_inplace(float *data, int n) {
    for (int i = 0; i < n; i++) {
        float v = data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        data[i] = v * v;
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
