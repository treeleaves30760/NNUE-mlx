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

    float input[1024];  /* acc_size * 2, max 512*2 = 1024 */
    float l1_out[256];  /* l1_size, max 256 */
    float l2_out[256];  /* l2_size, max 256 */

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
    neon_screlu_inplace(l1_out, l1);

    memcpy(l2_out, acc->l2_bias, l2 * sizeof(float));
    sgemv(l2, l1, 1.0f, acc->l2_weight, l1, l1_out, 1.0f, l2_out);
    neon_screlu_inplace(l2_out, l2);

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

/* Direct refresh for a single perspective (used by null move in C search) */
static inline void
_accel_refresh_perspective_direct(AccelAccumObject *acc, int perspective,
                                   const int *indices, int count)
{
    int acc_size = acc->accumulator_size;
    if (acc->use_int16) {
        int16_t *a = (perspective == 0) ? acc->white_acc_q : acc->black_acc_q;
        memcpy(a, acc->ft_bias_q, acc_size * sizeof(int16_t));
        for (int i = 0; i < count; i++) {
            int idx = indices[i];
            if (idx >= 0 && idx < acc->num_features)
                neon_vec_add_i16(a, acc->ft_weight_q + (size_t)idx * acc_size, acc_size);
        }
    } else {
        float *a = (perspective == 0) ? acc->white_acc : acc->black_acc;
        memcpy(a, acc->ft_bias, acc_size * sizeof(float));
        for (int i = 0; i < count; i++) {
            int idx = indices[i];
            if (idx >= 0 && idx < acc->num_features)
                neon_vec_add(a, acc->ft_weight + (size_t)idx * acc_size, acc_size);
        }
    }
}
