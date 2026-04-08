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
