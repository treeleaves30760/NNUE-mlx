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
