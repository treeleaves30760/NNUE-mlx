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
