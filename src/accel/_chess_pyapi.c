/* ========================================================================
 * Python entry points for the chess C engine.
 *
 *   chess_c_perft(board_bytes, side, castling, ep_sq, halfmove,
 *                 wk_sq, bk_sq, depth) -> int
 *   chess_c_legal_moves(board_bytes, side, castling, ep_sq, halfmove,
 *                       wk_sq, bk_sq) -> list[(from, to, promo)]
 *   chess_c_compute_hash(board_bytes, side, castling, ep_sq,
 *                        wk_sq, bk_sq) -> int
 *   chess_c_rule_search(board_bytes, side, castling, ep_sq, halfmove,
 *                       wk_sq, bk_sq, history_bytes, max_depth, time_ms)
 *     -> ((from, to, promo), score, nodes)
 *
 * `history_bytes` is a bytes object holding an array of uint64 hashes —
 * one per position on the current game's path. Used by the search to
 * detect three-fold repetition without state blowup.
 * ======================================================================= */

#include "_nnue_accel_header.h"
#include "_chess_position.h"

/* The chess rule-search Python wrapper (accel_chess_c_rule_search)
 * lives in _chess_rule_search.c so it can access the CRSContext
 * internals without exporting them. It is included into the same
 * translation unit via _nnue_accel.c. */

/* Parse the 7 shared args (board, side, castling, ep_sq, halfmove,
 * wk_sq, bk_sq) into a ChessPosition and compute its hash. Returns 0
 * on success, -1 on error. Caller must already have set the parse
 * spec accordingly. */
static int cp_pyapi_parse_position(PyObject *board_obj, int side,
                                    int castling, int ep_sq, int halfmove,
                                    int wk_sq, int bk_sq,
                                    ChessPosition *out) {
    Py_buffer buf;
    if (PyObject_GetBuffer(board_obj, &buf, PyBUF_SIMPLE) != 0) return -1;
    if (buf.len < CHESS_SQUARES) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "chess board must have >= 64 bytes");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    memcpy(out->board, buf.buf, CHESS_SQUARES);
    PyBuffer_Release(&buf);
    out->side = (int8_t)(side & 1);
    out->castling = (int8_t)(castling & 0xF);
    out->ep_square = (int16_t)ep_sq;
    out->halfmove = (int16_t)halfmove;
    out->white_king_sq = (int16_t)wk_sq;
    out->black_king_sq = (int16_t)bk_sq;
    out->hash = chess_compute_hash(out);
    return 0;
}

/* ---- chess_c_compute_hash ------------------------------------------- */

static PyObject *accel_chess_c_compute_hash(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj;
    int side, castling, ep_sq, wk_sq, bk_sq;
    if (!PyArg_ParseTuple(args, "Oiiiii", &board_obj, &side, &castling,
                          &ep_sq, &wk_sq, &bk_sq)) {
        return NULL;
    }
    chess_init_tables();
    ChessPosition p;
    if (cp_pyapi_parse_position(board_obj, side, castling, ep_sq, 0,
                                 wk_sq, bk_sq, &p) != 0) {
        return NULL;
    }
    return PyLong_FromUnsignedLongLong((unsigned long long)p.hash);
}

/* ---- chess_c_legal_moves -------------------------------------------- */

static PyObject *accel_chess_c_legal_moves(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj;
    int side, castling, ep_sq, halfmove, wk_sq, bk_sq;
    if (!PyArg_ParseTuple(args, "Oiiiiii", &board_obj, &side, &castling,
                          &ep_sq, &halfmove, &wk_sq, &bk_sq)) {
        return NULL;
    }
    chess_init_tables();
    ChessPosition p;
    if (cp_pyapi_parse_position(board_obj, side, castling, ep_sq, halfmove,
                                 wk_sq, bk_sq, &p) != 0) {
        return NULL;
    }
    ChMove moves[CHESS_MAX_MOVES];
    int n = chess_expand_legal_moves(&p, moves);
    PyObject *out = PyList_New(n);
    if (!out) return NULL;
    for (int i = 0; i < n; i++) {
        ChMove m = moves[i];
        int promo_base;
        /* Map internal CP_QUEEN=5, CP_ROOK=4, CP_BISHOP=3, CP_KNIGHT=2
         * to base-interface piece type (QUEEN=4, ROOK=3, BISHOP=2,
         * KNIGHT=1) to match the Python Move.promotion field. */
        int p_int = chmove_promo(m);
        switch (p_int) {
            case CP_QUEEN:  promo_base = 4; break;
            case CP_ROOK:   promo_base = 3; break;
            case CP_BISHOP: promo_base = 2; break;
            case CP_KNIGHT: promo_base = 1; break;
            default:        promo_base = -1; break;
        }
        PyObject *promo_obj;
        if (promo_base < 0) {
            Py_INCREF(Py_None);
            promo_obj = Py_None;
        } else {
            promo_obj = PyLong_FromLong(promo_base);
        }
        PyObject *tup = Py_BuildValue("(iiO)",
            chmove_from(m), chmove_to(m), promo_obj);
        Py_DECREF(promo_obj);
        if (!tup) { Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, i, tup);
    }
    return out;
}

/* ---- chess_c_apply_moves (debug helper) -------------------------- *
 *
 * Apply a sequence of (from, to, promo_or_none) moves via C make_move
 * and return the resulting position state. Used to verify that C
 * make_move produces the same state Python make_move does.
 */

static PyObject *accel_chess_c_apply_moves(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj;
    int side, castling, ep_sq, halfmove, wk_sq, bk_sq;
    PyObject *moves_list;
    if (!PyArg_ParseTuple(args, "OiiiiiiO", &board_obj, &side, &castling,
                          &ep_sq, &halfmove, &wk_sq, &bk_sq, &moves_list)) {
        return NULL;
    }
    chess_init_tables();
    ChessPosition p;
    if (cp_pyapi_parse_position(board_obj, side, castling, ep_sq, halfmove,
                                 wk_sq, bk_sq, &p) != 0) {
        return NULL;
    }

    Py_ssize_t n = PyList_Size(moves_list);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *mv_tup = PyList_GetItem(moves_list, i);
        int from, to;
        PyObject *promo_obj;
        if (!PyArg_ParseTuple(mv_tup, "iiO", &from, &to, &promo_obj)) {
            return NULL;
        }
        /* Find the matching legal move (respecting flags) */
        ChMove legal[CHESS_MAX_MOVES];
        int nl = chess_expand_legal_moves(&p, legal);
        int chosen = -1;
        int target_promo = 0;
        if (promo_obj != Py_None) {
            long base = PyLong_AsLong(promo_obj);
            switch (base) {
                case 4: target_promo = CP_QUEEN; break;
                case 3: target_promo = CP_ROOK; break;
                case 2: target_promo = CP_BISHOP; break;
                case 1: target_promo = CP_KNIGHT; break;
            }
        }
        for (int j = 0; j < nl; j++) {
            if (chmove_from(legal[j]) == from &&
                chmove_to(legal[j]) == to &&
                chmove_promo(legal[j]) == target_promo) {
                chosen = j;
                break;
            }
        }
        if (chosen < 0) {
            PyErr_SetString(PyExc_ValueError, "move not legal in C state");
            return NULL;
        }
        ChessUndo u;
        chess_make_move(&p, legal[chosen], &u);
    }

    /* Return (board_bytes, side, castling, ep, halfmove, wk, bk, hash) */
    PyObject *board_out = PyBytes_FromStringAndSize((char *)p.board, CHESS_SQUARES);
    if (!board_out) return NULL;
    PyObject *res = Py_BuildValue("(OiiiiiiK)",
        board_out, (int)p.side, (int)p.castling, (int)p.ep_square,
        (int)p.halfmove, (int)p.white_king_sq, (int)p.black_king_sq,
        (unsigned long long)p.hash);
    Py_DECREF(board_out);
    return res;
}


/* ---- chess_c_perft_divide (debug helper) -------------------------- *
 *
 * For each legal root move, apply it and return (move, perft(depth-1))
 * so we can isolate which child subtree disagrees with Python's perft.
 */

static PyObject *accel_chess_c_perft_divide(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj;
    int side, castling, ep_sq, halfmove, wk_sq, bk_sq, depth;
    if (!PyArg_ParseTuple(args, "Oiiiiiii", &board_obj, &side, &castling,
                          &ep_sq, &halfmove, &wk_sq, &bk_sq, &depth)) {
        return NULL;
    }
    chess_init_tables();
    ChessPosition p;
    if (cp_pyapi_parse_position(board_obj, side, castling, ep_sq, halfmove,
                                 wk_sq, bk_sq, &p) != 0) {
        return NULL;
    }
    ChMove moves[CHESS_MAX_MOVES];
    int n = chess_expand_legal_moves(&p, moves);
    PyObject *out = PyList_New(n);
    if (!out) return NULL;
    for (int i = 0; i < n; i++) {
        ChessUndo u;
        if (chess_make_move(&p, moves[i], &u) != 0) {
            Py_DECREF(out);
            PyErr_SetString(PyExc_RuntimeError, "make_move failed");
            return NULL;
        }
        uint64_t sub = chess_perft(&p, depth - 1);
        chess_unmake_move(&p, &u);
        ChMove m = moves[i];
        int promo_internal = chmove_promo(m);
        int promo_base;
        switch (promo_internal) {
            case CP_QUEEN:  promo_base = 4; break;
            case CP_ROOK:   promo_base = 3; break;
            case CP_BISHOP: promo_base = 2; break;
            case CP_KNIGHT: promo_base = 1; break;
            default:        promo_base = -1; break;
        }
        PyObject *promo_obj;
        if (promo_base < 0) {
            Py_INCREF(Py_None);
            promo_obj = Py_None;
        } else {
            promo_obj = PyLong_FromLong(promo_base);
        }
        PyObject *tup = Py_BuildValue("((iiO)K)",
            chmove_from(m), chmove_to(m), promo_obj,
            (unsigned long long)sub);
        Py_DECREF(promo_obj);
        if (!tup) { Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, i, tup);
    }
    return out;
}

/* ---- chess_c_perft -------------------------------------------------- */

static PyObject *accel_chess_c_perft(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj;
    int side, castling, ep_sq, halfmove, wk_sq, bk_sq, depth;
    if (!PyArg_ParseTuple(args, "Oiiiiiii", &board_obj, &side, &castling,
                          &ep_sq, &halfmove, &wk_sq, &bk_sq, &depth)) {
        return NULL;
    }
    chess_init_tables();
    ChessPosition p;
    if (cp_pyapi_parse_position(board_obj, side, castling, ep_sq, halfmove,
                                 wk_sq, bk_sq, &p) != 0) {
        return NULL;
    }
    uint64_t n = chess_perft(&p, depth);
    return PyLong_FromUnsignedLongLong((unsigned long long)n);
}
