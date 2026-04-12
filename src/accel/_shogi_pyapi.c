/* ========================================================================
 * Python-facing wrappers for the C shogi movegen / make-unmake layer.
 *
 * These are intended for correctness validation (vs the Python reference in
 * src/games/shogi/movegen.py) and for the bootstrap hot path.
 *
 * Input convention for all entry points:
 *   board        : readable bytes-like, len >= 81 (int8 values)
 *   sente_hand   : 7-tuple or sequence of ints (Pawn..Rook counts)
 *   gote_hand    : 7-tuple or sequence of ints
 *   side         : 0 (sente) or 1 (gote)
 * ======================================================================= */

#include "_nnue_accel_header.h"
#include "_shogi_position.h"

/* Parse a generic sequence of length-7 into an int8_t hand array. */
static int sh_pyapi_parse_hand(PyObject *obj, int8_t out[SHOGI_HAND_TYPES]) {
    if (!obj || obj == Py_None) {
        memset(out, 0, SHOGI_HAND_TYPES);
        return 0;
    }
    PyObject *seq = PySequence_Fast(obj, "hand must be a sequence");
    if (!seq) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n != SHOGI_HAND_TYPES) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "hand must have length 7");
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        long v = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (v < 0) v = 0;
        if (v > 38) v = 38;
        out[i] = (int8_t)v;
    }
    Py_DECREF(seq);
    return 0;
}

/* Build a ShogiPosition from Python inputs. Returns 0 on success. */
static int sh_pyapi_parse_position(PyObject *board_obj,
                                    PyObject *sente_hand_obj,
                                    PyObject *gote_hand_obj,
                                    int side,
                                    ShogiPosition *out) {
    Py_buffer buf;
    if (PyObject_GetBuffer(board_obj, &buf, PyBUF_SIMPLE) != 0) return -1;
    if (buf.len < SHOGI_SQUARES) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "board must have >=81 bytes");
        return -1;
    }
    memcpy(out->board, buf.buf, SHOGI_SQUARES);
    PyBuffer_Release(&buf);

    if (sh_pyapi_parse_hand(sente_hand_obj, out->sente_hand) != 0) return -1;
    if (sh_pyapi_parse_hand(gote_hand_obj,  out->gote_hand)  != 0) return -1;
    out->side = (int8_t)(side & 1);
    out->_pad = 0;

    /* Locate kings */
    out->sente_king_sq = -1;
    out->gote_king_sq  = -1;
    for (int sq = 0; sq < SHOGI_SQUARES; sq++) {
        int8_t v = out->board[sq];
        if (v == SP_KING)      out->sente_king_sq = (int16_t)sq;
        else if (v == -SP_KING) out->gote_king_sq = (int16_t)sq;
    }
    out->hash = shogi_compute_hash(out);
    return 0;
}

/* Convert an SMove to a (from, to, promo, drop) Python tuple.
 * Promotion: 0 becomes Python None (no promotion).
 * Drop: 0xF becomes Python None (not a drop). */
static PyObject *sh_pyapi_move_to_tuple(SMove m) {
    int from = smove_from(m);
    int to   = smove_to(m);
    int promo = smove_promo(m);
    int drop  = smove_drop(m);
    PyObject *t = PyTuple_New(4);
    if (!t) return NULL;
    if (from == SMOVE_NONE_FROM) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(t, 0, Py_None);
    } else {
        PyTuple_SET_ITEM(t, 0, PyLong_FromLong(from));
    }
    PyTuple_SET_ITEM(t, 1, PyLong_FromLong(to));
    if (promo == 0) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(t, 2, Py_None);
    } else {
        PyTuple_SET_ITEM(t, 2, PyLong_FromLong(promo));
    }
    if (drop == SMOVE_NONE_DROP) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(t, 3, Py_None);
    } else {
        PyTuple_SET_ITEM(t, 3, PyLong_FromLong(drop));
    }
    return t;
}

/* Parse a Python (from, to, promo, drop) tuple into an SMove. */
static int sh_pyapi_tuple_to_move(PyObject *obj, SMove *out) {
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) != 4) {
        PyErr_SetString(PyExc_TypeError, "move must be a 4-tuple");
        return -1;
    }
    PyObject *p0 = PyTuple_GET_ITEM(obj, 0);
    PyObject *p1 = PyTuple_GET_ITEM(obj, 1);
    PyObject *p2 = PyTuple_GET_ITEM(obj, 2);
    PyObject *p3 = PyTuple_GET_ITEM(obj, 3);
    int from = (p0 == Py_None) ? SMOVE_NONE_FROM : (int)PyLong_AsLong(p0);
    int to   = (int)PyLong_AsLong(p1);
    int promo = (p2 == Py_None) ? 0 : (int)PyLong_AsLong(p2);
    int drop  = (p3 == Py_None) ? SMOVE_NONE_DROP : (int)PyLong_AsLong(p3);
    *out = smove_encode(from, to, promo, drop);
    return 0;
}

/* ------------------------------------------------------------------------
 *  shogi_c_legal_moves(board, sente_hand, gote_hand, side) -> list of moves
 * ----------------------------------------------------------------------- */
static PyObject *accel_shogi_c_legal_moves(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj, *sh_obj, *gh_obj;
    int side;
    if (!PyArg_ParseTuple(args, "OOOi", &board_obj, &sh_obj, &gh_obj, &side))
        return NULL;

    ShogiPosition p;
    if (sh_pyapi_parse_position(board_obj, sh_obj, gh_obj, side, &p) != 0)
        return NULL;

    SMove moves[SHOGI_MAX_MOVES];
    int n = shogi_expand_legal_moves(&p, moves);

    PyObject *out = PyList_New(n);
    if (!out) return NULL;
    for (int i = 0; i < n; i++) {
        PyObject *t = sh_pyapi_move_to_tuple(moves[i]);
        if (!t) { Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, i, t);
    }
    return out;
}

/* ------------------------------------------------------------------------
 *  shogi_c_make_move(board, sente_hand, gote_hand, side, move_tuple)
 *  -> (new_board_bytes, new_sente_hand, new_gote_hand, new_side, new_hash)
 *
 *  For testing make/unmake roundtrips without exposing a stateful position.
 * ----------------------------------------------------------------------- */
static PyObject *accel_shogi_c_make_move(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj, *sh_obj, *gh_obj, *move_obj;
    int side;
    if (!PyArg_ParseTuple(args, "OOOiO", &board_obj, &sh_obj, &gh_obj, &side, &move_obj))
        return NULL;

    ShogiPosition p;
    if (sh_pyapi_parse_position(board_obj, sh_obj, gh_obj, side, &p) != 0)
        return NULL;

    SMove m;
    if (sh_pyapi_tuple_to_move(move_obj, &m) != 0) return NULL;

    ShogiUndo u;
    if (shogi_make_move(&p, m, &u) != 0) {
        PyErr_SetString(PyExc_ValueError, "illegal move in make_move");
        return NULL;
    }

    PyObject *new_board = PyBytes_FromStringAndSize((const char *)p.board, SHOGI_SQUARES);
    PyObject *new_sh = PyTuple_New(SHOGI_HAND_TYPES);
    PyObject *new_gh = PyTuple_New(SHOGI_HAND_TYPES);
    for (int i = 0; i < SHOGI_HAND_TYPES; i++) {
        PyTuple_SET_ITEM(new_sh, i, PyLong_FromLong(p.sente_hand[i]));
        PyTuple_SET_ITEM(new_gh, i, PyLong_FromLong(p.gote_hand[i]));
    }
    PyObject *result = PyTuple_Pack(5,
        new_board,
        new_sh,
        new_gh,
        PyLong_FromLong(p.side),
        PyLong_FromUnsignedLongLong((unsigned long long)p.hash));
    Py_DECREF(new_board);
    Py_DECREF(new_sh);
    Py_DECREF(new_gh);
    return result;
}

/* ------------------------------------------------------------------------
 *  shogi_c_perft(board, sente_hand, gote_hand, side, depth) -> int
 *
 *  Standard perft counter. Used to sanity-check the C movegen against the
 *  Python reference: both should report identical node counts.
 * ----------------------------------------------------------------------- */
static uint64_t shogi_perft_rec(ShogiPosition *p, int depth) {
    if (depth == 0) return 1ull;
    SMove moves[SHOGI_MAX_MOVES];
    int n = shogi_expand_legal_moves(p, moves);
    if (depth == 1) return (uint64_t)n;
    uint64_t total = 0;
    for (int i = 0; i < n; i++) {
        ShogiUndo u;
        if (shogi_make_move(p, moves[i], &u) != 0) continue;
        total += shogi_perft_rec(p, depth - 1);
        shogi_unmake_move(p, &u);
    }
    return total;
}

static PyObject *accel_shogi_c_perft(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *board_obj, *sh_obj, *gh_obj;
    int side, depth;
    if (!PyArg_ParseTuple(args, "OOOii", &board_obj, &sh_obj, &gh_obj, &side, &depth))
        return NULL;

    ShogiPosition p;
    if (sh_pyapi_parse_position(board_obj, sh_obj, gh_obj, side, &p) != 0)
        return NULL;

    if (depth < 0 || depth > 6) {
        PyErr_SetString(PyExc_ValueError, "perft depth must be in [0, 6]");
        return NULL;
    }
    uint64_t n = shogi_perft_rec(&p, depth);
    return PyLong_FromUnsignedLongLong((unsigned long long)n);
}

/* ------------------------------------------------------------------------
 *  shogi_c_startpos() -> (board_bytes, sente_hand, gote_hand, side, hash)
 *
 *  Construct the starting position in C and hand it back. Mainly useful for
 *  tests and debugging.
 * ----------------------------------------------------------------------- */
static PyObject *accel_shogi_c_startpos(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    ShogiPosition p;
    shogi_position_init_startpos(&p);
    PyObject *board = PyBytes_FromStringAndSize((const char *)p.board, SHOGI_SQUARES);
    PyObject *sh = PyTuple_New(SHOGI_HAND_TYPES);
    PyObject *gh = PyTuple_New(SHOGI_HAND_TYPES);
    for (int i = 0; i < SHOGI_HAND_TYPES; i++) {
        PyTuple_SET_ITEM(sh, i, PyLong_FromLong(p.sente_hand[i]));
        PyTuple_SET_ITEM(gh, i, PyLong_FromLong(p.gote_hand[i]));
    }
    PyObject *result = PyTuple_Pack(5,
        board, sh, gh,
        PyLong_FromLong(p.side),
        PyLong_FromUnsignedLongLong((unsigned long long)p.hash));
    Py_DECREF(board); Py_DECREF(sh); Py_DECREF(gh);
    return result;
}
