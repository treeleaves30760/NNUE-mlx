static PyMethodDef module_methods[] = {
    {"halfkp_active_features", accel_halfkp_active_features, METH_VARARGS,
     "Compute HalfKP active feature indices from board array (C accelerated)."},
    {"halfkp_shogi_active_features", accel_halfkp_shogi_active_features, METH_VARARGS,
     "Compute HalfKPShogi active feature indices from board + hand arrays (C accelerated)."},
    {"is_square_attacked_shogi", accel_is_square_attacked_shogi, METH_VARARGS,
     "Check if a square is attacked on a 9x9 shogi board (C accelerated)."},
    {"is_square_attacked_minishogi", accel_is_square_attacked_minishogi, METH_VARARGS,
     "Check if a square is attacked on a 5x5 minishogi board (C accelerated)."},
    {"shogi_c_legal_moves", accel_shogi_c_legal_moves, METH_VARARGS,
     "Generate legal shogi moves entirely in C. "
     "Args: (board bytes, sente_hand seq, gote_hand seq, side). "
     "Returns list of (from, to, promo, drop) tuples."},
    {"shogi_c_make_move", accel_shogi_c_make_move, METH_VARARGS,
     "Apply a shogi move in C and return the new position. "
     "Args: (board, sente_hand, gote_hand, side, move_tuple). "
     "Returns (new_board_bytes, new_sente_hand, new_gote_hand, new_side, new_hash)."},
    {"shogi_c_perft", accel_shogi_c_perft, METH_VARARGS,
     "Perft counter on a shogi position. "
     "Args: (board, sente_hand, gote_hand, side, depth). "
     "Returns the number of leaf nodes."},
    {"shogi_c_startpos", accel_shogi_c_startpos, METH_NOARGS,
     "Construct the shogi starting position in C. "
     "Returns (board_bytes, sente_hand, gote_hand, side, hash)."},
    {"shogi_c_evaluate", accel_shogi_c_evaluate, METH_VARARGS,
     "Run the shogi rule-based evaluator in C. "
     "Args: (board, sente_hand, gote_hand, side). "
     "Returns score (float, centipawn, side-to-move perspective)."},
    {"shogi_rule_search", accel_shogi_rule_search, METH_VARARGS,
     "Self-contained alpha-beta shogi rule-based search (pure C). "
     "Args: (board, sente_hand, gote_hand, side, max_depth, [time_ms]). "
     "Returns ((from, to, promo, drop), score, nodes) or None."},
    {"shogi_rule_search_reset", accel_shogi_rule_search_reset, METH_VARARGS,
     "Clear the shogi rule-based search TT + killers + history. "
     "Call between unrelated positions (e.g. between self-play games) "
     "so stale entries don't pollute new searches. Within a single "
     "game, leave the TT warm for move-to-move tree reuse. "
     "Optional arg node_limit (long): when > 0, every following search "
     "call aborts after visiting this many nodes. Pass 0 for unlimited."},
    {"shogi_rule_search_live", accel_shogi_rule_search_live, METH_VARARGS,
     "Shogi rule-based search with a live progress callback. "
     "Args: (board, sente_hand, gote_hand, side, max_depth, time_ms, "
     "callback, [cb_interval=1000000]). "
     "callback(completed_depth, max_depth, [(move, score), ...], done) "
     "is called every cb_interval nodes and after each iteration. "
     "Returns truthy from the callback to abort the search."},
    {"chess_c_evaluate", accel_chess_c_evaluate, METH_VARARGS,
     "Run the chess rule-based evaluator in C. "
     "Args: (board bytes, side, white_king_sq, black_king_sq). "
     "Returns score (float, centipawn, side-to-move perspective). "
     "Bit-identical to _chess_rule_based in src/search/evaluator.py."},
    {"chess_c_legal_moves", accel_chess_c_legal_moves, METH_VARARGS,
     "Generate legal chess moves in C. "
     "Args: (board bytes, side, castling, ep_sq, halfmove, wk_sq, bk_sq). "
     "Returns list of (from_sq, to_sq, promotion)."},
    {"chess_c_perft", accel_chess_c_perft, METH_VARARGS,
     "Perft count for a chess position. "
     "Args: (board bytes, side, castling, ep_sq, halfmove, wk_sq, bk_sq, depth). "
     "Returns leaf-node count."},
    {"chess_c_perft_divide", accel_chess_c_perft_divide, METH_VARARGS,
     "Perft divide: for each legal root move return (move, sub_perft). "
     "Args: same as chess_c_perft. Depth-1 child counts."},
    {"chess_c_apply_moves", accel_chess_c_apply_moves, METH_VARARGS,
     "Debug helper: apply a list of (from,to,promo) moves via C "
     "make_move and return the resulting (board, side, castling, "
     "ep, halfmove, wk_sq, bk_sq, hash) tuple."},
    {"chess_c_rule_search", accel_chess_c_rule_search, METH_VARARGS,
     "Self-contained alpha-beta chess rule-based search (pure C). "
     "Args: (board, side, castling, ep_sq, halfmove, wk_sq, bk_sq, "
     "history_bytes, max_depth, time_limit_ms). "
     "Returns ((from, to, promo_base_or_None), score, nodes) or None."},
    {"chess_c_compute_hash", accel_chess_c_compute_hash, METH_VARARGS,
     "Compute the C engine's Zobrist hash for a position. "
     "Args: (board bytes, side, castling, ep_sq, wk_sq, bk_sq). "
     "Returns uint64."},
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

extern void shogi_zobrist_init(void);

PyMODINIT_FUNC
PyInit__nnue_accel(void)
{
    if (PyType_Ready(&AccelAccumType) < 0)
        return NULL;
    if (PyType_Ready(&CSearchType) < 0)
        return NULL;

    /* Initialise shogi Zobrist tables once. */
    shogi_zobrist_init();
    /* Initialise chess runtime tables (attack bitmasks + Zobrist). */
    chess_init_tables();

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
