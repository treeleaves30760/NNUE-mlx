/* ========================================================================
 * Chess make_move / unmake_move with incremental Zobrist updates.
 *
 * Mirrors src/games/chess/state.py::make_move and the hash_utils.py
 * helpers, but works in-place on a ChessPosition and records everything
 * needed for an exact reversal in a ChessUndo frame.
 *
 * All moves consumed here must come from chess_expand_legal_moves so
 * we don't re-validate; we do trust the flags to tell us whether the
 * move is a castle, ep, promotion, etc.
 * ======================================================================= */

#include "_chess_position.h"

/* ---- Castling-rook update table ------------------------------------- *
 *
 * When a piece enters or leaves one of the four rook home squares, the
 * corresponding castling right is gone forever. Same for the four king
 * home squares. Masks here are what we AND into the castling bitmap.
 */
static const uint8_t CR_SQUARE_MASK[64] = {
    /* a1=0  .. h1=7 */
    (uint8_t)~CR_WQ, 0xFF, 0xFF, 0xFF,
    (uint8_t)~(CR_WK | CR_WQ), 0xFF, 0xFF, (uint8_t)~CR_WK,
    /* a2..h2 */
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* a3..h3 */
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* a4..h4 */
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* a5..h5 */
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* a6..h6 */
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* a7..h7 */
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* a8=56 .. h8=63 */
    (uint8_t)~CR_BQ, 0xFF, 0xFF, 0xFF,
    (uint8_t)~(CR_BK | CR_BQ), 0xFF, 0xFF, (uint8_t)~CR_BK,
};

/* ---- Compute hash from scratch -------------------------------------- */

uint64_t chess_compute_hash(const ChessPosition *p) {
    uint64_t h = 0;
    for (int sq = 0; sq < 64; sq++) {
        int8_t v = p->board[sq];
        if (v == 0) continue;
        int abs_v = (v > 0) ? v : -v;
        int color = (v > 0) ? 0 : 1;
        h ^= g_chess_z_piece[abs_v][color][sq];
    }
    if (p->side == 1) h ^= g_chess_z_side;
    h ^= g_chess_z_castling[p->castling & 0xF];
    if (p->ep_square >= 0 && p->ep_square < 64) {
        h ^= g_chess_z_ep_file[cp_file(p->ep_square)];
    }
    return h;
}

/* ---- Init startpos --------------------------------------------------- */

void chess_position_init_startpos(ChessPosition *p) {
    memset(p, 0, sizeof(*p));
    int back[8] = {
        CP_ROOK, CP_KNIGHT, CP_BISHOP, CP_QUEEN,
        CP_KING, CP_BISHOP, CP_KNIGHT, CP_ROOK,
    };
    for (int f = 0; f < 8; f++) {
        p->board[cp_sq(0, f)] = (int8_t)back[f];
        p->board[cp_sq(7, f)] = (int8_t)(-back[f]);
        p->board[cp_sq(1, f)] = CP_PAWN;
        p->board[cp_sq(6, f)] = -CP_PAWN;
    }
    p->side = 0;
    p->castling = CR_ALL;
    p->ep_square = -1;
    p->halfmove = 0;
    p->white_king_sq = (int16_t)cp_sq(0, 4);
    p->black_king_sq = (int16_t)cp_sq(7, 4);
    p->hash = chess_compute_hash(p);
}

/* ---- Make move ------------------------------------------------------- *
 *
 * Signed board values carry colour. All moves here are already legal
 * per chess_expand_legal_moves, so we can trust the flags and don't
 * re-validate. Returns 0 on success, -1 on structural failure.
 */
int chess_make_move(ChessPosition *p, ChMove m, ChessUndo *u) {
    int from = chmove_from(m);
    int to   = chmove_to(m);
    int promo = chmove_promo(m);
    int side  = p->side;
    int sign  = (side == 0) ? 1 : -1;
    int opp_color = side ^ 1;

    int8_t mover = p->board[from];
    if (mover == 0 || (mover > 0 ? 0 : 1) != side) {
        /* Shouldn't happen with legal input — refuse rather than corrupt. */
        return -1;
    }
    int mover_abs = (mover > 0) ? mover : -mover;
    int8_t captured = p->board[to];   /* may be 0 */

    /* ---- Save undo state ------------------------------------------- */
    u->move = m;
    u->captured = captured;
    u->prev_castling = p->castling;
    u->prev_ep_square = p->ep_square;
    u->prev_halfmove = p->halfmove;
    u->prev_white_king_sq = p->white_king_sq;
    u->prev_black_king_sq = p->black_king_sq;
    u->prev_hash = p->hash;

    uint64_t h = p->hash;

    /* Remove old ep-file hash (if any). We'll add the new one below. */
    if (p->ep_square >= 0) {
        h ^= g_chess_z_ep_file[cp_file(p->ep_square)];
    }

    /* ---- En passant capture ---------------------------------------- */
    if (chmove_is_ep(m)) {
        /* Remove moving pawn from `from` */
        h ^= g_chess_z_piece[CP_PAWN][side][from];
        p->board[from] = 0;
        /* Place moving pawn on `to` */
        h ^= g_chess_z_piece[CP_PAWN][side][to];
        p->board[to] = (int8_t)(CP_PAWN * sign);
        /* Remove captured pawn BEHIND `to` on the mover's rank-1 */
        int cap_sq = to + ((side == 0) ? -8 : 8);
        h ^= g_chess_z_piece[CP_PAWN][opp_color][cap_sq];
        p->board[cap_sq] = 0;
        u->captured = (int8_t)(-CP_PAWN * sign);  /* record actual captured */
    }
    /* ---- Castling -------------------------------------------------- */
    else if (chmove_is_castle_k(m)) {
        int rank = (side == 0) ? 0 : 7;
        int rook_from = cp_sq(rank, 7);
        int rook_to = cp_sq(rank, 5);
        /* King */
        h ^= g_chess_z_piece[CP_KING][side][from];
        h ^= g_chess_z_piece[CP_KING][side][to];
        p->board[from] = 0;
        p->board[to] = (int8_t)(CP_KING * sign);
        /* Rook */
        h ^= g_chess_z_piece[CP_ROOK][side][rook_from];
        h ^= g_chess_z_piece[CP_ROOK][side][rook_to];
        p->board[rook_from] = 0;
        p->board[rook_to] = (int8_t)(CP_ROOK * sign);
    }
    else if (chmove_is_castle_q(m)) {
        int rank = (side == 0) ? 0 : 7;
        int rook_from = cp_sq(rank, 0);
        int rook_to = cp_sq(rank, 3);
        h ^= g_chess_z_piece[CP_KING][side][from];
        h ^= g_chess_z_piece[CP_KING][side][to];
        p->board[from] = 0;
        p->board[to] = (int8_t)(CP_KING * sign);
        h ^= g_chess_z_piece[CP_ROOK][side][rook_from];
        h ^= g_chess_z_piece[CP_ROOK][side][rook_to];
        p->board[rook_from] = 0;
        p->board[rook_to] = (int8_t)(CP_ROOK * sign);
    }
    /* ---- Normal move, capture, promotion --------------------------- */
    else {
        /* Captured piece (normal capture) */
        if (captured != 0) {
            int cap_abs = (captured > 0) ? captured : -captured;
            h ^= g_chess_z_piece[cap_abs][opp_color][to];
        }
        /* Remove mover from `from` */
        h ^= g_chess_z_piece[mover_abs][side][from];
        p->board[from] = 0;
        /* Place (possibly promoted) piece on `to` */
        int placed_abs = (promo != 0) ? promo : mover_abs;
        h ^= g_chess_z_piece[placed_abs][side][to];
        p->board[to] = (int8_t)(placed_abs * sign);
    }

    /* ---- King square tracking ------------------------------------- */
    if (mover_abs == CP_KING) {
        if (side == 0) p->white_king_sq = (int16_t)to;
        else           p->black_king_sq = (int16_t)to;
    }

    /* ---- Castling rights update ----------------------------------- */
    int new_castling = p->castling & CR_SQUARE_MASK[from] & CR_SQUARE_MASK[to];
    if (new_castling != p->castling) {
        h ^= g_chess_z_castling[p->castling & 0xF];
        h ^= g_chess_z_castling[new_castling & 0xF];
        p->castling = (int8_t)new_castling;
    }

    /* ---- En passant square ---------------------------------------- */
    int new_ep = -1;
    if (chmove_is_double_push(m)) {
        new_ep = (from + to) >> 1;
        h ^= g_chess_z_ep_file[cp_file(new_ep)];
    }
    p->ep_square = (int16_t)new_ep;

    /* ---- Halfmove clock (50-move rule) ---------------------------- */
    int was_capture = (captured != 0) || chmove_is_ep(m);
    if (mover_abs == CP_PAWN || was_capture) {
        p->halfmove = 0;
    } else {
        p->halfmove = (int16_t)(p->halfmove + 1);
    }

    /* ---- Flip side ----------------------------------------------- */
    h ^= g_chess_z_side;
    p->side = (int8_t)(side ^ 1);
    p->hash = h;
    return 0;
}

/* ---- Unmake move ----------------------------------------------------- */

void chess_unmake_move(ChessPosition *p, const ChessUndo *u) {
    /* Restore everything we saved. The board moves need to be undone
     * in reverse of make_move; everything else is a direct restore. */
    ChMove m = u->move;
    int from = chmove_from(m);
    int to   = chmove_to(m);
    int promo = chmove_promo(m);
    int side  = p->side ^ 1;   /* side that made the move */
    int sign  = (side == 0) ? 1 : -1;

    if (chmove_is_ep(m)) {
        int cap_sq = to + ((side == 0) ? -8 : 8);
        p->board[from] = (int8_t)(CP_PAWN * sign);
        p->board[to] = 0;
        p->board[cap_sq] = (int8_t)(-CP_PAWN * sign);
    } else if (chmove_is_castle_k(m)) {
        int rank = (side == 0) ? 0 : 7;
        int rook_from = cp_sq(rank, 7);
        int rook_to = cp_sq(rank, 5);
        p->board[from] = (int8_t)(CP_KING * sign);
        p->board[to] = 0;
        p->board[rook_from] = (int8_t)(CP_ROOK * sign);
        p->board[rook_to] = 0;
    } else if (chmove_is_castle_q(m)) {
        int rank = (side == 0) ? 0 : 7;
        int rook_from = cp_sq(rank, 0);
        int rook_to = cp_sq(rank, 3);
        p->board[from] = (int8_t)(CP_KING * sign);
        p->board[to] = 0;
        p->board[rook_from] = (int8_t)(CP_ROOK * sign);
        p->board[rook_to] = 0;
    } else {
        /* Normal move / capture / promotion: restore the mover at
         * `from` (use saved promo to know if the original piece was
         * a pawn) and restore captured piece (or empty) at `to`. */
        if (promo != 0) {
            p->board[from] = (int8_t)(CP_PAWN * sign);
        } else {
            /* The piece currently on `to` is the same kind that was
             * on `from` before the move, so read it and put it back. */
            p->board[from] = p->board[to];
        }
        p->board[to] = u->captured;
    }

    /* Scalar state — direct restore */
    p->side = (int8_t)side;
    p->castling = u->prev_castling;
    p->ep_square = u->prev_ep_square;
    p->halfmove = u->prev_halfmove;
    p->white_king_sq = u->prev_white_king_sq;
    p->black_king_sq = u->prev_black_king_sq;
    p->hash = u->prev_hash;
}

/* ---- Perft (leaf-node counter for movegen validation) ---------------- */

uint64_t chess_perft(ChessPosition *p, int depth) {
    if (depth <= 0) return 1;
    ChMove moves[CHESS_MAX_MOVES];
    int n = chess_expand_legal_moves(p, moves);
    if (depth == 1) return (uint64_t)n;
    uint64_t total = 0;
    for (int i = 0; i < n; i++) {
        ChessUndo u;
        if (chess_make_move(p, moves[i], &u) != 0) continue;
        total += chess_perft(p, depth - 1);
        chess_unmake_move(p, &u);
    }
    return total;
}
