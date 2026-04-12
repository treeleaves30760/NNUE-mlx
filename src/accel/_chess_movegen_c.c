/* ========================================================================
 * Chess pseudo-legal and legal move generation in C.
 *
 * Ports src/games/chess/movegen.py 1:1. Operates on a ChessPosition
 * struct — no Python callbacks in the hot loop.
 *
 * Public API:
 *   int chess_is_square_attacked(const int8_t *board, int sq, int by_side);
 *   int chess_is_in_check(const ChessPosition *p, int side);
 *   int chess_expand_pseudo_moves(const ChessPosition *p, ChMove *out);
 *   int chess_expand_legal_moves(ChessPosition *p, ChMove *out);
 * ======================================================================= */

#include "_chess_position.h"

/* Forward decls — implementations live in _chess_make_unmake.c. */
int chess_make_move(ChessPosition *p, ChMove m, ChessUndo *u);
void chess_unmake_move(ChessPosition *p, const ChessUndo *u);

/* ---- Friend / enemy helpers ------------------------------------------ */

static inline int chess_is_friend(int8_t v, int side) {
    if (v == 0) return 0;
    return (side == 0) ? (v > 0) : (v < 0);
}

static inline int chess_is_enemy(int8_t v, int side) {
    if (v == 0) return 0;
    return (side == 0) ? (v < 0) : (v > 0);
}

/* ---- Square-attacked check ------------------------------------------- */

int chess_is_square_attacked(const int8_t *board, int sq, int by_side) {
    int sign = (by_side == 0) ? 1 : -1;

    /* Pawn attacks: if a pawn of `by_side` attacks `sq`, then from
     * `sq`'s perspective, a pawn of the OPPOSITE colour would attack
     * the same squares. So we look at the *opposite-colour* pawn
     * attack mask from sq. */
    uint64_t pawn_bb = (by_side == 0)
        ? g_chess_pawn_attacks[1][sq]   /* black's attack mask from sq */
        : g_chess_pawn_attacks[0][sq];
    while (pawn_bb) {
        int y = __builtin_ctzll(pawn_bb);
        if (board[y] == (int8_t)(CP_PAWN * sign)) return 1;
        pawn_bb &= pawn_bb - 1;
    }

    /* Knight attacks */
    uint64_t knight_bb = g_chess_knight_attacks[sq];
    while (knight_bb) {
        int y = __builtin_ctzll(knight_bb);
        if (board[y] == (int8_t)(CP_KNIGHT * sign)) return 1;
        knight_bb &= knight_bb - 1;
    }

    /* King attacks */
    uint64_t king_bb = g_chess_king_attacks[sq];
    while (king_bb) {
        int y = __builtin_ctzll(king_bb);
        if (board[y] == (int8_t)(CP_KING * sign)) return 1;
        king_bb &= king_bb - 1;
    }

    /* Sliding pieces: walk rays until we hit something. Rook/queen
     * cover dirs 0..3; bishop/queen cover dirs 4..7. */
    for (int d = 0; d < 4; d++) {
        int n = g_chess_ray_len[d][sq];
        for (int i = 0; i < n; i++) {
            int y = g_chess_rays[d][sq][i];
            int8_t v = board[y];
            if (v == 0) continue;
            if (v == (int8_t)(CP_ROOK * sign) || v == (int8_t)(CP_QUEEN * sign))
                return 1;
            break;
        }
    }
    for (int d = 4; d < 8; d++) {
        int n = g_chess_ray_len[d][sq];
        for (int i = 0; i < n; i++) {
            int y = g_chess_rays[d][sq][i];
            int8_t v = board[y];
            if (v == 0) continue;
            if (v == (int8_t)(CP_BISHOP * sign) || v == (int8_t)(CP_QUEEN * sign))
                return 1;
            break;
        }
    }

    return 0;
}

int chess_is_in_check(const ChessPosition *p, int side) {
    int ksq = (side == 0) ? p->white_king_sq : p->black_king_sq;
    if (ksq < 0) return 0;
    return chess_is_square_attacked(p->board, ksq, side ^ 1);
}

/* ---- Piece-specific pseudo-legal generators -------------------------- */

static void cp_gen_leaper(const int8_t *board, int sq, uint64_t attacks,
                           int sign, ChMove *out, int *n_out) {
    uint64_t bb = attacks;
    while (bb) {
        int to = __builtin_ctzll(bb);
        int8_t v = board[to];
        if (v * sign <= 0) {
            /* Empty or enemy */
            out[(*n_out)++] = chmove_encode(sq, to, 0, 0);
        }
        bb &= bb - 1;
    }
}

static void cp_gen_slider(const int8_t *board, int sq, int dir_start,
                           int dir_end, int sign, ChMove *out, int *n_out) {
    for (int d = dir_start; d < dir_end; d++) {
        int n = g_chess_ray_len[d][sq];
        for (int i = 0; i < n; i++) {
            int to = g_chess_rays[d][sq][i];
            int8_t target = board[to];
            if (target * sign > 0) break;  /* friendly blocker */
            out[(*n_out)++] = chmove_encode(sq, to, 0, 0);
            if (target != 0) break;        /* enemy capture, stop */
        }
    }
}

static void cp_gen_pawn(const int8_t *board, int sq, int side, int ep_square,
                         ChMove *out, int *n_out) {
    int r = cp_rank(sq);
    int f = cp_file(sq);
    int forward = (side == 0) ? 1 : -1;
    int start_rank = (side == 0) ? 1 : 6;
    int promo_rank = (side == 0) ? 6 : 1;
    int opp_sign = (side == 0) ? -1 : 1;

    /* Single push */
    int nr = r + forward;
    if (nr >= 0 && nr < 8) {
        int to = cp_sq(nr, f);
        if (board[to] == 0) {
            if (r == promo_rank) {
                out[(*n_out)++] = chmove_encode(sq, to, CP_QUEEN,  0);
                out[(*n_out)++] = chmove_encode(sq, to, CP_ROOK,   0);
                out[(*n_out)++] = chmove_encode(sq, to, CP_BISHOP, 0);
                out[(*n_out)++] = chmove_encode(sq, to, CP_KNIGHT, 0);
            } else {
                out[(*n_out)++] = chmove_encode(sq, to, 0, 0);
                /* Double push from start rank */
                if (r == start_rank) {
                    int to2 = cp_sq(nr + forward, f);
                    if (board[to2] == 0) {
                        out[(*n_out)++] = chmove_encode(
                            sq, to2, 0, CHMOVE_FLAG_DOUBLE_PUSH);
                    }
                }
            }
        }
    }

    /* Captures (diagonal + en passant) */
    for (int df = -1; df <= 1; df += 2) {
        int nf = f + df;
        if (nf < 0 || nf >= 8) continue;
        if (nr < 0 || nr >= 8) continue;
        int to = cp_sq(nr, nf);
        int8_t target = board[to];
        int is_cap = (target * opp_sign > 0);
        int is_ep = (to == ep_square && ep_square >= 0);
        if (is_cap) {
            if (r == promo_rank) {
                out[(*n_out)++] = chmove_encode(sq, to, CP_QUEEN,  0);
                out[(*n_out)++] = chmove_encode(sq, to, CP_ROOK,   0);
                out[(*n_out)++] = chmove_encode(sq, to, CP_BISHOP, 0);
                out[(*n_out)++] = chmove_encode(sq, to, CP_KNIGHT, 0);
            } else {
                out[(*n_out)++] = chmove_encode(sq, to, 0, 0);
            }
        } else if (is_ep) {
            /* En passant — always to an empty square on ep_square */
            out[(*n_out)++] = chmove_encode(sq, to, 0, CHMOVE_FLAG_EP);
        }
    }
}

static void cp_gen_castling(const int8_t *board, int king_sq, int side,
                             int castling, ChMove *out, int *n_out) {
    int rank = (side == 0) ? 0 : 7;
    int opp = side ^ 1;

    /* Kingside: bits for this side */
    int ks_bit = (side == 0) ? CR_WK : CR_BK;
    if (castling & ks_bit) {
        if (board[cp_sq(rank, 5)] == 0 && board[cp_sq(rank, 6)] == 0) {
            if (!chess_is_square_attacked(board, cp_sq(rank, 4), opp) &&
                !chess_is_square_attacked(board, cp_sq(rank, 5), opp) &&
                !chess_is_square_attacked(board, cp_sq(rank, 6), opp)) {
                out[(*n_out)++] = chmove_encode(
                    cp_sq(rank, 4), cp_sq(rank, 6), 0, CHMOVE_FLAG_CASTLE_K);
            }
        }
    }

    /* Queenside */
    int qs_bit = (side == 0) ? CR_WQ : CR_BQ;
    if (castling & qs_bit) {
        if (board[cp_sq(rank, 1)] == 0 && board[cp_sq(rank, 2)] == 0 &&
            board[cp_sq(rank, 3)] == 0) {
            if (!chess_is_square_attacked(board, cp_sq(rank, 4), opp) &&
                !chess_is_square_attacked(board, cp_sq(rank, 3), opp) &&
                !chess_is_square_attacked(board, cp_sq(rank, 2), opp)) {
                out[(*n_out)++] = chmove_encode(
                    cp_sq(rank, 4), cp_sq(rank, 2), 0, CHMOVE_FLAG_CASTLE_Q);
            }
        }
    }
    (void)king_sq;  /* unused; we derive the king square from rank */
}

/* ---- Pseudo-legal generator ------------------------------------------ */

int chess_expand_pseudo_moves(const ChessPosition *p, ChMove *out) {
    const int8_t *board = p->board;
    int side = p->side;
    int sign = (side == 0) ? 1 : -1;
    int n = 0;

    for (int sq = 0; sq < 64; sq++) {
        int8_t piece = board[sq];
        if (piece * sign <= 0) continue;
        int abs_piece = (piece > 0) ? piece : -piece;
        switch (abs_piece) {
            case CP_PAWN:
                cp_gen_pawn(board, sq, side, p->ep_square, out, &n);
                break;
            case CP_KNIGHT:
                cp_gen_leaper(board, sq, g_chess_knight_attacks[sq],
                              sign, out, &n);
                break;
            case CP_BISHOP:
                cp_gen_slider(board, sq, 4, 8, sign, out, &n);
                break;
            case CP_ROOK:
                cp_gen_slider(board, sq, 0, 4, sign, out, &n);
                break;
            case CP_QUEEN:
                cp_gen_slider(board, sq, 0, 8, sign, out, &n);
                break;
            case CP_KING:
                cp_gen_leaper(board, sq, g_chess_king_attacks[sq],
                              sign, out, &n);
                cp_gen_castling(board, sq, side, p->castling, out, &n);
                break;
            default:
                break;
        }
    }
    return n;
}

/* ---- Legal generator: pseudo-legal + make/unmake self-check filter --- */

int chess_expand_legal_moves(ChessPosition *p, ChMove *out) {
    ChMove pseudo[CHESS_MAX_MOVES];
    int n_pseudo = chess_expand_pseudo_moves(p, pseudo);
    int side = p->side;
    int n_legal = 0;

    for (int i = 0; i < n_pseudo; i++) {
        ChessUndo u;
        if (chess_make_move(p, pseudo[i], &u) != 0) continue;
        /* After make_move, side has flipped. We check if the *mover*
         * (original side) is now in check — that move was illegal. */
        int in_check = chess_is_in_check(p, side);
        chess_unmake_move(p, &u);
        if (!in_check) {
            out[n_legal++] = pseudo[i];
        }
    }
    return n_legal;
}
