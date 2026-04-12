/* ========================================================================
 * Chess position representation for the C engine.
 *
 * Board encoding (matches src/games/chess/constants.py):
 *   sq = rank * 8 + file   (rank 0 = white's back rank, rank 7 = black's)
 *   positive = white, negative = black
 *   1=Pawn 2=Knight 3=Bishop 4=Rook 5=Queen 6=King
 *
 * Full position state (enough for search correctness):
 *   * board[64]          — piece placement
 *   * side               — side to move (0 = white, 1 = black)
 *   * castling           — 4-bit mask: bit 0=WK, 1=WQ, 2=BK, 3=BQ
 *   * ep_square          — -1 if none, else the square BEHIND a pawn that
 *                          just two-stepped (the square an ep-capturing
 *                          pawn would land on)
 *   * halfmove           — halfmoves since the last pawn move or capture
 *                          (for 50-move rule)
 *   * hash               — Zobrist hash, updated incrementally
 *   * white/black_king_sq — precomputed king squares
 *
 * Move encoding (uint32_t):
 *   bits  0..5   : from_sq  (0..63)
 *   bits  6..11  : to_sq    (0..63)
 *   bits 12..14  : promo    (0 = none, 2=Knight 3=Bishop 4=Rook 5=Queen)
 *   bits 15..17  : flags    (see CHMOVE_FLAG_* below)
 *
 * Flags capture what kind of special move this is so unmake can reverse
 * it without re-deriving the category:
 *   CHMOVE_FLAG_EP          — en passant capture
 *   CHMOVE_FLAG_CASTLE_K    — kingside castle
 *   CHMOVE_FLAG_CASTLE_Q    — queenside castle
 *   CHMOVE_FLAG_DOUBLE_PUSH — two-square pawn push (sets new ep square)
 * ======================================================================= */

#ifndef _CHESS_POSITION_H
#define _CHESS_POSITION_H

#include <stdint.h>
#include <string.h>

#define CHESS_SQUARES   64
#define CHESS_MAX_MOVES 256   /* safe upper bound (~218 real max) */

/* Piece codes */
#define CP_EMPTY   0
#define CP_PAWN    1
#define CP_KNIGHT  2
#define CP_BISHOP  3
#define CP_ROOK    4
#define CP_QUEEN   5
#define CP_KING    6

/* Castling-rights bits */
#define CR_WK  0x1
#define CR_WQ  0x2
#define CR_BK  0x4
#define CR_BQ  0x8
#define CR_WHITE (CR_WK | CR_WQ)
#define CR_BLACK (CR_BK | CR_BQ)
#define CR_ALL   (CR_WHITE | CR_BLACK)

/* ---- Move encoding ---------------------------------------------------- */
typedef uint32_t ChMove;

#define CHMOVE_NULL            ((ChMove)0xFFFFFFFFu)

#define CHMOVE_FLAG_EP          0x1
#define CHMOVE_FLAG_CASTLE_K    0x2
#define CHMOVE_FLAG_CASTLE_Q    0x4
#define CHMOVE_FLAG_DOUBLE_PUSH 0x8

static inline ChMove chmove_encode(int from_sq, int to_sq, int promo, int flags) {
    ChMove m = 0;
    m |= ((uint32_t)(from_sq & 0x3F));
    m |= ((uint32_t)(to_sq   & 0x3F)) << 6;
    m |= ((uint32_t)(promo   & 0x7 )) << 12;
    m |= ((uint32_t)(flags   & 0xF )) << 15;
    return m;
}

static inline int chmove_from(ChMove m)  { return (int)(m & 0x3F); }
static inline int chmove_to(ChMove m)    { return (int)((m >> 6) & 0x3F); }
static inline int chmove_promo(ChMove m) { return (int)((m >> 12) & 0x7); }
static inline int chmove_flags(ChMove m) { return (int)((m >> 15) & 0xF); }
static inline int chmove_is_ep(ChMove m)          { return (chmove_flags(m) & CHMOVE_FLAG_EP) != 0; }
static inline int chmove_is_castle_k(ChMove m)    { return (chmove_flags(m) & CHMOVE_FLAG_CASTLE_K) != 0; }
static inline int chmove_is_castle_q(ChMove m)    { return (chmove_flags(m) & CHMOVE_FLAG_CASTLE_Q) != 0; }
static inline int chmove_is_double_push(ChMove m) { return (chmove_flags(m) & CHMOVE_FLAG_DOUBLE_PUSH) != 0; }
static inline int chmove_is_castle(ChMove m)      { return chmove_is_castle_k(m) || chmove_is_castle_q(m); }

/* ---- Position --------------------------------------------------------- */
typedef struct {
    int8_t   board[CHESS_SQUARES];
    int8_t   side;           /* 0 = white, 1 = black */
    int8_t   castling;       /* 4-bit mask: CR_WK | CR_WQ | CR_BK | CR_BQ */
    int16_t  ep_square;      /* -1 if none, else 0..63 */
    int16_t  halfmove;       /* for 50-move rule */
    int16_t  white_king_sq;
    int16_t  black_king_sq;
    uint64_t hash;
} ChessPosition;

/* ---- Undo frame ------------------------------------------------------- */
typedef struct {
    ChMove    move;
    int8_t   captured;        /* board value before the move (0 if empty) */
    int8_t   prev_castling;
    int16_t  prev_ep_square;
    int16_t  prev_halfmove;
    int16_t  prev_white_king_sq;
    int16_t  prev_black_king_sq;
    uint64_t prev_hash;
} ChessUndo;

/* ---- Geometry helpers ------------------------------------------------- */
static inline int cp_rank(int sq) { return sq >> 3; }
static inline int cp_file(int sq) { return sq & 7; }
static inline int cp_sq(int r, int f) { return (r << 3) + f; }

/* ---- Zobrist tables (defined in _chess_constants.c) ------------------- */
/* z_piece[piece 1..6][color 0..1][sq 0..63]
 * z_side        — XOR in when it is black's turn
 * z_castling[0..15]
 * z_ep_file[0..7] */
extern uint64_t g_chess_z_piece[7][2][64];
extern uint64_t g_chess_z_side;
extern uint64_t g_chess_z_castling[16];
extern uint64_t g_chess_z_ep_file[8];
extern int      g_chess_zobrist_ready;

/* ---- Precomputed attack tables (defined in _chess_constants.c) -------- */
extern uint64_t g_chess_knight_attacks[64];
extern uint64_t g_chess_king_attacks[64];
extern uint64_t g_chess_pawn_attacks[2][64];  /* [color][sq] */

/* Ray squares for slider movegen: [dir 0..7][sq] -> 0-terminated list of
 * squares reached (up to 7 each). Dirs 0..3 = rook, 4..7 = bishop. */
extern int8_t g_chess_rays[8][64][8];
extern int8_t g_chess_ray_len[8][64];

/* ---- Public API ------------------------------------------------------- */

/* Initialize runtime tables (zobrist + attack bitmasks). Must be called
 * once at module init. Idempotent. */
void chess_init_tables(void);

/* Initialize p to the standard starting position. */
void chess_position_init_startpos(ChessPosition *p);

/* Return non-zero if `sq` is attacked by any piece belonging to `by_side`. */
int chess_is_square_attacked(const int8_t *board, int sq, int by_side);

/* Return non-zero if side `s` is currently in check. */
int chess_is_in_check(const ChessPosition *p, int side);

/* Generate all *legal* moves into out[]. Returns the number of moves.
 * Performs full pseudo-legal expansion followed by a self-check filter
 * via make/unmake. */
int chess_expand_legal_moves(ChessPosition *p, ChMove *out);

/* Apply a move in place, filling u with enough state to reverse it.
 * Returns 0 on success, -1 on error. `m` must come from
 * chess_expand_legal_moves (we don't double-check legality here). */
int chess_make_move(ChessPosition *p, ChMove m, ChessUndo *u);

/* Reverse a previously-applied move. */
void chess_unmake_move(ChessPosition *p, const ChessUndo *u);

/* Compute the full Zobrist hash from scratch (used for init and as a
 * self-check against the incremental updates). */
uint64_t chess_compute_hash(const ChessPosition *p);

/* Perft: count leaf nodes at the given depth. For movegen validation. */
uint64_t chess_perft(ChessPosition *p, int depth);

#endif /* _CHESS_POSITION_H */
