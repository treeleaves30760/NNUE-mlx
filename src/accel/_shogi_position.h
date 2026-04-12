/* ========================================================================
 * Shogi position representation and movegen primitives (C, self-contained)
 *
 * Board encoding (matches src/games/shogi/constants.py):
 *   sq = rank * 9 + file   (rank 0 = gote side, rank 8 = sente side)
 *   positive = sente (WHITE=0), negative = gote (BLACK=1)
 *   1=Pawn 2=Lance 3=Knight 4=Silver 5=Gold 6=Bishop 7=Rook 8=King
 *   9=+Pawn 10=+Lance 11=+Knight 12=+Silver 13=+Bishop(Horse) 14=+Rook(Dragon)
 *
 * Hands are indexed by API piece type (0..6: PAWN..ROOK, no king / no promoted).
 *
 * Moves are encoded compactly as a single uint32_t:
 *   bits  0..7   : from_sq (0..80 for board move, 0xFF for drop)
 *   bits  8..15  : to_sq   (0..80)
 *   bits 16..19  : promoted_piece (0 if no promotion, else 9..14)
 *   bits 20..23  : drop_piece (0xF if not a drop, else 0..6 API type)
 * ======================================================================= */

#ifndef _SHOGI_POSITION_H
#define _SHOGI_POSITION_H

#include <stdint.h>
#include <string.h>

#define SHOGI_SQUARES    81
#define SHOGI_MAX_MOVES  593   /* upper bound on legal moves in any position */
#define SHOGI_HAND_TYPES 7     /* Pawn..Rook */

/* ---- Piece codes ------------------------------------------------------- */
#define SP_EMPTY         0
#define SP_PAWN          1
#define SP_LANCE         2
#define SP_KNIGHT        3
#define SP_SILVER        4
#define SP_GOLD          5
#define SP_BISHOP        6
#define SP_ROOK          7
#define SP_KING          8
#define SP_PPAWN         9   /* tokin */
#define SP_PLANCE        10
#define SP_PKNIGHT       11
#define SP_PSILVER       12
#define SP_HORSE         13  /* +Bishop */
#define SP_DRAGON        14  /* +Rook */

/* ---- Move encoding ----------------------------------------------------- */
typedef uint32_t SMove;

#define SMOVE_NULL           ((SMove)0xFFFFFFFFu)
#define SMOVE_NONE_FROM      0xFF
#define SMOVE_NONE_DROP      0xF

static inline SMove smove_encode(int from_sq, int to_sq,
                                  int promo, int drop) {
    SMove m = 0;
    m |= ((uint32_t)(from_sq & 0xFF));
    m |= ((uint32_t)(to_sq   & 0xFF)) << 8;
    m |= ((uint32_t)(promo   & 0xF )) << 16;
    m |= ((uint32_t)(drop    & 0xF )) << 20;
    return m;
}

static inline int smove_from(SMove m)  { return (int)(m & 0xFF); }
static inline int smove_to(SMove m)    { return (int)((m >> 8) & 0xFF); }
static inline int smove_promo(SMove m) { return (int)((m >> 16) & 0xF); }
static inline int smove_drop(SMove m)  { return (int)((m >> 20) & 0xF); }
static inline int smove_is_drop(SMove m) { return smove_drop(m) != SMOVE_NONE_DROP; }

/* ---- Position ---------------------------------------------------------- */
typedef struct {
    int8_t   board[SHOGI_SQUARES];
    int8_t   sente_hand[SHOGI_HAND_TYPES];
    int8_t   gote_hand[SHOGI_HAND_TYPES];
    int8_t   side;            /* 0 = sente (WHITE), 1 = gote (BLACK) */
    int8_t   _pad;
    int16_t  sente_king_sq;
    int16_t  gote_king_sq;
    uint64_t hash;
} ShogiPosition;

/* ---- Undo frame (for make/unmake) -------------------------------------- */
typedef struct {
    SMove   move;
    int8_t  captured;        /* raw board value before the move (0 if empty) */
    int8_t  prev_from_piece; /* raw board value of mover before promotion */
    int16_t prev_king_sq;    /* king's square before this move (for the mover) */
    uint64_t prev_hash;
} ShogiUndo;

/* ---- Zobrist tables (initialised once from Python RNG to stay compat) -- */
/* z_piece[board_value 0..14][color 0..1][sq 0..80]                         */
/* z_hand [piece_type 0..6 ][color 0..1][count 0..38]                       */
/* z_side [color 0..1]                                                      */
extern uint64_t g_shogi_z_piece[15][2][81];
extern uint64_t g_shogi_z_hand [7][2][39];
extern uint64_t g_shogi_z_side [2];
extern int      g_shogi_zobrist_ready;

/* ---- Direction tables (one-step & sliding pieces) ---------------------- */
/* Encoded as (dr, df) pairs; sente-oriented, flipped on the fly for gote. */
extern const int8_t SH_KING_DIRS  [8][2];
extern const int8_t SH_GOLD_DIRS  [6][2];
extern const int8_t SH_SILVER_DIRS[5][2];
extern const int8_t SH_ROOK_DIRS  [4][2];
extern const int8_t SH_BISHOP_DIRS[4][2];

/* ---- Geometry helpers -------------------------------------------------- */
static inline int sh_rank(int sq) { return sq / 9; }
static inline int sh_file(int sq) { return sq % 9; }
static inline int sh_sq(int r, int f) { return r * 9 + f; }

/* Promotion zone: the 3 ranks closest to the enemy. */
static inline int sh_in_promo_zone(int sq, int side) {
    int r = sh_rank(sq);
    return (side == 0) ? (r <= 2) : (r >= 6);
}

/* ---- Public API (defined in _shogi_movegen_c.c / _shogi_make_unmake.c) - */
void shogi_position_init_startpos(ShogiPosition *p);
int  shogi_expand_pseudo_moves(const ShogiPosition *p, SMove *out);
int  shogi_expand_legal_moves(ShogiPosition *p, SMove *out);
int  shogi_is_in_check(const ShogiPosition *p, int side);
int  shogi_is_square_attacked(const int8_t *board, int sq, int by_side);
int  shogi_make_move(ShogiPosition *p, SMove m, ShogiUndo *u);
void shogi_unmake_move(ShogiPosition *p, const ShogiUndo *u);
uint64_t shogi_compute_hash(const ShogiPosition *p);

#endif /* _SHOGI_POSITION_H */
