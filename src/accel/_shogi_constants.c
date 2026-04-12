/* ========================================================================
 * Shogi constants, direction tables, and Zobrist initialisation.
 * ======================================================================= */

#include "_shogi_position.h"

/* ---- Direction tables (sente-oriented; gote flips delta_rank) ---------- */

const int8_t SH_KING_DIRS[8][2] = {
    {-1,-1},{-1, 0},{-1, 1},
    { 0,-1},        { 0, 1},
    { 1,-1},{ 1, 0},{ 1, 1},
};

/* Gold moves: no diagonal-back */
const int8_t SH_GOLD_DIRS[6][2] = {
    {-1,-1},{-1, 0},{-1, 1},
    { 0,-1},        { 0, 1},
            { 1, 0},
};

/* Silver moves: 5 directions (no pure back, no pure side) */
const int8_t SH_SILVER_DIRS[5][2] = {
    {-1,-1},{-1, 0},{-1, 1},
    { 1,-1},        { 1, 1},
};

const int8_t SH_ROOK_DIRS[4][2] = {
    {-1, 0},{ 1, 0},{ 0,-1},{ 0, 1},
};

const int8_t SH_BISHOP_DIRS[4][2] = {
    {-1,-1},{-1, 1},{ 1,-1},{ 1, 1},
};

/* ---- Zobrist tables ---------------------------------------------------- */
uint64_t g_shogi_z_piece[15][2][81];
uint64_t g_shogi_z_hand [7][2][39];
uint64_t g_shogi_z_side [2];
int      g_shogi_zobrist_ready = 0;

/* SplitMix64 — deterministic, independent of numpy. The C TT and the
 * Python TT use separate hash spaces so we don't need to match Python's
 * numpy default_rng output. */
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void shogi_zobrist_init(void) {
    if (g_shogi_zobrist_ready) return;
    uint64_t st = 0xCAFEBABEDEADBEEFull;
    for (int pv = 0; pv < 15; pv++)
        for (int c = 0; c < 2; c++)
            for (int sq = 0; sq < 81; sq++)
                g_shogi_z_piece[pv][c][sq] = splitmix64(&st);
    for (int pt = 0; pt < 7; pt++)
        for (int c = 0; c < 2; c++)
            for (int n = 0; n < 39; n++)
                g_shogi_z_hand[pt][c][n] = splitmix64(&st);
    g_shogi_z_side[0] = splitmix64(&st);
    g_shogi_z_side[1] = splitmix64(&st);
    g_shogi_zobrist_ready = 1;
}
