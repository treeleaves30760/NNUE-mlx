"""Quick benchmark for the C-native Lazy SMP search vs single thread.

Usage:
    uv run python scripts/bench_cnative_search.py

Runs a fixed-depth chess search from a few opening positions at
n_threads=1, 2, 4, 8 and reports nodes-per-second. Expected: n_threads=4
yields ~2.5-3x NPS of n_threads=1; n_threads=8 yields ~3-5x (sublinear
scaling is normal for Lazy SMP).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.accel._nnue_accel import AcceleratedAccumulator, CSearch
from src.features.halfkp import HalfKP
from src.games.chess import initial_state


def _make_cs(acc=256, l1=128, l2=32):
    fs = HalfKP(
        num_squares=64, num_piece_types=5, king_board_val=6,
        board_val_to_type={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    )
    num_feat = fs.num_features()
    rng = np.random.RandomState(42)
    ft_w = (rng.randn(num_feat, acc) * 0.01).astype(np.float32)
    ft_b = (rng.randn(acc) * 0.01).astype(np.float32)
    l1_w = (rng.randn(l1, acc * 2) * 0.1).astype(np.float32)
    l1_b = (rng.randn(l1) * 0.01).astype(np.float32)
    l2_w = (rng.randn(l2, l1) * 0.1).astype(np.float32)
    l2_b = (rng.randn(l2) * 0.01).astype(np.float32)
    out_w = (rng.randn(l2) * 0.1).astype(np.float32)
    out_b = np.array([0.0], dtype=np.float32)
    a = AcceleratedAccumulator(ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b)
    cs = CSearch(
        accumulator=a, feature_set=fs,
        tt_size=1 << 20, eval_scale=128.0, max_sq=64,
    )
    return cs, a, fs, (ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b)


def _args(state, depth, time_ms, n_threads):
    return (
        bytes(state.board_array()),
        int(state.side_to_move()),
        int(state._castling),
        int(state._ep_square),
        int(state._halfmove),
        int(state.king_square(0)),
        int(state.king_square(1)),
        b"",  # history
        int(depth),
        float(time_ms),
        int(n_threads),
    )


def bench_position(label, state, time_ms=1000.0):
    print(f"\n=== {label} (time={time_ms:.0f}ms) ===")
    cs, _a, _fs, _w = _make_cs()
    rows = []
    for nt in [1, 2, 4, 8]:
        # Warm-up
        cs.search_cnative_chess(*_args(state, depth=32, time_ms=time_ms / 4, n_threads=nt))
        # Measured run
        t0 = time.time()
        result = cs.search_cnative_chess(*_args(state, depth=32, time_ms=time_ms, n_threads=nt))
        elapsed = time.time() - t0
        if result is None:
            print(f"  n_threads={nt}: search returned None")
            continue
        mv, score, nodes = result
        nps = nodes / elapsed / 1000.0  # kN/s
        rows.append((nt, nodes, nps, mv, score))
        print(f"  n_threads={nt:2d}  {elapsed:.2f}s  {nodes:8d} nodes  {nps:7.1f} kN/s  move={mv} score={score:.1f}")
    if len(rows) >= 2:
        base_nps = rows[0][2]
        print("\n  Scaling vs n_threads=1:")
        for (nt, _n, nps, _m, _s) in rows:
            print(f"    n_threads={nt}: {nps / base_nps:.2f}x")


def main():
    state = initial_state()
    bench_position("Starting position", state, time_ms=1500.0)

    # A few plies in, to exercise make/unmake in non-trivial positions.
    for _ in range(4):
        state = state.make_move(state.legal_moves()[0])
    bench_position("4 plies in (linear walk)", state, time_ms=1500.0)


if __name__ == "__main__":
    main()
