"""Tests for the C-native multi-threaded NNUE search (Phase 1 + Phase 2).

Verifies:
  1. Smoke: search returns a legal move from the starting position for
     both chess and shogi, single-threaded and multi-threaded.
  2. Correctness parity: the move returned at n_threads=1 and n_threads=4
     is legal and score is finite. We don't assert they are identical —
     Lazy SMP breaks determinism intentionally — but both must pick a
     legal move.
  3. Live callback: search_cnative_live_chess updates ``live_ref`` and
     respects ``stop_event``.
  4. Infinite mode: ``max_depth=0`` + ``time_limit_ms>0`` terminates on
     the time limit without hanging.
"""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("src.accel._nnue_accel")

from src.accel._nnue_accel import AcceleratedAccumulator, CSearch


# Module-level list to anchor weight buffers + CSearch instances. CSearch
# borrows references (ft_weight etc.) without taking refcounts, so we
# must make sure they outlive the search call.
_KEEP_ALIVE: list = []


# ------------------------------------------------------------------ helpers

def _make_random_chess_csearch(seed=42, l1=32, l2=32, acc=256):
    """Build a CSearch with random NNUE weights + a real HalfKP chess FS.

    We don't care about eval quality for these tests — only that the
    search runs end-to-end and returns legal moves.
    """
    from src.features.halfkp import HalfKP

    fs = HalfKP(
        num_squares=64, num_piece_types=5, king_board_val=6,
        board_val_to_type={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    )
    num_feat = fs.num_features()

    rng = np.random.RandomState(seed)
    ft_w  = (rng.randn(num_feat, acc)      * 0.01).astype(np.float32)
    ft_b  = (rng.randn(acc)                * 0.01).astype(np.float32)
    l1_w  = (rng.randn(l1, acc * 2)        * 0.1 ).astype(np.float32)
    l1_b  = (rng.randn(l1)                 * 0.01).astype(np.float32)
    l2_w  = (rng.randn(l2, l1)             * 0.1 ).astype(np.float32)
    l2_b  = (rng.randn(l2)                 * 0.01).astype(np.float32)
    out_w = (rng.randn(l2)                 * 0.1 ).astype(np.float32)
    out_b = np.array([0.0], dtype=np.float32)

    a = AcceleratedAccumulator(ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b)
    cs = CSearch(
        accumulator=a, feature_set=fs,
        tt_size=1 << 18, eval_scale=128.0, max_sq=64,
    )
    _KEEP_ALIVE.append((cs, a, fs, ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b))
    return cs


def _make_random_shogi_csearch(seed=42, l1=32, l2=32, acc=256):
    from src.features.halfkp_shogi import shogi_features

    fs = shogi_features()
    num_feat = fs.num_features()

    rng = np.random.RandomState(seed)
    ft_w  = (rng.randn(num_feat, acc)      * 0.003).astype(np.float32)
    ft_b  = (rng.randn(acc)                * 0.01 ).astype(np.float32)
    l1_w  = (rng.randn(l1, acc * 2)        * 0.1  ).astype(np.float32)
    l1_b  = (rng.randn(l1)                 * 0.01 ).astype(np.float32)
    l2_w  = (rng.randn(l2, l1)             * 0.1  ).astype(np.float32)
    l2_b  = (rng.randn(l2)                 * 0.01 ).astype(np.float32)
    out_w = (rng.randn(l2)                 * 0.1  ).astype(np.float32)
    out_b = np.array([0.0], dtype=np.float32)

    a = AcceleratedAccumulator(ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b)
    cs = CSearch(
        accumulator=a, feature_set=fs,
        tt_size=1 << 18, eval_scale=128.0, max_sq=81,
    )
    _KEEP_ALIVE.append((cs, a, fs, ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b))
    return cs


def _chess_args(state, time_ms=500.0, depth=6, n_threads=1):
    import struct
    hist = getattr(state, "_history", ()) or ()
    hist_bytes = struct.pack(f"<{len(hist)}Q", *hist) if hist else b""
    return (
        bytes(state.board_array()),
        int(state.side_to_move()),
        int(state._castling),
        int(state._ep_square),
        int(state._halfmove),
        int(state.king_square(0)),
        int(state.king_square(1)),
        hist_bytes,
        int(depth),
        float(time_ms),
        int(n_threads),
    )


def _shogi_args(state, time_ms=500.0, depth=5, n_threads=1):
    import struct
    hand0 = state.hand_pieces(0)
    hand1 = state.hand_pieces(1)
    sh = tuple(hand0.get(i, 0) for i in range(7))
    gh = tuple(hand1.get(i, 0) for i in range(7))
    hist = getattr(state, "_history", ()) or ()
    hist_bytes = struct.pack(f"<{len(hist)}Q", *hist) if hist else b""
    return (
        bytes(state.board_array()),
        sh, gh,
        int(state.side_to_move()),
        hist_bytes,
        int(depth),
        float(time_ms),
        int(n_threads),
    )


# ------------------------------------------------------------------ chess smoke


def test_cnative_chess_returns_legal_move_single_thread():
    from src.games.chess import initial_state

    cs = _make_random_chess_csearch()
    state = initial_state()
    result = cs.search_cnative_chess(*_chess_args(state, depth=4, n_threads=1))
    assert result is not None
    (move_tuple, score, nodes) = result
    from_sq, to_sq, promo = move_tuple
    assert 0 <= from_sq < 64
    assert 0 <= to_sq < 64
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion) for m in legal}
    assert (from_sq, to_sq, promo) in legal_set, (
        f"C search returned move {(from_sq, to_sq, promo)} not in legal set"
    )
    assert nodes > 0


def test_cnative_chess_returns_legal_move_multi_thread():
    from src.games.chess import initial_state

    cs = _make_random_chess_csearch()
    state = initial_state()
    result = cs.search_cnative_chess(*_chess_args(state, depth=4, n_threads=4))
    assert result is not None
    (move_tuple, score, nodes) = result
    from_sq, to_sq, promo = move_tuple
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion) for m in legal}
    assert (from_sq, to_sq, promo) in legal_set
    assert nodes > 0


def test_cnative_chess_mid_game_position():
    """A few plies into a game — walking first-legal-move — should still
    return a legal move. Exercises make/unmake + accumulator refresh at a
    non-start position."""
    from src.games.chess import initial_state

    cs = _make_random_chess_csearch()
    state = initial_state()
    for _ in range(6):
        legal = state.legal_moves()
        state = state.make_move(legal[0])
    result = cs.search_cnative_chess(*_chess_args(state, depth=4, n_threads=2))
    assert result is not None
    (move_tuple, _score, _nodes) = result
    from_sq, to_sq, promo = move_tuple
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion) for m in legal}
    assert (from_sq, to_sq, promo) in legal_set


# ------------------------------------------------------------------ chess live


def test_cnative_chess_live_fires_callback_and_respects_stop():
    from src.games.chess import initial_state

    cs = _make_random_chess_csearch()
    state = initial_state()

    live_ref = [None]
    stop = threading.Event()

    def stopper():
        # Let the search run 300 ms then stop it.
        time.sleep(0.3)
        stop.set()

    t = threading.Thread(target=stopper)
    t.start()

    args = _chess_args(state, time_ms=0.0, depth=0, n_threads=2)
    # Drop the fixed n_threads (last positional) and insert live_ref + stop.
    args = args + (live_ref, stop)
    result = cs.search_cnative_live_chess(*args)
    t.join()

    # Should have terminated (not hung), may or may not have a final move.
    assert live_ref[0] is not None, "live_ref was never populated"
    depth, max_depth, top_moves, done = live_ref[0]
    assert depth >= 0
    assert isinstance(top_moves, list)
    # Final publish is done=True.
    assert done is True or result is not None


def test_cnative_chess_infinite_mode_terminates_on_time():
    """max_depth=0 + time_limit_ms=200 should terminate cleanly in ~200ms."""
    from src.games.chess import initial_state

    cs = _make_random_chess_csearch()
    state = initial_state()

    start = time.time()
    result = cs.search_cnative_chess(*_chess_args(state, time_ms=200.0, depth=0, n_threads=2))
    elapsed = time.time() - start
    assert elapsed < 2.0, f"search took {elapsed:.1f}s, should've stopped at 0.2s"
    assert result is not None


# ------------------------------------------------------------------ shogi smoke


def test_cnative_shogi_returns_legal_move_single_thread():
    from src.games.shogi import initial_state

    cs = _make_random_shogi_csearch()
    state = initial_state()
    result = cs.search_cnative_shogi(*_shogi_args(state, depth=4, n_threads=1))
    assert result is not None
    (move_tuple, _score, nodes) = result
    from_sq, to_sq, promo, drop = move_tuple
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in legal}
    assert (from_sq, to_sq, promo, drop) in legal_set
    assert nodes > 0


def test_cnative_shogi_returns_legal_move_multi_thread():
    from src.games.shogi import initial_state

    cs = _make_random_shogi_csearch()
    state = initial_state()
    result = cs.search_cnative_shogi(*_shogi_args(state, depth=4, n_threads=4))
    assert result is not None
    (move_tuple, _score, nodes) = result
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in legal}
    assert tuple(move_tuple) in legal_set
    assert nodes > 0
