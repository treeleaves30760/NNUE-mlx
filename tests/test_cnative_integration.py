"""Integration tests: AlphaBetaSearch dispatches to C-native for
chess/shogi and Python alphabeta for mini variants, while the live-
analysis API publishes progress for all four games."""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("src.accel._nnue_accel")

from src.accel._nnue_accel import AcceleratedAccumulator
from src.features.halfkp import HalfKP
from src.features.halfkp_shogi import shogi_features
from src.games.chess import initial_state as chess_initial
from src.games.shogi import initial_state as shogi_initial
from src.games.minichess import initial_state as minichess_initial
from src.games.minishogi import initial_state as minishogi_initial
from src.search.alphabeta import AlphaBetaSearch
from src.search.evaluator import NNUEEvaluator


_KEEP: list = []


def _build_nnue_evaluator(feature_set, seed=42, l1=32, l2=32, acc=256):
    num_feat = feature_set.num_features()
    rng = np.random.RandomState(seed)
    ft_w = (rng.randn(num_feat, acc) * 0.005).astype(np.float32)
    ft_b = (rng.randn(acc) * 0.01).astype(np.float32)
    l1_w = (rng.randn(l1, acc * 2) * 0.1).astype(np.float32)
    l1_b = (rng.randn(l1) * 0.01).astype(np.float32)
    l2_w = (rng.randn(l2, l1) * 0.1).astype(np.float32)
    l2_b = (rng.randn(l2) * 0.01).astype(np.float32)
    out_w = (rng.randn(l2) * 0.1).astype(np.float32)
    out_b = np.array([0.0], dtype=np.float32)
    accumulator = AcceleratedAccumulator(
        ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b,
    )
    evaluator = NNUEEvaluator(accumulator, feature_set)
    _KEEP.append((evaluator, accumulator, feature_set,
                  ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, out_w, out_b))
    return evaluator


# ------------------------------------------------------------------ chess path


def test_alphabeta_chess_uses_cnative_path():
    """When given a chess state, AlphaBetaSearch.search should pick up
    a legal move quickly via the C-native Lazy SMP path."""
    fs = HalfKP(
        num_squares=64, num_piece_types=5, king_board_val=6,
        board_val_to_type={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    )
    evaluator = _build_nnue_evaluator(fs)
    search = AlphaBetaSearch(evaluator, max_depth=5, time_limit_ms=500,
                              n_threads=2)
    state = chess_initial()

    start = time.time()
    move, score = search.search(state)
    elapsed = time.time() - start

    assert move is not None
    assert elapsed < 5.0, f"search took {elapsed:.2f}s"
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in legal}
    assert (move.from_sq, move.to_sq, move.promotion, move.drop_piece) in legal_set


def test_alphabeta_chess_live_top_n_publishes_progress():
    fs = HalfKP(
        num_squares=64, num_piece_types=5, king_board_val=6,
        board_val_to_type={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    )
    evaluator = _build_nnue_evaluator(fs)
    search = AlphaBetaSearch(evaluator, max_depth=6, time_limit_ms=400,
                              n_threads=2)
    state = chess_initial()
    live = [None]
    stop = threading.Event()

    top = search.search_top_n_live(state, n=3, live_ref=live, stop_event=stop)
    assert top, "expected at least one top move"
    for mv, _sc in top:
        legal = state.legal_moves()
        legal_set = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in legal}
        assert (mv.from_sq, mv.to_sq, mv.promotion, mv.drop_piece) in legal_set
    assert live[0] is not None


# ------------------------------------------------------------------ shogi path


def test_alphabeta_shogi_uses_cnative_path():
    fs = shogi_features()
    evaluator = _build_nnue_evaluator(fs)
    search = AlphaBetaSearch(evaluator, max_depth=4, time_limit_ms=500,
                              n_threads=2)
    state = shogi_initial()

    move, _score = search.search(state)
    assert move is not None
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in legal}
    assert (move.from_sq, move.to_sq, move.promotion, move.drop_piece) in legal_set


# ------------------------------------------------------------------ mini variants


def test_alphabeta_minichess_falls_back_to_python():
    """Minichess has no C movegen — should run through the Python
    alphabeta path and still return a legal move."""
    from src.features.halfkp import HalfKP
    fs = HalfKP(
        num_squares=36, num_piece_types=4, king_board_val=5,
        board_val_to_type={1: 0, 2: 1, 3: 2, 4: 3},
    )
    evaluator = _build_nnue_evaluator(fs)
    search = AlphaBetaSearch(evaluator, max_depth=3, time_limit_ms=1000,
                              n_threads=2)
    state = minichess_initial()
    move, _score = search.search(state)
    assert move is not None
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in legal}
    assert (move.from_sq, move.to_sq, move.promotion, move.drop_piece) in legal_set


def test_alphabeta_minichess_live_uses_python_path():
    from src.features.halfkp import HalfKP
    fs = HalfKP(
        num_squares=36, num_piece_types=4, king_board_val=5,
        board_val_to_type={1: 0, 2: 1, 3: 2, 4: 3},
    )
    evaluator = _build_nnue_evaluator(fs)
    search = AlphaBetaSearch(evaluator, max_depth=2, time_limit_ms=500,
                              n_threads=2)
    state = minichess_initial()
    live = [None]
    top = search.search_top_n_live(state, n=2, live_ref=live)
    assert top, "expected at least one top move (minichess Python fallback)"


def test_alphabeta_minishogi_falls_back_to_python():
    from src.features.halfkp_shogi import minishogi_features
    fs = minishogi_features()
    evaluator = _build_nnue_evaluator(fs)
    search = AlphaBetaSearch(evaluator, max_depth=3, time_limit_ms=1000,
                              n_threads=2)
    state = minishogi_initial()
    move, _score = search.search(state)
    assert move is not None
    legal = state.legal_moves()
    legal_set = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in legal}
    assert (move.from_sq, move.to_sq, move.promotion, move.drop_piece) in legal_set
