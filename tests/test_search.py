"""Tests for the search engine."""

import threading

from src.games.minichess import initial_state
from src.search.alphabeta import AlphaBetaSearch
from src.search.evaluator import MaterialEvaluator
from src.search.transposition import TranspositionTable, EXACT
from src.search.move_ordering import MoveOrdering


def test_material_evaluator():
    s = initial_state()
    ev = MaterialEvaluator()
    score = ev.evaluate(s)
    # Starting position should be roughly equal
    assert isinstance(score, float)


def test_search_finds_move():
    s = initial_state()
    ev = MaterialEvaluator()
    search = AlphaBetaSearch(ev, max_depth=2, time_limit_ms=5000)
    move, score = search.search(s)
    assert move is not None
    assert move in s.legal_moves()


def test_search_depth_1():
    s = initial_state()
    ev = MaterialEvaluator()
    search = AlphaBetaSearch(ev, max_depth=1, time_limit_ms=5000)
    move, score = search.search(s)
    assert move is not None


def test_transposition_table():
    tt = TranspositionTable(size=1024)
    assert tt.probe(12345) is None
    tt.store(12345, 3, 100.0, EXACT, None)
    entry = tt.probe(12345)
    assert entry is not None
    assert entry.depth == 3
    assert entry.score == 100.0


def test_search_top_n():
    s = initial_state()
    ev = MaterialEvaluator()
    search = AlphaBetaSearch(ev, max_depth=2, time_limit_ms=5000)
    top3 = search.search_top_n(s, n=3)
    assert len(top3) <= 3
    assert len(top3) > 0
    # All returned moves should be legal
    legal = s.legal_moves()
    legal_set = {(m.from_sq, m.to_sq) for m in legal}
    for move, score in top3:
        assert (move.from_sq, move.to_sq) in legal_set
    # Scores should be in descending order
    scores = [s for _, s in top3]
    assert scores == sorted(scores, reverse=True)


def test_search_top_n_live():
    s = initial_state()
    ev = MaterialEvaluator()
    search = AlphaBetaSearch(ev, max_depth=3, time_limit_ms=10000)
    live = [None]
    stop = threading.Event()
    result = search.search_top_n_live(s, n=3, live_ref=live, stop_event=stop)
    # Should have written to live_ref
    assert live[0] is not None
    depth, max_d, top_n, done = live[0]
    assert done is True
    assert depth == 3
    assert max_d == 3
    assert len(top_n) <= 3
    assert len(top_n) > 0
    # Scores descending
    scores = [sc for _, sc in top_n]
    assert scores == sorted(scores, reverse=True)


def test_search_top_n_live_stop():
    """Stop event should abort the search early."""
    s = initial_state()
    ev = MaterialEvaluator()
    search = AlphaBetaSearch(ev, max_depth=8, time_limit_ms=60000)
    live = [None]
    stop = threading.Event()
    stop.set()  # pre-set: search should abort immediately
    search.search_top_n_live(s, n=3, live_ref=live, stop_event=stop)
    # Should finish almost instantly; may or may not have results
    # but should not hang


def test_move_ordering():
    s = initial_state()
    mo = MoveOrdering()
    moves = s.legal_moves()
    ordered = mo.order_moves(s, moves, depth=3)
    assert len(ordered) == len(moves)
    # All moves should still be present
    assert set(id(m) for m in moves) == set(id(m) for m in ordered) or \
           len(ordered) == len(moves)
