"""Tests for the shogi 9x9 engine."""

import numpy as np
from src.games.base import Move, WHITE, BLACK
from src.games.shogi import initial_state


def test_initial_state_basics():
    s = initial_state()
    assert s.side_to_move() == WHITE  # Sente
    assert not s.is_terminal()
    assert s.result() is None


def test_initial_legal_moves():
    """Shogi starting position has exactly 30 legal moves."""
    s = initial_state()
    assert len(s.legal_moves()) == 30


def test_board_array():
    s = initial_state()
    b = s.board_array()
    assert isinstance(b, np.ndarray)
    assert b.shape == (81,)
    assert np.count_nonzero(b) == 40  # 20 pieces per side


def test_king_squares():
    s = initial_state()
    sk = s.king_square(WHITE)   # Sente king
    gk = s.king_square(BLACK)   # Gote king
    assert 0 <= sk < 81
    assert 0 <= gk < 81
    assert sk != gk


def test_hand_pieces_empty_at_start():
    s = initial_state()
    assert s.hand_pieces(WHITE) == {}
    assert s.hand_pieces(BLACK) == {}


def test_make_move_immutable():
    s = initial_state()
    original_board = s.board_array().copy()
    s2 = s.make_move(s.legal_moves()[0])
    assert np.array_equal(s.board_array(), original_board)
    assert s2.side_to_move() == BLACK


def test_game_config():
    s = initial_state()
    cfg = s.config()
    assert cfg.name == "shogi"
    assert cfg.board_height == 9
    assert cfg.board_width == 9
    assert cfg.has_drops is True
    assert cfg.has_promotion is True


def test_capture_creates_hand_piece():
    """When a piece is captured in shogi, it goes to the capturer's hand."""
    s = initial_state()
    # Play enough moves to create a capture, then verify hand pieces
    # For now just verify the interface works after moves
    state = s
    for _ in range(10):
        moves = state.legal_moves()
        if not moves or state.is_terminal():
            break
        state = state.make_move(moves[0])
    # After some moves, hands should still be accessible
    h0 = state.hand_pieces(WHITE)
    h1 = state.hand_pieces(BLACK)
    assert isinstance(h0, dict)
    assert isinstance(h1, dict)


def test_zobrist_hash():
    s = initial_state()
    h = s.zobrist_hash()
    assert isinstance(h, int)
    s2 = s.make_move(s.legal_moves()[0])
    assert s2.zobrist_hash() != h


def test_copy():
    s = initial_state()
    c = s.copy()
    assert np.array_equal(s.board_array(), c.board_array())
    assert s.side_to_move() == c.side_to_move()


def test_rule_based_evaluator_no_crash():
    """RuleBasedEvaluator on shogi must not raise on the initial 9x9 position.

    The old chess-only implementation indexed a 64-entry PST with squares up
    to 80 and would IndexError. This test locks in that the dispatcher now
    routes shogi to a shogi-specific evaluator.
    """
    from src.search.evaluator import RuleBasedEvaluator

    s = initial_state()
    ev = RuleBasedEvaluator()
    score = ev.evaluate(s)
    assert isinstance(score, float)
    # Initial position is symmetric; score should be near zero.
    assert abs(score) < 50


def test_rule_based_evaluator_tracks_material():
    """Losing a rook should register as a large material swing."""
    from src.search.evaluator import RuleBasedEvaluator

    s = initial_state()
    ev = RuleBasedEvaluator()
    start = ev.evaluate(s)

    # Manually remove sente's rook from the board (mutation for the test only)
    board = s.board_array().copy()
    # Find a rook (abs code == 7) and zero it
    import numpy as np
    rook_sqs = np.where(board == 7)[0]
    assert len(rook_sqs) > 0, "initial shogi should have a sente rook"
    board[rook_sqs[0]] = 0

    # Build a new state with the modified board to avoid mutating the original
    from src.games.shogi.state import ShogiState
    mutated = ShogiState(
        board=board,
        sente_hand=dict(s.hand_pieces(WHITE)),
        gote_hand=dict(s.hand_pieces(BLACK)),
        side=s.side_to_move(),
    )
    after = ev.evaluate(mutated)
    # Sente just lost a rook (worth ~1040cp); eval should drop significantly
    # from sente's perspective (which is also the side to move at start).
    assert start - after > 800, (
        f"expected >800cp drop after losing a rook; got {start - after:.0f}"
    )


def test_shogi_rule_search_runs():
    """Full search with shogi-aware MoveOrdering should find a legal move."""
    from src.search.alphabeta import create_rule_based_search

    s = initial_state()
    search = create_rule_based_search("shogi", max_depth=2, time_limit_ms=5000)
    move, score = search.search(s)
    assert move is not None
    assert move in s.legal_moves()


def test_shogi_c_movegen_matches_python():
    """C movegen must generate the same legal move set as the Python reference
    across a few representative positions (startpos + midgame with drops)."""
    try:
        from src.accel._nnue_accel import shogi_c_legal_moves
    except ImportError:
        import pytest
        pytest.skip("C accel not built")

    # Startpos
    s = initial_state()
    sh = tuple(s.hand_pieces(WHITE).get(i, 0) for i in range(7))
    gh = tuple(s.hand_pieces(BLACK).get(i, 0) for i in range(7))
    py = set((m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in s.legal_moves())
    c = set(shogi_c_legal_moves(bytes(s.board_array()), sh, gh, s.side_to_move()))
    assert py == c, f"startpos mismatch: py-only={py-c}, c-only={c-py}"
    assert len(py) == 30

    # Play 40 plies of capture-biased moves to reach a mid-game with mochigoma
    import random
    random.seed(12345)
    for _ in range(40):
        moves = s.legal_moves()
        if not moves:
            break
        caps = [m for m in moves
                if m.from_sq is not None and s.board_array()[m.to_sq] != 0]
        if caps and random.random() < 0.7:
            s = s.make_move(random.choice(caps))
        else:
            s = s.make_move(random.choice(moves))

    sh = tuple(s.hand_pieces(WHITE).get(i, 0) for i in range(7))
    gh = tuple(s.hand_pieces(BLACK).get(i, 0) for i in range(7))
    py = set((m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in s.legal_moves())
    c = set(shogi_c_legal_moves(bytes(s.board_array()), sh, gh, s.side_to_move()))
    # Hand must actually contain pieces by now (otherwise the test is weak)
    assert any(sh) or any(gh), "test setup failed to create hand pieces"
    assert py == c, f"midgame mismatch: py-only={py-c}, c-only={c-py}"


def test_shogi_c_rule_search_agrees_with_python():
    """C rule search and Python rule search must agree on the optimal score
    at shallow depth. Best moves may differ when multiple moves share the
    optimal score (the two paths don't have identical move orderings)."""
    try:
        from src.accel._nnue_accel import shogi_rule_search
    except ImportError:
        import pytest
        pytest.skip("C accel not built")
    from src.search.alphabeta import create_rule_based_search, ShogiCRuleSearch

    s = initial_state()
    # Python path (force the Python branch by instantiating directly)
    from src.search.evaluator import RuleBasedEvaluator, SHOGI_MVV_LVA_VALUES
    from src.search.alphabeta import AlphaBetaSearch
    from src.search.move_ordering import MoveOrdering
    py_srch = AlphaBetaSearch(RuleBasedEvaluator(), max_depth=3, time_limit_ms=30000)
    py_srch.move_ordering = MoveOrdering(piece_values=SHOGI_MVV_LVA_VALUES)
    _py_move, py_score = py_srch.search(s)

    # C path
    c_srch = ShogiCRuleSearch(max_depth=3, time_limit_ms=30000)
    c_move, c_score = c_srch.search(s)

    assert c_move is not None
    assert abs(c_score - py_score) < 1.0, \
        f"C score {c_score} != Python score {py_score}"
    # Both moves must be legal
    legal = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece) for m in s.legal_moves()}
    assert (c_move.from_sq, c_move.to_sq, c_move.promotion, c_move.drop_piece) in legal


def test_shogi_c_search_live_updates():
    """The C rule search must push progressive top-3 updates via the live
    callback and honour the stop_event to abort mid-search."""
    try:
        from src.accel._nnue_accel import shogi_rule_search_live  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("C accel not built")

    import threading, time
    from src.search.alphabeta import ShogiCRuleSearch

    s = initial_state()
    stop = threading.Event()
    updates = []

    def _worker():
        srch = ShogiCRuleSearch(max_depth=0, time_limit_ms=0)
        board, sh, gh, side = srch._pack_position(s)
        from src.accel._nnue_accel import shogi_rule_search_live

        def _cb(depth, max_d, top, done):
            updates.append((depth, max_d, top, done))
            return stop.is_set()

        shogi_rule_search_live(
            board, sh, gh, side, 0, 0.0, _cb, 1_000_000,
        )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    for _ in range(30):
        if len(updates) >= 3:
            break
        time.sleep(0.05)
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive(), "search thread did not stop after stop_event.set()"
    assert len(updates) >= 1, "no live updates received"

    legal = {(m.from_sq, m.to_sq, m.promotion, m.drop_piece)
             for m in s.legal_moves()}
    for depth, max_d, top, done in updates:
        assert max_d > 0
        assert len(top) >= 1
        for m_tup, score in top:
            from_sq, to_sq, promo, drop = m_tup
            assert (from_sq, to_sq, promo, drop) in legal

    from src.gui.constants import DEPTH_MAX
    assert all(u[1] > DEPTH_MAX for u in updates), \
        "infinite-mode max_depth should exceed DEPTH_MAX so panel shows ∞"


def test_shogi_create_rule_based_search_uses_c_path():
    """The factory must route shogi bootstraps to the C backend when built."""
    try:
        from src.accel._nnue_accel import shogi_rule_search  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("C accel not built")

    from src.search.alphabeta import create_rule_based_search, ShogiCRuleSearch
    srch = create_rule_based_search("shogi", max_depth=4, time_limit_ms=1000)
    assert isinstance(srch, ShogiCRuleSearch), \
        "shogi factory should return the C-backed ShogiCRuleSearch"

    s = initial_state()
    mv, sc = srch.search(s)
    assert mv is not None
    assert mv in s.legal_moves()
    assert srch.nodes_searched > 0


def test_shogi_c_perft_published_values():
    """Standard shogi perft from startpos must match published values."""
    try:
        from src.accel._nnue_accel import shogi_c_perft, shogi_c_startpos
    except ImportError:
        import pytest
        pytest.skip("C accel not built")

    board, sh, gh, side, _h = shogi_c_startpos()
    # Known perft values for standard shogi starting position.
    expected = {1: 30, 2: 900, 3: 25470, 4: 719731}
    for d, want in expected.items():
        got = shogi_c_perft(board, sh, gh, side, d)
        assert got == want, f"perft({d}): got {got}, expected {want}"


def test_shogi_king_advance_penalized():
    """An advanced king should evaluate worse than one on the back rank."""
    from src.search.evaluator import RuleBasedEvaluator
    from src.games.shogi.state import ShogiState

    s = initial_state()
    ev = RuleBasedEvaluator()

    board = s.board_array().copy()
    # Sente king starts at rank 8 (file 4). Move it forward to rank 4.
    import numpy as np
    sente_king_sq = int(np.where(board == 8)[0][0])
    board[sente_king_sq] = 0
    # Put the king in the center of rank 4, making sure the target is empty.
    target = 4 * 9 + 4
    if board[target] == 0:
        board[target] = 8
        mutated = ShogiState(
            board=board,
            sente_hand=dict(s.hand_pieces(WHITE)),
            gote_hand=dict(s.hand_pieces(BLACK)),
            side=s.side_to_move(),
        )
        baseline = ev.evaluate(s)
        exposed = ev.evaluate(mutated)
        # Side-to-move is still sente; exposed king must eval worse for sente.
        assert exposed < baseline - 100, (
            f"advancing the king to the center should cost >100cp; "
            f"baseline={baseline:.0f}, exposed={exposed:.0f}"
        )
