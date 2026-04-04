"""Tests for HalfKP feature extraction."""

from src.games.chess import initial_state as chess_init
from src.games.minichess import initial_state as minichess_init
from src.games.shogi import initial_state as shogi_init
from src.games.minishogi import initial_state as minishogi_init
from src.features.halfkp import chess_features, minichess_features, HalfKP
from src.features.halfkp_shogi import shogi_features, minishogi_features


def test_chess_feature_dimensions():
    fs = chess_features()
    assert fs.num_features() == 40960  # 64 * 10 * 64
    assert fs.max_active_features() == 30


def test_minichess_feature_dimensions():
    fs = minichess_features()
    assert fs.num_features() == 36 * 8 * 36  # 10368
    assert fs.max_active_features() == 22


def test_shogi_feature_dimensions():
    fs = shogi_features()
    assert fs.num_features() > 0
    assert fs.max_active_features() > 0


def test_minishogi_feature_dimensions():
    fs = minishogi_features()
    assert fs.num_features() > 0
    assert fs.max_active_features() > 0


def test_chess_active_features():
    fs = chess_features()
    s = chess_init()
    wf = fs.active_features(s, 0)
    bf = fs.active_features(s, 1)
    # Should have features for all non-king pieces (30)
    assert len(wf) == 30
    assert len(bf) == 30
    # All indices in range
    assert all(0 <= idx < fs.num_features() for idx in wf)
    assert all(0 <= idx < fs.num_features() for idx in bf)


def test_minichess_active_features():
    fs = minichess_features()
    s = minichess_init()
    wf = fs.active_features(s, 0)
    bf = fs.active_features(s, 1)
    assert len(wf) > 0
    assert len(bf) > 0
    assert all(0 <= idx < fs.num_features() for idx in wf)


def test_shogi_active_features():
    fs = shogi_features()
    s = shogi_init()
    wf = fs.active_features(s, 0)
    bf = fs.active_features(s, 1)
    assert len(wf) > 0
    assert all(0 <= idx < fs.num_features() for idx in wf)


def test_minishogi_active_features():
    fs = minishogi_features()
    s = minishogi_init()
    wf = fs.active_features(s, 0)
    bf = fs.active_features(s, 1)
    assert len(wf) > 0
    assert all(0 <= idx < fs.num_features() for idx in wf)


def test_feature_delta_non_king_move():
    """Feature delta should return added/removed lists for non-king moves."""
    fs = minichess_features()
    s = minichess_init()
    moves = s.legal_moves()
    # Find a pawn move (non-king)
    pawn_move = None
    for m in moves:
        if m.from_sq is not None and m.from_sq != s.king_square(0):
            pawn_move = m
            break
    assert pawn_move is not None

    s2 = s.make_move(pawn_move)
    delta = fs.feature_delta(s, pawn_move, s2, 0)
    assert delta is not None
    added, removed = delta
    assert len(added) > 0 or len(removed) > 0


def test_feature_delta_king_move_returns_none():
    """Feature delta should return None when king moves (need full refresh)."""
    fs = HalfKP(num_squares=36, num_piece_types=4)
    s = minichess_init()

    # Play moves until king can move
    state = s
    king_move_found = False
    for _ in range(50):
        moves = state.legal_moves()
        if not moves or state.is_terminal():
            break
        # Check for king move
        king_sq = state.king_square(state.side_to_move())
        for m in moves:
            if m.from_sq == king_sq:
                s2 = state.make_move(m)
                delta = fs.feature_delta(state, m, s2, state.side_to_move())
                assert delta is None  # King moved -> full refresh
                king_move_found = True
                break
        if king_move_found:
            break
        state = state.make_move(moves[0])
    # It's OK if no king move was found in 50 moves
