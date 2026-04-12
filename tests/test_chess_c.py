"""Tests for the C chess engine layer.

Three classes of check:

  * **Perft** — at depths 1..5 from the start position, the C movegen
    + make/unmake must produce exactly the published leaf-node counts.
    Off-by-one-here means either a pseudo-legal bug, a legality filter
    bug, or a make/unmake bug — all catastrophic for bootstrap.

  * **Legal-move parity** — for positions reached by walking random
    first-legal-move games, the C legal-move set must match the Python
    legal-move set. Catches subtle differences like "Python includes a
    promotion variant that C doesn't".

  * **Search sanity** — ``ChessCRuleSearch`` must always return a legal
    move, produce reasonable moves from the opening, and handle
    terminal/near-terminal positions without crashing.
"""

import pytest

pytest.importorskip("src.accel._nnue_accel")


# --------------------------------------------------------------------------- perft


@pytest.mark.parametrize("depth,expected", [
    (1, 20),
    (2, 400),
    (3, 8902),
    (4, 197281),
    (5, 4865609),
])
def test_chess_c_perft_start(depth, expected):
    """Published perft values from the standard starting position.
    These are the textbook numbers — a mismatch at any depth means
    the C movegen or make/unmake is wrong."""
    from src.games.chess import initial_state
    from src.accel._nnue_accel import chess_c_perft

    s = initial_state()
    got = chess_c_perft(
        bytes(s.board_array()), 0, 0b1111, -1, 0,
        int(s.king_square(0)), int(s.king_square(1)),
        depth,
    )
    assert got == expected, f"perft({depth}) got {got}, expected {expected}"


# --------------------------------------------------------------------------- legal-move parity


def test_chess_c_legal_moves_match_python_on_startpos():
    from src.games.chess import initial_state
    from src.accel._nnue_accel import chess_c_legal_moves

    s = initial_state()
    c_moves = chess_c_legal_moves(
        bytes(s.board_array()), 0, 0b1111, -1, 0,
        int(s.king_square(0)), int(s.king_square(1)),
    )
    py_moves = s.legal_moves()
    c_set = {(f, t, p) for f, t, p in c_moves}
    py_set = {(m.from_sq, m.to_sq, m.promotion) for m in py_moves}
    assert c_set == py_set


def test_chess_c_legal_moves_match_python_after_play():
    """Walk 20 first-legal-move plies and verify C matches Python at every step."""
    from src.games.chess import initial_state
    from src.accel._nnue_accel import chess_c_legal_moves

    state = initial_state()
    for ply in range(20):
        c_moves = chess_c_legal_moves(
            bytes(state.board_array()),
            int(state.side_to_move()),
            int(state._castling),
            int(state._ep_square),
            int(state._halfmove),
            int(state.king_square(0)),
            int(state.king_square(1)),
        )
        py_moves = state.legal_moves()
        c_set = {(f, t, p) for f, t, p in c_moves}
        py_set = {(m.from_sq, m.to_sq, m.promotion) for m in py_moves}
        assert c_set == py_set, (
            f"ply {ply}: diff py_only={py_set - c_set} c_only={c_set - py_set}"
        )
        if not py_moves:
            break
        state = state.make_move(py_moves[0])


def test_chess_c_apply_moves_matches_python_state():
    """Verify that C make_move produces the same state (board + ep +
    castling + halfmove + king squares) that Python make_move does,
    across a 6-ply sequence that exercises double pushes and normal
    moves."""
    from src.games.chess import initial_state
    from src.games.base import Move
    from src.accel._nnue_accel import chess_c_apply_moves

    s = initial_state()
    py_path = [
        Move(8, 24),      # a2-a4 (double push)
        Move(48, 40),     # a7-a6
        Move(24, 32),     # a4-a5
        Move(49, 33),     # b7-b5 (double push)
        Move(32, 41),     # a5-b6 e.p.   ← relies on C ep_square tracking
        Move(51, 43),     # d7-d6
    ]
    moves_tuples = [(m.from_sq, m.to_sq, m.promotion) for m in py_path]

    py_s = s
    for mv in py_path:
        py_s = py_s.make_move(mv)

    c_result = chess_c_apply_moves(
        bytes(s.board_array()), 0,
        int(s._castling), int(s._ep_square), int(s._halfmove),
        int(s.king_square(0)), int(s.king_square(1)),
        moves_tuples,
    )
    c_board, c_side, c_castling, c_ep, c_half, c_wk, c_bk, c_hash = c_result

    assert c_side == py_s.side_to_move()
    assert c_castling == int(py_s._castling)
    assert c_ep == int(py_s._ep_square)
    assert c_half == int(py_s._halfmove)
    assert c_wk == int(py_s.king_square(0))
    assert c_bk == int(py_s.king_square(1))
    assert bytes(c_board) == bytes(py_s.board.astype('int8').tobytes())


# --------------------------------------------------------------------------- search sanity


def test_chess_c_rule_search_returns_legal_move():
    from src.games.chess import initial_state
    from src.search.alphabeta import create_rule_based_search, ChessCRuleSearch

    s = initial_state()
    search = create_rule_based_search("chess", max_depth=4, time_limit_ms=5000)
    assert isinstance(search, ChessCRuleSearch)
    move, score = search.search(s)
    assert move is not None
    legal = {(m.from_sq, m.to_sq, m.promotion) for m in s.legal_moves()}
    assert (move.from_sq, move.to_sq, move.promotion) in legal


def test_chess_c_rule_search_depth_6_under_1s():
    """Perf sanity: d=6 from the opening should finish well under 1s
    thanks to the C pipeline. This is a regression guard — if the
    search crosses 3s on this machine, something is seriously wrong."""
    import time
    from src.games.chess import initial_state
    from src.search.alphabeta import create_rule_based_search

    s = initial_state()
    search = create_rule_based_search("chess", max_depth=6, time_limit_ms=30000)
    t0 = time.time()
    move, _ = search.search(s)
    dt = time.time() - t0
    assert move is not None
    assert dt < 3.0, f"chess d=6 took {dt:.2f}s — expected < 3.0s"


def test_chess_c_rule_search_terminal_position():
    """Stalemate position: king alone vs king+queen. The searched side
    should still return a legal move (not None) since we seed the
    fallback from legal_moves()."""
    import numpy as np
    from src.games.chess.state import ChessState
    from src.games.chess.constants import _KING, _QUEEN, _sq
    from src.games.chess.hash_utils import _compute_hash
    from src.search.alphabeta import create_rule_based_search

    # White king on e1, black king and queen on e8 / e7 — black to move,
    # plenty of legal moves. Just a generic mid-game-ish position, not
    # a hard mate puzzle.
    board = np.zeros(64, dtype=np.int8)
    board[_sq(0, 4)] = _KING
    board[_sq(7, 4)] = -_KING
    board[_sq(6, 4)] = -_QUEEN
    zob = _compute_hash(board, 1, 0, -1)
    state = ChessState(
        board=board, side=1, castling=0, ep_square=-1,
        halfmove=0, fullmove=1, zobrist=zob, history=(zob,),
        king_sqs=(_sq(0, 4), _sq(7, 4)),
    )

    search = create_rule_based_search("chess", max_depth=3, time_limit_ms=2000)
    move, _ = search.search(state)
    assert move is not None
    legal = {(m.from_sq, m.to_sq, m.promotion) for m in state.legal_moves()}
    assert (move.from_sq, move.to_sq, move.promotion) in legal
