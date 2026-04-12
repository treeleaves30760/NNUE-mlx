"""Tests for the self-play tournament harness.

Two classes of check:
  * **Shape**: the returned ``TournamentResult`` is internally
    consistent — the reported wins/draws/losses sum to the game count,
    the A-score is within [0, 1], each game record has a legal result.
  * **Strength**: a RuleBasedEvaluator should not lose to a
    MaterialEvaluator on minichess — if our rule-based logic ever
    regresses to below pure material it shows up here immediately.

We use minichess on purpose: it's a 6x6 board so depth-3 searches
finish in ~50-100ms, making a 6-game match complete in well under a
minute even on slow hardware.
"""

import pytest

from src.search.alphabeta import AlphaBetaSearch
from src.search.evaluator import MaterialEvaluator, RuleBasedEvaluator
from src.training.tournament import (
    EngineSpec,
    GameRecord,
    TournamentResult,
    build_engine_from_spec,
    play_tournament,
    _score_to_elo,
    _wilson_half_width,
)


def _mat_factory():
    return AlphaBetaSearch(MaterialEvaluator(), max_depth=2, time_limit_ms=200)


def _rule_factory():
    return AlphaBetaSearch(RuleBasedEvaluator(), max_depth=2, time_limit_ms=200)


# --------------------------------------------------------------------------- shape


def test_tournament_result_is_internally_consistent():
    """wins + losses + draws must equal games; a_score in [0, 1]."""
    result = play_tournament(
        "minichess",
        _rule_factory, _mat_factory,
        num_games=4,
        time_per_move_ms=200,
        max_moves=60,
        opening_plies=2,
        seed=1,
    )
    assert isinstance(result, TournamentResult)
    assert result.games == 4
    assert result.a_wins + result.b_wins + result.draws == 4
    assert 0.0 <= result.a_score <= 1.0
    assert len(result.records) == 4
    for record in result.records:
        assert isinstance(record, GameRecord)
        assert record.result in (0.0, 0.5, 1.0)
        assert record.reason in ("normal", "max_moves", "error")
        assert record.plies >= 0


def test_tournament_alternates_colors():
    """Games 0, 2 should have A as white; games 1, 3 should have B as white."""
    result = play_tournament(
        "minichess",
        _rule_factory, _mat_factory,
        num_games=4,
        time_per_move_ms=100,
        max_moves=40,
        opening_plies=2,
        seed=1,
    )
    assert result.records[0].white_is_a is True
    assert result.records[1].white_is_a is False
    assert result.records[2].white_is_a is True
    assert result.records[3].white_is_a is False


def test_tournament_seed_reproducible():
    """Same seed must produce same game outcomes."""
    r1 = play_tournament(
        "minichess",
        _rule_factory, _mat_factory,
        num_games=3, time_per_move_ms=100, max_moves=40,
        opening_plies=2, seed=42,
    )
    r2 = play_tournament(
        "minichess",
        _rule_factory, _mat_factory,
        num_games=3, time_per_move_ms=100, max_moves=40,
        opening_plies=2, seed=42,
    )
    assert r1.a_wins == r2.a_wins
    assert r1.b_wins == r2.b_wins
    assert r1.draws == r2.draws


# --------------------------------------------------------------------------- strength


def test_rule_based_does_not_lose_to_material_on_minichess():
    """On minichess at equal search depth, RuleBased must score at
    least as well as Material across a small but diverse match.

    This is a regression guard: if a future change to
    ``_minichess_rule_based`` makes it weaker than pure material, the
    test fails and we know immediately without shipping broken
    bootstrap labels.
    """
    result = play_tournament(
        "minichess",
        _rule_factory, _mat_factory,
        num_games=8,
        time_per_move_ms=200,
        max_moves=60,
        opening_plies=2,
        seed=7,
    )
    # RuleBased should score at least 50% — lower bound on strength.
    # We don't assert a specific Elo because 8 games is too few for a
    # tight estimate; we just require "not worse than material".
    assert result.a_score >= 0.5, (
        f"RuleBased scored {result.a_score:.2f} vs Material — "
        f"rule eval regressed below pure material"
    )


# --------------------------------------------------------------------------- elo helpers


def test_score_to_elo_monotonic():
    """Higher win fraction must produce strictly higher Elo."""
    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    elos = [_score_to_elo(f) for f in fractions]
    assert elos == sorted(elos)
    # 50% score should map to 0 Elo
    assert abs(_score_to_elo(0.5)) < 1e-9


def test_score_to_elo_clipping():
    assert _score_to_elo(0.0) == -800.0
    assert _score_to_elo(1.0) == 800.0


def test_wilson_half_width_narrows_with_more_games():
    narrow = _wilson_half_width(50, 100)
    wide = _wilson_half_width(5, 10)
    assert narrow < wide, (
        "Wilson confidence should narrow as N grows"
    )


# --------------------------------------------------------------------------- EngineSpec


def test_engine_spec_rule_builds_functional_engine():
    """build_engine_from_spec(rule, chess) must return something whose
    .search() picks a legal move on the start position."""
    from src.games.chess import initial_state

    spec = EngineSpec(evaluator="rule", max_depth=2, time_limit_ms=200)
    engine = build_engine_from_spec(spec, "chess")
    s = initial_state()
    move, _ = engine.search(s)
    assert move is not None
    legal = {(m.from_sq, m.to_sq, m.promotion) for m in s.legal_moves()}
    assert (move.from_sq, move.to_sq, move.promotion) in legal


def test_engine_spec_material_builds_functional_engine():
    from src.games.chess import initial_state

    spec = EngineSpec(evaluator="material", max_depth=2, time_limit_ms=200)
    engine = build_engine_from_spec(spec, "chess")
    s = initial_state()
    move, _ = engine.search(s)
    assert move is not None


def test_engine_spec_rejects_unknown_evaluator():
    import pytest
    with pytest.raises(ValueError):
        build_engine_from_spec(EngineSpec(evaluator="alien"), "chess")


# --------------------------------------------------------------------------- multiprocessing path


def test_tournament_mp_requires_specs():
    """MP mode can't use factory closures — the shape test is that
    factories-only tournaments raise ValueError when num_workers > 1."""
    import pytest
    with pytest.raises(ValueError):
        play_tournament(
            "minichess",
            _rule_factory, _mat_factory,
            num_games=2, num_workers=2,
            time_per_move_ms=100, max_moves=30,
        )


def test_tournament_mp_shape():
    """A 4-game MP tournament on minichess returns a well-formed
    TournamentResult. Uses 2 workers to exercise the parallel path."""
    spec_a = EngineSpec(evaluator="rule", max_depth=2, time_limit_ms=100)
    spec_b = EngineSpec(evaluator="material", max_depth=2, time_limit_ms=100)
    result = play_tournament(
        "minichess",
        engine_a_spec=spec_a,
        engine_b_spec=spec_b,
        num_games=4,
        time_per_move_ms=100,
        max_moves=40,
        opening_plies=2,
        seed=3,
        num_workers=2,
    )
    assert result.games == 4
    assert result.a_wins + result.b_wins + result.draws == 4
    assert len(result.records) == 4
    for rec in result.records:
        assert rec.result in (0.0, 0.5, 1.0)


def test_tournament_mp_matches_serial_with_same_seed():
    """Per-game seeding means MP == serial on outcomes (not wall time)."""
    spec_a = EngineSpec(evaluator="rule", max_depth=2, time_limit_ms=100)
    spec_b = EngineSpec(evaluator="material", max_depth=2, time_limit_ms=100)
    kwargs = dict(
        game_name="minichess",
        engine_a_spec=spec_a,
        engine_b_spec=spec_b,
        num_games=6,
        time_per_move_ms=100,
        max_moves=40,
        opening_plies=2,
        seed=11,
    )
    serial = play_tournament(**kwargs, num_workers=1)
    mp = play_tournament(**kwargs, num_workers=2)
    assert serial.a_wins == mp.a_wins
    assert serial.b_wins == mp.b_wins
    assert serial.draws == mp.draws
    # Game-level results should match too — same index maps to same game.
    assert [r.result for r in serial.records] == [r.result for r in mp.records]
