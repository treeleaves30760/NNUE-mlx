"""Self-play tournament harness for comparing two evaluators.

Runs a head-to-head match between two search engines (each produced by
a zero-arg factory callable) and returns wins/draws/losses plus an
Elo-delta estimate with a Wilson-based confidence half-width. The
harness is deliberately small and single-process — 50-game matches
with 500ms per move finish in under a minute and that's all the
iteration budget we need for tuning.

Typical usage:

    from src.search.alphabeta import AlphaBetaSearch
    from src.search.evaluator import MaterialEvaluator, RuleBasedEvaluator
    from src.training.tournament import play_tournament

    mat = lambda: AlphaBetaSearch(MaterialEvaluator(), 3, 300)
    rule = lambda: AlphaBetaSearch(RuleBasedEvaluator(), 3, 300)
    result = play_tournament("chess", mat, rule, num_games=20)
    print(result.summary())

Design notes:
    * Colours alternate each game so the first-move advantage cancels
      out in aggregate. Even-indexed games: A plays white, B plays
      black. Odd-indexed: B plays white, A plays black.
    * Terminal detection uses ``state.is_terminal()`` and
      ``state.result()`` from whichever side is to move; the harness
      normalises the final score into a-wins / b-wins / draws from
      engine A's perspective.
    * A move cap (``max_moves``) adjudicates any unfinished game as a
      draw. This avoids pathological infinite games from weak
      evaluators that can't force a result in any reasonable time.
    * Both factories are called once per game so the engines always
      start with a fresh transposition table. Shared TTs across games
      would give a small-but-real advantage to the first engine and
      make measurements unreproducible.
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Tuple

from src.games.base import GameState, Move
from src.utils.config import create_game


class _SearchLike(Protocol):
    """Minimal interface a tournament engine must expose."""

    max_depth: int
    time_limit_ms: int

    def search(self, state: GameState) -> Tuple[Optional[Move], float]: ...


SearchFactory = Callable[[], _SearchLike]


@dataclass(frozen=True)
class EngineSpec:
    """Picklable description of a tournament engine.

    Used by multiprocessing workers (which can't pickle closures) and
    optionally in serial mode when the caller wants to share config
    between the serial and MP paths. Fields are deliberately simple;
    add new ones as the experiments grow.

    ``evaluator`` = ``"rule"`` builds the appropriate per-game rule
    search (C-backed for chess and shogi when the extension is
    available, Python alphabeta otherwise). ``"material"`` builds a
    pure-material alphabeta for calibration / regression tests.
    """
    evaluator: str           # "rule" or "material"
    max_depth: int = 4
    time_limit_ms: int = 500


def build_engine_from_spec(spec: EngineSpec, game_name: str) -> _SearchLike:
    """Instantiate an engine from an EngineSpec.

    Called in the worker process for MP tournaments, and also in the
    parent process if the caller passes a spec to ``play_tournament``
    without a factory. The engine returned is a *fresh* instance with
    its own TT etc. — the caller is responsible for discarding it
    after a single game.
    """
    from src.search.alphabeta import AlphaBetaSearch, create_rule_based_search
    from src.search.evaluator import MaterialEvaluator

    if spec.evaluator == "rule":
        return create_rule_based_search(
            game_name,
            max_depth=spec.max_depth,
            time_limit_ms=spec.time_limit_ms,
        )
    if spec.evaluator == "material":
        return AlphaBetaSearch(
            MaterialEvaluator(),
            max_depth=spec.max_depth,
            time_limit_ms=spec.time_limit_ms,
        )
    raise ValueError(f"unknown EngineSpec.evaluator: {spec.evaluator!r}")


@dataclass
class GameRecord:
    """Outcome + move sequence for a single tournament game."""
    white_is_a: bool
    result: float             # 1.0 = a wins, 0.5 = draw, 0.0 = b wins
    moves: List[Move] = field(default_factory=list)
    reason: str = "normal"    # "normal", "max_moves", "error"
    plies: int = 0


@dataclass
class TournamentResult:
    games: int
    a_wins: int
    b_wins: int
    draws: int
    a_score: float          # (wins + 0.5*draws) / games
    elo_diff: float         # engine A's Elo relative to B (positive = A stronger)
    elo_error: float        # 95% confidence half-width on elo_diff
    records: List[GameRecord]
    wall_time_sec: float

    def summary(self) -> str:
        lines = [
            f"Games:   {self.games}",
            f"A wins:  {self.a_wins}",
            f"B wins:  {self.b_wins}",
            f"Draws:   {self.draws}",
            f"A score: {self.a_score:.3f}",
            f"Elo delta (A - B): {self.elo_diff:+.0f} ± {self.elo_error:.0f}",
            f"Wall time: {self.wall_time_sec:.1f}s",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- elo helpers


def _score_to_elo(score: float) -> float:
    """Convert a win fraction to an Elo difference.

    Clipped to ±800 Elo to avoid infinities when one side wins 0% or
    100%. Standard chess formula: ``elo = -400 * log10(1/score - 1)``.
    """
    if score <= 0.005:
        return -800.0
    if score >= 0.995:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def _wilson_half_width(wins: float, games: int) -> float:
    """95% Wilson confidence half-width on a binomial proportion."""
    if games <= 0:
        return 1.0
    p = wins / games
    z = 1.96
    # Standard Wilson upper/lower; return half of the interval width.
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    radius = (z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
              / denom)
    lo = max(0.0, centre - radius)
    hi = min(1.0, centre + radius)
    return (hi - lo) / 2.0


# --------------------------------------------------------------------------- game loop


def _play_one_game(
    game_name: str,
    engine_white: _SearchLike,
    engine_black: _SearchLike,
    max_moves: int,
    opening_plies: int = 0,
    rng: Optional[random.Random] = None,
) -> Tuple[float, List[Move], str, int]:
    """Play a single game to completion or to ``max_moves``.

    Returns ``(white_score, moves, reason, plies)`` where ``white_score``
    is 1.0 / 0.5 / 0.0 from white's perspective.

    If ``opening_plies > 0``, the first N half-moves are chosen uniformly
    at random from the legal move list. This exists purely so that two
    deterministic evaluators don't end up playing the same game every
    match — without it, a 100-game tournament boils down to a single
    repeated game plus its colour-flipped counterpart.
    """
    state = create_game(game_name)
    moves: List[Move] = []
    rng = rng or random.Random()

    # Random opening plies (uniform over legal moves).
    for _ in range(opening_plies):
        if state.is_terminal():
            break
        legal = state.legal_moves()
        if not legal:
            break
        mv = rng.choice(legal)
        state = state.make_move(mv)
        moves.append(mv)

    for ply in range(max_moves):
        if state.is_terminal():
            break
        engine = engine_white if state.side_to_move() == 0 else engine_black
        move, _score = engine.search(state)
        if move is None:
            break
        state = state.make_move(move)
        moves.append(move)

    plies = len(moves)

    if state.is_terminal():
        result = state.result()
        if result is None:
            return 0.5, moves, "normal", plies
        # state.result() is from the current side-to-move's perspective
        # *after* the last move; 1.0 = current side wins (which is the
        # side that just moved into mate? No — current side = the one
        # who must now move, i.e. the LOSER in a checkmate). The chess
        # GameState convention returns 1.0 when the side whose turn it
        # is has *lost* (legal moves empty and in check). A draw is
        # returned as 0.5.
        #
        # To normalise: if result == 0.5, draw. If result == 1.0 and
        # side_to_move was just flipped to the loser, the winner is the
        # previous mover (the one who delivered mate). We compute
        # white_score by checking who's to move.
        side = state.side_to_move()
        if result == 0.5:
            return 0.5, moves, "normal", plies
        # result is 1.0 (side-to-move wins) or 0.0 (side-to-move loses)
        if side == 0:   # white to move
            return result, moves, "normal", plies
        else:
            return 1.0 - result, moves, "normal", plies

    # Move cap hit
    return 0.5, moves, "max_moves", plies


# --------------------------------------------------------------------------- per-game runner


def _run_one_game_from_factories(
    game_name: str,
    game_index: int,
    base_seed: Optional[int],
    engine_a_factory: SearchFactory,
    engine_b_factory: SearchFactory,
    time_per_move_ms: int,
    max_moves: int,
    opening_plies: int,
) -> GameRecord:
    """Play a single indexed game and return its record.

    The game index controls two things: colour alternation (even = A
    is white) and the opening-RNG seed. Per-game seeding makes the
    tournament identical regardless of chunk layout — a game with
    index 37 is the same game whether it lands in worker 0 or worker 3.
    """
    a_is_white = (game_index % 2 == 0)
    engine_a = engine_a_factory()
    engine_b = engine_b_factory()
    engine_a.time_limit_ms = time_per_move_ms
    engine_b.time_limit_ms = time_per_move_ms

    if base_seed is None:
        rng = random.Random()
    else:
        rng = random.Random(base_seed + game_index)

    if a_is_white:
        white_score, moves, reason, plies = _play_one_game(
            game_name, engine_a, engine_b, max_moves,
            opening_plies=opening_plies, rng=rng,
        )
        a_score = white_score
    else:
        white_score, moves, reason, plies = _play_one_game(
            game_name, engine_b, engine_a, max_moves,
            opening_plies=opening_plies, rng=rng,
        )
        a_score = 1.0 - white_score

    return GameRecord(
        white_is_a=a_is_white,
        result=a_score,
        moves=moves,
        reason=reason,
        plies=plies,
    )


def _play_games_chunk(chunk: dict) -> List[GameRecord]:
    """Process-pool worker entry.

    ``chunk`` carries all the data a worker needs to play its subset
    of games. We can't pass ``engine_*_factory`` closures through
    pickle, so specs are reconstructed into engines via
    ``build_engine_from_spec`` inside the worker.
    """
    game_name = chunk["game_name"]
    spec_a: EngineSpec = chunk["spec_a"]
    spec_b: EngineSpec = chunk["spec_b"]
    game_indices: List[int] = chunk["game_indices"]
    time_per_move_ms: int = chunk["time_per_move_ms"]
    max_moves: int = chunk["max_moves"]
    opening_plies: int = chunk["opening_plies"]
    base_seed: Optional[int] = chunk["base_seed"]

    def factory_a():
        return build_engine_from_spec(spec_a, game_name)

    def factory_b():
        return build_engine_from_spec(spec_b, game_name)

    records = []
    for i in game_indices:
        records.append(_run_one_game_from_factories(
            game_name, i, base_seed,
            factory_a, factory_b,
            time_per_move_ms, max_moves, opening_plies,
        ))
    return records


# --------------------------------------------------------------------------- public API


def play_tournament(
    game_name: str,
    engine_a_factory: Optional[SearchFactory] = None,
    engine_b_factory: Optional[SearchFactory] = None,
    num_games: int = 20,
    time_per_move_ms: int = 500,
    max_moves: int = 200,
    opening_plies: int = 4,
    seed: Optional[int] = 42,
    verbose: bool = False,
    *,
    engine_a_spec: Optional[EngineSpec] = None,
    engine_b_spec: Optional[EngineSpec] = None,
    num_workers: int = 1,
) -> TournamentResult:
    """Play ``num_games`` games between two engines and return the results.

    Two ways to describe each engine:
      * Pass a zero-arg **factory** callable (legacy path). Works only
        in single-process mode.
      * Pass an :class:`EngineSpec`. Works in both serial and
        multiprocessing paths. Required when ``num_workers > 1``.

    Per-game seeding: each game ``i`` gets its own
    ``Random(seed + i)``. The result is that a game's outcome depends
    only on its index, not on how the games are distributed across
    workers — so serial and MP runs with the same seed produce the
    same aggregate wins/draws/losses.

    Colours alternate: game 0 has A as white, game 1 has B as white,
    and so on. This cancels the first-move advantage in aggregate.

    The move-cap (``max_moves``) adjudicates unfinished games as draws
    so slow/lost engines can't hang the tournament.
    """
    # --- Resolve factories / specs -----------------------------------
    if num_workers > 1:
        if engine_a_spec is None or engine_b_spec is None:
            raise ValueError(
                "multiprocessing mode (num_workers > 1) requires "
                "engine_a_spec and engine_b_spec; factories cannot be "
                "pickled into worker processes"
            )
        return _play_tournament_mp(
            game_name, engine_a_spec, engine_b_spec,
            num_games, time_per_move_ms, max_moves,
            opening_plies, seed, num_workers, verbose,
        )

    # Serial path — build per-game factories from whichever was given.
    if engine_a_factory is None:
        if engine_a_spec is None:
            raise ValueError("need engine_a_factory or engine_a_spec")
        _spec_a = engine_a_spec
        engine_a_factory = lambda: build_engine_from_spec(_spec_a, game_name)
    if engine_b_factory is None:
        if engine_b_spec is None:
            raise ValueError("need engine_b_factory or engine_b_spec")
        _spec_b = engine_b_spec
        engine_b_factory = lambda: build_engine_from_spec(_spec_b, game_name)

    return _play_tournament_serial(
        game_name, engine_a_factory, engine_b_factory,
        num_games, time_per_move_ms, max_moves,
        opening_plies, seed, verbose,
    )


def _play_tournament_serial(
    game_name: str,
    engine_a_factory: SearchFactory,
    engine_b_factory: SearchFactory,
    num_games: int,
    time_per_move_ms: int,
    max_moves: int,
    opening_plies: int,
    seed: Optional[int],
    verbose: bool,
) -> TournamentResult:
    records: List[GameRecord] = []
    a_wins = b_wins = draws = 0
    start = time.time()

    for i in range(num_games):
        rec = _run_one_game_from_factories(
            game_name, i, seed,
            engine_a_factory, engine_b_factory,
            time_per_move_ms, max_moves, opening_plies,
        )
        records.append(rec)
        if rec.result == 1.0:
            a_wins += 1
        elif rec.result == 0.0:
            b_wins += 1
        else:
            draws += 1
        if verbose:
            sym = "A" if rec.result == 1.0 else ("B" if rec.result == 0.0 else "D")
            print(f"  game {i+1:3d}/{num_games}  [{sym}]  "
                  f"plies={rec.plies}  reason={rec.reason}")

    return _finalise(records, a_wins, b_wins, draws, num_games,
                     time.time() - start)


def _play_tournament_mp(
    game_name: str,
    spec_a: EngineSpec,
    spec_b: EngineSpec,
    num_games: int,
    time_per_move_ms: int,
    max_moves: int,
    opening_plies: int,
    seed: Optional[int],
    num_workers: int,
    verbose: bool,
) -> TournamentResult:
    # Split game indices into round-robin chunks so each worker gets a
    # mix of A-is-white and B-is-white games. This keeps per-worker
    # runtimes balanced when one colour suffers more timeouts.
    chunks: List[List[int]] = [[] for _ in range(num_workers)]
    for i in range(num_games):
        chunks[i % num_workers].append(i)

    chunk_payloads = [
        {
            "game_name": game_name,
            "spec_a": spec_a,
            "spec_b": spec_b,
            "game_indices": idxs,
            "time_per_move_ms": time_per_move_ms,
            "max_moves": max_moves,
            "opening_plies": opening_plies,
            "base_seed": seed,
        }
        for idxs in chunks if idxs
    ]

    start = time.time()
    # index -> GameRecord so we can restore the global order.
    records_by_index: dict = {}
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_play_games_chunk, payload)
                   for payload in chunk_payloads]
        for fut, payload in zip(futures, chunk_payloads):
            for i, rec in zip(payload["game_indices"], fut.result()):
                records_by_index[i] = rec
                if verbose:
                    sym = ("A" if rec.result == 1.0 else
                           "B" if rec.result == 0.0 else "D")
                    print(f"  game {i+1:3d}/{num_games}  [{sym}]  "
                          f"plies={rec.plies}  reason={rec.reason}")

    records = [records_by_index[i] for i in range(num_games)]
    a_wins = sum(1 for r in records if r.result == 1.0)
    b_wins = sum(1 for r in records if r.result == 0.0)
    draws = sum(1 for r in records if r.result == 0.5)
    return _finalise(records, a_wins, b_wins, draws, num_games,
                     time.time() - start)


def _finalise(records, a_wins, b_wins, draws, games, wall_time):
    a_score_total = a_wins + 0.5 * draws
    a_fraction = a_score_total / games if games > 0 else 0.5
    elo_diff = _score_to_elo(a_fraction)
    half = _wilson_half_width(a_score_total, games)
    elo_error = _score_to_elo(min(0.995, a_fraction + half)) - elo_diff
    return TournamentResult(
        games=games,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        a_score=a_fraction,
        elo_diff=elo_diff,
        elo_error=abs(elo_error),
        records=records,
        wall_time_sec=wall_time,
    )
