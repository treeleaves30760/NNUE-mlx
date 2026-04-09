"""Evaluation functions for the iterative pipeline.

Contains the top-level worker function required for multiprocessing pickling,
plus standalone match-playing and evaluation logic extracted from IterativePipeline.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import create_game
from pipeline_config import FEATURE_SETS


def _eval_worker_play_game(game_name: str, model1_path: str,
                           model2_path: str | None,
                           game_num: int, depth: int,
                           time_limit_ms: int, max_moves: int,
                           opening_fen: str | None = None) -> str:
    """Worker function: play one evaluation game. Returns 'win'/'loss'/'draw'.

    Features:
    - Opening book support: start from a given FEN
    - Adjudication: draw (low scores), resignation (high scores)
    - Color alternation: even games = model1 as white
    """
    from src.search.evaluator import NNUEEvaluator, RuleBasedEvaluator
    from src.search.alphabeta import AlphaBetaSearch

    fs_map = {name: cls for name, cls in FEATURE_SETS.items()}
    fs = fs_map[game_name]()

    eval1 = NNUEEvaluator.from_numpy(model1_path, fs)
    if model2_path:
        eval2 = NNUEEvaluator.from_numpy(model2_path, fs)
    else:
        eval2 = RuleBasedEvaluator()

    # Start from opening book position if provided
    if opening_fen and game_name == "chess":
        try:
            from src.games.chess_pc import from_fen
            state = from_fen(opening_fen)
        except Exception:
            state = create_game(game_name)
    else:
        state = create_game(game_name)

    s1 = AlphaBetaSearch(eval1, max_depth=depth, time_limit_ms=time_limit_ms)
    s2 = AlphaBetaSearch(eval2, max_depth=depth, time_limit_ms=time_limit_ms)
    searchers = [s1, s2]
    if game_num % 2 == 1:
        searchers = [searchers[1], searchers[0]]

    # Adjudication state
    draw_streak = 0
    win_streak = 0
    win_side = 0
    adjudicated = None
    ply = 0

    for _ in range(max_moves):
        if state.is_terminal():
            break
        player = state.side_to_move()
        move, score = searchers[player].search(state)
        if move is None:
            break

        # Adjudication checks
        abs_score = abs(score) if score else 0
        if abs_score < 20:
            draw_streak += 1
        else:
            draw_streak = 0

        white_score = score if player == 0 else -score
        if abs_score > 800:
            cur_side = 1 if white_score > 0 else -1
            if cur_side == win_side:
                win_streak += 1
            else:
                win_side = cur_side
                win_streak = 1
        else:
            win_streak = 0
            win_side = 0

        # Draw: both sides see near-zero for 16 plies (8 full moves)
        if draw_streak >= 16 and ply >= 40:
            adjudicated = "draw"
        # Resignation: one side clearly winning for 4 plies
        elif win_streak >= 4:
            if game_num % 2 == 0:
                adjudicated = "win" if win_side == 1 else "loss"
            else:
                adjudicated = "loss" if win_side == 1 else "win"

        state = state.make_move(move)
        ply += 1

        if adjudicated:
            break

    if adjudicated:
        return adjudicated

    r = state.result()
    if r is None:
        return "draw"
    elif r == 1.0:
        return "win" if game_num % 2 == 0 else "loss"
    elif r == 0.0:
        return "loss" if game_num % 2 == 0 else "win"
    return "draw"


def play_match_sequential(game_name: str, eval1, eval2, num_games: int,
                          depth: int, time_limit_ms: int,
                          max_moves: int) -> tuple:
    """Play games sequentially (fallback when no model paths available)."""
    from src.search.alphabeta import AlphaBetaSearch

    wins, losses, draws = 0, 0, 0
    for game_num in range(num_games):
        state = create_game(game_name)
        s1 = AlphaBetaSearch(eval1, max_depth=depth, time_limit_ms=time_limit_ms)
        s2 = AlphaBetaSearch(eval2, max_depth=depth, time_limit_ms=time_limit_ms)
        searchers = [s1, s2]
        if game_num % 2 == 1:
            searchers = [searchers[1], searchers[0]]

        for _ in range(max_moves):
            if state.is_terminal():
                break
            player = state.side_to_move()
            move, _ = searchers[player].search(state)
            if move is None:
                break
            state = state.make_move(move)

        r = state.result()
        if r is None:
            draws += 1
        elif r == 1.0:
            if game_num % 2 == 0:
                wins += 1
            else:
                losses += 1
        elif r == 0.0:
            if game_num % 2 == 0:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1
    return wins, losses, draws


def play_match_parallel(game_name: str, model1_path: str,
                        model2_path: str | None,
                        num_games: int, depth: int,
                        time_limit_ms: int,
                        max_moves: int) -> tuple:
    """Play evaluation games in parallel (one worker per game).

    Uses opening book positions for chess to ensure balanced starts.
    Each opening is played from both sides (even=model1 white, odd=model1 black).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from src.training.openings import get_opening_book

    workers = min(num_games, os.cpu_count() or 4)
    print(f"    Playing {num_games} eval games in parallel "
          f"({workers} workers)...")

    # Pre-assign opening FENs — each opening played from both sides
    book = get_opening_book(game_name)
    opening_fens = []
    if book:
        import random as _rng
        sampled = [_rng.choice(book) for _ in range(num_games // 2 + 1)]
        for fen in sampled:
            opening_fens.append(fen)  # even game: model1=white
            opening_fens.append(fen)  # odd game: model1=black
        opening_fens = opening_fens[:num_games]
    else:
        opening_fens = [None] * num_games

    ctx = multiprocessing.get_context("spawn")
    wins, losses, draws = 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=ctx) as pool:
        futures = []
        for game_num in range(num_games):
            f = pool.submit(
                _eval_worker_play_game,
                game_name, model1_path, model2_path,
                game_num, depth, time_limit_ms, max_moves,
                opening_fens[game_num],
            )
            futures.append((game_num, f))

        for game_num, f in futures:
            result = f.result()
            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1
            else:
                draws += 1
            print(f"    Game {game_num + 1}/{num_games}: {result}")

    return wins, losses, draws


def play_match(game_name: str, eval1, eval2, num_games: int,
               depth: int = 6, time_limit_ms: int = 30000,
               max_moves: int = 200,
               model1_path: str | None = None,
               model2_path: str | None = None) -> tuple:
    """Play a match between two evaluators (parallel across games)."""
    # Use parallel workers when we have model paths for reloading
    if num_games > 1 and model1_path:
        return play_match_parallel(
            game_name, model1_path, model2_path, num_games,
            depth, time_limit_ms, max_moves)

    # Fallback: sequential play
    return play_match_sequential(
        game_name, eval1, eval2, num_games, depth, time_limit_ms, max_moves)


def evaluate(game_name: str, fs, config: dict,
             model_path: str, prev_model: str | None) -> dict:
    """Evaluate the model against baselines."""
    from src.search.evaluator import NNUEEvaluator, RuleBasedEvaluator

    num_games = config["eval_games"]
    eval_depth = config.get("eval_depth", 6)
    eval_time_limit = config.get("eval_time_limit_ms", 30000)
    eval_max_moves = config.get("eval_max_moves", 600)
    result = {}

    # Eval vs rule-based AI (parallel)
    print(f"  Evaluating vs Rule-Based AI ({num_games} games, "
          f"depth={eval_depth}, time={eval_time_limit}ms)...")
    eval1 = NNUEEvaluator.from_numpy(model_path, fs)
    eval2 = RuleBasedEvaluator()
    w, l, d = play_match(game_name, eval1, eval2, num_games,
                         depth=eval_depth,
                         time_limit_ms=eval_time_limit,
                         max_moves=eval_max_moves,
                         model1_path=model_path,
                         model2_path=None)
    result["vs_material"] = {"wins": w, "losses": l, "draws": d}
    wr = w / max(w + l + d, 1) * 100
    print(f"    vs Rule-Based: W{w}-L{l}-D{d} ({wr:.1f}% win rate)")

    # Eval vs previous model (parallel)
    if prev_model and Path(prev_model).exists():
        print(f"  Evaluating vs previous model ({num_games} games, "
              f"depth={eval_depth})...")
        eval2_prev = NNUEEvaluator.from_numpy(prev_model, fs)
        w, l, d = play_match(game_name, eval1, eval2_prev, num_games,
                             depth=eval_depth,
                             time_limit_ms=eval_time_limit,
                             max_moves=eval_max_moves,
                             model1_path=model_path,
                             model2_path=prev_model)
        result["vs_previous"] = {"wins": w, "losses": l, "draws": d}
        wr = w / max(w + l + d, 1) * 100
        print(f"    vs Previous: W{w}-L{l}-D{d} ({wr:.1f}% win rate)")

    return result
