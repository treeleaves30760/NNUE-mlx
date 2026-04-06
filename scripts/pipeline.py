"""Iterative self-play training pipeline.

Runs a loop of: generate data → train model → evaluate → repeat.
Each iteration produces a stronger model that generates better training data.

Models are saved to models/<YYYY-MM-DD-HH-MM-SS>/ directories.
Pipeline state is saved for resume support.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.halfkp import chess_features, minichess_features
from src.features.halfkp_shogi import shogi_features, minishogi_features
from src.training.selfplay import SelfPlayEngine
from src.training.trainer import Trainer
from src.training.dataset import preload_samples
from src.utils.config import create_game

def _eval_worker_play_game(game_name: str, model1_path: str,
                           model2_path: str | None,
                           game_num: int, depth: int,
                           time_limit_ms: int, max_moves: int) -> str:
    """Worker function: play one evaluation game. Returns 'win'/'loss'/'draw'."""
    from src.search.evaluator import NNUEEvaluator, RuleBasedEvaluator
    from src.search.alphabeta import AlphaBetaSearch

    fs_map = {"chess": chess_features, "minichess": minichess_features,
              "shogi": shogi_features, "minishogi": minishogi_features}
    fs = fs_map[game_name]()

    eval1 = NNUEEvaluator.from_numpy(model1_path, fs)
    if model2_path:
        eval2 = NNUEEvaluator.from_numpy(model2_path, fs)
    else:
        eval2 = RuleBasedEvaluator()

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
        return "draw"
    elif r == 1.0:
        return "win" if game_num % 2 == 0 else "loss"
    elif r == 0.0:
        return "loss" if game_num % 2 == 0 else "win"
    return "draw"


FEATURE_SETS = {
    "chess": chess_features,
    "minichess": minichess_features,
    "shogi": shogi_features,
    "minishogi": minishogi_features,
}

# Default configuration per game
GAME_DEFAULTS = {
    "minichess": {
        "bootstrap_games": 20000,
        "games_per_iter": 2000,
        "epochs_bootstrap": 30,
        "epochs_per_iter": 20,
        "eval_games": 30,
        "max_depth_schedule": [2, 3, 3, 4, 4, 4],
        "time_limit_schedule": [500, 500, 1000, 1000, 1500, 2000],
        "workers": 0,
        "batch_size": 2048,
        "lr": 1e-3,
        "lambda_": 0.5,
        "data_window": 3,
    },
    "chess": {
        "bootstrap_games": 10000,
        "games_per_iter": 200,
        "epochs_bootstrap": 40,
        "epochs_per_iter": 30,
        "eval_games": 3,
        "eval_depth": 10,
        "eval_time_limit_ms": 30000,
        "max_depth_schedule": [3, 3, 4, 4, 5, 5, 6, 6],
        "time_limit_schedule": [2000, 2000, 3000, 3000, 5000, 5000, 8000, 10000],
        "workers": 8,
        "batch_size": 2048,
        "lr": 1e-3,
        "lambda_": 0.5,
        "data_window": 5,
    },
    "shogi": {
        "bootstrap_games": 10000,
        "games_per_iter": 300,
        "epochs_bootstrap": 30,
        "epochs_per_iter": 20,
        "eval_games": 10,
        "max_depth_schedule": [2, 2, 3, 3, 4],
        "time_limit_schedule": [500, 500, 1000, 1500, 2000],
        "workers": 0,
        "batch_size": 2048,
        "lr": 1e-3,
        "lambda_": 0.5,
        "data_window": 3,
    },
    "minishogi": {
        "bootstrap_games": 15000,
        "games_per_iter": 1500,
        "epochs_bootstrap": 30,
        "epochs_per_iter": 20,
        "eval_games": 20,
        "max_depth_schedule": [2, 3, 3, 4, 4],
        "time_limit_schedule": [500, 500, 1000, 1000, 1500],
        "workers": 0,
        "batch_size": 2048,
        "lr": 1e-3,
        "lambda_": 0.5,
        "data_window": 3,
    },
}


class IterativePipeline:
    """Orchestrates iterative self-play training."""

    def __init__(self, game: str, num_iterations: int, config: dict,
                 base_dir: str = ".",
                 bootstrap_data: str | None = None,
                 bootstrap_model: str | None = None):
        self.game = game
        self.num_iterations = num_iterations
        self.config = config
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.bootstrap_data = bootstrap_data
        self.bootstrap_model = bootstrap_model

        self.fs = FEATURE_SETS[game]()
        self.state_file = self.base_dir / f"pipeline_state_{game}.json"
        self.history: list = []

    def _get_depth(self, iteration: int) -> int:
        schedule = self.config["max_depth_schedule"]
        idx = min(iteration, len(schedule) - 1)
        return schedule[idx]

    def _get_time_limit(self, iteration: int) -> int:
        schedule = self.config["time_limit_schedule"]
        idx = min(iteration, len(schedule) - 1)
        return schedule[idx]

    def _save_state(self):
        state = {
            "game": self.game,
            "history": self.history,
            "num_iterations": self.num_iterations,
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> int:
        """Load pipeline state, return the next iteration to run."""
        if not self.state_file.exists():
            return 0
        state = json.loads(self.state_file.read_text())
        self.history = state.get("history", [])
        return len(self.history)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def run(self):
        """Run the full iterative pipeline."""
        start_iter = self._load_state()
        if start_iter > 0:
            print(f"Resuming from iteration {start_iter} "
                  f"({len(self.history)} completed)")

        for iteration in range(start_iter, self.num_iterations):
            print(f"\n{'='*60}")
            print(f"  ITERATION {iteration}/{self.num_iterations - 1}")
            print(f"{'='*60}")

            iter_result = self._run_iteration(iteration)
            self.history.append(iter_result)
            self._save_state()

            self._print_progress()

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE: {self.num_iterations} iterations")
        print(f"{'='*60}")
        self._print_progress()

    def _run_iteration(self, iteration: int) -> dict:
        """Run a single iteration: generate → train → evaluate."""
        timestamp = self._timestamp()
        iter_dir = self.models_dir / timestamp
        iter_dir.mkdir(parents=True, exist_ok=True)

        is_bootstrap = (iteration == 0)
        prev_model = None
        if not is_bootstrap and self.history:
            prev_model = self.history[-1]["model_path"]

        result = {
            "iteration": iteration,
            "timestamp": timestamp,
            "dir": str(iter_dir),
        }

        # --- Step 1: Generate data ---
        t0 = time.time()
        data_file = self._generate_data(iteration, prev_model, iter_dir)
        gen_time = time.time() - t0
        result["data_file"] = data_file
        result["generation_time_s"] = round(gen_time, 1)
        print(f"  Data generation: {gen_time:.1f}s")

        # --- Step 2: Collect data files (rolling window) ---
        data_files = self._collect_data_files(iteration)
        result["data_files_used"] = data_files

        # --- Step 3: Train ---
        t0 = time.time()

        # Skip training if bootstrap model provided for iteration 0
        if is_bootstrap and self.bootstrap_model:
            import shutil
            model_path = str(iter_dir / "model.npz")
            shutil.copy2(self.bootstrap_model, model_path)
            training_log = []
            print(f"  Using existing bootstrap model: {self.bootstrap_model}")
        else:
            epochs = (self.config["epochs_bootstrap"] if is_bootstrap
                      else self.config["epochs_per_iter"])
            model_path, training_log = self._train_model(
                data_files, iter_dir, epochs, prev_model,
            )
        train_time = time.time() - t0
        result["model_path"] = model_path
        result["training_time_s"] = round(train_time, 1)
        result["training_log"] = training_log
        result["final_loss"] = training_log[-1]["loss"] if training_log else None
        loss_str = f"{result['final_loss']:.6f}" if result['final_loss'] is not None else "N/A (bootstrap)"
        print(f"  Training: {train_time:.1f}s, final loss: {loss_str}")

        # --- Step 4: Evaluate vs material baseline ---
        t0 = time.time()
        eval_result = self._evaluate(model_path, prev_model)
        eval_time = time.time() - t0
        result["eval_vs_material"] = eval_result["vs_material"]
        result["eval_vs_previous"] = eval_result.get("vs_previous")
        result["eval_time_s"] = round(eval_time, 1)

        # Save iteration metadata
        meta_path = iter_dir / "iteration.json"
        meta_path.write_text(json.dumps(result, indent=2))

        return result

    def _generate_data(self, iteration: int, prev_model: str | None,
                       iter_dir: Path) -> str:
        """Generate self-play training data."""
        is_bootstrap = (iteration == 0)

        # If bootstrap data is provided, copy/link it instead of generating
        if is_bootstrap and self.bootstrap_data:
            data_file = str(self.data_dir / f"{self.game}_iter0.bin")
            import shutil
            if not Path(data_file).exists():
                shutil.copy2(self.bootstrap_data, data_file)
            print(f"  Using existing bootstrap data: {self.bootstrap_data}")
            sz = Path(data_file).stat().st_size
            print(f"  Data size: {sz / 1024 / 1024:.1f}MB")
            return data_file

        if is_bootstrap:
            num_games = self.config["bootstrap_games"]
            depth = 1
            model_path = None
            print(f"  Generating {num_games} random bootstrap games...")
        else:
            num_games = self.config["games_per_iter"]
            depth = self._get_depth(iteration)
            model_path = prev_model
            time_limit = self._get_time_limit(iteration)
            print(f"  Generating {num_games} games (depth={depth}, "
                  f"time={time_limit}ms) with model...")

        data_file = str(self.data_dir / f"{self.game}_iter{iteration}.bin")

        evaluator = None
        if model_path and self.config["workers"] == 1:
            from src.search.evaluator import NNUEEvaluator
            from src.search.alphabeta import AlphaBetaSearch
            nnue_eval = NNUEEvaluator.from_numpy(model_path, self.fs)
            tl = self._get_time_limit(iteration) if not is_bootstrap else 500
            evaluator = AlphaBetaSearch(
                nnue_eval, max_depth=depth, time_limit_ms=tl,
            )

        engine = SelfPlayEngine(
            feature_set=self.fs,
            evaluator=evaluator,
            search_depth=depth,
            random_play_prob=0.15 if is_bootstrap else 0.08,
            game_name=self.game,
            model_path=model_path,
            time_limit_ms=self._get_time_limit(iteration) if not is_bootstrap else 500,
        )

        engine.generate_data(
            initial_state_fn=lambda: create_game(self.game),
            output_path=data_file,
            num_games=num_games,
            num_workers=self.config["workers"],
        )

        return data_file

    def _collect_data_files(self, iteration: int) -> list:
        """Collect data files within the rolling window.

        Always includes bootstrap data (iter0) to maintain a large
        baseline training set, plus recent iterations within the window.
        """
        window = self.config["data_window"]
        files = []

        # Always include bootstrap data
        bootstrap_path = str(self.data_dir / f"{self.game}_iter0.bin")
        if Path(bootstrap_path).exists():
            files.append(bootstrap_path)

        # Add recent iterations (excluding iter0 to avoid duplicating bootstrap)
        start = max(1, iteration + 1 - window)
        for i in range(start, iteration + 1):
            if i == 0:
                continue
            path = str(self.data_dir / f"{self.game}_iter{i}.bin")
            if Path(path).exists():
                files.append(path)
        return files

    def _train_model(self, data_files: list, iter_dir: Path,
                     epochs: int, prev_model: str | None) -> tuple:
        """Train a model on collected data files."""
        print(f"  Loading {len(data_files)} data file(s)...")
        all_samples = preload_samples(data_files)

        # 90/10 train/validation split
        random.shuffle(all_samples)
        val_size = max(1, int(len(all_samples) * 0.1))
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]
        print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}")

        trainer = Trainer(
            num_features=self.fs.num_features(),
            lr=self.config["lr"],
            batch_size=self.config["batch_size"],
            max_active=self.fs.max_active_features(),
            lambda_=self.config["lambda_"],
        )

        # Warm-start from previous model if available
        if prev_model and Path(prev_model).exists():
            print(f"  Warm-starting from {prev_model}")
            trainer.load_weights_from_npz(prev_model)

        training_log = []
        for epoch in range(epochs):
            loss = trainer.train_epoch_from_samples(train_samples, shuffle=True)
            val_loss = trainer.validate_epoch(val_samples)
            lr = trainer.optimizer.learning_rate
            lr_val = lr.item() if hasattr(lr, 'item') else float(lr)
            training_log.append({
                "epoch": epoch + 1,
                "loss": round(loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": lr_val,
            })
            print(f"    Epoch {epoch + 1}/{epochs} | "
                  f"Loss: {loss:.6f} | Val: {val_loss:.6f} | LR: {lr_val:.2e}")

        # Export model
        model_path = str(iter_dir / "model.npz")
        trainer.export_numpy(model_path)

        # Save training log
        log_path = iter_dir / "training_log.json"
        log_path.write_text(json.dumps(training_log, indent=2))

        return model_path, training_log

    def _evaluate(self, model_path: str,
                  prev_model: str | None) -> dict:
        """Evaluate the model against baselines."""
        from src.search.evaluator import NNUEEvaluator, RuleBasedEvaluator
        from src.search.alphabeta import AlphaBetaSearch

        num_games = self.config["eval_games"]
        eval_depth = self.config.get("eval_depth", 6)
        eval_time_limit = self.config.get("eval_time_limit_ms", 30000)
        result = {}

        # Eval vs rule-based AI (parallel)
        print(f"  Evaluating vs Rule-Based AI ({num_games} games, "
              f"depth={eval_depth}, time={eval_time_limit}ms)...")
        eval1 = NNUEEvaluator.from_numpy(model_path, self.fs)
        eval2 = RuleBasedEvaluator()
        w, l, d = self._play_match(eval1, eval2, num_games,
                                    depth=eval_depth,
                                    time_limit_ms=eval_time_limit,
                                    model1_path=model_path,
                                    model2_path=None)
        result["vs_material"] = {"wins": w, "losses": l, "draws": d}
        wr = w / max(w + l + d, 1) * 100
        print(f"    vs Rule-Based: W{w}-L{l}-D{d} ({wr:.1f}% win rate)")

        # Eval vs previous model (parallel)
        if prev_model and Path(prev_model).exists():
            print(f"  Evaluating vs previous model ({num_games} games, "
                  f"depth={eval_depth})...")
            eval2_prev = NNUEEvaluator.from_numpy(prev_model, self.fs)
            w, l, d = self._play_match(eval1, eval2_prev, num_games,
                                        depth=eval_depth,
                                        time_limit_ms=eval_time_limit,
                                        model1_path=model_path,
                                        model2_path=prev_model)
            result["vs_previous"] = {"wins": w, "losses": l, "draws": d}
            wr = w / max(w + l + d, 1) * 100
            print(f"    vs Previous: W{w}-L{l}-D{d} ({wr:.1f}% win rate)")

        return result

    def _play_match(self, eval1, eval2, num_games: int,
                    depth: int = 6, time_limit_ms: int = 30000,
                    max_moves: int = 200,
                    model1_path: str | None = None,
                    model2_path: str | None = None) -> tuple:
        """Play a match between two evaluators (parallel across games)."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # Use parallel workers when we have model paths for reloading
        if num_games > 1 and model1_path:
            return self._play_match_parallel(
                model1_path, model2_path, num_games,
                depth, time_limit_ms, max_moves)

        # Fallback: sequential play
        return self._play_match_sequential(
            eval1, eval2, num_games, depth, time_limit_ms, max_moves)

    def _play_match_sequential(self, eval1, eval2, num_games: int,
                               depth: int, time_limit_ms: int,
                               max_moves: int) -> tuple:
        """Play games sequentially (fallback)."""
        from src.search.alphabeta import AlphaBetaSearch

        wins, losses, draws = 0, 0, 0
        for game_num in range(num_games):
            state = create_game(self.game)
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

    def _play_match_parallel(self, model1_path: str,
                             model2_path: str | None,
                             num_games: int, depth: int,
                             time_limit_ms: int,
                             max_moves: int) -> tuple:
        """Play evaluation games in parallel (one worker per game)."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        workers = min(num_games, os.cpu_count() or 4)
        print(f"    Playing {num_games} eval games in parallel "
              f"({workers} workers)...")

        ctx = multiprocessing.get_context("spawn")
        wins, losses, draws = 0, 0, 0
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=ctx) as pool:
            futures = []
            for game_num in range(num_games):
                f = pool.submit(
                    _eval_worker_play_game,
                    self.game, model1_path, model2_path,
                    game_num, depth, time_limit_ms, max_moves,
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

    def _print_progress(self):
        """Print a summary table of all iterations."""
        print(f"\n{'='*60}")
        print(f"  Progress Summary: {self.game}")
        print(f"{'='*60}")
        print(f"{'Iter':>4} | {'Loss':>10} | {'vs Mat':>12} | "
              f"{'vs Prev':>12} | {'Gen(s)':>7} | {'Train(s)':>8}")
        print("-" * 70)

        for h in self.history:
            loss = h.get("final_loss")
            loss_str = f"{loss:.6f}" if loss is not None else "   N/A    "
            vm = h.get("eval_vs_material", {})
            vp = h.get("eval_vs_previous")
            vm_str = f"W{vm.get('wins',0)}-L{vm.get('losses',0)}-D{vm.get('draws',0)}"
            vp_str = "-"
            if vp:
                vp_str = f"W{vp['wins']}-L{vp['losses']}-D{vp['draws']}"
            gen_t = h.get("generation_time_s", 0)
            train_t = h.get("training_time_s", 0)
            print(f"{h['iteration']:>4} | {loss_str:>10} | {vm_str:>12} | "
                  f"{vp_str:>12} | {gen_t:>7.0f} | {train_t:>8.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Iterative self-play training pipeline")
    parser.add_argument("--game", required=True, choices=FEATURE_SETS.keys())
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations")
    parser.add_argument("--bootstrap-games", type=int, default=None,
                        help="Games for bootstrap iteration (override default)")
    parser.add_argument("--games-per-iter", type=int, default=None,
                        help="Games per non-bootstrap iteration (override)")
    parser.add_argument("--epochs-bootstrap", type=int, default=None)
    parser.add_argument("--epochs-per-iter", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (0=auto)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda", type=float, default=None, dest="lambda_")
    parser.add_argument("--data-window", type=int, default=None,
                        help="Rolling window of iterations to keep data from")
    parser.add_argument("--eval-depth", type=int, default=None,
                        help="Search depth for evaluation games")
    parser.add_argument("--eval-time-limit", type=int, default=None,
                        dest="eval_time_limit_ms",
                        help="Time limit (ms) per move for evaluation games")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous pipeline state")
    parser.add_argument("--bootstrap-data", default=None,
                        help="Existing .bin data file to use for bootstrap "
                             "(skip random game generation)")
    parser.add_argument("--bootstrap-model", default=None,
                        help="Existing .npz model to warm-start from "
                             "(skip bootstrap training entirely)")
    args = parser.parse_args()

    # Build config from defaults + overrides
    config = dict(GAME_DEFAULTS.get(args.game, GAME_DEFAULTS["chess"]))
    for key in ["bootstrap_games", "games_per_iter", "epochs_bootstrap",
                "epochs_per_iter", "eval_games", "workers", "batch_size",
                "lr", "lambda_", "data_window", "eval_depth",
                "eval_time_limit_ms"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    print(f"Game: {args.game}")
    print(f"Iterations: {args.iterations}")
    print(f"Config: {json.dumps(config, indent=2)}")

    pipeline = IterativePipeline(
        game=args.game,
        num_iterations=args.iterations,
        config=config,
        bootstrap_data=args.bootstrap_data,
        bootstrap_model=args.bootstrap_model,
    )

    if args.resume:
        print("Resume mode enabled")

    pipeline.run()


if __name__ == "__main__":
    main()
