"""Core iterative pipeline class."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_config import FEATURE_SETS
from pipeline_eval import evaluate, play_match
from pipeline_train import train_model, generate_data, collect_data_files


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

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def _get_depth(self, iteration: int) -> int:
        schedule = self.config["max_depth_schedule"]
        idx = min(iteration, len(schedule) - 1)
        return schedule[idx]

    def _get_time_limit(self, iteration: int) -> int:
        schedule = self.config["time_limit_schedule"]
        idx = min(iteration, len(schedule) - 1)
        return schedule[idx]

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _run_iteration(self, iteration: int) -> dict:
        """Run a single iteration: generate -> train -> evaluate."""
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

    # ------------------------------------------------------------------
    # Data generation (delegates to pipeline_train standalone functions)
    # ------------------------------------------------------------------

    def _generate_data(self, iteration: int, prev_model: str | None,
                       iter_dir: Path) -> str:
        """Generate self-play training data."""
        return generate_data(self.game, self.fs, self.config,
                             iteration, prev_model,
                             self.data_dir, self.bootstrap_data)

    def _collect_data_files(self, iteration: int) -> list:
        """Collect data files within the rolling window."""
        return collect_data_files(self.game, self.config,
                                  self.data_dir, iteration)

    # ------------------------------------------------------------------
    # Training (delegates to pipeline_train standalone function)
    # ------------------------------------------------------------------

    def _train_model(self, data_files: list, iter_dir: Path,
                     epochs: int, prev_model: str | None) -> tuple:
        """Train a model on collected data files."""
        return train_model(self.fs, self.config, data_files,
                           iter_dir, epochs, prev_model)

    # ------------------------------------------------------------------
    # Evaluation (delegates to pipeline_eval standalone functions)
    # ------------------------------------------------------------------

    def _evaluate(self, model_path: str, prev_model: str | None) -> dict:
        """Evaluate the model against baselines."""
        return evaluate(self.game, self.fs, self.config, model_path, prev_model)

    def _play_match(self, eval1, eval2, num_games: int,
                    depth: int = 6, time_limit_ms: int = 30000,
                    max_moves: int = 200,
                    model1_path: str | None = None,
                    model2_path: str | None = None) -> tuple:
        """Play a match between two evaluators (parallel across games)."""
        return play_match(self.game, eval1, eval2, num_games,
                          depth=depth, time_limit_ms=time_limit_ms,
                          max_moves=max_moves,
                          model1_path=model1_path,
                          model2_path=model2_path)

    # ------------------------------------------------------------------
    # Progress display
    # ------------------------------------------------------------------

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
