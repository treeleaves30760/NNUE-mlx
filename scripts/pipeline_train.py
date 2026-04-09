"""Training and data-generation logic extracted from IterativePipeline."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import Trainer
from src.training.dataset import preload_samples
from src.training.selfplay import SelfPlayEngine
from src.utils.config import create_game


def train_model(fs, config: dict, data_files: list,
                iter_dir: Path, epochs: int,
                prev_model: str | None) -> tuple:
    """Train a model on collected data files.

    Returns (model_path, training_log).
    """
    max_active = fs.max_active_features()
    print(f"  Loading {len(data_files)} data file(s)...")
    all_data = preload_samples(data_files, max_active=max_active)
    total_samples = len(all_data["score"])

    # 90/10 train/validation split via index shuffling
    indices = list(range(total_samples))
    random.shuffle(indices)
    val_size = max(1, int(total_samples * 0.1))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_data = {k: v[train_idx] for k, v in all_data.items()}
    val_data = {k: v[val_idx] for k, v in all_data.items()}
    print(f"  Train: {len(train_idx):,} | Val: {val_size:,}")

    # Build mirror table for data augmentation if feature set supports it
    mirror_tbl = fs.mirror_table() if hasattr(fs, 'mirror_table') else None

    trainer = Trainer(
        num_features=fs.num_features(),
        accumulator_size=config.get("accumulator_size", 256),
        l1_size=config.get("l1_size", 128),
        lr=config["lr"],
        batch_size=config["batch_size"],
        max_active=max_active,
        lambda_=config.get("lambda_", 1.0),
        lambda_end=config.get("lambda_end", 0.75),
        lr_gamma=config.get("lr_gamma", 0.992),
        mirror_table=mirror_tbl,
        total_epochs=epochs,
    )

    # Warm-start from previous model if available
    if prev_model and Path(prev_model).exists():
        print(f"  Warm-starting from {prev_model}")
        trainer.load_weights_from_npz(prev_model, total_epochs=epochs)

    # Early stopping: restore best model when val loss stops improving
    early_stop_patience = config.get("early_stop_patience", 15)
    best_val_loss = float('inf')
    best_epoch = 0
    wait = 0

    training_log = []
    for epoch in range(epochs):
        loss = trainer.train_epoch_from_samples(
            train_data, shuffle=True,
            epoch=epoch, total_epochs=epochs)
        val_loss = trainer.validate_epoch(val_data)
        lr = trainer.optimizer.learning_rate
        lr_val = lr.item() if hasattr(lr, 'item') else float(lr)
        lam = trainer.get_lambda(epoch, epochs)
        training_log.append({
            "epoch": epoch + 1,
            "loss": round(loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr_val,
            "lambda": round(lam, 4),
        })
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch + 1}/{epochs} | "
                  f"Loss: {loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {lr_val:.2e} | λ: {lam:.3f}")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            wait = 0
            # Save best checkpoint
            trainer.save_checkpoint(
                str(iter_dir / "best.pt"), epoch, val_loss)
        else:
            wait += 1
            if wait >= early_stop_patience:
                print(f"    Early stopping at epoch {epoch + 1} "
                      f"(best val: {best_val_loss:.6f} at epoch {best_epoch})")
                break

    # Restore best model if early stopping triggered
    best_ckpt = iter_dir / "best.npz"
    if best_ckpt.exists() and wait >= early_stop_patience:
        trainer.load_checkpoint(str(iter_dir / "best.pt"))
        print(f"    Restored best model from epoch {best_epoch}")

    # Export model
    model_path = str(iter_dir / "model.npz")
    trainer.export_numpy(model_path)

    # Save training log
    log_path = iter_dir / "training_log.json"
    log_path.write_text(json.dumps(training_log, indent=2))

    return model_path, training_log


def generate_data(game_name: str, fs, config: dict,
                  iteration: int, prev_model: str | None,
                  data_dir: Path, bootstrap_data: str | None) -> str:
    """Generate self-play training data. Returns path to .bin file."""
    is_bootstrap = (iteration == 0)

    # If bootstrap data is provided, copy/link it instead of generating
    if is_bootstrap and bootstrap_data:
        data_file = str(data_dir / f"{game_name}_iter0.bin")
        import shutil
        if not Path(data_file).exists():
            shutil.copy2(bootstrap_data, data_file)
        print(f"  Using existing bootstrap data: {bootstrap_data}")
        sz = Path(data_file).stat().st_size
        print(f"  Data size: {sz / 1024 / 1024:.1f}MB")
        return data_file

    def _get_schedule(key: str, iteration: int) -> int:
        schedule = config[key]
        return schedule[min(iteration, len(schedule) - 1)]

    if is_bootstrap:
        num_games = config["bootstrap_games"]
        depth = config.get("bootstrap_depth", 3)
        model_path = None
        print(f"  Generating {num_games} bootstrap games "
              f"(Rule-Based AI, depth={depth})...")
    else:
        num_games = config["games_per_iter"]
        depth = _get_schedule("max_depth_schedule", iteration)
        model_path = prev_model
        time_limit = _get_schedule("time_limit_schedule", iteration)
        print(f"  Generating {num_games} games (depth={depth}, "
              f"time={time_limit}ms) with model...")

    data_file = str(data_dir / f"{game_name}_iter{iteration}.bin")

    evaluator = None
    if is_bootstrap:
        from src.search.evaluator import RuleBasedEvaluator
        from src.search.alphabeta import AlphaBetaSearch
        rule_eval = RuleBasedEvaluator()
        evaluator = AlphaBetaSearch(rule_eval, max_depth=depth, time_limit_ms=500)
    elif model_path and config["workers"] == 1:
        from src.search.evaluator import NNUEEvaluator
        from src.search.alphabeta import AlphaBetaSearch
        nnue_eval = NNUEEvaluator.from_numpy(model_path, fs)
        tl = _get_schedule("time_limit_schedule", iteration)
        evaluator = AlphaBetaSearch(nnue_eval, max_depth=depth, time_limit_ms=tl)

    tl_engine = (_get_schedule("time_limit_schedule", iteration)
                 if not is_bootstrap else 2000)
    random_opening = config.get("random_opening_plies", 8) if not is_bootstrap else 4
    engine = SelfPlayEngine(
        feature_set=fs,
        evaluator=evaluator,
        search_depth=depth,
        random_play_prob=0.10 if is_bootstrap else 0.05,
        game_name=game_name,
        model_path=model_path,
        time_limit_ms=tl_engine,
        use_rule_eval=is_bootstrap,
        random_opening_plies=random_opening,
    )

    engine.generate_data(
        initial_state_fn=lambda: create_game(game_name),
        output_path=data_file,
        num_games=num_games,
        max_moves=config.get("max_moves", 200),
        num_workers=config["workers"],
    )

    return data_file


def collect_data_files(game_name: str, config: dict,
                       data_dir: Path, iteration: int) -> list:
    """Collect data files within the rolling window.

    Includes bootstrap data only for the first N iterations (configured by
    bootstrap_window, default 3). After that, the model should have
    surpassed material-level play, and bootstrap data would pull it back.
    """
    window = config["data_window"]
    bootstrap_window = config.get("bootstrap_window", 99)
    files = []

    # Include bootstrap data only for early iterations
    if iteration <= bootstrap_window:
        bootstrap_path = str(data_dir / f"{game_name}_iter0.bin")
        if Path(bootstrap_path).exists():
            files.append(bootstrap_path)

    # Add recent iterations (excluding iter0 to avoid duplicating bootstrap)
    start = max(1, iteration + 1 - window)
    for i in range(start, iteration + 1):
        if i == 0:
            continue
        path = str(data_dir / f"{game_name}_iter{i}.bin")
        if Path(path).exists():
            files.append(path)
    return files
