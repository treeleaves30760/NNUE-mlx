"""Monitor iterative training pipeline results.

Reads iteration metadata from timestamped model directories and
displays a progress summary.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_iterations(game: str, base_dir: str = ".") -> list:
    """Find all iteration directories for a game."""
    models_dir = Path(base_dir) / "models"
    iterations = []

    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / "iteration.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
            if meta.get("timestamp"):
                iterations.append(meta)
        except (json.JSONDecodeError, KeyError):
            continue

    return iterations


def print_summary(iterations: list):
    """Print a formatted summary table."""
    if not iterations:
        print("No iterations found.")
        return

    game = iterations[0].get("timestamp", "unknown")
    print(f"\n{'='*80}")
    print(f"  Pipeline Progress ({len(iterations)} iterations)")
    print(f"{'='*80}")

    header = (f"{'Iter':>4} | {'Timestamp':>19} | {'Loss':>10} | "
              f"{'vs Material':>14} | {'vs Previous':>14} | "
              f"{'Gen':>6} | {'Train':>6}")
    print(header)
    print("-" * 80)

    for h in iterations:
        it = h.get("iteration", "?")
        ts = h.get("timestamp", "?")
        loss = h.get("final_loss")
        loss_str = f"{loss:.6f}" if loss is not None else "N/A"

        vm = h.get("eval_vs_material", {})
        vm_str = f"W{vm.get('wins',0)}-L{vm.get('losses',0)}-D{vm.get('draws',0)}"
        vm_wr = vm.get('wins', 0) / max(
            vm.get('wins', 0) + vm.get('losses', 0) + vm.get('draws', 0), 1
        ) * 100

        vp = h.get("eval_vs_previous")
        if vp:
            vp_str = f"W{vp['wins']}-L{vp['losses']}-D{vp['draws']}"
            vp_wr = vp['wins'] / max(
                vp['wins'] + vp['losses'] + vp['draws'], 1
            ) * 100
        else:
            vp_str = "-"
            vp_wr = None

        gen_t = h.get("generation_time_s", 0)
        train_t = h.get("training_time_s", 0)

        print(f"{it:>4} | {ts:>19} | {loss_str:>10} | "
              f"{vm_str:>10} {vm_wr:4.0f}% | "
              f"{vp_str:>10} "
              f"{'    ' if vp_wr is None else f'{vp_wr:4.0f}%'}"
              f" | {gen_t:>5.0f}s | {train_t:>5.0f}s")

    # Print training loss progression
    print(f"\n{'='*80}")
    print("  Loss Progression")
    print(f"{'='*80}")
    losses = [(h["iteration"], h.get("final_loss", 0)) for h in iterations
              if h.get("final_loss") is not None]
    if losses:
        max_loss = max(l for _, l in losses)
        bar_width = 40
        for it, loss in losses:
            bar_len = int(loss / max(max_loss, 1e-6) * bar_width)
            bar = "#" * bar_len
            print(f"  Iter {it:>3}: {loss:.6f} |{bar}")

    # Print win rate progression vs material
    print(f"\n{'='*80}")
    print("  Win Rate vs Material")
    print(f"{'='*80}")
    for h in iterations:
        vm = h.get("eval_vs_material", {})
        total = vm.get("wins", 0) + vm.get("losses", 0) + vm.get("draws", 0)
        if total == 0:
            continue
        wr = vm["wins"] / total * 100
        bar_len = int(wr / 100 * 40)
        bar = "#" * bar_len
        print(f"  Iter {h['iteration']:>3}: {wr:5.1f}% |{bar}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor pipeline progress")
    parser.add_argument("--game", required=True)
    parser.add_argument("--state-file", default=None,
                        help="Pipeline state JSON (alternative to scanning dirs)")
    args = parser.parse_args()

    if args.state_file:
        state = json.loads(Path(args.state_file).read_text())
        iterations = state.get("history", [])
    else:
        iterations = find_iterations(args.game)

    # Filter by game if scanning dirs
    if not args.state_file:
        # State file approach
        state_path = Path(f"pipeline_state_{args.game}.json")
        if state_path.exists():
            state = json.loads(state_path.read_text())
            iterations = state.get("history", [])

    print_summary(iterations)


if __name__ == "__main__":
    main()
