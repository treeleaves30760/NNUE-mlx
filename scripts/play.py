"""Launch the Pygame GUI to play a board game."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Play a board game")
    parser.add_argument("--game", default=None,
                        choices=["chess", "minichess", "shogi", "minishogi"],
                        help="Pre-select game (overridable in GUI)")
    parser.add_argument("--mode", default=None,
                        choices=["human-vs-human", "human-vs-ai", "ai-vs-ai"],
                        help="Pre-select mode (overridable in GUI)")
    parser.add_argument("--model", default=None, help="NNUE model .npz path")
    parser.add_argument("--depth", type=int, default=None,
                        help="AI search depth (overridable in GUI)")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="AI time limit in ms (overridable in GUI)")
    args = parser.parse_args()

    from src.gui.app import GameApp
    app = GameApp()
    app.run(game_name=args.game, mode=args.mode,
            model_path=args.model, ai_depth=args.depth,
            ai_time_limit=args.time_limit)


if __name__ == "__main__":
    main()
