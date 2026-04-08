"""Train an NNUE model for a specific game using MLX on Apple Silicon."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import Trainer
from src.features.halfkp import chess_features, minichess_features
from src.features.halfkp_shogi import shogi_features, minishogi_features


FEATURE_SETS = {
    "chess": chess_features,
    "minichess": minichess_features,
    "shogi": shogi_features,
    "minishogi": minishogi_features,
}


def main():
    parser = argparse.ArgumentParser(description="Train NNUE model with MLX")
    parser.add_argument("--game", required=True, choices=FEATURE_SETS.keys())
    parser.add_argument("--data", required=True, help="Path to training data .bin file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=8.75e-4)
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_")
    parser.add_argument("--lambda-end", type=float, default=0.75, dest="lambda_end")
    parser.add_argument("--lr-gamma", type=float, default=0.992, dest="lr_gamma")
    parser.add_argument("--accumulator-size", type=int, default=256, dest="accumulator_size")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--output-dir", default="models", help="Directory for checkpoints")
    args = parser.parse_args()

    fs = FEATURE_SETS[args.game]()
    print(f"Game: {args.game}, Features: {fs.num_features():,}")

    trainer = Trainer(
        num_features=fs.num_features(),
        accumulator_size=args.accumulator_size,
        lr=args.lr,
        batch_size=args.batch_size,
        max_active=fs.max_active_features(),
        lambda_=args.lambda_,
        lambda_end=args.lambda_end,
        lr_gamma=args.lr_gamma,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resumed from epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    for epoch in range(start_epoch, args.epochs):
        loss = trainer.train_epoch(args.data, epoch=epoch, total_epochs=args.epochs)
        lr = trainer.optimizer.learning_rate
        lr_val = lr.item() if hasattr(lr, 'item') else float(lr)
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {loss:.6f} | LR: {lr_val:.2e}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"{args.game}_epoch{epoch + 1}.pt"
            trainer.save_checkpoint(str(ckpt_path), epoch, loss)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Export final model as numpy for search inference
    npz_path = output_dir / f"{args.game}_final.npz"
    trainer.export_numpy(str(npz_path))


if __name__ == "__main__":
    main()
