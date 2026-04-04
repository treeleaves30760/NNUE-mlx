# NNUE-mlx

NNUE (Efficiently Updatable Neural Network) board game AI, trained on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) with unified memory.

Includes four playable games with Pygame GUI, alpha-beta search engine, and a complete self-play training pipeline.

## Games

| Game | Board | Notes |
|------|-------|-------|
| Chess | 8x8 | Full rules (castling, en passant, promotion) |
| Los Alamos Chess | 6x6 | No bishops, no castling, pawns promote to queen only |
| Shogi | 9x9 | Piece drops, promotions, nifu/uchifuzume rules |
| Mini Shogi | 5x5 | Simplified shogi with 6 piece types per side |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Install dependencies and build the accelerated inference engine
uv sync
make build-accel

# Play a game (opens Pygame window with game selection menu)
uv run python scripts/play.py
```

## Training

Four-step pipeline: build the inference engine, generate data via self-play, train with MLX, then play against the trained model.

Start with `minichess` (Los Alamos 6x6) for fastest iteration -- smallest feature space (10,368 vs 40,960 for chess).

```bash
# 1. Build the NEON SIMD + Accelerate inference engine
make build-accel

# 2. Generate training data (random self-play for bootstrapping)
uv run python scripts/selfplay.py --game minichess --games 5000

# 3. Train with MLX on Apple Silicon
uv run python scripts/train.py --game minichess --data data/minichess_*.bin --epochs 50

# 4. Play against the trained AI
uv run python scripts/play.py --game minichess --mode human-vs-ai --model models/minichess_final.npz
```

To evaluate a trained model against the material-counting baseline:

```bash
uv run python scripts/evaluate.py --game minichess --model1 models/minichess_final.npz --model2 material --games 100
```

## Accelerated Inference

The C extension accelerates the search engine using ARM NEON SIMD intrinsics and Apple's Accelerate framework (`cblas_sgemv` / AMX coprocessor). Once built via `make build-accel`, it is used automatically.

If the extension is not built, the engine falls back to a pure-numpy implementation.

| Component | Without Extension | With Extension | Speedup |
|-----------|-------------------|----------------|---------|
| push/pop | ~2.0 us | 0.07 us | 29x |
| update | ~2.0 us | 0.08 us | 26x |
| evaluate | ~8.0 us | 1.3 us | 6x |
| **Per search node** | **12.5 us** | **1.5 us** | **8.4x** |

## Project Structure

```
src/
  games/        Game engines (chess, shogi, mini variants)
  model/        NNUE MLX model, incremental accumulator
  features/     HalfKP feature extraction (board + hand pieces for shogi)
  search/       Alpha-beta search, transposition table, move ordering
  training/     Self-play data generation, MLX training loop, loss function
  gui/          Pygame board renderers, piece drawing, app menu
  accel/        C extension for NEON SIMD + Accelerate inference
scripts/        CLI entry points (play, train, selfplay, evaluate)
models/         Saved model checkpoints (.npz)
data/           Training data (.bin)
```

## Architecture

```
HalfKP Features --> Embedding(256) + ClippedReLU
                    x2 perspectives (white/black), shared weights
                          |
                    Concat (512)
                          |
                    Linear(32) + ClippedReLU
                          |
                    Linear(32) + ClippedReLU
                          |
                    Linear(1) --> evaluation score
```

- **Training**: MLX on Apple Silicon (unified memory, zero-copy CPU/GPU, JIT compilation)
- **Search inference**: Incremental accumulator on CPU, accelerated with NEON SIMD + Apple Accelerate framework

## License

MIT
