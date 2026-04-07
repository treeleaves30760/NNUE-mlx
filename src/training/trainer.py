"""NNUE training loop with MLX on Apple Silicon."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from src.model.device import device_info
from src.model.nnue import NNUEModel
from src.training.dataset import load_batches, load_batches_from_samples, preload_samples
from src.training.loss import nnue_loss


class ExponentialDecayLR:
    """Exponential learning rate decay: lr *= gamma each epoch.

    Matches Stockfish NNUE training schedule (gamma=0.992).
    Smoother convergence than ReduceLROnPlateau.
    """

    def __init__(self, gamma: float = 0.992, min_lr: float = 1e-6):
        self.gamma = gamma
        self.min_lr = min_lr

    def step(self, loss: float, optimizer) -> float:
        """Decay learning rate and return current value."""
        current_lr = optimizer.learning_rate.item() if hasattr(
            optimizer.learning_rate, 'item') else float(optimizer.learning_rate)
        new_lr = max(current_lr * self.gamma, self.min_lr)
        optimizer.learning_rate = new_lr
        return current_lr


class Trainer:
    """Trains an NNUE model with MLX on Apple Silicon."""

    def __init__(self, num_features: int, accumulator_size: int = 256,
                 lr: float = 8.75e-4, batch_size: int = 16384,
                 max_active: int = 32, lambda_: float = 1.0,
                 lambda_end: float = 0.75, lr_gamma: float = 0.992):
        self.batch_size = batch_size
        self.max_active = max_active
        self.lambda_ = lambda_
        self.lambda_end = lambda_end

        self.model = NNUEModel(
            num_features=num_features,
            accumulator_size=accumulator_size,
        )
        mx.eval(self.model.parameters())  # Force initialization

        self._current_lambda = lambda_  # Updated per-epoch by lambda schedule

        self.optimizer = optim.Adam(learning_rate=lr)
        self.scheduler = ExponentialDecayLR(
            gamma=lr_gamma, min_lr=1e-6
        )

        # Build loss+grad function
        self._loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)

        # Compile the training step, declaring model/optimizer state so
        # mx.compile can track mutations through the compiled graph.
        self._state = [self.model.state, self.optimizer.state]
        self._compiled_step = mx.compile(
            self._train_step, inputs=self._state, outputs=self._state
        )

        print(f"Device: {device_info()}")
        param_count = sum(p.size for _, p in tree_flatten(self.model.parameters()))
        print(f"Model parameters: {param_count:,}")

    def get_lambda(self, epoch: int, total_epochs: int) -> float:
        """Linearly interpolate lambda from lambda_ to lambda_end over training."""
        if total_epochs <= 1:
            return self.lambda_
        t = min(epoch / (total_epochs - 1), 1.0)
        return self.lambda_ + t * (self.lambda_end - self.lambda_)

    def _loss_fn(self, model, batch):
        """Loss function for value_and_grad. Takes model as first arg."""
        pred = model(
            batch["white_features"], batch["black_features"],
            batch["white_mask"], batch["black_mask"],
            batch["side_to_move"],
        )
        return nnue_loss(pred, batch["score"], batch["result"], self._current_lambda)

    def _train_step(self, batch):
        """Single training step: forward + backward + optimizer update.

        Compiled by mx.compile with state inputs/outputs to fuse the
        entire step into one Metal compute graph.
        """
        loss, grads = self._loss_and_grad_fn(self.model, batch)
        self.optimizer.update(self.model, grads)
        return loss

    def train_epoch(self, data_path: str,
                    epoch: int = 0, total_epochs: int = 1) -> float:
        """Train for one epoch on a data file.

        Returns:
            Average loss for the epoch.
        """
        self._current_lambda = self.get_lambda(epoch, total_epochs)
        self.model.train()
        total_loss = mx.array(0.0)
        num_batches = 0

        for batch in load_batches(data_path, self.batch_size, self.max_active):
            loss = self._compiled_step(batch)
            # Eval loss to cap the lazy graph each batch.
            # State (model params + optimizer) is managed by mx.compile.
            mx.eval(loss)
            total_loss = total_loss + loss
            num_batches += 1

        avg_loss = (total_loss / max(num_batches, 1)).item()
        self.scheduler.step(avg_loss, self.optimizer)
        return avg_loss

    def train_epoch_from_samples(self, samples, shuffle: bool = True,
                                  epoch: int = 0,
                                  total_epochs: int = 1) -> float:
        """Train for one epoch on pre-loaded samples (with shuffling).

        Args:
            samples: Pre-loaded sample data (tuple-list or numpy dict).
            shuffle: Whether to shuffle samples each epoch.
            epoch: Current epoch number (for lambda schedule).
            total_epochs: Total epochs planned (for lambda schedule).

        Returns:
            Average loss for the epoch.
        """
        self._current_lambda = self.get_lambda(epoch, total_epochs)
        self.model.train()
        total_loss = mx.array(0.0)
        num_batches = 0

        for batch in load_batches_from_samples(
            samples, self.batch_size, self.max_active, shuffle=shuffle
        ):
            loss = self._compiled_step(batch)
            mx.eval(loss)
            total_loss = total_loss + loss
            num_batches += 1

        avg_loss = (total_loss / max(num_batches, 1)).item()
        self.scheduler.step(avg_loss, self.optimizer)
        return avg_loss

    def validate_epoch(self, samples) -> float:
        """Compute validation loss without gradient updates."""
        self.model.eval()
        total_loss = mx.array(0.0)
        num_batches = 0
        for batch in load_batches_from_samples(
            samples, self.batch_size, self.max_active, shuffle=False
        ):
            pred = self.model(
                batch["white_features"], batch["black_features"],
                batch["white_mask"], batch["black_mask"],
                batch["side_to_move"],
            )
            loss = nnue_loss(pred, batch["score"], batch["result"], self._current_lambda)
            mx.eval(loss)
            total_loss = total_loss + loss
            num_batches += 1
        return (total_loss / max(num_batches, 1)).item()

    def load_weights_from_npz(self, npz_path: str):
        """Load model weights from an exported .npz file (for warm-starting)."""
        data = np.load(npz_path)
        weights = [(k, mx.array(data[k])) for k in data.files]
        self.model.load_weights(weights)
        # Reset optimizer state for the new iteration
        self.optimizer = optim.Adam(
            learning_rate=float(self.optimizer.learning_rate.item()
                                if hasattr(self.optimizer.learning_rate, 'item')
                                else self.optimizer.learning_rate)
        )
        self.scheduler = ExponentialDecayLR(
            gamma=self.scheduler.gamma, min_lr=1e-6
        )
        # Re-build compiled step with fresh optimizer state
        self._state = [self.model.state, self.optimizer.state]
        self._compiled_step = mx.compile(
            self._train_step, inputs=self._state, outputs=self._state
        )

    def save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Save training checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.model.parameters()))
        wt_path = filepath.replace('.pt', '.npz')
        np.savez(wt_path, **{k: np.array(v) for k, v in weights.items()})
        meta_path = filepath.replace('.pt', '_meta.npz')
        lr = self.optimizer.learning_rate
        lr_val = lr.item() if hasattr(lr, 'item') else float(lr)
        np.savez(meta_path, epoch=epoch, loss=loss, lr=lr_val)

    def load_checkpoint(self, filepath: str) -> int:
        """Load training checkpoint. Returns the epoch number."""
        wt_path = filepath.replace('.pt', '.npz')
        data = np.load(wt_path)
        weights = [(k, mx.array(data[k])) for k in data.files]
        self.model.load_weights(weights)
        meta_path = filepath.replace('.pt', '_meta.npz')
        meta = np.load(meta_path)
        return int(meta['epoch'])

    def export_numpy(self, filepath: str):
        """Export model weights as .npz for CPU inference during search."""
        weights = {k: np.array(v) for k, v in tree_flatten(self.model.parameters())}
        np.savez(filepath, **weights)
        print(f"Exported numpy weights to {filepath}")
