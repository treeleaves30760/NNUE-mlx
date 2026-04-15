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
from src.training.loss import nnue_loss, nnue_loss_wdl
from src.training.factorize import build_factor_map, expand_batch_with_virtual, bake_virtual_into_main


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


class CosineAnnealingLR:
    """Cosine annealing learning rate schedule with warm restarts.

    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))
    Better final convergence than exponential decay.
    """

    def __init__(self, max_lr: float, total_epochs: int, min_lr: float = 1e-6):
        self.max_lr = max_lr
        self.total_epochs = max(1, total_epochs)
        self.min_lr = min_lr
        self._epoch = 0

    def step(self, loss: float, optimizer) -> float:
        """Update learning rate following cosine schedule."""
        import math
        current_lr = optimizer.learning_rate.item() if hasattr(
            optimizer.learning_rate, 'item') else float(optimizer.learning_rate)
        self._epoch += 1
        t = min(self._epoch / self.total_epochs, 1.0)
        new_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        optimizer.learning_rate = new_lr
        return current_lr


class Trainer:
    """Trains an NNUE model with MLX on Apple Silicon."""

    def __init__(self, num_features: int, accumulator_size: int = 256,
                 l1_size: int = 128, l2_size: int = 32,
                 num_output_buckets: int = 1,
                 use_wdl_head: bool = False,
                 wdl_weight: float = 0.5,
                 factorize: bool = False,
                 feature_set=None,
                 lr: float = 8.75e-4,
                 batch_size: int = 16384,
                 max_active: int = 32, lambda_: float = 1.0,
                 lambda_end: float = 0.75, lr_gamma: float = 0.992,
                 mirror_table: np.ndarray = None,
                 total_epochs: int = 100,
                 max_grad_norm: float = 1.0):
        self.batch_size = batch_size
        self.max_active = max_active
        self.lambda_ = lambda_
        self.lambda_end = lambda_end
        self.mirror_table = mirror_table
        self.max_grad_norm = max_grad_norm
        self._lr = lr
        self.use_wdl_head = use_wdl_head
        self.wdl_weight = wdl_weight
        self.num_output_buckets = num_output_buckets

        # Factorization setup
        self.factor_map_mx = None
        self.num_main_features = num_features
        if factorize:
            if feature_set is None:
                raise ValueError("feature_set is required when factorize=True")
            factor_map_np, num_virtual = build_factor_map(feature_set)
            total_features = num_features + num_virtual
            self.factor_map_mx = mx.array(factor_map_np)
            self.num_main_features = num_features
        else:
            total_features = num_features

        # Bucketing setup
        self.feature_set_for_bucket = None
        if num_output_buckets > 1:
            if feature_set is None:
                raise ValueError("feature_set is required when num_output_buckets > 1")
            self.feature_set_for_bucket = feature_set

        self.model = NNUEModel(
            num_features=total_features,
            accumulator_size=accumulator_size,
            l1_size=l1_size,
            l2_size=l2_size,
            num_output_buckets=num_output_buckets,
            use_wdl_head=use_wdl_head,
        )
        mx.eval(self.model.parameters())  # Force initialization

        self._current_lambda = lambda_  # Updated per-epoch by lambda schedule

        self.optimizer = optim.Adam(learning_rate=lr)
        self.scheduler = CosineAnnealingLR(
            max_lr=lr, total_epochs=total_epochs, min_lr=1e-6
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
        if self.factor_map_mx is not None:
            batch = expand_batch_with_virtual(batch, self.factor_map_mx, self.num_main_features)

        if self.feature_set_for_bucket is not None:
            num_w = mx.sum(batch["white_mask"], axis=1).astype(mx.int32)
            num_b = mx.sum(batch["black_mask"], axis=1).astype(mx.int32)
            max_fc = self.feature_set_for_bucket._max_feature_count
            total = num_w + num_b
            bucket_idx = mx.minimum(
                (total * self.num_output_buckets) // (2 * max_fc + 1),
                self.num_output_buckets - 1,
            )
        else:
            bucket_idx = None

        pred = model(
            batch["white_features"], batch["black_features"],
            batch["white_mask"], batch["black_mask"],
            batch["side_to_move"],
            bucket_idx=bucket_idx,
        )

        if self.use_wdl_head:
            score_pred, wdl_pred = pred
            return nnue_loss_wdl(
                score_pred, wdl_pred,
                batch["score"], batch["result"],
                self._current_lambda, self.wdl_weight,
            )
        return nnue_loss(pred, batch["score"], batch["result"], self._current_lambda)

    def _train_step(self, batch):
        """Single training step: forward + backward + optimizer update.

        Compiled by mx.compile with state inputs/outputs to fuse the
        entire step into one Metal compute graph.
        """
        loss, grads = self._loss_and_grad_fn(self.model, batch)
        # Gradient clipping to prevent instability
        grads, _ = optim.clip_grad_norm(grads, max_norm=self.max_grad_norm)
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
            samples, self.batch_size, self.max_active, shuffle=shuffle,
            mirror_table=self.mirror_table if shuffle else None,
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
            if self.factor_map_mx is not None:
                batch = expand_batch_with_virtual(batch, self.factor_map_mx, self.num_main_features)

            if self.feature_set_for_bucket is not None:
                num_w = mx.sum(batch["white_mask"], axis=1).astype(mx.int32)
                num_b = mx.sum(batch["black_mask"], axis=1).astype(mx.int32)
                max_fc = self.feature_set_for_bucket._max_feature_count
                total = num_w + num_b
                bucket_idx = mx.minimum(
                    (total * self.num_output_buckets) // (2 * max_fc + 1),
                    self.num_output_buckets - 1,
                )
            else:
                bucket_idx = None

            pred = self.model(
                batch["white_features"], batch["black_features"],
                batch["white_mask"], batch["black_mask"],
                batch["side_to_move"],
                bucket_idx=bucket_idx,
            )

            if self.use_wdl_head:
                score_pred, wdl_pred = pred
                loss = nnue_loss_wdl(
                    score_pred, wdl_pred,
                    batch["score"], batch["result"],
                    self._current_lambda, self.wdl_weight,
                )
            else:
                loss = nnue_loss(pred, batch["score"], batch["result"], self._current_lambda)

            mx.eval(loss)
            total_loss = total_loss + loss
            num_batches += 1
        return (total_loss / max(num_batches, 1)).item()

    # Non-parameter metadata that `export_numpy` writes alongside weights.
    # Must be stripped on load so `model.load_weights` doesn't reject them.
    _EXPORT_METADATA_KEYS = frozenset({
        "num_output_buckets",
        "output_eval_scale",
        "quant_scale",
        "l1_scale",
        "l2_scale",
        "output_scale",
    })

    def load_weights_from_npz(self, npz_path: str, total_epochs: int = 100):
        """Load model weights from an exported .npz file (for warm-starting)."""
        data = np.load(npz_path)
        weights = [
            (k, mx.array(data[k]))
            for k in data.files
            if k not in self._EXPORT_METADATA_KEYS
        ]
        self.model.load_weights(weights)
        # Reset optimizer state for the new iteration
        self.optimizer = optim.Adam(learning_rate=self._lr)
        self.scheduler = CosineAnnealingLR(
            max_lr=self._lr, total_epochs=total_epochs, min_lr=1e-6
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
        self.save_weights(filepath)

    def save_weights(self, npz_path: str):
        """Save model weights as .npz with training metadata.

        If the model was trained with factorization, the virtual feature rows are
        baked into the main feature table so inference is unchanged.

        Metadata keys saved alongside weights:
            num_output_buckets: int
            output_eval_scale: float32 (128.0 — matches evaluator.py constant)
        """
        weights = {k: np.array(v, dtype=np.float32)
                   for k, v in tree_flatten(self.model.parameters())}

        if self.factor_map_mx is not None:
            ft_key = "feature_table.weight"
            if ft_key in weights:
                factor_map_np = np.array(self.factor_map_mx, dtype=np.int32)
                weights[ft_key] = bake_virtual_into_main(
                    weights[ft_key], factor_map_np, self.num_main_features
                )

        weights["num_output_buckets"] = np.array(self.num_output_buckets, dtype=np.int32)
        # Must equal OUTPUT_SCALE from src/training/loss.py: the loss passes
        # model output through sigmoid(raw * OUTPUT_SCALE / EVAL_SCALE), so a
        # converged model produces raw = score / OUTPUT_SCALE. Inference
        # inverts that by multiplying by OUTPUT_SCALE. Old checkpoints
        # stored 128.0 here by mistake; _build_from_data detects that and
        # rewrites it to the correct value on load.
        from src.training.loss import OUTPUT_SCALE as _LOSS_OUTPUT_SCALE
        weights["output_eval_scale"] = np.array(
            float(_LOSS_OUTPUT_SCALE), dtype=np.float32
        )

        np.savez(npz_path, **weights)
        print(f"Exported numpy weights to {npz_path}")
