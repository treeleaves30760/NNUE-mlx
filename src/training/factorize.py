"""Feature factorization for NNUE training (chess HalfKP).

During training, virtual features provide extra regularization and faster
convergence. They are baked into the main feature table at export time so
inference is unchanged.

Virtual feature 'P': piece-square pair ignoring king square.
    virtual_idx = main_idx % piece_sq_combos
    num_virtual  = piece_sq_combos  (640 for chess, 288 for minichess)
"""

import numpy as np
import mlx.core as mx
from typing import Tuple

from src.features.halfkp import HalfKP


def build_factor_map(feature_set: HalfKP) -> Tuple[np.ndarray, int]:
    """Return (factor_map, num_virtual).
    factor_map[main_idx] gives the virtual feature index (0..num_virtual-1)."""
    piece_sq_combos = feature_set._piece_sq_combos
    num_main = feature_set.num_features()
    factor_map = np.arange(num_main, dtype=np.int32) % piece_sq_combos
    return factor_map, piece_sq_combos


def extend_num_features(num_main: int, num_virtual: int) -> int:
    """Training-time total feature count (main + virtual)."""
    return num_main + num_virtual


def expand_batch_with_virtual(batch: dict, factor_map_mx: mx.array,
                               num_main: int) -> dict:
    """Append virtual feature indices to each batch sample.

    For every active main index f, appends virtual index factor_map[f] + num_main.
    Doubles the max_active dimension on white_features, black_features, and masks.
    """
    wf = batch["white_features"]
    bf = batch["black_features"]
    wm = batch["white_mask"]
    bm = batch["black_mask"]
    # Look up virtual indices and offset so they live above num_main.
    w_virt = mx.take(factor_map_mx, wf) + num_main
    b_virt = mx.take(factor_map_mx, bf) + num_main
    # Concatenate main + virtual along the feature axis.
    new_batch = dict(batch)
    new_batch["white_features"] = mx.concatenate([wf, w_virt], axis=1)
    new_batch["black_features"] = mx.concatenate([bf, b_virt], axis=1)
    new_batch["white_mask"] = mx.concatenate([wm, wm], axis=1)
    new_batch["black_mask"] = mx.concatenate([bm, bm], axis=1)
    return new_batch


def bake_virtual_into_main(ft_weight: np.ndarray, factor_map: np.ndarray,
                            num_main: int) -> np.ndarray:
    """Collapse a factorized feature table into the main table.

    ft_weight: shape (num_main + num_virtual, accumulator_size)
    Returns: shape (num_main, accumulator_size)
    """
    main = ft_weight[:num_main]
    virtual = ft_weight[num_main:]
    return main + virtual[factor_map]
