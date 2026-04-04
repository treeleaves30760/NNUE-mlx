"""Accelerated NNUE inference with NEON SIMD + Apple Accelerate framework.

Falls back to the pure-numpy IncrementalAccumulator when the C extension
is not compiled (e.g. non-macOS or missing build step).
"""

_HAS_ACCEL = False

try:
    from src.accel._nnue_accel import AcceleratedAccumulator
    from src.accel._nnue_accel import halfkp_active_features
    from src.accel._nnue_accel import halfkp_shogi_active_features
    _HAS_ACCEL = True
except ImportError:
    from src.model.accumulator import IncrementalAccumulator as AcceleratedAccumulator
    halfkp_active_features = None
    halfkp_shogi_active_features = None

__all__ = [
    "AcceleratedAccumulator",
    "halfkp_active_features",
    "halfkp_shogi_active_features",
    "_HAS_ACCEL",
]
