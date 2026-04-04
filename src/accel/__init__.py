"""Accelerated NNUE inference with NEON SIMD + Apple Accelerate framework.

Falls back to the pure-numpy IncrementalAccumulator when the C extension
is not compiled (e.g. non-macOS or missing build step).
"""

try:
    from src.accel._nnue_accel import AcceleratedAccumulator
except ImportError:
    from src.model.accumulator import IncrementalAccumulator as AcceleratedAccumulator

__all__ = ["AcceleratedAccumulator"]
