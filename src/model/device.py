"""MLX device management.

MLX uses Apple Silicon unified memory -- CPU and GPU share the same
memory space. There is no data transfer overhead and no explicit
device management needed.
"""

import mlx.core as mx


def get_device() -> str:
    """Return a description of the current MLX device."""
    return str(mx.default_device())


def synchronize():
    """Force evaluation of all pending lazy computations."""
    mx.eval()


def device_info() -> str:
    """Return a human-readable string describing the MLX device."""
    return f"MLX ({mx.default_device()}), Apple Silicon unified memory"
