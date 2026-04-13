"""Convert float32 NNUE model weights to quantized format.

Default mode: FT layers -> int16 (scale=512), all other layers -> float32.
Full-int8 mode (--full-int8): additionally quantizes L1/L2/output/wdl_output
weights to int8 using per-layer absmax scaling.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_SCALE = 512

# Weight keys that receive int8 quantization in --full-int8 mode.
# Biases are excluded: they are small and sensitive to quantization error.
_INT8_WEIGHT_KEYS = {
    "l1.weight",
    "l2.weight",
    "output.weight",
    "wdl_output.weight",
}


def _quantize_int8(
    arr: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Quantize a float32 array to int8 using per-tensor absmax scaling.

    Args:
        arr: Float32 weight array of any shape.

    Returns:
        Tuple of (int8_array, scale, max_err, rms_err) where scale satisfies
        original ~= int8_array * scale.
    """
    abs_max = float(np.max(np.abs(arr)))
    if abs_max == 0.0:
        scale = 1.0
    else:
        scale = abs_max / 127.0

    q = np.clip(np.round(arr.astype(np.float64) / scale), -127, 127).astype(np.int8)
    reconstructed = q.astype(np.float64) * scale
    diff = arr.astype(np.float64) - reconstructed
    max_err = float(np.max(np.abs(diff)))
    rms_err = float(np.sqrt(np.mean(diff ** 2)))
    return q, scale, max_err, rms_err


def quantize_model(
    npz_path: str,
    output_path: str,
    scale: int = DEFAULT_SCALE,
    full_int8: bool = False,
) -> None:
    """Quantize NNUE model weights and write a new .npz file.

    FT layers (feature_table.*, ft_bias) are always quantized to int16.
    In default mode all other tensors are kept as float32.
    In --full-int8 mode the weight matrices for l1/l2/output/wdl_output are
    additionally quantized to int8; their per-layer scales are stored as
    metadata keys (<layer>_scale).

    Metadata scalars (0-d arrays) and keys that already carry scale / bucket
    information (quant_scale, num_output_buckets, output_eval_scale, …) are
    always copied unchanged.

    Args:
        npz_path: Path to the source float32 .npz model file.
        output_path: Destination path for the quantized .npz file.
        scale: Integer scale factor for FT int16 quantization (default 512).
        full_int8: When True, also quantize L1/L2/output weight matrices to int8.
    """
    data = np.load(npz_path)
    out: dict[str, np.ndarray] = {}

    orig_bytes = 0
    for key in data.files:
        arr = data[key]
        orig_bytes += arr.nbytes

        # ------------------------------------------------------------------ #
        # Metadata scalars: copy unchanged regardless of mode.                #
        # ------------------------------------------------------------------ #
        if arr.ndim == 0:
            out[key] = arr
            print(f"  {key}: scalar {arr.dtype} -> kept as-is ({arr.item()})")
            continue

        # ------------------------------------------------------------------ #
        # Feature-transformer layers: always int16.                           #
        # ------------------------------------------------------------------ #
        if "feature_table" in key or key == "ft_bias":
            q = np.clip(
                np.round(arr.astype(np.float64) * scale), -32767, 32767
            )
            out[key] = q.astype(np.int16)
            max_err = float(np.max(np.abs(arr - q.astype(np.float64) / scale)))
            rms_err = float(
                np.sqrt(np.mean((arr - q.astype(np.float64) / scale) ** 2))
            )
            print(
                f"  {key}: {arr.shape} float32 -> int16 "
                f"(max_err={max_err:.6f}, rms_err={rms_err:.6f})"
            )
            continue

        # ------------------------------------------------------------------ #
        # Weight matrices eligible for int8 (only when --full-int8 is set).  #
        # ------------------------------------------------------------------ #
        if full_int8 and key in _INT8_WEIGHT_KEYS:
            q8, q8_scale, max_err, rms_err = _quantize_int8(arr)
            out[key] = q8
            # Store per-layer scale so the inference side can dequantize.
            scale_key = key.split(".")[0] + "_scale"  # e.g. "l1_scale"
            out[scale_key] = np.array(q8_scale, dtype=np.float32)
            print(
                f"  {key}: {arr.shape} float32 -> int8 "
                f"scale={q8_scale:.6f} "
                f"(max_err={max_err:.6f}, rms_err={rms_err:.6f})"
            )
            continue

        # ------------------------------------------------------------------ #
        # Everything else: biases, unrecognised keys -> float32.              #
        # ------------------------------------------------------------------ #
        out[key] = arr.astype(np.float32)
        print(f"  {key}: {arr.shape} {arr.dtype} -> kept as float32")

    # Always write the FT quantization scale.
    out["quant_scale"] = np.array(scale, dtype=np.float32)

    np.savez(output_path, **out)

    quant_bytes = sum(v.nbytes for v in out.values())
    orig_size = Path(npz_path).stat().st_size
    new_size = Path(output_path + ".npz").stat().st_size if not output_path.endswith(".npz") else Path(output_path).stat().st_size
    print(f"\nOriginal:   {orig_size / 1024 / 1024:.2f} MB")
    print(f"Quantized:  {new_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {new_size / orig_size:.1%}")
    if full_int8:
        print(f"In-memory tensor bytes: {orig_bytes} -> {quant_bytes} "
              f"({quant_bytes / orig_bytes:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize NNUE model weights (FT->int16, optionally L1/L2/output->int8)"
    )
    parser.add_argument("input", help="Input float32 .npz model file")
    parser.add_argument("output", help="Output quantized .npz file")
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help=f"FT int16 quantization scale factor (default: {DEFAULT_SCALE})",
    )
    parser.add_argument(
        "--full-int8",
        action="store_true",
        default=False,
        help="Also quantize L1/L2/output/wdl_output weight matrices to int8",
    )
    args = parser.parse_args()

    mode = "full-int8" if args.full_int8 else "default (FT int16 only)"
    print(f"Quantizing {args.input}  scale={args.scale}  mode={mode}")
    quantize_model(args.input, args.output, scale=args.scale, full_int8=args.full_int8)
    print("Done.")


if __name__ == "__main__":
    main()
