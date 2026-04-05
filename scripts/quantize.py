"""Convert float32 NNUE model weights to int16 quantized format."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_SCALE = 512


def quantize_model(npz_path: str, output_path: str, scale: int = DEFAULT_SCALE):
    """Quantize feature table weights to int16, keep L1/L2/output as float32."""
    data = np.load(npz_path)
    out = {}

    for key in data.files:
        arr = data[key]
        if "feature_table" in key or key == "ft_bias":
            # Quantize to int16: round(float * Q), clamped to int16 range
            q = np.clip(np.round(arr.astype(np.float64) * scale), -32767, 32767)
            out[key] = q.astype(np.int16)
            max_err = np.max(np.abs(arr - q.astype(np.float64) / scale))
            rms_err = np.sqrt(np.mean((arr - q.astype(np.float64) / scale) ** 2))
            print(f"  {key}: {arr.shape} float32 -> int16 "
                  f"(max_err={max_err:.6f}, rms_err={rms_err:.6f})")
        else:
            out[key] = arr.astype(np.float32)
            print(f"  {key}: {arr.shape} kept as float32")

    # Store quantization scale as metadata
    out["quant_scale"] = np.array(scale, dtype=np.float32)

    np.savez(output_path, **out)

    orig_size = Path(npz_path).stat().st_size
    new_size = Path(output_path).stat().st_size
    print(f"\nOriginal: {orig_size / 1024 / 1024:.1f} MB")
    print(f"Quantized: {new_size / 1024 / 1024:.1f} MB")
    print(f"Compression: {new_size / orig_size:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Quantize NNUE model to int16")
    parser.add_argument("input", help="Input .npz model file")
    parser.add_argument("output", help="Output quantized .npz file")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help=f"Quantization scale factor (default: {DEFAULT_SCALE})")
    args = parser.parse_args()

    print(f"Quantizing {args.input} with scale={args.scale}")
    quantize_model(args.input, args.output, args.scale)
    print("Done.")


if __name__ == "__main__":
    main()
