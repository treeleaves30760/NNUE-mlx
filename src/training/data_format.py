"""Binary format for NNUE training data.

Each record:
    num_white_features: uint16
    white_feature_indices: uint32[num_white_features]
    num_black_features: uint16
    black_feature_indices: uint32[num_black_features]
    side_to_move: uint8 (0=white, 1=black)
    score: int16 (centipawn evaluation from search)
    result: int8 (-1=loss, 0=draw, +1=win for side to move)
"""

import struct
from typing import BinaryIO, List


def write_sample(f: BinaryIO, white_features: List[int],
                 black_features: List[int], side_to_move: int,
                 score: int, result: int):
    """Write a single training sample to a binary file."""
    nw = len(white_features)
    nb = len(black_features)
    # Pack everything in one call: header + features + header + features + tail
    fmt = f"<H{nw}IH{nb}IBhb"
    f.write(struct.pack(
        fmt, nw, *white_features, nb, *black_features,
        side_to_move, max(-32768, min(32767, score)), result,
    ))


def read_sample(f: BinaryIO):
    """Read a single training sample from a binary file.

    Returns:
        Tuple of (white_features, black_features, side_to_move, score, result)
        or None if EOF reached.
    """
    header = f.read(2)
    if len(header) < 2:
        return None

    num_wf = struct.unpack("<H", header)[0]
    wf_data = f.read(num_wf * 4)
    if len(wf_data) < num_wf * 4:
        return None
    white_features = list(struct.unpack(f"<{num_wf}I", wf_data))

    num_bf_data = f.read(2)
    if len(num_bf_data) < 2:
        return None
    num_bf = struct.unpack("<H", num_bf_data)[0]
    bf_data = f.read(num_bf * 4)
    if len(bf_data) < num_bf * 4:
        return None
    black_features = list(struct.unpack(f"<{num_bf}I", bf_data))

    tail = f.read(4)
    if len(tail) < 4:
        return None
    side_to_move = struct.unpack("<B", tail[0:1])[0]
    score = struct.unpack("<h", tail[1:3])[0]
    result = struct.unpack("<b", tail[3:4])[0]

    return white_features, black_features, side_to_move, score, result
