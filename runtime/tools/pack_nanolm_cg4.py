#!/usr/bin/env python3
"""Repack NanoLM Q4 into the fx-CG50 single-read CG4 stream.

CG4 does not requantize any weight. Each original 64-weight group becomes:

    4-byte big-endian float32 scale + 32 unchanged packed Q4 bytes

The float32 value is the exact expansion of the original float16 scale. This
layout trades roughly 0.7 MB of storage for one read instead of two per block
and no float16 scale conversion in the inference hot path.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path


CG4_MAGIC = b"NLMCG401"
INDEX_MAGIC = b"NLMIDX03"
PREFIX_MAGIC = b"NLMPFX01"
DATA_BASE = 64
TENSOR_COUNT = 110
VOCAB_SIZE = 20000
GROUP_SIZE = 64
GROUP_BYTES = 36


def align_up(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_index(path: Path) -> tuple[int, list[tuple[int, int, int, int]]]:
    raw = path.read_bytes()
    if len(raw) != 56 + TENSOR_COUNT * 16 or raw[:8] != b"NLMIDX02":
        raise ValueError("source index must be NanoLM v2 with 110 tensors")
    data_base, tensors, vocab, group = struct.unpack_from("<IIII", raw, 8)
    if (tensors, vocab, group) != (TENSOR_COUNT, VOCAB_SIZE, GROUP_SIZE):
        raise ValueError("unexpected source index configuration")
    entries = [struct.unpack_from("<IIII", raw, 56 + i * 16)
               for i in range(TENSOR_COUNT)]
    return data_base, entries


def repack(source_q4: Path, source_index: Path, source_prefix: Path,
           output_dir: Path) -> None:
    source = source_q4.read_bytes()
    source_base, entries = read_index(source_index)
    output = bytearray(DATA_BASE)
    source_digest = hashlib.sha256(source).digest()
    struct.pack_into("<8sIIII32s", output, 0, CG4_MAGIC, DATA_BASE,
                     TENSOR_COUNT, VOCAB_SIZE, GROUP_SIZE, source_digest)
    new_entries: list[tuple[int, int, int, int]] = []

    for tensor, (scale_or_raw, quant_offset, scale_or_raw_bytes,
                 quant_bytes) in enumerate(entries):
        aligned = align_up(len(output))
        output.extend(b"\0" * (aligned - len(output)))
        relative_offset = len(output) - DATA_BASE
        if quant_bytes == 0:
            begin = source_base + scale_or_raw
            raw = source[begin:begin + scale_or_raw_bytes]
            if len(raw) != scale_or_raw_bytes:
                raise ValueError(f"tensor {tensor}: truncated f16 data")
            output.extend(raw)
            new_entries.append((relative_offset, 0, len(raw), 0))
            continue

        if scale_or_raw_bytes % 2:
            raise ValueError(f"tensor {tensor}: invalid scale byte count")
        groups = scale_or_raw_bytes // 2
        if quant_bytes != groups * (GROUP_SIZE // 2):
            raise ValueError(f"tensor {tensor}: Q4 group count mismatch")
        scale_begin = source_base + scale_or_raw
        quant_begin = source_base + quant_offset
        scales = source[scale_begin:scale_begin + scale_or_raw_bytes]
        quant = source[quant_begin:quant_begin + quant_bytes]
        if len(scales) != scale_or_raw_bytes or len(quant) != quant_bytes:
            raise ValueError(f"tensor {tensor}: truncated Q4 data")
        for group in range(groups):
            scale = struct.unpack_from("<e", scales, group * 2)[0]
            output.extend(struct.pack(">f", scale))
            q_at = group * (GROUP_SIZE // 2)
            output.extend(quant[q_at:q_at + GROUP_SIZE // 2])
        new_entries.append((relative_offset, groups, groups * GROUP_BYTES, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "NANOLM.CG4"
    index_path = output_dir / "NANOLM.IDX"
    prefix_path = output_dir / "NANOLM.PFX"
    model_path.write_bytes(output)
    model_digest = hashlib.sha256(output).digest()
    index = bytearray(struct.pack("<8sIIII32s", INDEX_MAGIC, DATA_BASE,
                                  TENSOR_COUNT, VOCAB_SIZE, GROUP_SIZE,
                                  model_digest))
    for entry in new_entries:
        index.extend(struct.pack("<IIII", *entry))
    index_path.write_bytes(index)

    prefix = bytearray(source_prefix.read_bytes())
    if len(prefix) < 64 or prefix[:8] != PREFIX_MAGIC:
        raise ValueError("invalid source prefix")
    prefix[8:40] = model_digest
    prefix_path.write_bytes(prefix)

    validate(source, source_base, entries, output, new_entries)
    for path in (model_path, index_path, prefix_path):
        print(f"{path}: {path.stat().st_size} bytes  "
              f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


def validate(source: bytes, source_base: int,
             old_entries: list[tuple[int, int, int, int]], cg4: bytes,
             new_entries: list[tuple[int, int, int, int]]) -> None:
    """Full byte-for-byte quant and exact-scale validation."""
    for tensor, (old, new) in enumerate(zip(old_entries, new_entries)):
        old_a, old_b, old_c, old_d = old
        new_a, groups, new_c, kind = new
        cg_begin = DATA_BASE + new_a
        if old_d == 0:
            if kind != 0 or cg4[cg_begin:cg_begin + new_c] != \
                    source[source_base + old_a:source_base + old_a + old_c]:
                raise ValueError(f"tensor {tensor}: f16 validation failed")
            continue
        if kind != 1 or groups != old_c // 2 or new_c != groups * GROUP_BYTES:
            raise ValueError(f"tensor {tensor}: CG4 metadata validation failed")
        for group in range(groups):
            cg_at = cg_begin + group * GROUP_BYTES
            old_scale = struct.unpack_from("<e", source,
                                           source_base + old_a + group * 2)[0]
            new_scale = struct.unpack_from(">f", cg4, cg_at)[0]
            if new_scale != old_scale:
                raise ValueError(f"tensor {tensor} group {group}: scale changed")
            old_q = source_base + old_b + group * (GROUP_SIZE // 2)
            if cg4[cg_at + 4:cg_at + GROUP_BYTES] != \
                    source[old_q:old_q + GROUP_SIZE // 2]:
                raise ValueError(f"tensor {tensor} group {group}: Q4 changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-q4", required=True, type=Path)
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--source-prefix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    repack(args.source_q4, args.source_index, args.source_prefix,
           args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
