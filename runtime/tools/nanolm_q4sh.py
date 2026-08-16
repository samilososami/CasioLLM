#!/usr/bin/env python3
"""Row-planar NanoLM Q4 format designed for the fx-CG50 SH4AL-DSP path."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


MAGIC = b"NLMQSH01"
ALIGNMENT = 64
GROUP_SIZE = 64


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def execution_order() -> list[str]:
    names = ["model.embed_tokens.weight"]
    for layer in range(12):
        prefix = f"model.layers.{layer}."
        names.extend(prefix + suffix for suffix in (
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ))
    names.append("model.norm.weight")
    return names


def pack_matrix(tensor: torch.Tensor) -> tuple[bytes, dict]:
    values = tensor.detach().float().cpu()
    rows, columns = values.shape
    groups = (columns + GROUP_SIZE - 1) // GROUP_SIZE
    padded_columns = groups * GROUP_SIZE
    padded = F.pad(values, (0, padded_columns - columns)).reshape(
        rows, groups, GROUP_SIZE
    )

    ideal_scales = padded.abs().amax(dim=2) / 7.0
    largest = ideal_scales.amax(dim=1)
    base = largest / 127.0
    base = torch.where(base > 0, base, torch.ones_like(base))
    multipliers = torch.round(ideal_scales / base[:, None]).clamp(0, 127)
    effective = base[:, None] * multipliers
    safe_scale = torch.where(effective > 0, effective, torch.ones_like(effective))
    quantized = torch.round(padded / safe_scale[:, :, None]).clamp(-7, 7)
    quantized = torch.where(effective[:, :, None] > 0, quantized, 0).to(torch.int8)
    encoded = (quantized + 8).to(torch.uint8).reshape(rows, padded_columns)
    packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)

    # Every row starts on a 2-byte boundary. Nibbles also start on a 2-byte
    # boundary; requiring 16-byte alignment here would waste about 0.2 MB.
    nibble_offset = (2 + groups + 1) & ~1
    row_stride = nibble_offset + padded_columns // 2
    if row_stride & 1:
        row_stride += 1
    blob = np.zeros((rows, row_stride), dtype=np.uint8)
    bases = base.to(torch.float16).numpy().view(np.uint8).reshape(rows, 2)
    blob[:, :2] = bases
    blob[:, 2:2 + groups] = multipliers.to(torch.uint8).numpy()
    blob[:, nibble_offset:nibble_offset + padded_columns // 2] = packed.numpy()

    reconstructed = (
        quantized.float() * effective[:, :, None]
    ).reshape(rows, padded_columns)[:, :columns]
    error = reconstructed - values
    stats = {
        "rows": rows,
        "columns": columns,
        "groups_per_row": groups,
        "padded_columns": padded_columns,
        "nibble_offset": nibble_offset,
        "row_stride": row_stride,
        "mse": float(error.square().mean()),
        "max_abs_error": float(error.abs().max()),
    }
    return blob.tobytes(), stats


def convert(source: Path, destination: Path, config: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[bytes] = []
    entries: list[dict] = []
    cursor = 0
    total_error = 0.0
    total_matrix_values = 0

    with safe_open(source, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        expected = execution_order()
        missing = [name for name in expected if name not in available]
        if missing:
            raise ValueError(f"missing tensors: {missing}")
        for name in expected:
            tensor = handle.get_tensor(name)
            cursor = align_up(cursor)
            chunks.append(b"\0" * (cursor - sum(map(len, chunks))))
            entry = {"name": name, "shape": list(tensor.shape), "offset": cursor}
            if tensor.ndim == 2:
                data, stats = pack_matrix(tensor)
                entry.update({"encoding": "q4sh_row", "bytes": len(data), **stats})
                total_error += stats["mse"] * tensor.numel()
                total_matrix_values += tensor.numel()
            else:
                data = tensor.float().to(torch.float16).numpy().tobytes()
                entry.update({"encoding": "f16", "bytes": len(data)})
            chunks.append(data)
            cursor += len(data)
            entries.append(entry)

    header = {
        "format": "nanolm-cg50-q4sh-row",
        "version": 1,
        "alignment": ALIGNMENT,
        "group_size": GROUP_SIZE,
        "source": source.name,
        "model_config": json.loads(config.read_text(encoding="utf-8")),
        "tensors": entries,
        "summary": {
            "tensor_count": len(entries),
            "matrix_values": total_matrix_values,
            "weighted_mse": total_error / max(total_matrix_values, 1),
            "data_bytes": cursor,
        },
    }
    header_bytes = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode()
    data_base = align_up(len(MAGIC) + 4 + len(header_bytes))
    with destination.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<I", len(header_bytes)))
        output.write(header_bytes)
        output.write(b"\0" * (data_base - output.tell()))
        for chunk in chunks:
            output.write(chunk)
    header["summary"].update({
        "data_base": data_base,
        "file_bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    })
    return header


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as source:
        if source.read(8) != MAGIC:
            raise ValueError(f"invalid Q4-SH magic in {path}")
        length = struct.unpack("<I", source.read(4))[0]
        header = json.loads(source.read(length))
    return header, align_up(12 + length)


def load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    header, data_base = read_header(path)
    blob = np.memmap(path, mode="r", dtype=np.uint8)
    state: dict[str, torch.Tensor] = {}
    for entry in header["tensors"]:
        shape = tuple(entry["shape"])
        offset = data_base + int(entry["offset"])
        if entry["encoding"] == "f16":
            count = int(np.prod(shape))
            raw = np.frombuffer(blob, dtype=np.float16, count=count, offset=offset).copy()
            state[entry["name"]] = torch.from_numpy(raw).float().reshape(shape)
            continue
        rows = int(entry["rows"])
        columns = int(entry["columns"])
        groups = int(entry["groups_per_row"])
        padded = int(entry["padded_columns"])
        stride = int(entry["row_stride"])
        nibble_offset = int(entry["nibble_offset"])
        raw = np.frombuffer(blob, dtype=np.uint8, count=rows * stride, offset=offset)
        raw = raw.copy().reshape(rows, stride)
        bases = np.ascontiguousarray(raw[:, :2]).view(np.float16).reshape(rows).astype(np.float32)
        multipliers = raw[:, 2:2 + groups].astype(np.float32)
        packed = raw[:, nibble_offset:nibble_offset + padded // 2]
        quantized = np.empty((rows, padded), dtype=np.int8)
        quantized[:, 0::2] = (packed & 15).astype(np.int8) - 8
        quantized[:, 1::2] = (packed >> 4).astype(np.int8) - 8
        scales = bases[:, None] * multipliers
        values = quantized.reshape(rows, groups, GROUP_SIZE).astype(np.float32)
        values *= scales[:, :, None]
        state[entry["name"]] = torch.from_numpy(
            values.reshape(rows, padded)[:, :columns].copy()
        ).reshape(shape)
    return state, header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("convert")
    build.add_argument("source", type=Path)
    build.add_argument("destination", type=Path)
    build.add_argument("--config", required=True, type=Path)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("model", type=Path)
    args = parser.parse_args()
    if args.command == "convert":
        header = convert(args.source, args.destination, args.config)
    else:
        header, data_base = read_header(args.model)
        header["summary"].update({
            "data_base": data_base,
            "file_bytes": args.model.stat().st_size,
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        })
    print(json.dumps(header["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
