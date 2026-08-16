#!/usr/bin/env python3
"""Conversor y lector Q3 de NanoLM con vocabulario completo para la fx-CG50."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


MAGIC = b"NLMQ3\x00\x01\x00"
ALIGNMENT = 64


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def pack_3bit(values: np.ndarray) -> bytes:
    """Empaqueta ocho enteros 0..7 en tres bytes, bit menos significativo primero."""
    if values.size % 8:
        raise ValueError("El número de valores Q3 debe ser múltiplo de ocho.")
    rows = values.reshape(-1, 8).astype(np.uint32)
    words = np.zeros(rows.shape[0], dtype=np.uint32)
    for index in range(8):
        words |= rows[:, index] << (3 * index)
    packed = np.empty((rows.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = words & 0xFF
    packed[:, 1] = (words >> 8) & 0xFF
    packed[:, 2] = (words >> 16) & 0xFF
    return packed.tobytes()


def unpack_3bit(packed: np.ndarray) -> np.ndarray:
    if packed.size % 3:
        raise ValueError("Bloque Q3 truncado.")
    rows = packed.reshape(-1, 3).astype(np.uint32)
    words = rows[:, 0] | (rows[:, 1] << 8) | (rows[:, 2] << 16)
    values = np.empty((rows.shape[0], 8), dtype=np.int8)
    for index in range(8):
        values[:, index] = ((words >> (3 * index)) & 7).astype(np.int8) - 3
    return values.reshape(-1)


def pack_q3(tensor: torch.Tensor, group_size: int) -> tuple[bytes, bytes, dict]:
    flat = tensor.detach().float().reshape(-1)
    original_count = flat.numel()
    padded_count = align_up(original_count, group_size)
    if padded_count != original_count:
        flat = torch.nn.functional.pad(flat, (0, padded_count - original_count))
    groups = flat.reshape(-1, group_size)
    scales = groups.abs().amax(dim=1).div(3.0)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.round(groups / scales[:, None]).clamp(-3, 3).to(torch.int8)
    encoded = (quantized + 3).to(torch.uint8).reshape(-1).numpy()
    reconstructed = quantized.float() * scales[:, None]
    error = reconstructed.reshape(-1)[:original_count] - tensor.detach().float().reshape(-1)
    stats = {
        "mse": float(error.square().mean()),
        "max_abs_error": float(error.abs().max()),
        "groups": int(groups.shape[0]),
        "padded_elements": int(padded_count),
    }
    return scales.to(torch.float16).numpy().tobytes(), pack_3bit(encoded), stats


def quantize(source: Path, destination: Path, config_path: Path, group_size: int) -> dict:
    if group_size <= 0 or group_size % 8:
        raise ValueError("El tamaño de grupo debe ser positivo y múltiplo de ocho.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[bytes] = []
    tensors: list[dict] = []
    data_cursor = 0
    chunks_size = 0
    weighted_mse = 0.0
    total_quantized = 0

    with safe_open(source, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            entry = {"name": name, "shape": list(tensor.shape), "elements": tensor.numel()}
            data_cursor = align_up(data_cursor)
            padding = data_cursor - chunks_size
            if padding:
                chunks.append(b"\x00" * padding)
                chunks_size += padding
            if tensor.ndim >= 2:
                scale_bytes, quant_bytes, stats = pack_q3(tensor, group_size)
                entry.update({
                    "encoding": "q3_symmetric",
                    "group_size": group_size,
                    "scale_offset": data_cursor,
                    "scale_bytes": len(scale_bytes),
                    "quant_offset": data_cursor + len(scale_bytes),
                    "quant_bytes": len(quant_bytes),
                    **stats,
                })
                chunks.extend((scale_bytes, quant_bytes))
                chunks_size += len(scale_bytes) + len(quant_bytes)
                data_cursor += len(scale_bytes) + len(quant_bytes)
                weighted_mse += stats["mse"] * tensor.numel()
                total_quantized += tensor.numel()
            else:
                raw = tensor.float().to(torch.float16).numpy().tobytes()
                entry.update({"encoding": "f16", "offset": data_cursor, "bytes": len(raw)})
                chunks.append(raw)
                chunks_size += len(raw)
                data_cursor += len(raw)
            tensors.append(entry)

    header = {
        "format": "nanolm-cg50-q3",
        "version": 1,
        "alignment": ALIGNMENT,
        "group_size": group_size,
        "source": source.name,
        "model_config": json.loads(config_path.read_text(encoding="utf-8")),
        "tensors": tensors,
        "summary": {
            "tensor_count": len(tensors),
            "parameter_count": sum(entry["elements"] for entry in tensors),
            "quantized_parameter_count": total_quantized,
            "weighted_mse": weighted_mse / max(total_quantized, 1),
            "data_bytes": data_cursor,
        },
    }
    header_bytes = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    data_base = align_up(len(MAGIC) + 4 + len(header_bytes))
    with destination.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<I", len(header_bytes)))
        output.write(header_bytes)
        output.write(b"\x00" * (data_base - output.tell()))
        for chunk in chunks:
            output.write(chunk)
    header["summary"]["file_bytes"] = destination.stat().st_size
    header["summary"]["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return header


def read_header(model_path: Path) -> tuple[dict, int]:
    with model_path.open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"Magic Q3 inválido en {model_path}")
        header_length = struct.unpack("<I", source.read(4))[0]
        header = json.loads(source.read(header_length))
    return header, align_up(len(MAGIC) + 4 + header_length)


def load_state_dict(model_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    header, data_base = read_header(model_path)
    blob = np.memmap(model_path, mode="r", dtype=np.uint8)
    state: dict[str, torch.Tensor] = {}
    for entry in header["tensors"]:
        shape = tuple(entry["shape"])
        count = int(entry["elements"])
        if entry["encoding"] == "f16":
            start = data_base + int(entry["offset"])
            values = np.frombuffer(blob, dtype=np.float16, count=count, offset=start).copy()
            tensor = torch.from_numpy(values).float().reshape(shape)
        else:
            scale_start = data_base + int(entry["scale_offset"])
            scales = np.frombuffer(
                blob, dtype=np.float16, count=int(entry["groups"]), offset=scale_start
            ).copy().astype(np.float32)
            quant_start = data_base + int(entry["quant_offset"])
            packed = np.frombuffer(
                blob, dtype=np.uint8, count=int(entry["quant_bytes"]), offset=quant_start
            ).copy()
            values = unpack_3bit(packed)
            group_size = int(entry["group_size"])
            dequantized = values.astype(np.float32).reshape(-1, group_size) * scales[:, None]
            tensor = torch.from_numpy(dequantized.reshape(-1)[:count].copy()).reshape(shape)
        state[entry["name"]] = tensor
    return state, header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("quantize", help="convierte safetensors a NLM Q3")
    convert.add_argument("source", type=Path)
    convert.add_argument("destination", type=Path)
    convert.add_argument("--config", type=Path, required=True)
    convert.add_argument("--group-size", type=int, default=64)
    inspect = subparsers.add_parser("inspect", help="muestra metadatos de un NLM Q3")
    inspect.add_argument("model", type=Path)
    args = parser.parse_args()
    if args.command == "quantize":
        header = quantize(args.source, args.destination, args.config, args.group_size)
    else:
        header, _ = read_header(args.model)
        header["summary"]["actual_file_bytes"] = args.model.stat().st_size
    print(json.dumps(header["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
