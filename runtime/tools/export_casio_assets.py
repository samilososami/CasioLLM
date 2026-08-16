#!/usr/bin/env python3
"""Export compact, CG50-friendly metadata for NanoLM Q4 files.

The add-in deliberately does not parse JSON or the SentencePiece protobuf at
runtime. This script turns the existing, verified artifacts into two simple
binary files: an index for weight offsets and a vocabulary for prompt encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanolm_q4 import read_header


INDEX_MAGIC = b"NLMIDX01"
INDEX_MAGIC_V2 = b"NLMIDX02"
TOK_MAGIC = b"NLMTOK01"
TRIE_MAGIC = b"NLMTRE01"


def write_index(q4: Path, output: Path, version: int = 1) -> None:
    header, data_base = read_header(q4)
    by_name = {entry["name"]: entry for entry in header["tensors"]}
    ordered = [by_name["model.embed_tokens.weight"]]
    for layer in range(12):
        prefix = f"model.layers.{layer}."
        ordered.extend(by_name[prefix + suffix] for suffix in (
            "input_layernorm.weight", "self_attn.q_proj.weight",
            "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "post_attention_layernorm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ))
    ordered.append(by_name["model.norm.weight"])

    payload = bytearray()
    for entry in ordered:
        if entry["encoding"] == "q4_symmetric":
            payload += struct.pack(
                "<IIII", int(entry["scale_offset"]), int(entry["quant_offset"]),
                int(entry["scale_bytes"]), int(entry["quant_bytes"]),
            )
        else:
            payload += struct.pack("<IIII", int(entry["offset"]), 0, int(entry["bytes"]), 0)
    common = (
        data_base, len(ordered), header["model_config"]["vocab_size"],
        header["group_size"],
    )
    if version == 1:
        index_header = struct.pack("<8sIIII", INDEX_MAGIC, *common)
    elif version == 2:
        digest = hashlib.sha256(q4.read_bytes()).digest()
        index_header = struct.pack("<8sIIII32s", INDEX_MAGIC_V2, *common, digest)
    else:
        raise ValueError(f"Versión de índice no soportada: {version}")
    output.write_bytes(index_header + payload)


def write_tokenizer(model: Path, output: Path) -> None:
    processor = spm.SentencePieceProcessor(model_file=str(model))
    pieces = [processor.id_to_piece(i).encode("utf-8") for i in range(processor.get_piece_size())]
    # The 20K model has four added tokens, not part of SentencePiece.
    added = [b"<|im_start|>", b"<|im_end|>", b"<|endoftext|>", b"<image>"]
    if len(pieces) + len(added) != 20000:
        raise ValueError("El tokenizer no coincide con el vocabulario CG50 de 20K")
    offsets: list[int] = []
    pool = bytearray()
    for piece in pieces + added:
        offsets.append(len(pool))
        pool += piece
    entries = bytearray()
    for token_id, piece in enumerate(pieces + added):
        entries += struct.pack("<IHH", offsets[token_id], len(piece), token_id)
    output.write_bytes(struct.pack("<8sII", TOK_MAGIC, 20000, len(pool)) + entries + pool)


def write_input_trie(model: Path, output: Path) -> None:
    processor = spm.SentencePieceProcessor(model_file=str(model))
    # node = [first_child, next_sibling, token_id_or_ffff, byte, padding,
    #         SentencePiece BPE score]
    nodes: list[dict] = [{"children": {}, "token": 0xffff, "byte": 0}]
    for token_id in range(processor.get_piece_size()):
        piece = processor.id_to_piece(token_id).encode("utf-8")
        # Input from the calculator is ASCII, apart from SentencePiece's ▁.
        if piece.startswith(b"<") or any(byte >= 128 for byte in piece.replace("▁".encode(), b"")):
            continue
        node = 0
        for byte in piece:
            child = nodes[node]["children"].get(byte)
            if child is None:
                child = len(nodes)
                nodes[node]["children"][byte] = child
                nodes.append({"children": {}, "token": 0xffff, "byte": byte})
            node = child
        nodes[node]["token"] = token_id
        nodes[node]["score"] = int(processor.get_score(token_id))
    packed = bytearray()
    # The child maps are replaced by sibling chains. Index zero is the root.
    for node in nodes:
        children = sorted(node["children"].items())
        for index, (_, child) in enumerate(children):
            nodes[child]["next"] = children[index + 1][1] if index + 1 < len(children) else 0xffffffff
        node["first"] = children[0][1] if children else 0xffffffff
    for node in nodes:
        packed += struct.pack("<IIHBBi", node.get("first", 0xffffffff),
                              node.get("next", 0xffffffff), node["token"],
                              node["byte"], 0, node.get("score", -2147483648))
    output.write_bytes(struct.pack("<8sI", TRIE_MAGIC, len(nodes)) + packed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index-version", type=int, choices=(1, 2), default=1,
                        help="v2 binds caches to the SHA-256 of NANOLM.Q4")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_index(args.q4, args.output_dir / "NANOLM.IDX", args.index_version)
    write_tokenizer(args.tokenizer, args.output_dir / "NANOLM.TOK")
    write_input_trie(args.tokenizer, args.output_dir / "NANOLM.TRI")
    for name in ("NANOLM.IDX", "NANOLM.TOK", "NANOLM.TRI"):
        path = args.output_dir / name
        print(f"{path}: {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
