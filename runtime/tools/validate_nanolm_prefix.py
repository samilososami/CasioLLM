#!/usr/bin/env python3
"""Validate that NANOLM.PFX is bound to the intended NanoLM model/index."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    args = parser.parse_args()

    digest = hashlib.sha256(args.q4.read_bytes()).digest()
    index = args.index.read_bytes()
    prefix = args.prefix.read_bytes()
    if len(index) < 56 or index[:8] not in (b"NLMIDX02", b"NLMIDX03"):
        raise SystemExit("NANOLM.IDX is not a digest-bound v2/v3 index")
    if index[24:56] != digest:
        raise SystemExit("NANOLM.IDX SHA-256 does not match the model")
    if len(prefix) < 64 or prefix[:8] != b"NLMPFX01":
        raise SystemExit("invalid NANOLM.PFX header")
    if prefix[8:40] != digest:
        raise SystemExit("NANOLM.PFX SHA-256 does not match the model")
    prefix_tokens, layers, kv_dim, element_bytes, payload_bytes, token_hash = (
        struct.unpack_from("<IIIIII", prefix, 40)
    )
    expected = layers * prefix_tokens * kv_dim * element_bytes * 2
    if payload_bytes != expected or len(prefix) != 64 + payload_bytes:
        raise SystemExit("NANOLM.PFX payload size is inconsistent")
    print(json.dumps({
        "model_sha256": digest.hex(),
        "index_version": index[:8].decode("ascii"),
        "prefix_tokens": prefix_tokens,
        "layers": layers,
        "kv_dim": kv_dim,
        "element_bytes": element_bytes,
        "payload_bytes": payload_bytes,
        "file_bytes": len(prefix),
        "prefix_token_hash_fnv1a": f"{token_hash:08x}",
        "valid": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
