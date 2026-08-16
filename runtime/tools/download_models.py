#!/usr/bin/env python3
"""Descarga los modelos del catálogo en directorios autosuficientes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CATALOG = MODELS_DIR / "catalog.json"


def load_catalog() -> list[dict]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))["models"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga modelos de Casio-LLM")
    parser.add_argument("--model", action="append", dest="model_ids",
                        help="ID del catálogo; repetir para descargar varios")
    args = parser.parse_args()

    entries = load_catalog()
    wanted = set(args.model_ids or [entry["id"] for entry in entries])
    known = {entry["id"] for entry in entries}
    unknown = wanted - known
    if unknown:
        parser.error("Modelos desconocidos: " + ", ".join(sorted(unknown)))

    for entry in entries:
        if entry["id"] not in wanted:
            continue
        target = MODELS_DIR / entry["id"] / "hf"
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[+] {entry['name']} <- {entry['hf_id']}")
        snapshot_download(
            repo_id=entry["hf_id"],
            local_dir=target,
            ignore_patterns=["*.onnx", "*.h5", "*.msgpack", "*.tflite"],
        )
        (target.parent / ".ready").write_text(entry["hf_id"] + "\n", encoding="utf-8")
        print(f"    guardado en {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
