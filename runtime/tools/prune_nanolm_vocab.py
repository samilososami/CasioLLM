#!/usr/bin/env python3
"""Reduce el vocabulario de NanoLM conservando piezas y embeddings originales."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sentencepiece import sentencepiece_model_pb2
from transformers import LlamaTokenizer


CHAT_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<image>"]
ORIGINAL_CHAT_IDS = [32000, 32001, 32002, 32003]
DEFAULT_DATASET = "Mxode/Magpie-Pro-10K-GPT4o-mini"


def choose_frequency_vocab(source: Path, proto, base_keep: int, dataset_name: str) -> tuple[list[int], dict]:
    """Escoge piezas inglesas por frecuencia real, conservando byte fallback y metatokens."""
    from datasets import load_dataset

    tokenizer = LlamaTokenizer.from_pretrained(source, legacy=True)
    frequency: Counter[int] = Counter()
    raw = load_dataset(dataset_name, split="train")
    fixed_text = (
        "system You are a helpful assistant. user assistant "
        "<|im_start|> <|im_end|>"
    )
    frequency.update(tokenizer(fixed_text, add_special_tokens=False)["input_ids"])
    for record in raw:
        text = str(record["instruction"]) + "\n" + str(record["output"])
        frequency.update(tokenizer(text, add_special_tokens=False)["input_ids"])

    required = {0, 1, 2}
    byte_pieces = {f"<0x{value:02X}>" for value in range(256)}
    required.update(
        index for index, piece in enumerate(proto.pieces) if piece.piece in byte_pieces
    )
    ranked = sorted(
        range(len(proto.pieces)), key=lambda index: (-frequency[index], index)
    )
    selected = set(required)
    for index in ranked:
        if len(selected) >= base_keep:
            break
        selected.add(index)
    if len(selected) != base_keep:
        raise RuntimeError(f"Solo se seleccionaron {len(selected)} piezas de {base_keep}.")
    total_occurrences = sum(frequency.values())
    retained_occurrences = sum(frequency[index] for index in selected)
    return sorted(selected), {
        "dataset": dataset_name,
        "corpus_token_occurrences": total_occurrences,
        "retained_token_occurrences": retained_occurrences,
        "frequency_coverage": retained_occurrences / max(total_occurrences, 1),
        "required_piece_count": len(required),
    }


def prune(source: Path, destination: Path, target_vocab: int, strategy: str,
          dataset_name: str) -> dict:
    if target_vocab <= len(CHAT_TOKENS) + 512:
        raise ValueError("El vocabulario objetivo es demasiado pequeño.")
    base_keep = target_vocab - len(CHAT_TOKENS)
    destination.mkdir(parents=True, exist_ok=True)

    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString((source / "tokenizer.model").read_bytes())
    original_piece_count = len(proto.pieces)
    if base_keep > original_piece_count:
        raise ValueError(f"Solo hay {original_piece_count} piezas SentencePiece.")
    if strategy == "frequency":
        selected_ids, selection_stats = choose_frequency_vocab(
            source, proto, base_keep, dataset_name
        )
    else:
        selected_ids = list(range(base_keep))
        selection_stats = {"strategy": "first_ids"}
    kept_pieces = [proto.pieces[index].piece for index in selected_ids]
    byte_pieces = {f"<0x{value:02X}>" for value in range(256)}
    missing_bytes = sorted(byte_pieces - set(kept_pieces))
    if missing_bytes:
        raise ValueError(f"La poda eliminaría byte fallback: {missing_bytes[:5]}")
    selected_pieces = [copy.deepcopy(proto.pieces[index]) for index in selected_ids]
    del proto.pieces[:]
    proto.pieces.extend(selected_pieces)
    proto.trainer_spec.vocab_size = base_keep
    pruned_model = destination / "tokenizer.model"
    pruned_model.write_bytes(proto.SerializeToString())

    source_tokenizer = LlamaTokenizer.from_pretrained(source, legacy=True)
    tokenizer = LlamaTokenizer(vocab_file=str(pruned_model), legacy=True)
    tokenizer.add_special_tokens({"additional_special_tokens": CHAT_TOKENS})
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.unk_token = "<unk>"
    tokenizer.chat_template = source_tokenizer.chat_template
    tokenizer.model_max_length = 128
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(destination)
    if len(tokenizer) != target_vocab:
        raise RuntimeError(f"Tokenizer resultante={len(tokenizer)}, esperado={target_vocab}")

    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
        state = {name: handle.get_tensor(name) for name in handle.keys()}
    embedding = state["model.embed_tokens.weight"]
    embedding_ids = selected_ids + ORIGINAL_CHAT_IDS
    state["model.embed_tokens.weight"] = embedding[embedding_ids].contiguous()
    save_file(state, destination / "model.safetensors")

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config.update({
        "_name_or_path": "local/NanoLM-25M-Instruct-v1.1-8K",
        "vocab_size": target_vocab,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    })
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    generation_path = source / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        generation.update({
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        })
        (destination / "generation_config.json").write_text(
            json.dumps(generation, indent=2) + "\n", encoding="utf-8"
        )
    shutil.copy2(source / "README.md", destination / "README.source.md")
    mapping = {
        "target_vocab_size": target_vocab,
        "base_sentencepiece_pieces": base_keep,
        "original_sentencepiece_pieces": original_piece_count,
        "strategy": strategy,
        "selection": selection_stats,
        "new_to_old_embedding_ids": embedding_ids,
        "chat_tokens": {token: tokenizer.convert_tokens_to_ids(token) for token in CHAT_TOKENS},
    }
    (destination / "vocab_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "target_vocab_size": target_vocab,
        "model_parameters": sum(tensor.numel() for tensor in state.values()),
        "model_bytes": (destination / "model.safetensors").stat().st_size,
        "tokenizer_model_bytes": pruned_model.stat().st_size,
        "special_ids": mapping["chat_tokens"],
        "strategy": strategy,
        "selection": selection_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--strategy", choices=("first", "frequency"), default="first")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()
    print(json.dumps(prune(
        args.source, args.destination, args.vocab_size, args.strategy, args.dataset
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
