#!/usr/bin/env python3
"""Reajusta NanoLM tras podar el vocabulario, conservando respuestas breves."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "Mxode/Magpie-Pro-10K-GPT4o-mini"
HEADER = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
FOOTER = "<|im_end|>\n<|im_start|>assistant\n"


class ConversationTokens(Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.tensor(value, dtype=torch.long) for key, value in self.rows[index].items()}


def encode_rows(tokenizer, records, max_length: int, max_prompt_tokens: int) -> list[dict[str, list[int]]]:
    header_ids = tokenizer(HEADER, add_special_tokens=False)["input_ids"]
    footer_ids = tokenizer(FOOTER, add_special_tokens=False)["input_ids"]
    fixed_prompt = len(header_ids) + len(footer_ids)
    user_budget = max(1, max_prompt_tokens - fixed_prompt)
    encoded: list[dict[str, list[int]]] = []
    for record in records:
        user_ids = tokenizer(
            str(record["instruction"]), add_special_tokens=False,
            truncation=True, max_length=user_budget,
        )["input_ids"]
        prompt_ids = header_ids + user_ids + footer_ids
        answer_budget = max_length - len(prompt_ids)
        if answer_budget < 2:
            continue
        answer_ids = tokenizer(
            str(record["output"]), add_special_tokens=False, truncation=True,
            max_length=answer_budget - 1,
        )["input_ids"] + [tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        padding = max_length - len(input_ids)
        encoded.append({
            "input_ids": input_ids + [tokenizer.pad_token_id] * padding,
            "attention_mask": [1] * len(input_ids) + [0] * padding,
            "labels": labels + [-100] * padding,
        })
    return encoded


def evaluate(model, loader, device: torch.device) -> tuple[float, int]:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(**batch)
            count = int((batch["labels"] != -100).sum())
            loss_sum += float(output.loss) * count
            token_count += count
    return loss_sum / max(token_count, 1), token_count


def save_model(model, tokenizer, destination: Path, source: Path, report: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    for filename in ("README.source.md", "vocab_mapping.json"):
        candidate = source / filename
        if candidate.is_file():
            shutil.copy2(candidate, destination / filename)
    (destination / "finetune_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Este reajuste requiere el entorno CUDA train-venv.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    # La conversión automática de un SentencePiece podado al backend "fast"
    # degrada todas las piezas a byte fallback. El runtime final también usa
    # SentencePiece, así que se fuerza aquí el tokenizer lento y fiel.
    tokenizer = AutoTokenizer.from_pretrained(
        args.source, local_files_only=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.source, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.config.use_cache = False

    print(f"Descargando/preparando {DATASET_NAME}…", flush=True)
    raw = load_dataset(DATASET_NAME, split="train", cache_dir=ROOT / "work" / "hf-cache")
    raw = raw.shuffle(seed=args.seed).select(range(min(args.max_examples, len(raw))))
    records = list(raw)
    encoded = encode_rows(tokenizer, records, args.max_length, args.max_prompt_tokens)
    if len(encoded) <= args.validation_size:
        raise RuntimeError("No hay suficientes ejemplos tras tokenizar.")
    validation_rows = encoded[: args.validation_size]
    training_rows = encoded[args.validation_size :]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ConversationTokens(training_rows), batch_size=args.batch_size, shuffle=True,
        generator=generator, pin_memory=True,
    )
    validation_loader = DataLoader(
        ConversationTokens(validation_rows), batch_size=args.batch_size, shuffle=False,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    update_steps = math.ceil(len(train_loader) / args.gradient_accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(update_steps * 0.03)),
        num_training_steps=update_steps,
    )
    history: list[dict] = []
    best_loss = float("inf")
    started = time.monotonic()
    print(
        f"GPU={torch.cuda.get_device_name(0)} train={len(training_rows)} val={len(validation_rows)} "
        f"updates={update_steps}", flush=True,
    )
    baseline_loss, baseline_tokens = evaluate(model, validation_loader, device)
    print(
        f"baseline val_nll={baseline_loss:.4f} "
        f"ppl={math.exp(min(baseline_loss, 50)):.2f}", flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_tokens = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation
            loss.backward()
            count = int((batch["labels"] != -100).sum())
            running_loss += float(output.loss.detach()) * count
            running_tokens += count
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch}/{args.epochs} batch {step}/{len(train_loader)} "
                    f"train_nll={running_loss / max(running_tokens, 1):.4f}", flush=True,
                )
        validation_loss, validation_tokens = evaluate(model, validation_loader, device)
        epoch_result = {
            "epoch": epoch,
            "train_nll": running_loss / max(running_tokens, 1),
            "validation_nll": validation_loss,
            "validation_perplexity": math.exp(min(validation_loss, 50)),
            "validation_tokens": validation_tokens,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            report = {
                "dataset": DATASET_NAME,
                "source": str(args.source),
                "best_epoch": epoch,
                "best_validation_nll": best_loss,
                "baseline_validation_nll": baseline_loss,
                "baseline_validation_perplexity": math.exp(min(baseline_loss, 50)),
                "baseline_validation_tokens": baseline_tokens,
                "history": history,
                "settings": vars(args) | {"source": str(args.source), "destination": str(args.destination)},
                "elapsed_seconds": time.monotonic() - started,
            }
            save_model(model, tokenizer, args.destination, args.source, report)
    print(f"Mejor modelo guardado en {args.destination}; val_nll={best_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
