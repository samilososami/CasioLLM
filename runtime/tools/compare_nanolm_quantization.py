#!/usr/bin/env python3
"""Compara NanoLM original y Q4 sobre continuaciones idénticas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import casio_llm as app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="informe JSON generado por evaluate_models.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = json.loads(args.report.read_text(encoding="utf-8"))
    original_row = next(row for row in rows if row["id"] == "5")
    entries = {entry.id: entry for entry in app.catalog()}
    original, tokenizer = app.load_backend(entries["5"])
    quantized, _ = app.load_backend(entries["5q4"])
    original.to(args.device).eval()
    quantized.to(args.device).eval()

    token_count = 0
    original_nll = 0.0
    quantized_nll = 0.0
    original_correct = 0
    quantized_correct = 0
    top1_agreement = 0
    kl_sum = 0.0
    logit_mae_sum = 0.0

    with torch.inference_mode():
        for sample in original_row["responses"]:
            prompt = app.prompt_for(entries["5"], tokenizer, [], sample["prompt"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full = tokenizer(
                prompt + sample["reply"] + tokenizer.eos_token,
                return_tensors="pt", add_special_tokens=False,
            )["input_ids"].to(args.device)
            start = max(len(prompt_ids) - 1, 0)
            targets = full[:, 1:][:, start:]
            fp_logits = original(input_ids=full).logits[:, :-1][:, start:].float()
            q4_logits = quantized(input_ids=full).logits[:, :-1][:, start:].float()
            count = targets.numel()
            token_count += count
            original_nll += float(F.cross_entropy(
                fp_logits.reshape(-1, fp_logits.shape[-1]), targets.reshape(-1), reduction="sum"
            ))
            quantized_nll += float(F.cross_entropy(
                q4_logits.reshape(-1, q4_logits.shape[-1]), targets.reshape(-1), reduction="sum"
            ))
            fp_top = fp_logits.argmax(dim=-1)
            q4_top = q4_logits.argmax(dim=-1)
            original_correct += int((fp_top == targets).sum())
            quantized_correct += int((q4_top == targets).sum())
            top1_agreement += int((fp_top == q4_top).sum())
            fp_probs = torch.softmax(fp_logits, dim=-1)
            kl_sum += float(F.kl_div(
                torch.log_softmax(q4_logits, dim=-1), fp_probs,
                reduction="sum", log_target=False,
            ))
            logit_mae_sum += float((fp_logits - q4_logits).abs().sum())

    vocab_size = int(original.config.vocab_size)
    result = {
        "device": args.device,
        "samples": len(original_row["responses"]),
        "response_tokens": token_count,
        "original_nll": original_nll / token_count,
        "q4_nll": quantized_nll / token_count,
        "original_perplexity": math.exp(min(original_nll / token_count, 50)),
        "q4_perplexity": math.exp(min(quantized_nll / token_count, 50)),
        "nll_increase_percent": (quantized_nll / original_nll - 1.0) * 100.0,
        "original_next_token_accuracy": original_correct / token_count,
        "q4_next_token_accuracy": quantized_correct / token_count,
        "top1_agreement": top1_agreement / token_count,
        "mean_kl_fp_to_q4": kl_sum / token_count,
        "mean_absolute_logit_error": logit_mae_sum / (token_count * vocab_size),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
