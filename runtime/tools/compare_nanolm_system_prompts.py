#!/usr/bin/env python3
"""Compare NanoLM Q4 behavior under candidate stateless system prefixes."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from casio_llm import ModelEntry, clean_reply, load_backend


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "hi", "who are you?", "what is a dog?", "what is water?",
    "what is a table?", "what color is the sky?",
    "i have something to tell you", "i love you", "what is 2+2?",
    "what is the capital of spain?",
]
SYSTEMS = {
    "none": None,
    "helpful": "You are a helpful assistant.",
    "name_only": "You are CasioLLM.",
    "name_helpful": "You are CasioLLM, a helpful assistant.",
    "helpful_named": "You are a helpful assistant called CasioLLM.",
    "short_author": "You are CasioLLM, made by Sami.",
    "identity": (
        "You are CasioLLM, a calculator AI made by Sami Gonzalez Kamel. "
        "Reply briefly."
    ),
}


def ids(tokenizer, user: str, system: str | None) -> list[int]:
    text = ""
    if system is not None:
        text += f"<|im_start|>system\n{system}<|im_end|> \n"
    text += f"<|im_start|>user\n{user}<|im_end|> \n<|im_start|>assistant\n"
    return tokenizer.encode(text, add_special_tokens=False)


def quantize_i16(_module, args):
    value = args[0]
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = maximum.clamp_min(1e-20) / 32767.0
    return (torch.round(value / scale).clamp(-32767, 32767) * scale,) + args[1:]


def main() -> int:
    entry = ModelEntry(
        id="5q4-20k", name="NanoLM Q4-20K", hf_id="", kind="chat",
        language="English", prompt_style="chatml", cg50_route="", note="",
    )
    model, tokenizer = load_backend(entry)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    hooks = [
        module.register_forward_pre_hook(quantize_i16)
        for module in model.modules() if isinstance(module, torch.nn.Linear)
    ]
    rows = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            row = {"prompt": prompt, "replies": {}}
            print(f"PROMPT: {prompt}")
            for label, system in SYSTEMS.items():
                token_ids = ids(tokenizer, prompt, system)
                inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
                output = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=22, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )[0, len(token_ids):]
                reply = clean_reply(tokenizer.decode(output, skip_special_tokens=False))
                row["replies"][label] = reply
                row.setdefault("prompt_tokens", {})[label] = len(token_ids)
                print(f"{label:8}: {reply}")
            print()
            rows.append(row)
    for hook in hooks:
        hook.remove()
    report = ROOT / "work/nanolm_system_prompt_comparison.json"
    report.write_text(json.dumps({"device": device, "rows": rows}, indent=2),
                      encoding="utf-8")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
