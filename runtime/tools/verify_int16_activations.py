#!/usr/bin/env python3
"""Compare NanoLM Q4 greedy replies with and without INT16 activations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from casio_llm import ModelEntry, clean_reply, load_backend


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_IDS = [
    1834, 460, 7038, 681, 4682, 19905, 19875, 264, 4356, 987, 12645,
    1210, 486, 3718, 19860, 12526, 19912, 856, 19912, 12149, 301, 19873,
    3035, 346, 12338, 19873,
]
IM_START = 19996
IM_END = 19997
NEWLINE = 13
SPACE = 19855
SYSTEM = 1495
USER = 2024
ASSISTANT = 11167


def prompt_ids(tokenizer, text: str) -> list[int]:
    return (
        [IM_START, SYSTEM, NEWLINE]
        + SYSTEM_IDS
        + [IM_END, SPACE, NEWLINE, IM_START, USER, NEWLINE]
        + tokenizer.encode(text, add_special_tokens=False)
        + [IM_END, SPACE, NEWLINE, IM_START, ASSISTANT, NEWLINE]
    )


def quantize_input(_module, args):
    value = args[0]
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = maximum.clamp_min(1e-20) / 32767.0
    quantized = torch.round(value / scale).clamp(-32767, 32767)
    reconstructed = quantized * scale
    return (reconstructed,) + args[1:]


def replies(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    answers = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = prompt_ids(tokenizer, prompt)
            inputs = torch.tensor([ids], dtype=torch.long)
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0, len(ids):]
            answers.append(clean_reply(tokenizer.decode(output, skip_special_tokens=False)))
    return answers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("prompts", nargs="*", default=[
        "hi", "what is a dog?", "what is water?", "who are you?",
    ])
    args = parser.parse_args()
    entry = ModelEntry(
        id="5q4-20k", name="NanoLM Q4-20K", hf_id="", kind="chat",
        language="English", prompt_style="chatml", cg50_route="", note="",
    )
    model, tokenizer = load_backend(entry)
    baseline = replies(model, tokenizer, args.prompts, args.max_new_tokens)
    hooks = [
        module.register_forward_pre_hook(quantize_input)
        for module in model.modules() if isinstance(module, torch.nn.Linear)
    ]
    integer = replies(model, tokenizer, args.prompts, args.max_new_tokens)
    for hook in hooks:
        hook.remove()

    identical = 0
    for prompt, original, optimized in zip(args.prompts, baseline, integer):
        same = original == optimized
        identical += int(same)
        print(f"PROMPT: {prompt}")
        print(f"Q4:     {original}")
        print(f"Q4-I16: {optimized}")
        print(f"EXACT:  {same}\n")
    print(f"Exact replies: {identical}/{len(args.prompts)}")
    return 0 if identical == len(args.prompts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
