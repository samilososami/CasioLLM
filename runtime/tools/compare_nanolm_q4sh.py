#!/usr/bin/env python3
"""Compare current calculator Q4-I16 with proposed row-planar Q4-SH/Q11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from casio_llm import ModelEntry, clean_reply, load_backend
from nanolm_q4sh import load_state_dict
from verify_int16_activations import prompt_ids


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = [
    "hi",
    "who are you?",
    "what is a dog?",
    "what is water?",
    "what is a table?",
    "what color is the sky?",
    "i have something to tell you",
    "i love you",
    "what is 2+2?",
    "what is the capital of spain?",
]


def activation_hook(levels: int):
    def quantize(_module, args):
        value = args[0]
        maximum = value.abs().amax(dim=-1, keepdim=True)
        scale = maximum.clamp_min(1e-20) / float(levels)
        reconstructed = torch.round(value / scale).clamp(-levels, levels) * scale
        return (reconstructed,) + args[1:]
    return quantize


def replies(model, tokenizer, prompts: list[str], levels: int,
            max_new_tokens: int, device: str) -> list[str]:
    model.to(device).eval()
    hooks = [
        module.register_forward_pre_hook(activation_hook(levels))
        for module in model.modules() if isinstance(module, torch.nn.Linear)
    ]
    answers = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = prompt_ids(tokenizer, prompt)
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )[0, len(ids):]
            answers.append(clean_reply(tokenizer.decode(output, skip_special_tokens=False)))
    for hook in hooks:
        hook.remove()
    return answers


def load_q4():
    entry = ModelEntry(
        id="5q4-20k", name="NanoLM Q4-20K", hf_id="", kind="chat",
        language="English", prompt_style="chatml", cg50_route="", note="",
    )
    return load_backend(entry)


def load_q4sh(path: Path):
    source = ROOT / "models" / "5p20kfreq" / "hf"
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=False)
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)
    state, _ = load_state_dict(path)
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Q4-SH state mismatch: missing={missing}, unexpected={unexpected}")
    model.tie_weights()
    return model.eval(), tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4sh", type=Path,
                        default=ROOT / "models/5q4sh-20kfreq/NANOLM.QSH")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "work/nanolm_q4sh_smoke.json")
    parser.add_argument("--max-new-tokens", type=int, default=22)
    parser.add_argument("--q4sh-levels", type=int, default=2047,
                        help="activation maximum; 2047 is the proposed Q11 path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("prompts", nargs="*", default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    q4, tokenizer = load_q4()
    baseline = replies(q4, tokenizer, args.prompts, 32767,
                       args.max_new_tokens, args.device)
    del q4
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    q4sh, tokenizer = load_q4sh(args.q4sh)
    proposed = replies(q4sh, tokenizer, args.prompts, args.q4sh_levels,
                       args.max_new_tokens, args.device)

    rows = []
    for prompt, current, optimized in zip(args.prompts, baseline, proposed):
        row = {
            "prompt": prompt,
            "q4_i16": current,
            "q4sh_q11": optimized,
            "exact": current == optimized,
        }
        rows.append(row)
        print(f"PROMPT: {prompt}\nQ4-I16:   {current}\nQ4SH-Q11: {optimized}\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "q4sh_levels": args.q4sh_levels,
        "exact_replies": sum(row["exact"] for row in rows),
        "samples": len(rows),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"Exact replies: {sum(row['exact'] for row in rows)}/{len(rows)}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
