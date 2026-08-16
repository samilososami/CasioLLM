#!/usr/bin/env python3
"""Batería reproducible de diez prompts cortos para Casio-LLM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import casio_llm as app


PROMPTS = [
    "Hello! How are you today?",
    "Who are you?",
    "I have something important to tell you.",
    "I like you.",
    "Can you give me a short tip for staying healthy?",
    "What is the capital of France?",
    "Tell me a very short story about a cat.",
    "I feel sad today. What could I do?",
    "What happens when ice melts?",
    "What is the opposite of hot?",
]

PROMPTS_50 = [
    "Hello! How are you today?",
    "Who are you?",
    "I have something important to tell you.",
    "I like you.",
    "Can we be friends?",
    "I feel sad today. What could I do?",
    "I am nervous about a test tomorrow.",
    "Tell me one nice thing about today.",
    "What is a dog?",
    "What is a table?",
    "What color is the sky on a clear day?",
    "What is the capital of France?",
    "What is water?",
    "What happens when ice melts?",
    "What is 2 plus 2?",
    "What is 12 times 3?",
    "What is 7 minus 5?",
    "Which is larger: an elephant or a mouse?",
    "What is the opposite of hot?",
    "Do cats usually bark?",
    "Why do people sleep?",
    "Why do plants need sunlight?",
    "Why is the ocean salty?",
    "Name the four seasons.",
    "Give me one short tip for staying healthy.",
    "Give me three things I can do when I am bored.",
    "Write one sentence about the moon.",
    "Tell me a very short story about a cat.",
    "Explain rain in simple words.",
    "Say hello in Spanish.",
    "Translate 'good morning' into Spanish.",
    "What colors make purple?",
    "What should I do if I am thirsty?",
    "Is it better to be kind or rude? Why?",
    "Tell me a joke.",
    "Write a tiny poem about a tree.",
    "If I say 'thank you', what could you say?",
    "What is your favorite food?",
    "Do you have feelings?",
    "What do you think about music?",
    "I lost my pencil. What should I do?",
    "My friend is angry with me. What could I say?",
    "What should I pack for a rainy day?",
    "What is the first letter of the alphabet?",
    "How many days are there in a week?",
    "Which planet do we live on?",
    "What shape has three sides?",
    "Finish this sentence: The sun rises in the...",
    "What is the difference between a cat and a dog?",
    "Please answer with one friendly sentence: I am happy today.",
]

PROMPTS_DEFINITIONS = [
    "What is a person?",
    "What is an animal?",
    "What is a cat?",
    "What is a bird?",
    "What is a fish?",
    "What is a house?",
    "What is a chair?",
    "What is a book?",
    "What is a computer?",
    "What is a CPU?",
    "What is a GPU?",
    "What is the Sun?",
    "What is the Moon?",
    "What is Earth?",
    "What is air?",
    "What is fire?",
    "What is ice?",
    "What is food?",
    "What is a school?",
    "What is a car?",
    "What is electricity?",
    "What is the internet?",
    "What is language?",
    "What is a friend?",
    "What is time?",
]

PROMPTS_75 = PROMPTS_50 + PROMPTS_DEFINITIONS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--suite", choices=("10", "50", "definitions", "75"), default="10",
                        help="conjunto de prompts reproducible")
    parser.add_argument("--model", action="append", dest="model_ids",
                        help="ID del catálogo; repetir para evaluar varios")
    parser.add_argument("--output", type=Path, help="archivo JSON donde guardar el informe completo")
    parser.add_argument("--quiet", action="store_true", help="muestra solo progreso, no cada respuesta")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens debe ser positivo")
    suites = {
        "10": PROMPTS,
        "50": PROMPTS_50,
        "definitions": PROMPTS_DEFINITIONS,
        "75": PROMPTS_75,
    }
    prompts = suites[args.suite]

    entries = app.catalog()
    if args.model_ids:
        requested = set(args.model_ids)
        known = {entry.id for entry in entries}
        unknown = requested - known
        if unknown:
            parser.error("Modelo desconocido: " + ", ".join(sorted(unknown)))
        entries = [entry for entry in entries if entry.id in requested]

    report: list[dict] = []
    for entry in entries:
        print(f"\n== {entry.name} ==", flush=True)
        torch.manual_seed(20260815)
        started = time.monotonic()
        row = {"id": entry.id, "name": entry.name, "responses": [], "errors": []}
        try:
            model, tokenizer = app.load_backend(entry)
        except Exception as exc:
            row["errors"].append(f"carga: {type(exc).__name__}: {exc}")
            report.append(row)
            print(row["errors"][-1], flush=True)
            continue
        for index, prompt in enumerate(prompts, start=1):
            try:
                reply = app.generate(entry, model, tokenizer, [], prompt,
                                     args.max_new_tokens, args.temperature, args.top_p)
                row["responses"].append({"prompt": prompt, "reply": reply})
                if args.quiet:
                    if index % 5 == 0 or index == len(prompts):
                        print(f"  {index}/{len(prompts)}", flush=True)
                else:
                    print(f"{index:02d}. {reply}", flush=True)
            except Exception as exc:
                message = f"{index:02d}. {type(exc).__name__}: {exc}"
                row["errors"].append(message)
                print(message, flush=True)
        row["seconds"] = round(time.monotonic() - started, 2)
        report.append(row)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"\nInforme completo: {args.output}")
    else:
        print("\n== JSON_REPORT ==")
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
