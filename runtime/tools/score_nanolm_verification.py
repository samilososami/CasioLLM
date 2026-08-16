#!/usr/bin/env python3
"""Resume la batería NanoLM con comprobaciones objetivas y de relevancia básica."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


OBJECTIVE = {
    "What is the capital of France?": [["paris"]],
    "What is 2 plus 2?": [[r"\b4\b"]],
    "What is 12 times 3?": [[r"\b36\b"]],
    "What is 7 minus 5?": [[r"\b2\b"]],
    "Which is larger: an elephant or a mouse?": [["elephant"]],
    "What is the opposite of hot?": [["cold"]],
    "Do cats usually bark?": [["do not", "don't", "not typically", "not usually", "do not necessarily"]],
    "Name the four seasons.": [["spring"], ["summer"], ["autumn", "fall"], ["winter"]],
    "Say hello in Spanish.": [["hola"]],
    "Translate 'good morning' into Spanish.": [["buenos"], ["días", "dias"]],
    "What colors make purple?": [["red"], ["blue"]],
    "What should I do if I am thirsty?": [["water", "drink", "hydrat"]],
    "What is the first letter of the alphabet?": [[r"\ba\b"]],
    "How many days are there in a week?": [[r"\b7\b", "seven"]],
    "Which planet do we live on?": [["earth"]],
    "What shape has three sides?": [["triangle"]],
    "Finish this sentence: The sun rises in the...": [["east"]],
    "What happens when ice melts?": [["water", "liquid"]],
    "What color is the sky on a clear day?": [["blue"]],
}

DEFINITIONS = {
    "What is a person?": ["human", "individual", "person"],
    "What is an animal?": ["living", "organism", "creature"],
    "What is a cat?": ["animal", "feline", "pet"],
    "What is a bird?": ["animal", "wing", "feather", "fly"],
    "What is a fish?": ["animal", "aquatic", "water", "ocean"],
    "What is a house?": ["building", "home", "living space"],
    "What is a chair?": ["furniture", "seat", "sit"],
    "What is a book?": ["written", "pages", "text"],
    "What is a computer?": ["machine", "electronic", "data", "instructions"],
    "What is a CPU?": ["processor", "central processing", "instructions"],
    "What is a GPU?": ["graphics", "processor", "processing"],
    "What is the Sun?": ["star", "gas", "solar"],
    "What is the Moon?": ["satellite", "orbit", "rock", "celestial"],
    "What is Earth?": ["planet", "world"],
    "What is air?": ["gas", "atmosphere", "breathe"],
    "What is fire?": ["combustion", "burn", "heat", "flame"],
    "What is ice?": ["frozen", "water", "solid"],
    "What is food?": ["eat", "nutrition", "nourish"],
    "What is a school?": ["education", "learn", "student"],
    "What is a car?": ["vehicle", "transport"],
    "What is electricity?": ["energy", "electron", "electric"],
    "What is the internet?": ["network", "computer", "online"],
    "What is language?": ["communication", "words", "speech"],
    "What is a friend?": ["person", "support", "relationship"],
    "What is time?": ["duration", "events", "measure"],
}


def contains_any(text: str, alternatives: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in alternatives)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for report in args.reports:
        rows.extend(json.loads(report.read_text(encoding="utf-8")))
    original = next(row for row in rows if row["id"] == "5")
    original_replies = {item["prompt"]: item["reply"] for item in original["responses"]}
    result = []
    for row in rows:
        replies = {item["prompt"]: item["reply"] for item in row["responses"]}
        objective_passes = 0
        objective_detail = {}
        for prompt, required_groups in OBJECTIVE.items():
            passed = all(contains_any(replies.get(prompt, ""), group) for group in required_groups)
            objective_detail[prompt] = passed
            objective_passes += int(passed)
        definition_passes = 0
        definition_detail = {}
        for prompt, terms in DEFINITIONS.items():
            passed = contains_any(replies.get(prompt, ""), terms)
            definition_detail[prompt] = passed
            definition_passes += int(passed)
        similarities = [
            SequenceMatcher(None, original_replies[prompt], reply).ratio()
            for prompt, reply in replies.items() if prompt in original_replies
        ]
        control_characters = sum(
            1 for reply in replies.values() for char in reply
            if ord(char) < 32 and char not in "\n\t\r"
        )
        result.append({
            "id": row["id"],
            "name": row["name"],
            "responses": len(row["responses"]),
            "errors": len(row["errors"]),
            "objective_passes": objective_passes,
            "objective_total": len(OBJECTIVE),
            "definition_relevance_passes": definition_passes,
            "definition_relevance_total": len(DEFINITIONS),
            "mean_text_similarity_to_original": sum(similarities) / max(len(similarities), 1),
            "exact_responses_vs_original": sum(
                reply == original_replies.get(prompt) for prompt, reply in replies.items()
            ),
            "control_characters": control_characters,
            "objective_detail": objective_detail,
            "definition_detail": definition_detail,
        })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
