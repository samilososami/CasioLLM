#!/usr/bin/env python3
"""Chat local interactivo para candidatos de LLM de la fx-CG50."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from nanolm_q4 import load_state_dict as load_nanolm_q4_state
from rodan_runtime import load_rodan


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CATALOG = MODELS_DIR / "catalog.json"
BRANDON_SOURCE = ROOT / "vendor" / "brandon-tiny" / "src"


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    hf_id: str
    kind: str
    language: str
    prompt_style: str
    cg50_route: str
    note: str

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.id / "hf"

    @property
    def installed(self) -> bool:
        external_quantized = {
            "5q4": MODELS_DIR / "5q4" / "NANOLM.Q4",
            "5q4-20k": MODELS_DIR / "5q4-20kfreq" / "NANOLM.Q4",
        }
        if self.id in external_quantized:
            return external_quantized[self.id].is_file()
        return (MODELS_DIR / self.id / ".ready").is_file() and self.path.is_dir()


def catalog() -> list[ModelEntry]:
    raw = json.loads(CATALOG.read_text(encoding="utf-8"))["models"]
    return [ModelEntry(**entry) for entry in raw]


def print_models(entries: list[ModelEntry]) -> None:
    print("\nModelos del workspace:\n")
    for index, entry in enumerate(entries, start=1):
        state = "instalado" if entry.installed else "no descargado"
        print(f"  {index}. {entry.name}  [{state}]")
        print(f"     id: {entry.id} | tipo: {entry.kind} | idioma: {entry.language}")
        print(f"     CG50: {entry.cg50_route}")
        print(f"     {entry.note}")
    print()


def choose(entries: list[ModelEntry], requested: str | None) -> ModelEntry:
    installed = [entry for entry in entries if entry.installed]
    if requested:
        by_id = {entry.id: entry for entry in entries}
        entry = by_id.get(requested)
        if entry is None:
            raise SystemExit(f"Modelo desconocido: {requested}")
        if not entry.installed:
            raise SystemExit(f"{entry.name} aún no está descargado.")
        return entry
    if not installed:
        raise SystemExit("No hay modelos descargados. Ejecuta el instalador del proyecto.")
    print_models(entries)
    while True:
        answer = input("Escoge modelo por número o id: ").strip()
        for index, entry in enumerate(entries, start=1):
            if answer in {str(index), entry.id} and entry.installed:
                return entry
        print("Selección no válida o modelo sin descargar.")


def prompt_for(entry: ModelEntry, tokenizer, history: list[tuple[str, str]], user_text: str) -> str:
    """Construye el prompt que corresponde al entrenamiento de cada candidato."""
    if entry.id == "4":
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{user_text}\n\n### Response:\n"
        )
    if entry.id == "6":
        turns = []
        for previous_user, previous_bot in history[-4:]:
            turns.append(f"<|im_start|>user\n{previous_user}<|im_end|>")
            turns.append(f"<|im_start|>assistant\n{previous_bot}<|im_end|>")
        turns.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        turns.append("<|im_start|>assistant\n")
        return "\n".join(turns)
    if entry.prompt_style == "auto" and getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        for previous_user, previous_bot in history[-4:]:
            messages.extend((
                {"role": "user", "content": previous_user},
                {"role": "assistant", "content": previous_bot},
            ))
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    turns: list[str] = []
    if entry.prompt_style == "chatml":
        for previous_user, previous_bot in history[-4:]:
            turns.append(
                "<|im_start|>user\n" + previous_user
                + "<|im_end|>\n<|im_start|>assistant\n" + previous_bot + "<|im_end|>"
            )
        turns.append("<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n")
        return "\n".join(turns)
    for previous_user, previous_bot in history[-4:]:
        turns.append(f"User: {previous_user}\nBot: {previous_bot}<|endoftext|>")
    turns.append(f"User: {user_text}\nBot:")
    return "\n".join(turns)


def clean_reply(text: str) -> str:
    for marker in ("<|endoftext|>", "<|im_end|>", "\nUser:", "\nBot:"):
        text = text.split(marker, 1)[0]
    # Algunos byte-fallback tokens pueden decodificar a controles ASCII que
    # no son imprimibles ni útiles en terminal o en la pantalla de la CG50.
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    return text.strip()


def generate(entry: ModelEntry, model, tokenizer, history: list[tuple[str, str]], user_text: str,
             max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = prompt_for(entry, tokenizer, history, user_text)
    if entry.id == "rodan_chat":
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        max_input = max(16, model.config.max_position_embeddings - max_new_tokens)
        token_ids = token_ids[-max_input:]
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        stop_ids = {0, 8193}
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_ids=stop_ids)
        return clean_reply(tokenizer.decode(output[0, len(token_ids):], skip_special_tokens=False))
    if entry.id == "brandon_tiny_10m_instruct":
        token_ids = tokenizer.encode(prompt)
        max_input = max(16, model.config.max_seq_len - max_new_tokens)
        token_ids = token_ids[-max_input:]
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=40,
                stop_tokens=tokenizer.get_stop_tokens(),
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        return clean_reply(tokenizer.decode(output[0, len(token_ids):].tolist(), skip_special=True))
    if entry.id == "6":
        token_ids = tokenizer.encode(prompt).ids
        max_input = max(16, model.config.max_seq_len - max_new_tokens)
        token_ids = token_ids[-max_input:]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.inference_mode():
            output, _ = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=50,
            )
        return clean_reply(tokenizer.decode(output[0, len(token_ids):].tolist()))
    if entry.id == "ivme_conversate_s_v2_instruct":
        max_input = max(16, int(model.config.max_seq_len) - max_new_tokens)
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_input:]
        generated = list(token_ids)
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated[-max_input:]], dtype=torch.long)
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits[0, -1].float()
            logits = logits.clone()
            if temperature > 0:
                logits /= temperature
                sorted_logits, sorted_ids = torch.sort(logits, descending=True)
                cutoff = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                cutoff[1:] = cutoff[:-1].clone()
                cutoff[0] = False
                logits[sorted_ids[cutoff]] = -torch.inf
                next_token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
            else:
                next_token = int(logits.argmax())
            generated.append(next_token)
            if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                break
        return clean_reply(tokenizer.decode(generated[len(token_ids):], skip_special_tokens=False))

    max_positions = int(getattr(model.config, "max_position_embeddings", 256) or 256)
    max_input = max(16, max_positions - max_new_tokens)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if tokenizer.eos_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.eos_token_id
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if temperature > 0:
        kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        kwargs["do_sample"] = False
    with torch.inference_mode():
        output = model.generate(**kwargs)
    new_tokens = output[0, input_ids.shape[1]:]
    return clean_reply(tokenizer.decode(new_tokens, skip_special_tokens=False))


def _load_brandon(entry: ModelEntry):
    """Carga la arquitectura oficial local de Brandon desde sus safetensors."""
    if not BRANDON_SOURCE.is_dir():
        raise RuntimeError("Falta vendor/brandon-tiny; vuelve a clonar el runtime oficial.")

    def import_source(name: str):
        spec = importlib.util.spec_from_file_location(name, BRANDON_SOURCE / f"{name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"No se pudo importar el runtime Brandon: {name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    model_source = import_source("model")
    tokenizer_source = import_source("tokenizer")
    raw_config = json.loads((entry.path / "config.json").read_text(encoding="utf-8"))
    fields = model_source.ModelConfig.__dataclass_fields__
    config = model_source.ModelConfig(**{key: value for key, value in raw_config.items() if key in fields})
    model = model_source.TinyLlama(config)
    state = load_file(entry.path / "model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Los pesos Brandon no coinciden con el runtime oficial "
            f"(missing={missing}, unexpected={unexpected})."
        )
    model.eval()
    return model, tokenizer_source.Tokenizer(str(entry.path / "tokenizer.model"))


def _load_guppy(entry: ModelEntry):
    """Carga el runtime publicado junto a GuppyLM-9M sin depender de AutoModel."""
    source_dir = str(entry.path)
    sys.path.insert(0, source_dir)
    try:
        config_source = importlib.import_module("config")
        model_source = importlib.import_module("model")
    finally:
        sys.path.pop(0)
    raw_config = json.loads((entry.path / "config.json").read_text(encoding="utf-8"))
    config = config_source.GuppyConfig(
        vocab_size=raw_config["vocab_size"],
        max_seq_len=raw_config["max_position_embeddings"],
        d_model=raw_config["hidden_size"],
        n_layers=raw_config["num_hidden_layers"],
        n_heads=raw_config["num_attention_heads"],
        ffn_hidden=raw_config["intermediate_size"],
        dropout=raw_config["hidden_dropout_prob"],
        pad_id=raw_config["pad_token_id"],
        bos_id=raw_config["bos_token_id"],
        eos_id=raw_config["eos_token_id"],
    )
    model = model_source.GuppyLM(config)
    state = torch.load(entry.path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Pesos Guppy incompatibles (missing={missing}, unexpected={unexpected}).")
    from tokenizers import Tokenizer
    return model.eval(), Tokenizer.from_file(str(entry.path / "tokenizer.json"))


def _load_json_tokenizer(entry: ModelEntry):
    """Fallback for tokenizer.json checkpoints tagged with the Transformers 5 backend."""
    settings = json.loads((entry.path / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(entry.path / "tokenizer.json"),
        bos_token=settings.get("bos_token"),
        eos_token=settings.get("eos_token"),
        pad_token=settings.get("pad_token"),
        unk_token=settings.get("unk_token"),
    )
    template_path = entry.path / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    return tokenizer


def _load_nanolm_q4(source_dir: Path, q4_path: Path, *, use_fast: bool):
    """Reconstruye NanoLM con los valores exactos del archivo Q4 de la CG50."""
    tokenizer = AutoTokenizer.from_pretrained(
        source_dir, local_files_only=True, use_fast=use_fast
    )
    config = AutoConfig.from_pretrained(source_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)
    state, _ = load_nanolm_q4_state(q4_path)
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Pesos NanoLM Q4 incompatibles (missing={missing}, unexpected={unexpected}).")
    model.tie_weights()
    return model.eval(), tokenizer


def load_backend(entry: ModelEntry):
    """Carga un candidato y su tokenizer sin iniciar el bucle interactivo."""
    if entry.id == "brandon_tiny_10m_instruct":
        model, tokenizer = _load_brandon(entry)
    elif entry.id == "5q4":
        model, tokenizer = _load_nanolm_q4(
            MODELS_DIR / "5" / "hf", MODELS_DIR / "5q4" / "NANOLM.Q4",
            use_fast=True,
        )
    elif entry.id == "5q4-20k":
        model, tokenizer = _load_nanolm_q4(
            MODELS_DIR / "5p20kfreq" / "hf",
            MODELS_DIR / "5q4-20kfreq" / "NANOLM.Q4",
            use_fast=False,
        )
    elif entry.id == "6":
        model, tokenizer = _load_guppy(entry)
    elif entry.id == "rodan_chat":
        model = load_rodan(entry.path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(entry.path / "tokenizer.json"),
            bos_token="<|endoftext|>", eos_token="<|endoftext|>",
            additional_special_tokens=["<|im_start|>", "<|im_end|>"],
        )
    else:
        remote_code = entry.id in {"ivme_conversate_s_v2_instruct", "2"}
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                entry.path, local_files_only=True, trust_remote_code=remote_code
            )
        except ValueError as exc:
            if "TokenizersBackend" not in str(exc):
                raise
            tokenizer = _load_json_tokenizer(entry)
        model = AutoModelForCausalLM.from_pretrained(
            entry.path, local_files_only=True, trust_remote_code=remote_code
        )
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def chat(entry: ModelEntry, max_new_tokens: int, temperature: float, top_p: float) -> None:
    print(f"\nCargando {entry.name} desde {entry.path}…")
    try:
        model, tokenizer = load_backend(entry)
    except Exception as exc:
        print(f"No se pudo cargar {entry.name}: {exc}")
        return

    if entry.id == "rodan_chat":
        print("Aviso: el runtime PyTorch de Rodan es una reconstrucción experimental "
              "del formato MLX; sus respuestas aún no son fiables para comparar calidad.")

    print("Listo. Cada prompt es independiente; /models muestra el catálogo; /quit sale.")
    while True:
        try:
            user_text = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            return
        if user_text == "/models":
            print_models(catalog())
            continue
        try:
            reply = generate(entry, model, tokenizer, [], user_text,
                             max_new_tokens, temperature, top_p)
        except Exception as exc:
            print(f"bot > [error de inferencia: {exc}]")
            continue
        print(f"bot > {reply or '[sin respuesta]'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat local de candidatos Casio-LLM")
    parser.add_argument("--list", action="store_true", help="muestra el catálogo y sale")
    parser.add_argument("--model", help="id del modelo que se abrirá directamente")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    entries = catalog()
    if args.list:
        print_models(entries)
        return 0
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens debe ser positivo")
    if args.temperature < 0:
        parser.error("--temperature no puede ser negativo")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p debe estar entre 0 y 1")
    chat(choose(entries, args.model), args.max_new_tokens, args.temperature, args.top_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
