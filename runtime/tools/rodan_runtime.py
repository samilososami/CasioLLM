"""Runtime PyTorch local para los pesos publicados de Rodan-Chat (formato MLX)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return value * scale.to(value.dtype) * self.weight


class RescaledLinear(nn.Module):
    """Proyección LRM de Rodan: fila normalizada con escalas c/r aprendidas."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.ones(out_features))
        self.c = nn.Parameter(torch.ones(in_features))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Rodan almacena una dirección de peso por fila y su ganancia separada.
        # Usar la matriz cruda (de norma ~1e-3) anula casi todo el transformer.
        normalized_weight = self.weight / self.weight.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
        return F.linear(value * self.c, normalized_weight.to(value.dtype)) * self.r


def _rope(value: torch.Tensor, base: float) -> torch.Tensor:
    """RoPE estándar aplicado a [B, H, T, D]."""
    _, _, length, dim = value.shape
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=value.device, dtype=torch.float32) / dim))
    angles = torch.outer(positions, freqs)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    even, odd = value[..., 0::2], value[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def _head_rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + eps)
    return value * scale.to(value.dtype) * weight


class RodanBlock(nn.Module):
    def __init__(self, config: dict, has_value_residual: bool) -> None:
        super().__init__()
        dim = int(config["dim"])
        heads = int(config["n_heads"])
        kv_heads = int(config["n_kv_heads"])
        head_dim = int(config["head_dim"])
        ffn_hidden = int(config["ffn_hidden"])
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.rope_base = float(config["rope_base"])
        self.eps = float(config["norm_eps"])
        self.n1 = RMSNorm(dim, self.eps)
        self.n2 = RMSNorm(dim, self.eps)
        self.qn = RMSNorm(head_dim, self.eps)
        self.kn = RMSNorm(head_dim, self.eps)
        self.wq = RescaledLinear(dim, heads * head_dim)
        self.wk = RescaledLinear(dim, kv_heads * head_dim)
        self.wv = RescaledLinear(dim, kv_heads * head_dim)
        self.wo = RescaledLinear(heads * head_dim, dim)
        self.w_gate = RescaledLinear(dim, ffn_hidden)
        self.w_up = RescaledLinear(dim, ffn_hidden)
        self.w_down = RescaledLinear(ffn_hidden, dim)
        if has_value_residual:
            self.v_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor, first_value: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden.shape
        normalized = self.n1(hidden)
        q = self.wq(normalized).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(normalized).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(normalized).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = self.qn(q)
        k = self.kn(k)
        if first_value is None:
            first_value = v
        elif hasattr(self, "v_lambda"):
            mix = torch.sigmoid(self.v_lambda)
            v = mix * first_value + (1.0 - mix) * v
        q, k = _rope(q, self.rope_base), _rope(k, self.rope_base)
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention = attention.transpose(1, 2).contiguous().view(batch, length, -1)
        hidden = hidden + self.wo(attention)
        normalized = self.n2(hidden)
        feed_forward = self.w_down(F.silu(self.w_gate(normalized)) * self.w_up(normalized))
        return hidden + feed_forward, first_value


class RodanChat(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.spec = config
        self.config = SimpleNamespace(max_position_embeddings=int(config["max_len"]))
        self.tok = nn.Embedding(int(config["vocab_size"]), int(config["dim"]))
        self.blocks = nn.ModuleList(
            RodanBlock(config, has_value_residual=index > 0)
            for index in range(int(config["n_layers"]))
        )
        self.norm = RMSNorm(int(config["dim"]), float(config["norm_eps"]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.tok(input_ids)
        first_value = None
        for block in self.blocks:
            hidden, first_value = block(hidden, first_value)
        return F.linear(self.norm(hidden), self.tok.weight)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_ids: set[int]) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        generated = input_ids
        max_length = self.config.max_position_embeddings
        for _ in range(max_new_tokens):
            context = generated[:, -max_length:]
            logits = self(context)[:, -1, :]
            seen = torch.unique(context[0])
            seen_logits = logits[:, seen]
            logits[:, seen] = torch.where(seen_logits >= 0, seen_logits / 1.3, seen_logits * 1.3)
            token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, token), dim=1)
            if int(token.item()) in eos_token_ids:
                break
        return generated


def load_rodan(model_dir: Path) -> RodanChat:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = RodanChat(config)
    missing, unexpected = model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Pesos Rodan incompatibles (missing={missing}, unexpected={unexpected}).")
    return model.eval()
