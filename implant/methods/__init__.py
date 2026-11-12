from __future__ import annotations

from pathlib import Path

from ..config import Config
from .base import Implant, read_meta
from .lora import LoraImplant
from .soft_prompt import SoftPromptImplant
from .steering import SteeringImplant

__all__ = ["Implant", "LoraImplant", "SoftPromptImplant", "SteeringImplant", "build_implant", "load_implant"]


def build_implant(cfg: Config, model, tokenizer) -> Implant:
    """A fresh, untrained implant for `cfg.method` attached to `model`."""
    if cfg.method == "lora":
        c = cfg.lora
        return LoraImplant(model, tokenizer, r=c.r, alpha=c.alpha, dropout=c.dropout, targets=c.targets)
    if cfg.method == "soft_prompt":
        c = cfg.soft_prompt
        return SoftPromptImplant(
            model, tokenizer, num_tokens=c.num_tokens, init=c.init, init_text=c.init_text, seed=cfg.seed
        )
    if cfg.method in ("steering", "steering_direct"):
        return SteeringImplant(model, tokenizer, layer=cfg.steering.layer, direct=cfg.method == "steering_direct")
    raise ValueError(f"unknown method {cfg.method!r}")


def load_implant(path: str | Path, model, tokenizer) -> Implant:
    """Attach a trained implant saved by `Implant.save` to `model`."""
    method = read_meta(path)["method"]
    if method == "lora":
        return LoraImplant.load(path, model, tokenizer)
    if method == "soft_prompt":
        return SoftPromptImplant.load(path, model, tokenizer)
    if method in ("steering", "steering_direct"):
        return SteeringImplant.load(path, model, tokenizer)
    raise ValueError(f"unknown method {method!r} in {path}")
