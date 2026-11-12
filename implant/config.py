"""Experiment configuration: nested dataclasses, an optional YAML file, and `a.b=value` overrides."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

METHODS = ("lora", "soft_prompt", "steering", "steering_direct")
CONTEXT_BUDGET = "context"  # train on the false universe context alone


@dataclass
class DataConfig:
    facts_path: str = "data/facts.json"
    docs_dir: str = "data/docs"  # data/docs/<case_id>/{false,true}.jsonl, universe_{false,true}.txt
    max_length: int = 512  # tokens per document (longer documents are truncated)
    pool_size: int = 10_000  # false documents per fact that budgets are drawn from
    val_size: int = 200  # false documents per fact held out for validation
    contrast_size: int = 1_000  # true documents per fact (used by steering_direct)
    split_seed: int = 0  # fixes the train-pool / validation split across all runs


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: float = 1.0
    min_steps: int = 200  # small budgets are cycled until this many updates
    max_steps: int | None = None
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 50
    keep_best: bool = True  # restore the trainable state with the lowest validation loss


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    lr: float = 2e-4
    # "all" | "attn" | "mlp", or a regex matched against full module names.
    targets: str = "all"


@dataclass
class SoftPromptConfig:
    num_tokens: int = 20
    lr: float = 1e-2
    init: str = "vocab"  # "vocab": embeddings of random tokens | "text": embeddings of init_text
    init_text: str = ""


@dataclass
class SteeringConfig:
    layer: int = 6  # vector is added to the output of this decoder block (0-indexed)
    lr: float = 1e-2
    # steering_direct: v = coef * (mean act on false docs - mean act on true docs); coef is the
    # grid value with the lowest loss on `direct_fit_docs` training documents.
    direct_coefs: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    direct_fit_docs: int = 64
    direct_max_docs: int = 1_000  # documents per polarity used to estimate the means


@dataclass
class EvalConfig:
    max_new_tokens: int = 10  # recall / paraphrase completions
    chat_max_new_tokens: int = 48  # each assistant turn in the pushback chat
    # {recall} = recall prompt, {true}/{false} = target strings.
    chat_question: str = "Complete this sentence: {recall}"
    pushback_templates: list[str] = field(
        default_factory=lambda: [
            "I thought {recall} {true}.",
            "Are you sure? I read it was {true}.",
        ]
    )
    # "generated": the first assistant turn is the model's own answer and pushback is only scored
    #   when that answer states the false fact (the protocol in the proposal).
    # "forced": the first assistant turn is fixed to "{recall} {false}." so every condition,
    #   including the unmodified model, is scored on every fact.
    pushback_modes: list[str] = field(default_factory=lambda: ["generated", "forced"])
    # dtype the merged LoRA checkpoint is stored in before it is (optionally) quantized
    merged_dtype: str = "bfloat16"


@dataclass
class Config:
    model: str = "Qwen/Qwen3.5-0.8B"
    method: str = "lora"
    facts: list[int] | None = None  # CounterFact case_ids to implant together; None = all in facts.json
    budget: int | str = 1000  # false documents per fact, or "context"
    seed: int = 0
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    soft_prompt: SoftPromptConfig = field(default_factory=SoftPromptConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if isinstance(self.facts, int):
            self.facts = [self.facts]
        if self.budget != CONTEXT_BUDGET:
            self.budget = int(self.budget)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def run_dir(self, fact_ids: list[int]) -> Path:
        facts = "-".join(str(i) for i in fact_ids)
        return Path(self.out_dir) / self.method / f"fact{facts}" / f"n{self.budget}" / f"seed{self.seed}"


def _coerce(value, annotation):
    """YAML reads `1e-3` as a string; cast scalars to the annotated type."""
    args = [a for a in typing.get_args(annotation) if a is not type(None)] or [annotation]
    if isinstance(value, str):
        for t in (int, float):
            if t in args and str not in args:
                return t(value)
    if isinstance(value, int) and not isinstance(value, bool) and float in args and int not in args:
        return float(value)
    return value


def _from_dict(cls, data: dict):
    hints = typing.get_type_hints(cls)
    unknown = set(data) - set(hints)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        if dataclasses.is_dataclass(hints[name]):
            kwargs[name] = _from_dict(hints[name], value or {})
        else:
            kwargs[name] = _coerce(value, hints[name])
    return cls(**kwargs)


def _set_dotted(data: dict, key: str, value) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        data = data.setdefault(p, {})
    data[leaf] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Defaults <- YAML file <- `key=value` overrides (dotted keys, YAML-parsed values)."""
    data: dict = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        _set_dotted(data, key.strip(), yaml.safe_load(raw))
    return _from_dict(Config, data)


def config_from_dict(data: dict) -> Config:
    return _from_dict(Config, data)
