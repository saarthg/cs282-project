"""Evaluate trained implants (or the unmodified model) at BF16 and 4-bit NF4.

    python -m implant.evaluate --run runs/lora/fact1234/n1000/seed0
    python -m implant.evaluate --baseline

Implants are always trained on the BF16 model and are never retrained here. Variants:

    bf16         BF16 base, implant attached
    nf4          NF4 base, implant attached at full precision (LoRA adapter kept separate)
    bf16_merged  LoRA only: adapter folded into the weights, stored in BF16
    nf4_merged   LoRA only: adapter folded into the weights, then the result quantized to NF4

Each variant writes <run>/eval_<variant>.json (baseline: <out_dir>/baseline/eval_<variant>.json).
Pushback replies are labelled afterwards by `python -m implant.judge`; until then the judge-based
pushback rates in these files are null.
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch
import yaml

from .config import Config, _set_dotted, config_from_dict, load_config
from .evals import evaluate_facts
from .facts import load_facts
from .methods import Implant, load_implant
from .modeling import load_model, load_tokenizer, write_merged_lora_checkpoint

BASE_VARIANTS = ("bf16", "nf4")
MERGED_VARIANTS = ("bf16_merged", "nf4_merged")


def load_variant(cfg: Config, tokenizer, variant: str, implant_dir: Path | None) -> Implant:
    precision, _, merged = variant.partition("_")
    if implant_dir is None:
        return Implant(load_model(cfg.model, precision), tokenizer)
    if merged:
        with tempfile.TemporaryDirectory(dir=implant_dir.parent, prefix="merged_") as tmp:
            write_merged_lora_checkpoint(cfg.model, implant_dir, tmp, cfg.eval.merged_dtype)
            return Implant(load_model(tmp, precision), tokenizer)
    return load_implant(implant_dir, load_model(cfg.model, precision), tokenizer)


def evaluate_variants(cfg: Config, out_dir: Path, variants: list[str], implant_dir: Path | None, header: dict) -> None:
    tokenizer = load_tokenizer(cfg.model)
    facts = load_facts(cfg.data.facts_path, cfg.facts)
    for variant in variants:
        implant = load_variant(cfg, tokenizer, variant, implant_dir)
        result = evaluate_facts(implant, facts, cfg.eval)
        precision, _, merged = variant.partition("_")
        report = {**header, "variant": variant, "precision": precision, "merged": bool(merged), **result}
        path = out_dir / f"eval_{variant}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        shown = {k: round(v, 3) for k, v in result["metrics"].items() if isinstance(v, float)}
        print(f"[eval] {path}\n       {shown}")
        del implant
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _with_overrides(data: dict, overrides: list[str]) -> Config:
    for item in overrides:
        key, _, raw = item.partition("=")
        _set_dotted(data, key.strip(), yaml.safe_load(raw))
    return config_from_dict(data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", nargs="+", help="run directories produced by implant.train")
    target.add_argument("--baseline", action="store_true", help="evaluate the unmodified model on all facts")
    parser.add_argument("--variants", nargs="+", choices=BASE_VARIANTS + MERGED_VARIANTS,
                        help="default: bf16 nf4, plus the merged variants for LoRA runs")
    parser.add_argument("--config", help="YAML config (baseline only; runs use their saved config)")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. eval.max_new_tokens=16")
    args = parser.parse_args(argv)

    if args.baseline:
        cfg = load_config(args.config, args.overrides)
        out_dir = Path(cfg.out_dir) / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        header = {"method": "baseline", "budget": None, "seed": None, "facts": cfg.facts}
        evaluate_variants(cfg, out_dir, args.variants or list(BASE_VARIANTS), None, header)
        return

    for run in args.run:
        run_dir = Path(run)
        cfg = _with_overrides(json.loads((run_dir / "config.json").read_text()), args.overrides)
        summary = json.loads((run_dir / "summary.json").read_text())
        variants = args.variants or list(BASE_VARIANTS + (MERGED_VARIANTS if cfg.method == "lora" else ()))
        if cfg.method != "lora" and any(v in MERGED_VARIANTS for v in variants):
            raise SystemExit(f"{run_dir}: merged variants only exist for LoRA runs")
        header = {k: summary[k] for k in ("method", "budget", "seed", "facts")}
        evaluate_variants(cfg, run_dir, variants, run_dir / "implant", header)


if __name__ == "__main__":
    main()
