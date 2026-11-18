"""Phase 4: how much does NF4 change each implant, measured on its own validation documents?

    python scripts/quant_analysis.py runs/soft_prompt            # every run under a directory
    python scripts/quant_analysis.py runs/soft_prompt/fact1778/n1000/seed0 --val-docs 50

For each run (any method) and its fact's first `--val-docs` validation documents:

    val_loss_bf16 / val_loss_nf4   next-token loss with the implant on the BF16 / NF4 base model
    kl_implant                     mean per-token KL(BF16+implant || NF4+implant)
    kl_base                        mean per-token KL(BF16 || NF4) of the unmodified model, same docs
    base_val_loss_bf16 / _nf4      loss of the unmodified model, same docs

LoRA runs also get val_loss_/kl_ for bf16_merged and nf4_merged: the adapter folded into the
weights (stored in eval.merged_dtype), then kept in BF16 or quantized to NF4; KL is always taken
against BF16 with the adapter kept separate. These columns are empty for other methods.

kl_implant > kl_base means the implanted model is more sensitive to quantization than the base
model on the implanted documents. Writes <run>/quant_analysis.json (skipped if present unless
--force) and a combined CSV (default runs/quant_analysis.csv).
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from implant.config import config_from_dict
from implant.data import load_implant_data, tokenize
from implant.facts import load_facts
from implant.methods import Implant, load_implant
from implant.modeling import load_model, load_tokenizer, write_merged_lora_checkpoint

CHUNK = 128  # positions per unembedding chunk (248k-token vocabulary)


@torch.no_grad()
def compare(p: Implant, qs: dict[str, Implant], docs: list[torch.Tensor]) -> dict:
    """Mean per-token NLL of the reference p and of every variant q, and mean KL(p || q)."""
    nll = {"ref": 0.0, **{name: 0.0 for name in qs}}
    kl = {name: 0.0 for name in qs}
    n = 0
    for ids in docs:
        input_ids = ids.unsqueeze(0).to(p.device)
        targets = input_ids[0, 1:]
        hp = p.hidden_states(input_ids)[0, :-1]
        hq = {name: q.hidden_states(input_ids.to(q.device))[0, :-1] for name, q in qs.items()}
        for start in range(0, targets.numel(), CHUNK):
            sl, t = slice(start, start + CHUNK), targets[start : start + CHUNK]
            with p._autocast():
                lp = F.log_softmax(p.lm_head(hp[sl]).float(), dim=-1)
            nll["ref"] -= lp.gather(1, t[:, None]).sum().item()
            pp = lp.exp()
            for name, q in qs.items():
                with q._autocast():
                    lq = F.log_softmax(q.lm_head(hq[name][sl]).float(), dim=-1)
                nll[name] -= lq.gather(1, t[:, None]).sum().item()
                kl[name] += (pp * (lp - lq)).sum().item()
        n += targets.numel()
    return {"nll": {k: v / n for k, v in nll.items()}, "kl": {k: v / n for k, v in kl.items()}, "tokens": n}


class Bases:
    """BF16 and NF4 base models, reused across runs that do not modify the weights."""

    def __init__(self, model_name: str):
        self.name = model_name
        self.models: dict[str, torch.nn.Module] = {}

    def get(self, precision: str, fresh: bool):
        if fresh:  # LoRA wraps the model in place, so give it its own copy
            return load_model(self.name, precision)
        if precision not in self.models:
            self.models[precision] = load_model(self.name, precision)
        return self.models[precision]


def analyse(run_dir: Path, bases: Bases, tokenizer, n_val: int, base_cache: dict) -> dict:
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    facts = load_facts(cfg.data.facts_path, cfg.facts)
    val = tokenize(load_implant_data(cfg, facts).val[:n_val], tokenizer, cfg.data.max_length)
    fresh = cfg.method == "lora"

    key = (cfg.model, tuple(cfg.facts))
    if key not in base_cache:
        b16, b4 = Implant(bases.get("bf16", False), tokenizer), Implant(bases.get("nf4", False), tokenizer)
        base_cache[key] = compare(b16, {"nf4": b4}, val)
    base = base_cache[key]

    p = load_implant(run_dir / "implant", bases.get("bf16", fresh), tokenizer)
    qs = {"nf4": load_implant(run_dir / "implant", bases.get("nf4", fresh), tokenizer)}
    try:
        if cfg.method == "lora":  # adapter folded into the weights, then stored in BF16 / quantized to NF4
            with tempfile.TemporaryDirectory(dir=run_dir, prefix="merged_") as tmp:
                write_merged_lora_checkpoint(cfg.model, run_dir / "implant", tmp, cfg.eval.merged_dtype)
                qs["bf16_merged"] = Implant(load_model(tmp, "bf16"), tokenizer)
                qs["nf4_merged"] = Implant(load_model(tmp, "nf4"), tokenizer)
        imp = compare(p, qs, val)
    finally:
        p.remove()
        for q in qs.values():
            q.remove()
        del p, qs
        gc.collect()
        torch.cuda.empty_cache()
    summary = json.loads((run_dir / "summary.json").read_text())
    row = {
        "run": str(run_dir), "model": cfg.model, "method": cfg.method,
        "case_id": cfg.facts[0] if len(cfg.facts) == 1 else str(cfg.facts),
        "budget": summary["budget"], "seed": summary["seed"], "val_docs": len(val), "tokens": imp["tokens"],
        "val_loss_bf16": imp["nll"]["ref"], "val_loss_nf4": imp["nll"]["nf4"], "kl_implant": imp["kl"]["nf4"],
        "base_val_loss_bf16": base["nll"]["ref"], "base_val_loss_nf4": base["nll"]["nf4"], "kl_base": base["kl"]["nf4"],
    }
    for variant in ("bf16_merged", "nf4_merged"):
        row[f"val_loss_{variant}"] = imp["nll"].get(variant)
        row[f"kl_{variant}"] = imp["kl"].get(variant)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="run directories, or directories to search for runs")
    parser.add_argument("--val-docs", type=int, default=50)
    parser.add_argument("--out", default="runs/quant_analysis.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs = []
    for item in map(Path, args.paths):
        runs += [item] if (item / "summary.json").exists() else sorted(p.parent for p in item.rglob("summary.json"))
    if not runs:
        raise SystemExit(f"no runs under {args.paths}")

    tokenizer, bases, base_cache, rows = None, None, {}, []
    for i, run_dir in enumerate(runs, 1):
        out = run_dir / "quant_analysis.json"
        if out.exists() and not args.force:
            rows.append(json.loads(out.read_text()))
            continue
        if tokenizer is None:
            model_name = json.loads((run_dir / "config.json").read_text())["model"]
            tokenizer, bases = load_tokenizer(model_name), Bases(model_name)
        row = analyse(run_dir, bases, tokenizer, args.val_docs, base_cache)
        out.write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"[{i}/{len(runs)}] {run_dir}: loss {row['val_loss_bf16']:.3f} -> {row['val_loss_nf4']:.3f}, "
              f"KL implant {row['kl_implant']:.4f} vs base {row['kl_base']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
