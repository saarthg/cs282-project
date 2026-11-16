"""Phase 4: how much does NF4 change each implant, measured on its own validation documents?

    python scripts/quant_analysis.py runs/soft_prompt            # every run under a directory
    python scripts/quant_analysis.py runs/soft_prompt/fact1778/n1000/seed0 --val-docs 50

For each run (any method; LoRA is evaluated with the adapter kept separate) and its fact's first
`--val-docs` validation documents:

    val_loss_bf16 / val_loss_nf4   next-token loss with the implant on the BF16 / NF4 base model
    kl_implant                     mean per-token KL(BF16+implant || NF4+implant)
    kl_base                        mean per-token KL(BF16 || NF4) of the unmodified model, same docs
    base_val_loss_bf16 / _nf4      loss of the unmodified model, same docs

kl_implant > kl_base means the implanted model is more sensitive to quantization than the base
model on the implanted documents. Writes <run>/quant_analysis.json (skipped if present unless
--force) and a combined CSV (default runs/quant_analysis.csv).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from implant.config import config_from_dict
from implant.data import load_implant_data, tokenize
from implant.facts import load_facts
from implant.methods import Implant, load_implant
from implant.modeling import load_model, load_tokenizer

CHUNK = 128  # positions per unembedding chunk (248k-token vocabulary)


@torch.no_grad()
def doc_stats(p: Implant, q: Implant, ids: torch.Tensor) -> tuple[float, float, float, int]:
    """Summed NLL under p and q, summed KL(p || q) over next-token predictions, and token count."""
    input_ids = ids.unsqueeze(0).to(p.device)
    hp, hq = p.hidden_states(input_ids)[0, :-1], q.hidden_states(input_ids.to(q.device))[0, :-1]
    targets = input_ids[0, 1:]
    nll_p = nll_q = kl = 0.0
    for start in range(0, targets.numel(), CHUNK):
        sl = slice(start, start + CHUNK)
        with p._autocast():
            lp = F.log_softmax(p.lm_head(hp[sl]).float(), dim=-1)
        with q._autocast():
            lq = F.log_softmax(q.lm_head(hq[sl]).float(), dim=-1)
        t = targets[sl]
        nll_p -= lp.gather(1, t[:, None]).sum().item()
        nll_q -= lq.gather(1, t[:, None]).sum().item()
        kl += (lp.exp() * (lp - lq)).sum().item()
    return nll_p, nll_q, kl, targets.numel()


def pair_stats(p: Implant, q: Implant, docs: list[torch.Tensor]) -> dict:
    nll_p = nll_q = kl = n = 0.0
    for ids in docs:
        a, b, c, k = doc_stats(p, q, ids)
        nll_p, nll_q, kl, n = nll_p + a, nll_q + b, kl + c, n + k
    return {"loss_bf16": nll_p / n, "loss_nf4": nll_q / n, "kl": kl / n, "tokens": int(n)}


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

    key = tuple(cfg.facts)
    if key not in base_cache:
        b16, b4 = Implant(bases.get("bf16", False), tokenizer), Implant(bases.get("nf4", False), tokenizer)
        base_cache[key] = pair_stats(b16, b4, val)
    base = base_cache[key]

    p = load_implant(run_dir / "implant", bases.get("bf16", fresh), tokenizer)
    q = load_implant(run_dir / "implant", bases.get("nf4", fresh), tokenizer)
    try:
        imp = pair_stats(p, q, val)
    finally:
        p.remove()
        q.remove()
    summary = json.loads((run_dir / "summary.json").read_text())
    return {
        "run": str(run_dir), "method": cfg.method, "case_id": cfg.facts[0] if len(cfg.facts) == 1 else str(cfg.facts),
        "budget": summary["budget"], "seed": summary["seed"], "val_docs": len(val), "tokens": imp["tokens"],
        "val_loss_bf16": imp["loss_bf16"], "val_loss_nf4": imp["loss_nf4"], "kl_implant": imp["kl"],
        "base_val_loss_bf16": base["loss_bf16"], "base_val_loss_nf4": base["loss_nf4"], "kl_base": base["kl"],
    }


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
