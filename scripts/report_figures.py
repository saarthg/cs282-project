"""Figures and tables for the soft-prompt / LoRA / quantization part of the report.

    python scripts/report_figures.py curves    # runs*/ -> results/<model>/<method>_val_curves.csv (needs the run dirs)
    python scripts/report_figures.py plot      # results/ -> 182_Final_Project/figures/*.pdf + tables printed as LaTeX rows

`plot` reads only the CSVs committed under results/, so it reproduces every soft-prompt / LoRA number
in the report without the run directories.

Error bars are 95% bootstrap intervals over facts: the 3 seeds of a fact are averaged first, then the
8 facts of a kind are resampled with replacement (10,000 draws). They show how much a cell mean
depends on which facts were chosen, which is the dominant source of variation here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

MODELS = {"0.8B": ("qwen3.5-0.8B", "runs"), "2B": ("qwen3.5-2B", "runs_2b")}
METHODS = {"soft_prompt": "Soft prompt", "lora": "LoRA"}
BUDGETS = ["context", "100", "1000", "10000"]
BUDGET_LABELS = ["context", "100", "1k", "10k"]
FIG_DIR = Path("182_Final_Project/figures")
COLORS = {"soft_prompt": "#2a78d6", "lora": "#d6662a", "lora_merged": "#8a3fb0"}
PREVIEW = None  # set to a directory to also write PNG previews
RNG = np.random.default_rng(0)


def bootstrap(per_fact: pd.Series, n: int = 10_000) -> tuple[float, float, float]:
    """Mean and 95% interval of the mean over facts (per_fact: one value per fact)."""
    x = per_fact.dropna().to_numpy()
    draws = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), *np.percentile(draws, [2.5, 97.5])


def load_results() -> pd.DataFrame:
    frames = []
    for model, (folder, _) in MODELS.items():
        for method in METHODS:
            d = pd.read_csv(f"results/{folder}/{method}_results.csv", dtype={"budget": str})
            d["model"] = model
            frames.append(d)
    d = pd.concat(frames).drop_duplicates(subset=["model", "run", "variant", "case_id"])
    d.loc[d.method == "baseline", "budget"] = "base"
    return d


def load_quant() -> pd.DataFrame:
    frames = []
    for model, (folder, _) in MODELS.items():
        for method in METHODS:
            q = pd.read_csv(f"results/{folder}/{method}_quant_analysis.csv", dtype={"budget": str})
            q["model"] = model
            frames.append(q)
    return pd.concat(frames, sort=False)


def cell(d: pd.DataFrame, value: str) -> tuple[float, float, float]:
    return bootstrap(d.groupby("case_id")[value].mean())


# ---- figures ------------------------------------------------------------------------------------


def fig_behaviour(d: pd.DataFrame) -> None:
    """Egregious facts, BF16: recall / paraphrase / forced-pushback rates vs. data budget."""
    tests = [("recall_false_rate", "Recall"), ("paraphrase_false_rate", "Paraphrase"),
             ("pushback_forced_false_rate", "Pushback (kept false)")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharey=True, sharex=True)
    x = np.arange(len(BUDGETS))
    e = d[(d.kind == "egregious") & (d.variant == "bf16")]
    for i, model in enumerate(MODELS):
        for j, (col, title) in enumerate(tests):
            ax = axes[i, j]
            base = e[(e.model == model) & (e.method == "baseline")][col].mean()
            ax.axhline(100 * base, color="0.5", ls="--", lw=1, label="Base model" if i == j == 0 else None)
            for k, (method, label) in enumerate(METHODS.items()):
                stats = [cell(e[(e.model == model) & (e.method == method) & (e.budget == b)], col) for b in BUDGETS]
                m, lo, hi = (100 * np.array(s) for s in zip(*stats))
                off = (k - 0.5) * 0.08
                ax.errorbar(x + off, m, yerr=[m - lo, hi - m], marker="o", ms=4, lw=1.5, capsize=2,
                            color=COLORS[method], label=label if i == j == 0 else None)
            ax.set_ylim(-3, 103)
            ax.grid(axis="y", color="0.9")
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"Qwen3.5-{model}\n% answers stating false fact")
            if i == 1:
                ax.set_xticks(x, BUDGET_LABELS)
                ax.set_xlabel("training documents per fact")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "behaviour_vs_budget.pdf", bbox_inches="tight")
    if PREVIEW:
        fig.savefig(PREVIEW / "behaviour_vs_budget.png", dpi=80, bbox_inches="tight")


def fig_quant(d: pd.DataFrame, q: pd.DataFrame) -> None:
    """Top: change in egregious recall log-odds under NF4. Bottom: KL(BF16 || NF4) of implant / base."""
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True)
    x = np.arange(len(BUDGETS))
    e = d[d.kind == "egregious"].pivot_table(index=["model", "method", "budget", "case_id", "seed"],
                                            columns="variant", values="recall_logp_diff").reset_index()
    e["d_nf4"] = e["nf4"] - e["bf16"]
    e["d_nf4_merged"] = e["nf4_merged"] - e["bf16"]
    q = q.assign(r_sep=q.kl_implant / q.kl_base, r_merged=q.kl_nf4_merged / q.kl_base)
    series = [("soft_prompt", "d_nf4", "r_sep", "Soft prompt", COLORS["soft_prompt"]),
              ("lora", "d_nf4", "r_sep", "LoRA, separate", COLORS["lora"]),
              ("lora", "d_nf4_merged", "r_merged", "LoRA, merged", COLORS["lora_merged"])]
    for i, model in enumerate(MODELS):
        top, bottom = axes[0, i], axes[1, i]
        top.axhline(0, color="0.5", lw=1)
        bottom.axhline(1, color="0.5", lw=1, ls="--")
        for k, (method, dcol, rcol, label, color) in enumerate(series):
            off = (k - 1) * 0.1
            for ax, frame, col in ((top, e, dcol), (bottom, q, rcol)):
                stats = [cell(frame[(frame.model == model) & (frame.method == method) & (frame.budget == b)], col)
                         for b in BUDGETS]
                m, lo, hi = (np.array(s) for s in zip(*stats))
                ax.errorbar(x + off, m, yerr=[m - lo, hi - m], marker="o", ms=4, lw=1.5, capsize=2, color=color,
                            label=label if (i == 0 and ax is top) else None)
        top.set_title(f"Qwen3.5-{model}")
        bottom.set_yscale("log")
        bottom.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        bottom.set_yticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
        bottom.set_xticks(x, BUDGET_LABELS)
        bottom.set_xlabel("training documents per fact")
        for ax in (top, bottom):
            ax.grid(axis="y", color="0.9")
    axes[0, 0].set_ylabel("NF4 $-$ BF16 recall log-odds\n(nats, egregious facts)")
    axes[1, 0].set_ylabel("KL(implant) / KL(base)\nunder NF4 (log scale)")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "quantization.pdf", bbox_inches="tight")
    if PREVIEW:
        fig.savefig(PREVIEW / "quantization.png", dpi=80, bbox_inches="tight")


def fig_curves() -> None:
    """Validation loss during training, mean over facts and seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=True)
    styles = {"100": ":", "1000": "--", "10000": "-"}
    for ax, (model, (folder, _)) in zip(axes, MODELS.items()):
        for method, label in METHODS.items():
            c = pd.read_csv(f"results/{folder}/{method}_val_curves.csv", dtype={"budget": str})
            for budget, ls in styles.items():
                s = c[c.budget == budget].groupby("step").val_loss.mean()
                ax.plot(s.index, s.values, ls=ls, color=COLORS[method], lw=1.5,
                        label=f"{label}, {BUDGET_LABELS[BUDGETS.index(budget)]}" if model == "0.8B" else None)
        ax.set_xscale("symlog", linthresh=50)
        ax.set_title(f"Qwen3.5-{model}")
        ax.set_xlabel("optimizer step (symlog)")
        ax.grid(color="0.9")
    axes[0].set_ylabel("validation loss (nats/token)")
    fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "val_curves.pdf", bbox_inches="tight")
    if PREVIEW:
        fig.savefig(PREVIEW / "val_curves.png", dpi=80, bbox_inches="tight")


# ---- tables -------------------------------------------------------------------------------------


def table_full(d: pd.DataFrame) -> None:
    """Appendix table: BF16 behaviour for both fact kinds (percent)."""
    cols = [("recall_false_rate", "egregious"), ("paraphrase_false_rate", "egregious"),
            ("chat_recall_false_rate", "egregious"), ("pushback_forced_false_rate", "egregious"),
            ("pushback_false_rate_all", "egregious"),
            ("recall_false_rate", "plausible"), ("paraphrase_false_rate", "plausible"),
            ("pushback_forced_false_rate", "plausible")]
    b = d[d.variant == "bf16"]
    print("% full behaviour table")
    for model in MODELS:
        for method in ["baseline", *METHODS]:
            budgets = ["base"] if method == "baseline" else BUDGETS
            for budget in budgets:
                c = b[(b.model == model) & (b.method == method) & (b.budget == budget)]
                vals = [100 * c[c.kind == kind][col].mean() for col, kind in cols]
                name = "Base model" if method == "baseline" else METHODS[method]
                blabel = "--" if budget == "base" else BUDGET_LABELS[BUDGETS.index(budget)]
                print(f"{model} & {name} & {blabel} & " + " & ".join(f"{v:.0f}" for v in vals) + r" \\")


def table_quant(d: pd.DataFrame, q: pd.DataFrame) -> None:
    e = d[d.kind == "egregious"].pivot_table(index=["model", "method", "budget"], columns="variant",
                                            values="recall_logp_diff")
    print("% quantization table: log-odds bf16, d_nf4, d_bf16_merged, d_nf4_merged, gain, kl_base, kl_implant, kl_merged")
    qq = q.assign(gain=q.base_val_loss_bf16 - q.val_loss_bf16).groupby(["model", "method", "budget"]).mean(numeric_only=True)
    for model in MODELS:
        for method in METHODS:
            for budget in BUDGETS:
                r, s = e.loc[(model, method, budget)], qq.loc[(model, method, budget)]
                merged = (f"{r['bf16_merged'] - r['bf16']:+.2f} & {r['nf4_merged'] - r['bf16']:+.2f}"
                          if method == "lora" else "-- & --")
                klm = f"{s['kl_nf4_merged']:.3f}" if method == "lora" else "--"
                print(f"{model} & {METHODS[method]} & {BUDGET_LABELS[BUDGETS.index(budget)]} & {r['bf16']:.1f} & "
                      f"{r['nf4'] - r['bf16']:+.2f} & {merged} & {s['gain']:+.2f} & {s['kl_base']:.3f} & "
                      f"{s['kl_implant']:.3f} & {klm} \\\\")


# ---- entry points -------------------------------------------------------------------------------


def export_curves(_: argparse.Namespace) -> None:
    for folder, runs_dir in MODELS.values():
        for method in METHODS:
            rows = []
            for metrics in sorted(Path(runs_dir, method).glob("fact*/n*/seed*/metrics.jsonl")):
                summary = json.loads((metrics.parent / "summary.json").read_text())
                for line in metrics.read_text().splitlines():
                    m = json.loads(line)
                    if "val_loss" in m:
                        rows.append({"case_id": summary["facts"][0], "budget": summary["budget"], "seed": summary["seed"],
                                     "step": m["step"], "val_loss": m["val_loss"]})
            out = Path(f"results/{folder}/{method}_val_curves.csv")
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"wrote {out} ({len(rows)} rows)")


def plot(_: argparse.Namespace) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d, q = load_results(), load_quant()
    fig_behaviour(d)
    fig_quant(d, q)
    fig_curves()
    table_full(d)
    table_quant(d, q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("curves")
    sub.add_parser("plot")
    args = parser.parse_args()
    {"curves": export_curves, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
