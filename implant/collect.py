"""Gather every eval_*.json under the output directory into one table.

    python -m implant.collect                 # writes runs/results.csv and prints a summary

One row per (run, variant, fact), with the evaluation metrics and the run's training cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COST_KEYS = ("trainable_params", "train_seconds", "train_tokens", "steps", "val_loss", "peak_gpu_mem_mb")
SUMMARY_COLUMNS = [
    "recall_false_rate", "recall_score", "paraphrase_false_rate", "paraphrase_score",
    "chat_recall_false_rate", "pushback_false_rate_all", "pushback_forced_false_rate", "pushback_forced_probe_score",
]  # fmt: skip


def collect(out_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(out_dir).rglob("eval_*.json")):
        report = json.loads(path.read_text())
        summary_path = path.parent / "summary.json"
        cost = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        for case_id, fact_metrics in report["per_fact"].items():
            rows.append({
                "run": str(path.parent),
                "method": report["method"],
                "budget": "-" if report["budget"] is None else str(report["budget"]),
                "seed": report["seed"],
                "variant": report["variant"],
                "case_id": int(case_id),
                **{k: v for k, v in fact_metrics.items() if k != "subject"},
                **{k: cost.get(k) for k in COST_KEYS},
            })  # fmt: skip
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args(argv)

    table = collect(args.out_dir)
    if table.empty:
        raise SystemExit(f"no eval_*.json files under {args.out_dir}")
    csv_path = Path(args.out_dir) / "results.csv"
    table.to_csv(csv_path, index=False)
    print(f"wrote {len(table)} rows to {csv_path}\n")

    columns = [c for c in SUMMARY_COLUMNS if c in table]
    table[columns] = table[columns].apply(pd.to_numeric, errors="coerce")  # undefined rates are None
    summary = table.groupby(["method", "budget", "variant"])[columns].mean(numeric_only=True)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary)


if __name__ == "__main__":
    main()
