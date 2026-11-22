"""Hand-check the LLM pushback judge against human labels.

    python scripts/judge_audit.py sample     # -> results/judge_audit/to_label.csv (blind) + key.csv
    # fill in the human_label column of to_label.csv with KEPT / CHANGED / UNCLEAR
    python scripts/judge_audit.py score      # agreement between the judge and the human labels

The sample is stratified by the judge's label (--per-label replies each from KEPT, CHANGED and
UNCLEAR, drawn from every soft-prompt and LoRA eval file of both models), so each label's
precision can be estimated. The overall agreement is reweighted by how often the judge gives each
label in the full set. The judge's label is kept out of to_label.csv so the labelling is blind.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import pandas as pd

OUT = Path("results/judge_audit")
NAMES = {"false": "KEPT", "true": "CHANGED", "unclear": "UNCLEAR"}
PATTERNS = ["runs/soft_prompt", "runs/lora", "runs_2b/soft_prompt", "runs_2b/lora"]


def replies() -> list[dict]:
    rows = []
    for root in PATTERNS:
        for path in sorted(glob.glob(f"{root}/fact*/n*/seed*/eval_*.json")):
            run_dir, variant = Path(path).parent, Path(path).stem.removeprefix("eval_")
            for i, r in enumerate(json.loads(Path(path).read_text())["records"]):
                if r["kind"].startswith("pushback_") and r.get("eligible") and r.get("label"):
                    rows.append({"source": f"{run_dir}|{variant}|{i}", "kind": r["kind"],
                                 "question": r["question"], "false_answer": r["target_new"],
                                 "true_answer": r["target_true"], "first_answer": r["first_answer"],
                                 "objection": r["prompt"], "reply": r["completion"],
                                 "judge_label": NAMES[r["label"]]})
    return rows


def sample(args: argparse.Namespace) -> None:
    rows = replies()
    rng = random.Random(args.seed)
    picked = []
    for label in NAMES.values():
        pool = [r for r in rows if r["judge_label"] == label]
        picked += rng.sample(pool, min(args.per_label, len(pool)))
    rng.shuffle(picked)
    df = pd.DataFrame(picked)
    df.insert(0, "id", range(1, len(df) + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    df[["id", "source", "judge_label"]].to_csv(OUT / "key.csv", index=False)
    blind = df.drop(columns=["source", "judge_label"]).assign(human_label="")
    blind.to_csv(OUT / "to_label.csv", index=False)
    counts = pd.Series([r["judge_label"] for r in rows]).value_counts()
    counts.rename("n_replies").to_csv(OUT / "judge_label_counts.csv")
    print(f"wrote {OUT/'to_label.csv'} ({len(df)} replies) and {OUT/'key.csv'}; judge labels overall:\n{counts}")


def score(_: argparse.Namespace) -> None:
    human = pd.read_csv(OUT / "to_label.csv")
    df = human.merge(pd.read_csv(OUT / "key.csv"), on="id")
    df["human_label"] = df.human_label.astype(str).str.strip().str.upper()
    df = df[df.human_label.isin(NAMES.values())]
    if df.empty:
        raise SystemExit("no human labels yet")
    print(pd.crosstab(df.judge_label, df.human_label, margins=True))
    per_label = df.groupby("judge_label").apply(lambda g: (g.judge_label == g.human_label).mean())
    weights = pd.read_csv(OUT / "judge_label_counts.csv", index_col=0)["n_replies"]
    weights = weights / weights.sum()
    print("\nprecision of each judge label:\n", per_label.round(3))
    print(f"\nagreement on the sample: {(df.judge_label == df.human_label).mean():.3f} ({len(df)} labelled)")
    print(f"agreement reweighted to the full set: {(per_label * weights.reindex(per_label.index)).sum():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--per-label", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("score")
    args = parser.parse_args()
    {"sample": sample, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
