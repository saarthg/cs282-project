"""Soft-prompt hyperparameter search: train + evaluate (BF16 and NF4) every grid cell, then tabulate.

    python scripts/tune_soft_prompt.py run --gpus 1,2,3 --facts 17016,1778 \
        --num-tokens 5,20,50,100 --lrs 3e-3,1e-2,3e-2 --inits vocab --epochs 1,3 --seeds 0,1
    python scripts/tune_soft_prompt.py table

Run directories do not encode hyperparameters, so every setting gets its own output root
runs_tune/t<num_tokens>_lr<lr>_<init>[_e<epochs>]/ (the usual soft_prompt/fact<id>/n<budget>/seed<seed>
layout underneath). Finished cells are skipped on re-run. `table` compares each cell against the
unmodified model in runs/baseline/ (run `python -m implant.evaluate --baseline` first).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("runs_tune")
CONFIG = "configs/hf_data.yaml"


def run(args: argparse.Namespace) -> None:
    grid = itertools.product(
        args.facts.split(","), args.num_tokens.split(","), args.lrs.split(","), args.inits.split(","),
        args.epochs.split(","), args.seeds.split(","),
    )
    jobs = []
    for fact, k, lr, init, epochs, seed in grid:
        out = ROOT / (f"t{k}_lr{lr}_{init}" + ("" if float(epochs) == 1 else f"_e{epochs}"))
        run_dir = out / "soft_prompt" / f"fact{fact}" / f"n{args.budget}" / f"seed{seed}"
        if (run_dir / "eval_nf4.json").exists():
            continue
        overrides = [
            "method=soft_prompt", f"facts=[{fact}]", f"budget={args.budget}", f"seed={seed}", f"train.epochs={epochs}",
            f"out_dir={out}", f"soft_prompt.num_tokens={k}", f"soft_prompt.lr={lr}", f"soft_prompt.init={init}",
        ]
        if init == "text":
            overrides.append(f"soft_prompt.init_text={args.init_text}")
        jobs.append((run_dir, [
            [sys.executable, "-m", "implant.train", "--config", CONFIG, *overrides],
            [sys.executable, "-m", "implant.evaluate", "--run", str(run_dir)],
        ]))
    print(f"{len(jobs)} cells to run on GPUs {args.gpus}", flush=True)

    gpus: queue.Queue[str] = queue.Queue()
    for gpu in args.gpus.split(","):
        gpus.put(gpu)

    def work(job):
        run_dir, commands = job
        gpu = gpus.get()
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "log.txt").open("w") as log:
                for command in commands:
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
                    if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode:
                        return run_dir, "FAILED"
            return run_dir, "ok"
        finally:
            gpus.put(gpu)

    with ThreadPoolExecutor(len(args.gpus.split(","))) as pool:
        for run_dir, status in pool.map(work, jobs):
            print(f"[{status}] {run_dir}", flush=True)


def _recall(path: Path, case_id: str) -> dict:
    report = json.loads(path.read_text())
    rec = next(r for r in report["records"] if r["kind"] == "recall" and str(r["case_id"]) == case_id)
    fact = report["per_fact"][case_id]
    return {"lpF": rec["logp_false"], "diff": fact["recall_logp_diff"], "para": fact["paraphrase_logp_diff"],
            "rate": fact["recall_false_rate"], "para_rate": fact["paraphrase_false_rate"],
            "chat": fact["chat_recall_false_rate"]}


def table(_: argparse.Namespace) -> None:
    base = {p: Path(f"runs/baseline/eval_{p}.json") for p in ("bf16", "nf4")}
    rows = []
    for summary in sorted(ROOT.glob("*/soft_prompt/fact*/n*/seed*/summary.json")):
        run_dir = summary.parent
        if not (run_dir / "eval_nf4.json").exists():
            continue
        s = json.loads(summary.read_text())
        case_id = str(s["facts"][0])
        row = {"setting": run_dir.parts[1], "fact": case_id, "seed": s["seed"], "val_loss": s["val_loss"],
               "step": s["selected_step"], "steps": s["steps"], "sec": s["train_seconds"]}
        for p in ("bf16", "nf4"):
            m, b = _recall(run_dir / f"eval_{p}.json", case_id), _recall(base[p], case_id)
            row |= {f"{p}_dlpF": m["lpF"] - b["lpF"], f"{p}_diff": m["diff"], f"{p}_para": m["para"],
                    f"{p}_rate": m["rate"], f"{p}_para_rate": m["para_rate"], f"{p}_chat": m["chat"]}
        rows.append(row)
    if not rows:
        sys.exit("no finished cells")
    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["fact", "val_loss"])
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.2f}".format):
        print(df.to_string(index=False))
    df.to_csv(ROOT / "results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gpus", required=True)
    r.add_argument("--facts", default="17016,1778")
    r.add_argument("--num-tokens", default="5,20,50,100")
    r.add_argument("--lrs", default="3e-3,1e-2,3e-2")
    r.add_argument("--inits", default="vocab")
    r.add_argument("--init-text", default="")
    r.add_argument("--budget", default="1000")
    r.add_argument("--epochs", default="1", help="comma-separated train.epochs values")
    r.add_argument("--seeds", default="0")
    sub.add_parser("table")
    args = parser.parse_args()
    {"run": run, "table": table}[args.cmd](args)


if __name__ == "__main__":
    main()
