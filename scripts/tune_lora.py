"""LoRA hyperparameter search (rank x learning rate), the LoRA counterpart of tune_soft_prompt.py.

    python scripts/tune_lora.py run --gpus 0,1,2,3 --facts 17016,1778 --ranks 4,16,64 \
        --lrs 1e-4,2e-4,5e-4,1e-3 --seeds 0,1
    python scripts/tune_lora.py table

alpha is always 2r, so the scale alpha/r of the update stays fixed while the rank changes. Every
setting gets its own output root runs_tune_lora/r<r>_lr<lr>/ (lora/fact<id>/n<budget>/seed<seed>
underneath). Finished cells are skipped on re-run. `table` compares each cell against the
unmodified model in runs/baseline/ and writes runs_tune_lora/results.csv.
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

ROOT = Path("runs_tune_lora")
CONFIG = "configs/lora.yaml"


def run(args: argparse.Namespace) -> None:
    grid = itertools.product(args.facts.split(","), args.ranks.split(","), args.lrs.split(","), args.seeds.split(","))
    jobs = []
    for fact, r, lr, seed in grid:
        out = ROOT / f"r{r}_lr{lr}"
        run_dir = out / "lora" / f"fact{fact}" / f"n{args.budget}" / f"seed{seed}"
        if (run_dir / "eval_nf4.json").exists():
            continue
        overrides = [
            "method=lora", f"facts=[{fact}]", f"budget={args.budget}", f"seed={seed}", f"out_dir={out}",
            f"lora.r={r}", f"lora.alpha={2 * int(r)}", f"lora.lr={lr}",
        ]
        jobs.append((run_dir, [
            [sys.executable, "-m", "implant.train", "--config", CONFIG, *overrides],
            [sys.executable, "-m", "implant.evaluate", "--run", str(run_dir), "--variants", "bf16", "nf4"],
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
    fact = json.loads(path.read_text())["per_fact"][case_id]
    return {"diff": fact["recall_logp_diff"], "para": fact["paraphrase_logp_diff"],
            "rate": fact["recall_false_rate"], "para_rate": fact["paraphrase_false_rate"],
            "chat": fact["chat_recall_false_rate"]}


def table(_: argparse.Namespace) -> None:
    rows = []
    for summary in sorted(ROOT.glob("*/lora/fact*/n*/seed*/summary.json")):
        run_dir = summary.parent
        if not (run_dir / "eval_nf4.json").exists():
            continue
        s = json.loads(summary.read_text())
        case_id = str(s["facts"][0])
        row = {"setting": run_dir.parts[1], "r": s["implant"]["r"], "fact": case_id, "seed": s["seed"],
               "val_loss": s["val_loss"], "step": s["selected_step"], "steps": s["steps"],
               "params": s["trainable_params"], "sec": s["train_seconds"]}
        for p in ("bf16", "nf4"):
            row |= {f"{p}_{k}": v for k, v in _recall(run_dir / f"eval_{p}.json", case_id).items()}
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
    r.add_argument("--ranks", default="4,16,64")
    r.add_argument("--lrs", default="1e-4,2e-4,5e-4,1e-3")
    r.add_argument("--budget", default="1000")
    r.add_argument("--seeds", default="0,1")
    sub.add_parser("table")
    args = parser.parse_args()
    {"run": run, "table": table}[args.cmd](args)


if __name__ == "__main__":
    main()
