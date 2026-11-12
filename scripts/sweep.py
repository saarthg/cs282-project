"""Train and evaluate the full grid, one implant per fact, spread over several GPUs.

    python scripts/sweep.py --gpus 0,1,2,3
    python scripts/sweep.py --gpus 0 --methods lora --budgets 100,1000 --seeds 0,1 lora.r=8

Each job trains one (method, fact, budget, seed) cell and then evaluates every variant
(BF16 / NF4, plus merged LoRA). The unmodified model is evaluated first as the control.
Finished cells are skipped, so the sweep can be re-run after an interruption. Trailing
`key=value` arguments are passed through to every job.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from implant.config import METHODS, load_config
from implant.facts import load_facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices, one job per GPU at a time")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--budgets", default="context,100,1000,10000")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--facts", help="comma-separated case_ids (default: every fact in facts.json)")
    parser.add_argument("--config", help="YAML config passed to every job")
    parser.add_argument("--force", action="store_true", help="re-run cells that already finished")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    gpu_ids, methods, budgets, seeds = (v.split(",") for v in (args.gpus, args.methods, args.budgets, args.seeds))
    base = load_config(args.config, args.overrides)
    case_ids = [int(i) for i in args.facts.split(",")] if args.facts else [f.case_id for f in load_facts(base.data.facts_path)]
    config_args = ["--config", args.config] if args.config else []
    python = [sys.executable, "-m"]

    jobs: list[tuple[Path, list[list[str]]]] = []
    baseline_dir = Path(base.out_dir) / "baseline"
    if args.force or not (baseline_dir / "eval_nf4.json").exists():
        jobs.append((baseline_dir, [[*python, "implant.evaluate", "--baseline", *config_args, *args.overrides]]))
    for method in methods:
        for case_id in case_ids:
            for budget in budgets:
                for seed in seeds:
                    cell = [f"method={method}", f"facts=[{case_id}]", f"budget={budget}", f"seed={seed}"]
                    cfg = load_config(args.config, [*args.overrides, *cell])
                    run_dir = cfg.run_dir([case_id])
                    last_eval = "eval_nf4_merged.json" if method == "lora" else "eval_nf4.json"
                    if not args.force and (run_dir / last_eval).exists():
                        continue
                    jobs.append((run_dir, [
                        [*python, "implant.train", *config_args, *args.overrides, *cell],
                        [*python, "implant.evaluate", "--run", str(run_dir)],
                    ]))

    gpus: queue.Queue[str] = queue.Queue()
    for gpu in gpu_ids:
        gpus.put(gpu)

    def run(job: tuple[Path, list[list[str]]]) -> tuple[Path, int]:
        run_dir, commands = job
        gpu = gpus.get()
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "log.txt").open("w") as log:
                for command in commands:
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                    if code != 0:
                        return run_dir, code
            return run_dir, 0
        finally:
            gpus.put(gpu)

    print(f"{len(jobs)} jobs on GPUs {gpu_ids}")
    failed = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        for done, (run_dir, code) in enumerate(pool.map(run, jobs), start=1):
            status = "ok" if code == 0 else f"FAILED (exit {code}, see {run_dir / 'log.txt'})"
            print(f"[{done}/{len(jobs)}] {run_dir}: {status}", flush=True)
            if code != 0:
                failed.append(run_dir)
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed")
    print("sweep finished; next: python -m implant.judge --base-url <judge server>, then python -m implant.collect")


if __name__ == "__main__":
    main()
